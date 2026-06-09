"""Tests for `pydantic_ai.budget`."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from pydantic_ai import Agent, UsageLimitExceeded
from pydantic_ai.budget import (
    DEFAULT_BUDGET_DB_PATH,
    AnthropicAdminCostSource,
    BudgetGuard,
    CompositeBudgetStore,
    InMemoryBudgetStore,
    OpenAICostSource,
    ProviderBudgetStore,
    ProviderCostSource,
    SQLiteBudgetStore,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

pytestmark = pytest.mark.anyio


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


async def test_in_memory_store_sums_within_window() -> None:
    store = InMemoryBudgetStore()
    now = _utcnow()
    await store.add_spend('default', Decimal('1.5'), now)
    await store.add_spend('default', Decimal('2.5'), now)

    assert await store.get_spend('default', now - timedelta(hours=1)) == Decimal('4.0')


async def test_in_memory_store_window_excludes_old_entries() -> None:
    store = InMemoryBudgetStore()
    now = _utcnow()
    await store.add_spend('default', Decimal('100'), now - timedelta(hours=48))
    await store.add_spend('default', Decimal('5'), now)

    assert await store.get_spend('default', now - timedelta(hours=24)) == Decimal('5')
    assert await store.get_spend('default', now - timedelta(hours=72)) == Decimal('105')


async def test_in_memory_store_partitions_are_isolated() -> None:
    store = InMemoryBudgetStore()
    now = _utcnow()
    await store.add_spend('client_a', Decimal('10'), now)
    await store.add_spend('client_b', Decimal('5'), now)

    assert await store.get_spend('client_a', now - timedelta(hours=1)) == Decimal('10')
    assert await store.get_spend('client_b', now - timedelta(hours=1)) == Decimal('5')
    assert await store.get_spend('client_c', now - timedelta(hours=1)) == Decimal('0')


async def test_in_memory_store_poison_propagates() -> None:
    """One unknown-cost entry within the window poisons the sum."""
    store = InMemoryBudgetStore()
    now = _utcnow()
    await store.add_spend('default', Decimal('1.0'), now)
    await store.add_spend('default', None, now)
    await store.add_spend('default', Decimal('2.0'), now)

    assert await store.get_spend('default', now - timedelta(hours=1)) is None


async def test_sqlite_store_persists_and_sums(tmp_path: Path) -> None:
    db = tmp_path / 'budget.db'
    store = SQLiteBudgetStore(str(db))
    now = _utcnow()
    await store.add_spend('default', Decimal('1.5'), now)
    await store.add_spend('default', Decimal('2.5'), now)

    assert await store.get_spend('default', now - timedelta(hours=1)) == Decimal('4.0')


async def test_sqlite_store_survives_reopen(tmp_path: Path) -> None:
    """State persists across store instances."""
    db = tmp_path / 'budget.db'
    now = _utcnow()

    store_a = SQLiteBudgetStore(str(db))
    await store_a.add_spend('default', Decimal('7'), now)

    store_b = SQLiteBudgetStore(str(db))
    assert await store_b.get_spend('default', now - timedelta(hours=1)) == Decimal('7')


async def test_sqlite_store_window_excludes_old_entries(tmp_path: Path) -> None:
    store = SQLiteBudgetStore(str(tmp_path / 'budget.db'))
    now = _utcnow()
    await store.add_spend('default', Decimal('100'), now - timedelta(hours=48))
    await store.add_spend('default', Decimal('5'), now)

    assert await store.get_spend('default', now - timedelta(hours=24)) == Decimal('5')


async def test_sqlite_store_poison_propagates(tmp_path: Path) -> None:
    store = SQLiteBudgetStore(str(tmp_path / 'budget.db'))
    now = _utcnow()
    await store.add_spend('default', Decimal('1.0'), now)
    await store.add_spend('default', None, now)

    assert await store.get_spend('default', now - timedelta(hours=1)) is None


async def test_sqlite_store_creates_parent_directory(tmp_path: Path) -> None:
    """The store creates any missing parent directories on first use."""
    nested = tmp_path / 'nested' / 'deep' / 'budget.db'
    assert not nested.parent.exists()

    store = SQLiteBudgetStore(nested)
    await store.add_spend('default', Decimal('1'), _utcnow())

    assert nested.exists()


async def test_sqlite_store_purge_before_deletes_old_rows(tmp_path: Path) -> None:
    """`purge_before` removes rows older than the cutoff and reports the count deleted."""
    store = SQLiteBudgetStore(str(tmp_path / 'budget.db'))
    now = _utcnow()
    await store.add_spend('default', Decimal('1'), now - timedelta(hours=48))
    await store.add_spend('default', Decimal('2'), now - timedelta(hours=36))
    await store.add_spend('default', Decimal('3'), now)

    deleted = await store.purge_before(now - timedelta(hours=24))

    assert deleted == 2
    assert await store.get_spend('default', now - timedelta(hours=1)) == Decimal('3')


async def test_sqlite_store_lazy_cleanup_on_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the cleanup roll fires, `add_spend` purges rows older than `retention_hours`."""
    monkeypatch.setattr('pydantic_ai.budget._CLEANUP_PROBABILITY', 1.0)

    store = SQLiteBudgetStore(str(tmp_path / 'budget.db'), retention_hours=1)
    now = _utcnow()

    await store.add_spend('default', Decimal('100'), now - timedelta(hours=48))
    await store.add_spend('default', Decimal('5'), now)
    assert await store.get_spend('default', now - timedelta(hours=48)) == Decimal('5')


async def test_sqlite_store_retention_none_disables_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`retention_hours=None` keeps rows forever, even when the cleanup roll fires."""
    monkeypatch.setattr('pydantic_ai.budget._CLEANUP_PROBABILITY', 1.0)

    store = SQLiteBudgetStore(str(tmp_path / 'budget.db'), retention_hours=None)
    now = _utcnow()
    await store.add_spend('default', Decimal('100'), now - timedelta(days=365))
    await store.add_spend('default', Decimal('1'), now)

    assert await store.get_spend('default', now - timedelta(days=400)) == Decimal('101')


def test_sqlite_store_default_retention_is_seven_days() -> None:
    """Default retention matches 7× the default `BudgetGuard.window_hours` (24h) → 168h."""
    assert SQLiteBudgetStore(':memory:').retention_hours == 24 * 7


async def test_sqlite_store_uses_default_path_when_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`SQLiteBudgetStore()` with no args resolves to `DEFAULT_BUDGET_DB_PATH`."""
    fake_default = tmp_path / 'pydantic_ai_test_home' / 'budget.db'
    monkeypatch.setattr('pydantic_ai.budget.DEFAULT_BUDGET_DB_PATH', fake_default)

    store = SQLiteBudgetStore()
    await store.add_spend('default', Decimal('1'), _utcnow())

    assert fake_default.exists()


def test_budget_guard_limit_usd_accepts_int_and_str() -> None:
    """`limit_usd` coerces `int` / `str` to `Decimal`."""
    from_int = BudgetGuard(limit_usd=1000, store=InMemoryBudgetStore())
    from_str = BudgetGuard(limit_usd='1000.00', store=InMemoryBudgetStore())

    assert from_int.limit_usd == Decimal('1000')
    assert isinstance(from_int.limit_usd, Decimal)
    assert from_str.limit_usd == Decimal('1000.00')


def test_budget_guard_limit_usd_rejects_float() -> None:
    """`float` is rejected with `TypeError`."""
    with pytest.raises(TypeError, match='does not accept float'):
        BudgetGuard(limit_usd=1000.0, store=InMemoryBudgetStore())  # pyright: ignore[reportArgumentType]


def test_sqlite_store_rejects_invalid_table_name() -> None:
    """`table` must be a valid SQL identifier to prevent injection via config."""
    with pytest.raises(ValueError, match='valid SQL identifier'):
        SQLiteBudgetStore(table='budget; DROP TABLE users')


def test_default_budget_db_path_under_home() -> None:
    """Default path lives under the user's home directory."""
    assert DEFAULT_BUDGET_DB_PATH.is_absolute()
    assert DEFAULT_BUDGET_DB_PATH.name == 'budget.db'
    assert str(Path.home()) in str(DEFAULT_BUDGET_DB_PATH)


def _ok_response(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    """500k input + 100k output gpt-4o tokens ≈ $2.25 per request."""
    return ModelResponse(
        parts=[TextPart('ok')],
        usage=RequestUsage(input_tokens=500_000, output_tokens=100_000),
    )


async def test_budget_guard_blocks_after_limit_exceeded() -> None:
    store = InMemoryBudgetStore()
    guard = BudgetGuard(limit_usd=Decimal('1.50'), store=store)
    agent = Agent(FunctionModel(_ok_response, model_name='gpt-4o'), capabilities=[guard])

    await agent.run('first')

    with pytest.raises(UsageLimitExceeded, match=re.escape('BudgetGuard limit_usd=1.50 exceeded')):
        await agent.run('second')


async def test_budget_guard_allows_runs_under_limit() -> None:
    store = InMemoryBudgetStore()
    guard = BudgetGuard(limit_usd=Decimal('100'), store=store)
    agent = Agent(FunctionModel(_ok_response, model_name='gpt-4o'), capabilities=[guard])

    await agent.run('one')
    await agent.run('two')

    spend = await store.get_spend('default', _utcnow() - timedelta(hours=1))
    assert spend is not None
    assert spend == Decimal('4.50')


async def test_budget_guard_per_tenant_partitioning() -> None:
    """Each `key_fn` partition gets its own budget bucket."""

    @dataclass
    class Deps:
        client_id: str

    store = InMemoryBudgetStore()
    guard = BudgetGuard[Deps](
        limit_usd=Decimal('1.50'),
        store=store,
        key_fn=lambda ctx: ctx.deps.client_id,
    )
    agent = Agent(FunctionModel(_ok_response, model_name='gpt-4o'), deps_type=Deps, capabilities=[guard])

    await agent.run('first', deps=Deps(client_id='client_a'))
    with pytest.raises(UsageLimitExceeded, match=r"bucket 'client_a'"):
        await agent.run('second', deps=Deps(client_id='client_a'))

    await agent.run('first', deps=Deps(client_id='client_b'))


async def test_budget_guard_window_resets_after_expiry() -> None:
    """Spend recorded before the window starts does not count toward the limit."""
    store = InMemoryBudgetStore()
    guard = BudgetGuard(limit_usd=Decimal('1.50'), window_hours=1, store=store)
    agent = Agent(FunctionModel(_ok_response, model_name='gpt-4o'), capabilities=[guard])

    await store.add_spend('default', Decimal('100'), _utcnow() - timedelta(hours=2))

    await agent.run('hello')


async def test_budget_guard_fail_closed_on_poisoned_window() -> None:
    """Default `fail_open=False` blocks requests on a poisoned window."""
    store = InMemoryBudgetStore()
    await store.add_spend('default', None, _utcnow())

    guard = BudgetGuard(limit_usd=Decimal('1000'), store=store)
    agent = Agent(FunctionModel(_ok_response, model_name='gpt-4o'), capabilities=[guard])

    with pytest.raises(UsageLimitExceeded, match='window contains unpriced events'):
        await agent.run('hello')


async def test_budget_guard_fail_open_on_poisoned_window() -> None:
    """`fail_open=True` lets requests through on a poisoned window."""
    store = InMemoryBudgetStore()
    await store.add_spend('default', None, _utcnow())

    guard = BudgetGuard(limit_usd=Decimal('1000'), store=store, fail_open=True)
    agent = Agent(FunctionModel(_ok_response, model_name='gpt-4o'), capabilities=[guard])

    await agent.run('hello')


async def test_budget_guard_records_none_for_unknown_model() -> None:
    """A response from an unknown model contributes a `None` (unpriced) entry."""
    store = InMemoryBudgetStore()
    guard = BudgetGuard(limit_usd=Decimal('1000'), store=store, fail_open=True)
    agent = Agent(
        FunctionModel(_ok_response, model_name='no-such-model-zzz'),
        capabilities=[guard],
    )

    await agent.run('hello')

    assert await store.get_spend('default', _utcnow() - timedelta(hours=1)) is None


def test_budget_guard_limit_usd_rejects_bool() -> None:
    """`bool` (an `int` subclass) is rejected rather than silently coerced to `Decimal('1')`."""
    with pytest.raises(TypeError, match='does not accept bool'):
        BudgetGuard(limit_usd=True, store=InMemoryBudgetStore())


def test_budget_guard_limit_usd_rejects_none() -> None:
    """`limit_usd` is required; `None` fails at construction (also under `python -O`, where asserts vanish)."""
    with pytest.raises(ValueError, match='limit_usd is required'):
        BudgetGuard(limit_usd=None, store=InMemoryBudgetStore())  # pyright: ignore[reportArgumentType]


async def test_in_memory_store_treats_naive_datetimes_as_utc() -> None:
    """Naive datetimes are interpreted as UTC instead of shifting by the host's local offset."""
    store = InMemoryBudgetStore()
    aware = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    naive = datetime(2026, 1, 1, 12, 0)

    await store.add_spend('default', Decimal('3'), naive)

    assert await store.get_spend('default', aware - timedelta(hours=1)) == Decimal('3')


async def test_sqlite_store_treats_naive_datetimes_as_utc(tmp_path: Path) -> None:
    """Naive datetimes passed to the SQLite store are interpreted as UTC."""
    store = SQLiteBudgetStore(str(tmp_path / 'budget.db'))
    aware = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    naive = datetime(2026, 1, 1, 12, 0)

    await store.add_spend('default', Decimal('3'), naive)

    assert await store.get_spend('default', aware - timedelta(hours=1)) == Decimal('3')


# --- Provider cost sources -------------------------------------------------------------------------


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    """An httpx client whose transport is driven by `handler` — no network, real httpx parsing."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_anthropic_cost_source_sums_and_normalizes_to_usd() -> None:
    """Anthropic `amount` is in cents and is filtered to the requested workspace."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                'data': [
                    {'results': [{'amount': '1000', 'workspace_id': 'wrk_a'}]},
                    {
                        'results': [
                            {'amount': '500', 'workspace_id': 'wrk_a'},
                            {'amount': '9999', 'workspace_id': 'wrk_b'},
                        ]
                    },
                ],
                'has_more': False,
                'next_page': None,
            },
        )

    async with _mock_client(handler) as client:
        source = AnthropicAdminCostSource(api_key='admin-key', http_client=client)
        cost = await source.get_cost(group_key='wrk_a', since=_utcnow() - timedelta(hours=1), until=_utcnow())

    assert cost == Decimal('15')
    assert captured[0].headers['x-api-key'] == 'admin-key'
    assert captured[0].headers['anthropic-version'] == '2023-06-01'
    assert captured[0].url.params['group_by[]'] == 'workspace_id'
    assert captured[0].url.params['bucket_width'] == '1d'


async def test_openai_cost_source_sums_dollar_amounts() -> None:
    """OpenAI `amount.value` is already USD and is filtered to the requested api key."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                'object': 'page',
                'data': [
                    {
                        'object': 'bucket',
                        'results': [
                            {'amount': {'value': 0.06, 'currency': 'usd'}, 'api_key_id': 'key_a'},
                            {'amount': {'value': 1.5, 'currency': 'usd'}, 'api_key_id': 'key_b'},
                        ],
                    }
                ],
                'has_more': False,
                'next_page': None,
            },
        )

    async with _mock_client(handler) as client:
        source = OpenAICostSource(api_key='admin-key', http_client=client)
        cost = await source.get_cost(group_key='key_a', since=_utcnow() - timedelta(hours=1), until=_utcnow())

    assert cost == Decimal('0.06')
    assert captured[0].headers['authorization'] == 'Bearer admin-key'
    assert captured[0].url.params['group_by[]'] == 'api_key_id'


async def test_openai_cost_source_follows_pagination() -> None:
    """`has_more=true` triggers a follow-up request carrying the `next_page` cursor."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if 'page' not in request.url.params:
            return httpx.Response(
                200,
                json={
                    'data': [{'results': [{'amount': {'value': 1.0, 'currency': 'usd'}, 'api_key_id': 'k'}]}],
                    'has_more': True,
                    'next_page': 'cursor-2',
                },
            )
        return httpx.Response(
            200,
            json={
                'data': [{'results': [{'amount': {'value': 2.0, 'currency': 'usd'}, 'api_key_id': 'k'}]}],
                'has_more': False,
                'next_page': None,
            },
        )

    async with _mock_client(handler) as client:
        source = OpenAICostSource(api_key='admin-key', http_client=client)
        cost = await source.get_cost(group_key='k', since=_utcnow() - timedelta(hours=1), until=_utcnow())

    assert cost == Decimal('3')
    assert len(captured) == 2
    assert captured[1].url.params['page'] == 'cursor-2'


async def test_cost_source_returns_none_on_http_error() -> None:
    """A non-2xx response is poison: the source cannot determine spend, so it returns `None`."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={'error': 'boom'})

    async with _mock_client(handler) as client:
        source = OpenAICostSource(api_key='admin-key', http_client=client)
        cost = await source.get_cost(group_key='k', since=_utcnow() - timedelta(hours=1), until=_utcnow())

    assert cost is None


async def test_cost_source_returns_zero_when_group_absent() -> None:
    """A successful response with no rows for the group means zero spend, not poison."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={'data': [{'results': [{'amount': {'value': 5.0, 'currency': 'usd'}, 'api_key_id': 'other'}]}]},
        )

    async with _mock_client(handler) as client:
        source = OpenAICostSource(api_key='admin-key', http_client=client)
        cost = await source.get_cost(group_key='mine', since=_utcnow() - timedelta(hours=1), until=_utcnow())

    assert cost == Decimal('0')


async def test_cost_source_creates_and_closes_its_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no `http_client`, the source creates one and closes it after the read."""
    created: list[httpx.AsyncClient] = []

    def fake_create_async_http_client(**_kwargs: Any) -> httpx.AsyncClient:
        client = _mock_client(
            lambda _request: httpx.Response(
                200,
                json={'data': [{'results': [{'amount': '4200', 'workspace_id': 'wrk'}]}]},
            )
        )
        created.append(client)
        return client

    monkeypatch.setattr('pydantic_ai.models.create_async_http_client', fake_create_async_http_client)

    source = AnthropicAdminCostSource(api_key='admin-key')
    cost = await source.get_cost(group_key='wrk', since=_utcnow() - timedelta(hours=1), until=_utcnow())

    assert cost == Decimal('42')
    assert created and created[0].is_closed


def test_anthropic_cost_source_requires_admin_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ANTHROPIC_ADMIN_API_KEY', raising=False)
    with pytest.raises(UserError, match='ANTHROPIC_ADMIN_API_KEY'):
        AnthropicAdminCostSource()


def test_openai_cost_source_requires_admin_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OPENAI_ADMIN_KEY', raising=False)
    with pytest.raises(UserError, match='OPENAI_ADMIN_KEY'):
        OpenAICostSource()


def test_cost_sources_read_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both adapters fall back to their documented environment variables."""
    monkeypatch.setenv('ANTHROPIC_ADMIN_API_KEY', 'anthropic-env')
    monkeypatch.setenv('OPENAI_ADMIN_KEY', 'openai-env')

    assert isinstance(AnthropicAdminCostSource(), AnthropicAdminCostSource)
    assert isinstance(OpenAICostSource(), OpenAICostSource)


# --- ProviderBudgetStore ---------------------------------------------------------------------------


@dataclass
class _CountingCostSource:
    """A fake `ProviderCostSource` that records how often it is queried."""

    value: Decimal | None
    calls: int = 0

    async def get_cost(self, *, group_key: str, since: datetime, until: datetime) -> Decimal | None:
        self.calls += 1
        return self.value


def test_counting_cost_source_satisfies_protocol() -> None:
    assert isinstance(_CountingCostSource(Decimal('1')), ProviderCostSource)


async def test_provider_budget_store_caches_within_ttl() -> None:
    """A second read inside the TTL window is served from cache without re-querying the source."""
    source = _CountingCostSource(Decimal('7'))
    store = ProviderBudgetStore(source, cache_ttl_seconds=60)
    since = _utcnow() - timedelta(hours=1)

    assert await store.get_spend('k', since) == Decimal('7')
    assert await store.get_spend('k', since) == Decimal('7')
    assert source.calls == 1


async def test_provider_budget_store_refetches_after_ttl_expiry() -> None:
    """With a zero TTL every read re-queries the source."""
    source = _CountingCostSource(Decimal('7'))
    store = ProviderBudgetStore(source, cache_ttl_seconds=0)
    since = _utcnow() - timedelta(hours=1)

    await store.get_spend('k', since)
    await store.get_spend('k', since)
    assert source.calls == 2


async def test_provider_budget_store_caches_none() -> None:
    """A `None` (poison) result is cached just like a numeric one."""
    source = _CountingCostSource(None)
    store = ProviderBudgetStore(source, cache_ttl_seconds=60)
    since = _utcnow() - timedelta(hours=1)

    assert await store.get_spend('k', since) is None
    assert await store.get_spend('k', since) is None
    assert source.calls == 1


async def test_provider_budget_store_add_spend_is_noop() -> None:
    """The provider is the source of truth, so writes are dropped."""
    source = _CountingCostSource(Decimal('1'))
    store = ProviderBudgetStore(source)

    assert await store.add_spend('k', Decimal('5'), _utcnow()) is None


# --- CompositeBudgetStore --------------------------------------------------------------------------


@dataclass
class _FakeStore:
    """A fake `BudgetStore` with a fixed `get_spend` and a record of `add_spend` calls."""

    value: Decimal | None
    added: list[tuple[str, Decimal | None, datetime]] = field(
        default_factory=list['tuple[str, Decimal | None, datetime]']
    )

    async def get_spend(self, key: str, since: datetime) -> Decimal | None:
        return self.value

    async def add_spend(self, key: str, amount: Decimal | None, when: datetime) -> None:
        self.added.append((key, amount, when))


async def test_composite_adds_live_self_tracking_to_authoritative_base() -> None:
    """Total = authoritative base + self-tracked spend recorded since the base last moved."""
    primary = InMemoryBudgetStore()
    authoritative = _FakeStore(Decimal('10'))
    composite = CompositeBudgetStore(primary=primary, authoritative=authoritative)
    since = _utcnow() - timedelta(hours=1)

    assert await composite.get_spend('k', since) == Decimal('10')

    await composite.add_spend('k', Decimal('2'), _utcnow())
    assert await composite.get_spend('k', since) == Decimal('12')


async def test_composite_resets_live_window_when_authoritative_moves() -> None:
    """When the billed figure catches up, previously-live spend folds into the new base."""
    primary = InMemoryBudgetStore()
    authoritative = _FakeStore(Decimal('10'))
    composite = CompositeBudgetStore(primary=primary, authoritative=authoritative)
    since = _utcnow() - timedelta(hours=1)

    await composite.get_spend('k', since)
    await composite.add_spend('k', Decimal('5'), _utcnow())

    authoritative.value = Decimal('20')
    assert await composite.get_spend('k', since) == Decimal('20')


async def test_composite_falls_back_to_primary_when_authoritative_unavailable() -> None:
    """A `None` authoritative read degrades to pure self-tracking instead of erroring."""
    primary = _FakeStore(Decimal('3'))
    authoritative = _FakeStore(None)
    composite = CompositeBudgetStore(primary=primary, authoritative=authoritative)

    assert await composite.get_spend('k', _utcnow() - timedelta(hours=1)) == Decimal('3')


async def test_composite_propagates_live_poison() -> None:
    """An unpriced live event poisons the total even when the base is known."""
    primary = _FakeStore(None)
    authoritative = _FakeStore(Decimal('10'))
    composite = CompositeBudgetStore(primary=primary, authoritative=authoritative)

    assert await composite.get_spend('k', _utcnow() - timedelta(hours=1)) is None


async def test_composite_add_spend_only_reaches_primary() -> None:
    """Writes go to the real-time `primary`; the authoritative store stays read-only."""
    primary = _FakeStore(Decimal('0'))
    authoritative = _FakeStore(Decimal('0'))
    composite = CompositeBudgetStore(primary=primary, authoritative=authoritative)

    when = _utcnow()
    await composite.add_spend('k', Decimal('1'), when)

    assert primary.added == [('k', Decimal('1'), when)]
    assert authoritative.added == []


async def test_budget_guard_blocks_on_live_spend_before_provider_sees_it() -> None:
    """Hybrid mode: `primary` self-tracking blocks in real time while the provider still reports zero."""
    source = _CountingCostSource(Decimal('0'))  # provider hasn't billed anything yet
    composite = CompositeBudgetStore(primary=InMemoryBudgetStore(), authoritative=ProviderBudgetStore(source))
    guard = BudgetGuard(limit_usd=Decimal('1.50'), store=composite)
    agent = Agent(FunctionModel(_ok_response, model_name='gpt-4o'), capabilities=[guard])

    await agent.run('first')

    with pytest.raises(UsageLimitExceeded, match=re.escape('BudgetGuard limit_usd=1.50 exceeded')):
        await agent.run('second')
