"""Tests for `pydantic_ai.budget`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from pydantic_ai import Agent, UsageLimitExceeded
from pydantic_ai.budget import DEFAULT_BUDGET_DB_PATH, BudgetGuard, InMemoryBudgetStore, SQLiteBudgetStore
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
