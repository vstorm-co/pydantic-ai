"""Cross-run budget tracking via the [capabilities][pydantic_ai.capabilities] system."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import UsageLimitExceeded, UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import coerce_decimal_usd

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models import ModelRequestContext

__all__ = (
    'DEFAULT_BUDGET_DB_PATH',
    'AnthropicAdminCostSource',
    'BudgetGuard',
    'BudgetStore',
    'CompositeBudgetStore',
    'InMemoryBudgetStore',
    'OpenAICostSource',
    'ProviderBudgetStore',
    'ProviderCostSource',
    'SQLiteBudgetStore',
)

DEFAULT_BUDGET_DB_PATH = Path.home() / '.pydantic_ai' / 'budget.db'
"""Default path for [`SQLiteBudgetStore`][pydantic_ai.budget.SQLiteBudgetStore]."""

_DEFAULT_RETENTION_HOURS: float = 24 * 7
"""Default SQLite retention — 7× the default 24h window."""

_CLEANUP_PROBABILITY: float = 0.001
"""Probability that a write triggers a lazy purge of expired rows."""

_BUSY_TIMEOUT_SECONDS: float = 30.0
"""How long a connection waits for a contended lock before raising `database is locked`."""

_VALID_TABLE_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
"""SQLite identifiers are interpolated into SQL, so `table` must match this safe pattern."""


def _default_key_fn(ctx: RunContext[AgentDepsT]) -> str:
    return 'default'


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _ensure_aware(when: datetime) -> datetime:
    """Treat a naive datetime as UTC so window math doesn't silently shift by the host's local offset."""
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


@runtime_checkable
class BudgetStore(Protocol):
    """Persistence backend for [`BudgetGuard`][pydantic_ai.budget.BudgetGuard]."""

    async def get_spend(self, key: str, since: datetime) -> Decimal | None:
        """Return total spend for `key` at or after `since`, or `None` if any record is unpriced."""
        ...

    async def add_spend(self, key: str, amount: Decimal | None, when: datetime) -> None:
        """Record a spend event (`amount=None` marks unknown cost)."""
        ...


class InMemoryBudgetStore:
    """In-memory store for tests, single-process scripts, and short-lived workers."""

    def __init__(self) -> None:
        self._entries: dict[str, list[tuple[datetime, Decimal | None]]] = {}
        self._lock = asyncio.Lock()

    async def get_spend(self, key: str, since: datetime) -> Decimal | None:
        since = _ensure_aware(since)
        async with self._lock:
            entries = self._entries.get(key)
            if not entries:
                return Decimal(0)
            total: Decimal | None = Decimal(0)
            for when, amount in entries:
                if when < since:
                    continue
                if amount is None:
                    total = None
                elif total is not None:
                    total += amount
            return total

    async def add_spend(self, key: str, amount: Decimal | None, when: datetime) -> None:
        when = _ensure_aware(when)
        async with self._lock:
            self._entries.setdefault(key, []).append((when, amount))


class SQLiteBudgetStore:
    """SQLite-backed store for single-host multi-worker deployments."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        table: str = 'pydantic_ai_budget',
        retention_hours: float | None = _DEFAULT_RETENTION_HOURS,
    ) -> None:
        if not _VALID_TABLE_NAME.match(table):
            raise ValueError(f'table must be a valid SQL identifier, got {table!r}')
        self._path = Path(path) if path is not None else DEFAULT_BUDGET_DB_PATH
        self._table = table
        self._retention_hours = retention_hours
        self._initialised = False
        self._lock = asyncio.Lock()

    @property
    def retention_hours(self) -> float | None:
        """How long entries are kept before lazy cleanup deletes them (`None` disables cleanup)."""
        return self._retention_hours

    @contextmanager
    def _open_conn(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived autocommit connection.

        WAL lets concurrent readers proceed alongside a writer, and the busy timeout makes
        contending writers wait for the lock instead of immediately raising `database is locked` —
        both matter for the documented single-host multi-worker use case.
        """
        conn = sqlite3.connect(self._path, isolation_level=None, timeout=_BUSY_TIMEOUT_SECONDS)
        try:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute(f'PRAGMA busy_timeout={int(_BUSY_TIMEOUT_SECONDS * 1000)}')
            yield conn
        finally:
            conn.close()

    def _ensure_schema_sync(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._open_conn() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bucket TEXT NOT NULL,
                    cost_usd TEXT,
                    timestamp_unix REAL NOT NULL
                )
                """
            )
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS ix_{self._table}_bucket_timestamp ON {self._table}(bucket, timestamp_unix)'
            )

    async def _ensure_schema(self) -> None:
        if self._initialised:
            return
        async with self._lock:
            if self._initialised:  # pragma: no cover
                return
            await asyncio.to_thread(self._ensure_schema_sync)
            self._initialised = True

    def _get_spend_sync(self, key: str, since_unix: float) -> Decimal | None:
        with self._open_conn() as conn:
            cursor = conn.execute(
                f'SELECT cost_usd FROM {self._table} WHERE bucket = ? AND timestamp_unix >= ?',
                (key, since_unix),
            )
            total = Decimal(0)
            for (cost_str,) in cursor:
                if cost_str is None:
                    return None
                total += Decimal(cost_str)
            return total

    async def get_spend(self, key: str, since: datetime) -> Decimal | None:
        await self._ensure_schema()
        return await asyncio.to_thread(self._get_spend_sync, key, _ensure_aware(since).timestamp())

    def _add_spend_sync(self, key: str, amount_str: str | None, timestamp_unix: float) -> None:
        with self._open_conn() as conn:
            conn.execute(
                f'INSERT INTO {self._table} (bucket, cost_usd, timestamp_unix) VALUES (?, ?, ?)',
                (key, amount_str, timestamp_unix),
            )
            if self._retention_hours is not None and random.random() < _CLEANUP_PROBABILITY:
                cutoff = timestamp_unix - self._retention_hours * 3600
                conn.execute(f'DELETE FROM {self._table} WHERE timestamp_unix < ?', (cutoff,))

    async def add_spend(self, key: str, amount: Decimal | None, when: datetime) -> None:
        await self._ensure_schema()
        amount_str = str(amount) if amount is not None else None
        await asyncio.to_thread(self._add_spend_sync, key, amount_str, _ensure_aware(when).timestamp())

    def _purge_before_sync(self, before_unix: float) -> int:
        with self._open_conn() as conn:
            cursor = conn.execute(f'DELETE FROM {self._table} WHERE timestamp_unix < ?', (before_unix,))
            return cursor.rowcount

    async def purge_before(self, before: datetime) -> int:
        """Delete every row with `ts < before`; returns the number of rows deleted."""
        await self._ensure_schema()
        return await asyncio.to_thread(self._purge_before_sync, _ensure_aware(before).timestamp())


_ANTHROPIC_COST_URL = 'https://api.anthropic.com/v1/organizations/cost_report'
"""Anthropic organization-level cost report admin endpoint."""

_OPENAI_COSTS_URL = 'https://api.openai.com/v1/organization/costs'
"""OpenAI organization-level costs admin endpoint."""

_MAX_COST_PAGES = 1000
"""Upper bound on `next_page` follows, guarding against a misbehaving pagination loop."""

_QueryParams = dict[str, 'str | int | list[str]']
"""Query parameters accepted by the admin cost endpoints (httpx encodes lists as repeated keys)."""


def _format_rfc3339(when: datetime) -> str:
    return _ensure_aware(when).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _floor_to_utc_day(when: datetime) -> datetime:
    """Both cost APIs bucket by whole UTC days, so the window start is floored to midnight UTC."""
    return _ensure_aware(when).astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


async def _fetch_paginated_cost(
    *,
    url: str,
    headers: dict[str, str],
    params: _QueryParams,
    parse_page: Callable[[Any], tuple[Decimal, bool, str | None]],
    http_client: httpx.AsyncClient | None,
) -> Decimal | None:
    """Sum cost across every page, returning `None` if the read fails for any reason (poison).

    JSON numbers are parsed with `parse_float=Decimal` so monetary values never round-trip through
    binary float. Any transport, status, or schema failure collapses to `None` so the caller can
    decide (via `fail_open`) whether to block or allow the request.
    """
    from pydantic_ai.models import create_async_http_client

    client = http_client or create_async_http_client()
    own_client = http_client is None
    try:
        total = Decimal(0)
        current = dict(params)
        for _ in range(_MAX_COST_PAGES):
            response = await client.get(url, params=current, headers=headers)
            response.raise_for_status()
            payload = json.loads(response.content, parse_float=Decimal)
            page_total, has_more, next_page = parse_page(payload)
            total += page_total
            if not has_more or not next_page:
                return total
            current['page'] = next_page
        return total  # pragma: no cover
    except (httpx.HTTPError, ValidationError, ValueError, KeyError):
        return None
    finally:
        if own_client:
            await client.aclose()


@runtime_checkable
class ProviderCostSource(Protocol):
    """Fetches authoritative spend (already normalized to USD) from a provider's admin cost API."""

    async def get_cost(self, *, group_key: str, since: datetime, until: datetime) -> Decimal | None:
        """Return authoritative cost in USD for `group_key` over `[since, until)`, or `None` if unavailable."""
        ...


class _AnthropicCostResultModel(BaseModel):
    model_config = ConfigDict(extra='ignore')

    amount: Decimal
    """Cost in minor units (cents); divided by 100 to get USD."""

    workspace_id: str | None = None
    """`None` for the organization's default workspace."""


class _AnthropicCostBucketModel(BaseModel):
    model_config = ConfigDict(extra='ignore')

    results: list[_AnthropicCostResultModel] = []


class _AnthropicCostReportModel(BaseModel):
    model_config = ConfigDict(extra='ignore')

    data: list[_AnthropicCostBucketModel] = []
    has_more: bool = False
    next_page: str | None = None


class AnthropicAdminCostSource:
    """Authoritative spend from Anthropic's [organization cost report](https://docs.anthropic.com/en/api/admin-api/usage-cost/get-cost-report) admin API.

    The cost report groups only by `workspace_id`, so per-tenant attribution requires a separate
    Anthropic workspace per tenant, with `group_key` set to that workspace's id. Cost is reported
    in whole UTC days, so spend is day-granular regardless of the requested window.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        bucket_width: str = '1d',
    ) -> None:
        """Build the cost source.

        Args:
            api_key: Anthropic **admin** API key. Falls back to the `ANTHROPIC_ADMIN_API_KEY`
                environment variable; a missing key raises [`UserError`][pydantic_ai.exceptions.UserError].
            http_client: Optional shared client. When omitted, a client is created and closed per request.
            bucket_width: Cost report bucket width (the API currently only supports `'1d'`).
        """
        key = api_key or os.getenv('ANTHROPIC_ADMIN_API_KEY')
        if not key:
            raise UserError(
                'Set the `ANTHROPIC_ADMIN_API_KEY` environment variable or pass it via '
                '`AnthropicAdminCostSource(api_key=...)` to read authoritative cost from Anthropic.'
            )
        self._api_key = key
        self._http_client = http_client
        self._bucket_width = bucket_width

    async def get_cost(self, *, group_key: str, since: datetime, until: datetime) -> Decimal | None:
        def parse_page(payload: Any) -> tuple[Decimal, bool, str | None]:
            report = _AnthropicCostReportModel.model_validate(payload)
            total = Decimal(0)
            for bucket in report.data:
                for result in bucket.results:
                    if result.workspace_id == group_key:
                        total += result.amount / 100
            return total, report.has_more, report.next_page

        params: _QueryParams = {
            'starting_at': _format_rfc3339(_floor_to_utc_day(since)),
            'ending_at': _format_rfc3339(until),
            'group_by[]': ['workspace_id'],
            'bucket_width': self._bucket_width,
        }
        return await _fetch_paginated_cost(
            url=_ANTHROPIC_COST_URL,
            headers={'x-api-key': self._api_key, 'anthropic-version': '2023-06-01'},
            params=params,
            parse_page=parse_page,
            http_client=self._http_client,
        )


class _OpenAICostAmountModel(BaseModel):
    model_config = ConfigDict(extra='ignore')

    value: Decimal
    """Cost already expressed in USD."""


class _OpenAICostResultModel(BaseModel):
    model_config = ConfigDict(extra='ignore')

    amount: _OpenAICostAmountModel
    api_key_id: str | None = None


class _OpenAICostBucketModel(BaseModel):
    model_config = ConfigDict(extra='ignore')

    results: list[_OpenAICostResultModel] = []


class _OpenAICostsPageModel(BaseModel):
    model_config = ConfigDict(extra='ignore')

    data: list[_OpenAICostBucketModel] = []
    has_more: bool = False
    next_page: str | None = None


class OpenAICostSource:
    """Authoritative spend from OpenAI's [organization costs](https://platform.openai.com/docs/api-reference/usage/costs) admin API.

    The costs endpoint can group by `api_key_id`, so per-tenant attribution works directly when each
    tenant uses a distinct API key, with `group_key` set to that key's id. Cost is reported in whole
    UTC days, so spend is day-granular regardless of the requested window.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        bucket_width: str = '1d',
    ) -> None:
        """Build the cost source.

        Args:
            api_key: OpenAI **admin** API key. Falls back to the `OPENAI_ADMIN_KEY` environment
                variable; a missing key raises [`UserError`][pydantic_ai.exceptions.UserError].
            http_client: Optional shared client. When omitted, a client is created and closed per request.
            bucket_width: Costs bucket width (the API currently only supports `'1d'`).
        """
        key = api_key or os.getenv('OPENAI_ADMIN_KEY')
        if not key:
            raise UserError(
                'Set the `OPENAI_ADMIN_KEY` environment variable or pass it via '
                '`OpenAICostSource(api_key=...)` to read authoritative cost from OpenAI.'
            )
        self._api_key = key
        self._http_client = http_client
        self._bucket_width = bucket_width

    async def get_cost(self, *, group_key: str, since: datetime, until: datetime) -> Decimal | None:
        def parse_page(payload: Any) -> tuple[Decimal, bool, str | None]:
            page = _OpenAICostsPageModel.model_validate(payload)
            total = Decimal(0)
            for bucket in page.data:
                for result in bucket.results:
                    if result.api_key_id == group_key:
                        total += result.amount.value
            return total, page.has_more, page.next_page

        params: _QueryParams = {
            'start_time': int(_floor_to_utc_day(since).timestamp()),
            'end_time': int(_ensure_aware(until).timestamp()),
            'group_by[]': ['api_key_id'],
            'bucket_width': self._bucket_width,
        }
        return await _fetch_paginated_cost(
            url=_OPENAI_COSTS_URL,
            headers={'Authorization': f'Bearer {self._api_key}'},
            params=params,
            parse_page=parse_page,
            http_client=self._http_client,
        )


class ProviderBudgetStore:
    """A read-only [`BudgetStore`][pydantic_ai.budget.BudgetStore] backed by a provider's authoritative cost API.

    `get_spend` is called before every model request, so the authoritative read is cached per key for
    `cache_ttl_seconds` to stay within admin-API rate limits. The cache is keyed by partition only (not
    by `since`): the rolling window start slides by milliseconds between calls, so caching on it would
    never hit. `add_spend` is a no-op — the provider, not this store, is the source of truth.
    """

    def __init__(self, source: ProviderCostSource, *, cache_ttl_seconds: float = 60.0) -> None:
        self._source = source
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, tuple[float, Decimal | None]] = {}
        self._lock = asyncio.Lock()

    async def get_spend(self, key: str, since: datetime) -> Decimal | None:
        """Return cached authoritative spend for `key`, refreshing from the source once per TTL."""
        now = time.monotonic()
        async with self._lock:
            cached = self._cache.get(key)
            if cached is not None and now - cached[0] < self._cache_ttl_seconds:
                return cached[1]
        cost = await self._source.get_cost(group_key=key, since=_ensure_aware(since), until=_now_utc())
        async with self._lock:
            self._cache[key] = (time.monotonic(), cost)
        return cost

    async def add_spend(self, key: str, amount: Decimal | None, when: datetime) -> None:
        """No-op: the provider's billing system is the source of truth."""
        return


class CompositeBudgetStore:
    """Hybrid [`BudgetStore`][pydantic_ai.budget.BudgetStore]: a real-time self-tracking guard corrected by authoritative billing.

    `primary` (e.g. [`SQLiteBudgetStore`][pydantic_ai.budget.SQLiteBudgetStore]) sees every request the
    instant it happens; `authoritative` (a [`ProviderBudgetStore`][pydantic_ai.budget.ProviderBudgetStore])
    supplies the invoice-accurate figure on a TTL. The reported total is `authoritative + live`, where
    `live` is the self-tracked spend recorded since the authoritative value last moved — so the total is
    both fresh enough to block in real time and consistent with the provider's billing over time.

    When the authoritative read is unavailable (`None`), this degrades to pure self-tracking from
    `primary` without erroring. A `None` from `primary`'s live window is treated as poison and propagates.
    """

    def __init__(self, *, primary: BudgetStore, authoritative: BudgetStore) -> None:
        self._primary = primary
        self._authoritative = authoritative
        self._lock = asyncio.Lock()
        self._reconcile: dict[str, tuple[Decimal, datetime]] = {}

    async def get_spend(self, key: str, since: datetime) -> Decimal | None:
        base = await self._authoritative.get_spend(key, since)
        if base is None:
            return await self._primary.get_spend(key, since)
        async with self._lock:
            previous = self._reconcile.get(key)
            if previous is None or previous[0] != base:
                # The authoritative figure moved (or this is the first read): everything `primary`
                # recorded up to now is considered billed, so the live window restarts here.
                reconcile_since = _now_utc()
                self._reconcile[key] = (base, reconcile_since)
            else:
                reconcile_since = previous[1]
        live = await self._primary.get_spend(key, reconcile_since)
        if live is None:
            return None
        return base + live

    async def add_spend(self, key: str, amount: Decimal | None, when: datetime) -> None:
        """Record into `primary` only; the authoritative store is read-only."""
        await self._primary.add_spend(key, amount, when)


@dataclass(init=False)
class BudgetGuard(AbstractCapability[AgentDepsT]):
    """Capability that enforces a cumulative cost budget across runs over a rolling time window."""

    limit_usd: Decimal
    """Maximum cumulative cost in USD allowed within the rolling window."""

    store: BudgetStore
    """Backend that persists spend events across processes."""

    window_hours: float = 24
    """Length of the rolling window in hours."""

    key_fn: Callable[[RunContext[AgentDepsT]], str] = _default_key_fn
    """Derives the budget partition key from `RunContext` (default returns `'default'`)."""

    fail_open: bool = False
    """If `True`, allow requests through when the window contains unpriced events."""

    def __init__(
        self,
        limit_usd: Decimal | int | str,
        store: BudgetStore,
        *,
        window_hours: float = 24,
        key_fn: Callable[[RunContext[AgentDepsT]], str] = _default_key_fn,
        fail_open: bool = False,
    ) -> None:
        coerced = coerce_decimal_usd(limit_usd, 'limit_usd')
        if coerced is None:
            raise ValueError('limit_usd is required')
        self.limit_usd = coerced
        self.store = store
        self.window_hours = window_hours
        self.key_fn = key_fn
        self.fail_open = fail_open

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Block the request if recorded spend in the window meets or exceeds `limit_usd`."""
        key = self.key_fn(ctx)
        since = _now_utc() - timedelta(hours=self.window_hours)
        spend = await self.store.get_spend(key, since)
        if spend is None:
            if self.fail_open:
                return request_context
            raise UsageLimitExceeded(
                f'BudgetGuard cannot evaluate spend for bucket {key!r}: window contains '
                f'unpriced events. Set fail_open=True to allow these requests through.'
            )
        if spend >= self.limit_usd:
            raise UsageLimitExceeded(
                f'BudgetGuard limit_usd={self.limit_usd} exceeded for bucket {key!r} '
                f'over the last {self.window_hours}h (current spend: {spend})'
            )
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Record this response's cost (or `None` if unpriced) into the store."""
        key = self.key_fn(ctx)
        await self.store.add_spend(key, response.cost_or_none(), _now_utc())
        return response
