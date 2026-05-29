"""Cross-run budget tracking via the [capabilities][pydantic_ai.capabilities] system."""

from __future__ import annotations

import asyncio
import random
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import coerce_decimal_usd

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models import ModelRequestContext

__all__ = (
    'DEFAULT_BUDGET_DB_PATH',
    'BudgetGuard',
    'BudgetStore',
    'InMemoryBudgetStore',
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
