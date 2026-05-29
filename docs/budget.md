# Budget tracking

The [`BudgetGuard`][pydantic_ai.budget.BudgetGuard] capability enforces a **cumulative cost budget across many runs** over a rolling time window. It complements [`UsageLimits.cost_limit_usd`](agent.md#capping-cost-in-usd), which caps cost for a single run, by persisting per-request spend to a pluggable [`BudgetStore`][pydantic_ai.budget.BudgetStore] so the budget survives process restarts and is shared across workers.

Use `BudgetGuard` when you need to enforce limits like:

* "No more than \$1000 of model usage on this account per 24 hours"
* "Each customer in our SaaS gets \$10/day of model usage"
* "Staging environment caps at \$50/day, production at \$5000/day"

## Quickstart

The minimal setup is a global budget shared by every run of the agent:

```python {title="global_budget.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.budget import BudgetGuard, SQLiteBudgetStore

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        BudgetGuard(
            limit_usd=1000,
            window_hours=24,
            store=SQLiteBudgetStore(),
        ),
    ],
)
```

`limit_usd` accepts `int`, `str` (`'1000.50'`), or `Decimal` — internally normalised to `Decimal` so arithmetic stays exact. Passing a `float` raises `TypeError` to prevent silent binary-float precision errors from creeping into budget tracking.

[`SQLiteBudgetStore`][pydantic_ai.budget.SQLiteBudgetStore] with no arguments writes to [`DEFAULT_BUDGET_DB_PATH`][pydantic_ai.budget.DEFAULT_BUDGET_DB_PATH] (`~/.pydantic_ai/budget.db`), which is convenient for quick setup. Pass an explicit path to isolate budgets per project or environment — for example `SQLiteBudgetStore('/var/lib/my_app/budget.db')` for a production deployment.

Once attached, the guard checks the recorded spend in its window before each model request and raises [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] when the cumulative total has met or exceeded `limit_usd`. After each successful response it records that request's cost (computed via [`response.cost()`][pydantic_ai.messages.ModelResponse.cost]) so future runs see an up-to-date total.

## Per-tenant budgets

In a multi-tenant deployment a single agent typically serves many clients through one provider API key. To give each client an independent budget bucket, supply a `key_fn` that derives a partition key from each run's [`RunContext`][pydantic_ai.tools.RunContext]:

```python {title="per_tenant_budget.py" test="skip"}
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.budget import BudgetGuard, SQLiteBudgetStore


@dataclass
class Deps:
    client_id: str


agent = Agent[Deps](
    'anthropic:claude-sonnet-4-6',
    deps_type=Deps,
    capabilities=[
        BudgetGuard[Deps](
            limit_usd=100,
            window_hours=24,
            store=SQLiteBudgetStore(),
            key_fn=lambda ctx: ctx.deps.client_id,
        ),
    ],
)

await agent.run('summarise', deps=Deps(client_id='client_42'))
```

Each unique value returned by `key_fn` gets its own bucket; one client hitting its limit does not affect any other. The default `key_fn` returns `'default'`, so a guard configured without it behaves as a single global bucket.

## Choosing a store

[`BudgetStore`][pydantic_ai.budget.BudgetStore] is a [`Protocol`][typing.Protocol], so any backend can be plugged in. Pydantic AI ships two implementations:

| Store | Persistence | Cross-process | Use for |
|---|---|---|---|
| [`InMemoryBudgetStore`][pydantic_ai.budget.InMemoryBudgetStore] | Lost on exit | No | Tests, single-process scripts, development |
| [`SQLiteBudgetStore`][pydantic_ai.budget.SQLiteBudgetStore] | File-based | Yes (same host) | Production single-host multi-worker deployments |

For multi-host deployments (or where you want the budget to integrate with existing infrastructure) implement the `BudgetStore` protocol against Redis, Postgres, or your own datastore. Two async methods are required:

```python {title="custom_store.py" test="skip"}
from datetime import datetime
from decimal import Decimal


class MyRedisBudgetStore:
    async def get_spend(self, key: str, since: datetime) -> Decimal | None:
        ...  # sum recorded spend for `key` at or after `since`; return None if any record is unpriced

    async def add_spend(self, key: str, amount: Decimal | None, when: datetime) -> None:
        ...  # record one spend event; `amount=None` marks unknown cost
```

`BudgetGuard` always passes timezone-aware UTC datetimes; if your store persists them, store them as UTC so window comparisons stay correct across hosts.

## Combining with `UsageLimits.cost_limit_usd`

`BudgetGuard` and [`UsageLimits.cost_limit_usd`](agent.md#capping-cost-in-usd) solve different problems and are designed to work together as **defense in depth**:

| | `UsageLimits.cost_limit_usd` | `BudgetGuard` |
|---|---|---|
| Scope | Single `agent.run()` | Cumulative across runs |
| Persistence | None — resets every run | Required (via `BudgetStore`) |
| Multi-tenant | No | Yes (via `key_fn`) |
| Protects against | Runaway loop within one run | Long-tail cumulative overspend |

```python {title="defense_in_depth.py" test="skip"}
from dataclasses import dataclass

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.budget import BudgetGuard, SQLiteBudgetStore


@dataclass
class Deps:
    client_id: str


agent = Agent[Deps](
    'openai:gpt-5.2',
    deps_type=Deps,
    capabilities=[
        BudgetGuard[Deps](
            limit_usd=100,
            store=SQLiteBudgetStore(),
            key_fn=lambda ctx: ctx.deps.client_id,
        ),
    ],
)

result = await agent.run(
    'hello',
    deps=Deps(client_id='client_42'),
    usage_limits=UsageLimits(cost_limit_usd=5),
)
```

Here the guard prevents `client_42` from ever exceeding \$100 of cumulative spend in the rolling window, and `cost_limit_usd=Decimal('5')` ensures no single run can spend more than \$5 even if the guard would still allow it.

## Behavior details

### Optimistic check, post-response recording

`BudgetGuard` checks **before** the request whether the recorded spend has already crossed the limit, and records the actual cost **after** the response. This means a single request can overshoot the budget by its own cost (typically a few cents on top of a multi-dollar budget). Pre-request cost estimation would require knowing the request's token count up front, which most providers don't expose without an extra round trip — the trade-off is not worth it for the budgets `BudgetGuard` is intended to protect.

### Unpriced models — `fail_open`

When a request uses a model whose pricing is not in [`genai-prices`][pydantic_ai.messages.ModelResponse.cost] (for example, self-hosted models served via an OpenAI-compatible endpoint), `BudgetGuard` records that event as having unknown cost. Once any window contains an unpriced event, the sum is **poisoned** — the guard cannot know whether the limit has actually been exceeded.

The `fail_open` field controls the behavior in that case:

* `fail_open=False` (default) — block the request with `UsageLimitExceeded`. Conservative; surfaces pricing gaps loudly so they get fixed.
* `fail_open=True` — allow the request through. Useful in mixed fleets where some models legitimately lack pricing data and under-enforcement is preferred over breaking the agent.

### Concurrency and overshoot

Both shipped stores serialize their operations, so concurrent runs of the same partition produce consistent sums. `SQLiteBudgetStore` opens its database in WAL mode with a busy timeout, so multiple worker processes on one host can read and write concurrently without tripping over `database is locked`. Even so, two concurrent requests can both pass the pre-check at the same time (each sees the spend before the other writes) and together overshoot the limit by the cost of one request. This is the same trade-off as a single-process pre-check and matches what most budget systems do — accurate accounting after the fact, not perfect synchronous enforcement.

If perfect synchronous enforcement matters more than throughput, implement `BudgetStore` against a system that supports check-and-set semantics (Redis with Lua, or Postgres with a serializable transaction) and add the conditional update inside `add_spend`.

### Retention and table growth

[`SQLiteBudgetStore`][pydantic_ai.budget.SQLiteBudgetStore] records one row per model request. To keep the table from growing without bound, rows older than `retention_hours` are purged lazily on writes — about one purge per ~1000 writes, so the amortised cost is negligible.

The default `retention_hours=168` (7 days) is sized at 7× the default `BudgetGuard.window_hours` of 24h, so a rolling window query never misses a row that is about to be purged. If you increase `window_hours`, increase retention to match:

```python {title="long_window_retention.py" test="skip"}
from pydantic_ai.budget import BudgetGuard, SQLiteBudgetStore

# 30-day window → keep at least ~60 days of history.
guard = BudgetGuard(
    limit_usd=10_000,
    window_hours=24 * 30,
    store=SQLiteBudgetStore(retention_hours=24 * 60),
)
```

Pass `retention_hours=None` to disable cleanup entirely — only do this if you purge externally (e.g. via a cron job calling [`purge_before`][pydantic_ai.budget.SQLiteBudgetStore.purge_before]):

```python {title="manual_retention.py" test="skip"}
from datetime import datetime, timedelta, timezone

from pydantic_ai.budget import SQLiteBudgetStore

store = SQLiteBudgetStore(retention_hours=None)
# ... later, from a scheduled job:
deleted = await store.purge_before(datetime.now(tz=timezone.utc) - timedelta(days=90))
```

Custom stores (Redis, Postgres) typically have better-suited native retention primitives — Redis `EXPIRE`, Postgres partitioning — so the `BudgetStore` protocol intentionally does not require a retention method.

### Window semantics

The window is a **rolling** wall-clock interval ending at the moment of the check. `window_hours=24` means "spend recorded in the last 24 hours". Calendar-day windows (for example, "midnight to midnight in `America/New_York`") are out of scope for the shipped stores — implement them in a custom `BudgetStore` if needed.
