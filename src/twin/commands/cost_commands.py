"""``/费用 [今天|本月]``: what the model calls cost, from the ledger (R-LLM-006, R-LLM-008).

The figures are the ones the budget manager works with: the **daily** account of ``cost_ledger``
(one-time batches are not part of the budget and are shown on a line of their own), for the local
day or the local month of the bot, with the split by purpose, the cache hit rate of the prompts
and what is left of the budget.  The text is the answer of the command only; ``/状态`` carries the
short form of the same numbers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from twin.commands import texts
from twin.commands.registry import CommandCall
from twin.commands.status import StatusSources
from twin.llm.ledger import LedgerStore, SpendSummary


class CostCommands:
    """The handler of ``/费用``."""

    def __init__(self, sources: StatusSources) -> None:
        self._s = sources

    async def cost(self, call: CommandCall) -> str:
        monthly = call.spec.choose(call.args) == "month" if call.args else False
        return await asyncio.to_thread(self._report, monthly)

    def _report(self, monthly: bool) -> str:
        s = self._s
        today = s.time.local_date()
        start, end = s.time.month_bounds_utc(today) if monthly else s.time.day_bounds_utc(today)
        title = (
            texts.COST_TITLE_MONTH.format(month=f"{today:%Y-%m}")
            if monthly
            else texts.COST_TITLE_DAY.format(date=today.isoformat())
        )
        lines = [title]
        ledger, budget = s.ledger, s.budget
        if ledger is None:
            return "\n".join([*lines, texts.COST_NO_CALLS])
        daily = ledger.totals(start, end, account="daily")
        if budget is not None:
            status = budget.status()
            spent = status.monthly_spent if monthly else status.daily_spent
            limit = status.monthly_budget if monthly else status.daily_budget
        else:
            spent = daily.cost_usd
            limit = s.settings.budget.monthly_usd if monthly else s.settings.budget.daily_usd
        left = max(0.0, limit - spent)
        used = f"{spent / limit:.0%}" if limit > 0 else "-"
        lines.append(texts.COST_TOTAL.format(spent=spent, budget=limit, left=left, used=used))
        if daily.calls == 0:
            lines.append(texts.COST_NO_CALLS)
        else:
            lines.append(
                texts.COST_PURPOSES.format(items="、".join(self._items(ledger, start, end, daily)))
            )
            prompt = daily.cache_hit_tokens + daily.cache_miss_tokens
            lines.append(
                texts.COST_CACHE.format(
                    ratio=f"{daily.cache_hit_ratio:.0%}", hit=daily.cache_hit_tokens, total=prompt
                )
            )
        batch = ledger.totals(start, end, account="one_time")
        if batch.calls:
            lines.append(texts.COST_ONE_TIME.format(spent=batch.cost_usd))
        if budget is not None:
            level = budget.current_level()
            meaning = texts.STATUS_BUDGET_LEVELS[min(level, len(texts.STATUS_BUDGET_LEVELS) - 1)]
            lines.append(texts.COST_LEVEL.format(level=level, meaning=meaning))
        return "\n".join(lines)

    @staticmethod
    def _items(
        ledger: LedgerStore, start: datetime, end: datetime, daily: SpendSummary
    ) -> list[str]:
        rows = ledger.by_purpose(start, end, account="daily")
        rows.sort(key=lambda row: -row.cost_usd)
        total = daily.cost_usd or 1.0
        return [
            texts.COST_PURPOSE_ITEM.format(
                purpose=texts.PURPOSE_NAMES.get(row.key, row.key),
                cost=row.cost_usd,
                share=f"{row.cost_usd / total:.0%}",
            )
            for row in rows
        ]
