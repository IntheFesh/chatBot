"""``/不像 [正确说法]``: tell the bot that its last reply did not sound like her (R-LRN-002).

The command is the chat side of :class:`~twin.learning.dislike.NotLikeRecorder`:

* the reply is marked in ``feedback`` (``not_like``) - a negative example the weekly consolidation
  turns into a rule of the persona card (R-LRN-003);
* with a wording ("她会这么说…") the situation and the two texts are stored as a **preference pair**
  (R-LRN-002): the wording is ``chosen``, the reply is ``rejected``, the situation is the
  structured prompt sample.  Pairs are only ever used for DPO (R-LRN-004).

The answer says what was stored and, when there are enough pairs, that DPO can be done.
"""

from __future__ import annotations

from collections.abc import Callable

from twin.commands import texts
from twin.commands.registry import CommandCall
from twin.learning.dislike import NotLikeRecorder, NotLikeResult
from twin.learning.pairs import dpo_hint
from twin.ops.alerts import AlertSink


class LearnCommands:
    """The handler of ``/不像``."""

    def __init__(
        self, recorder: NotLikeRecorder, alerts: AlertSink, dpo_min_pairs: Callable[[], int]
    ) -> None:
        self._recorder = recorder
        self._alerts = alerts
        self._min_pairs = dpo_min_pairs

    async def not_like(self, call: CommandCall) -> str:
        wording = call.args.strip()
        result = await self._recorder.arecord(wording or None)
        return self.render(result, wording=bool(wording))

    def render(self, result: NotLikeResult, *, wording: bool) -> str:
        """The answer to a verdict."""
        if result.status == "unknown_reply" or (
            result.status == "no_reply" and result.reply_id is None
        ):
            return texts.NOT_LIKE_NOTHING
        if result.status == "no_reply":
            return texts.NOT_LIKE_NO_TEXT
        if result.status == "not_hers":
            return texts.NOT_LIKE_NOT_HERS
        lines = [texts.NOT_LIKE_DONE]
        if wording:
            if result.pair is not None and result.pair_created:
                lines.append(texts.NOT_LIKE_PAIR.format(total=result.pairs_total))
                self._announce_dpo(result.pairs_total)
            elif result.pair is not None:
                lines.append(texts.NOT_LIKE_PAIR_AGAIN)
            else:
                lines.append(texts.NOT_LIKE_PAIR_SKIPPED)
        else:
            lines.append(texts.NOT_LIKE_HINT)
        lines.append(texts.NOT_LIKE_WEEKLY)
        hint = dpo_hint(result.pairs_total, self._min_pairs())
        if hint:
            lines.append(hint)
        return "\n".join(lines)

    def _announce_dpo(self, total: int) -> None:
        """One alert when the pairs reach the number DPO needs (R-TRN-012)."""
        if total >= self._min_pairs():
            self._alerts.raise_alert(
                "dpo_ready",
                f"{total} preference pairs: DPO can be done (twin train export-dpo)",
                severity="info",
                detail={"pairs": total},
                dedup_key="dpo_ready",
            )
