"""What the engine keeps in ``conversation_state.data`` while a round is under way (R-ENG-001).

The row has fixed columns for the state, the waiting messages, the bubbles already out and the
planned send time; everything else the state machine must remember to go on after a restart is
here, as one typed record that round-trips through JSON: the quiet window's clocks, the ids that
are being answered and the ids that were screened for a crisis, the decision (when and why), the
failures so far, and the reply that is being sent (its numbers and the bubbles still to send).

None of it is ever the text of a message of the user; the bubbles still to send are the bot's own
words and the whole column is sealed like the rest of the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from twin.clock import ensure_aware
from twin.engine.decision import Decision
from twin.engine.sender import OutBubble
from twin.engine.state_store import ConversationSnapshot
from twin.engine.turns import ReplyMeta


def _moment(raw: Any) -> datetime | None:
    return ensure_aware(datetime.fromisoformat(str(raw))) if raw else None


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None


def meta_to_json(meta: ReplyMeta) -> dict[str, Any]:
    return {
        "backend": meta.backend,
        "thinking": meta.thinking,
        "plan": meta.plan,
        "cost_usd": meta.cost_usd,
        "timings_ms": meta.timings_ms,
        "actions": [dict(action) for action in meta.actions],
    }


def meta_from_json(data: dict[str, Any]) -> ReplyMeta:
    return ReplyMeta(
        backend=str(data["backend"]),
        thinking=data.get("thinking"),
        plan=data.get("plan"),
        cost_usd=data.get("cost_usd"),
        timings_ms=data.get("timings_ms"),
        actions=tuple(dict(action) for action in data.get("actions") or ()),
    )


def bubble_to_json(bubble: OutBubble) -> dict[str, Any]:
    return {"kind": bubble.kind, "text": bubble.text, "md5": bubble.sticker_md5}


def bubble_from_json(data: dict[str, Any]) -> OutBubble:
    return OutBubble(data["kind"], str(data["text"]), data.get("md5"))


@dataclass(frozen=True)
class Outgoing:
    """The reply that is being sent: its numbers, the id of its first bubble's reply, the rest."""

    meta: ReplyMeta
    unsent: tuple[OutBubble, ...]
    reply_id: str | None = None  # set when the first bubble is out
    quote_id: str | None = None  # the message the first text bubble quotes
    quote_text: str | None = None
    extra_actions: tuple[dict[str, Any], ...] = ()  # what the engine did besides the pipeline

    def to_json(self) -> dict[str, Any]:
        return {
            "meta": meta_to_json(self.meta),
            "unsent": [bubble_to_json(b) for b in self.unsent],
            "reply_id": self.reply_id,
            "quote_id": self.quote_id,
            "quote_text": self.quote_text,
            "extra_actions": [dict(a) for a in self.extra_actions],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Outgoing:
        return cls(
            meta=meta_from_json(data["meta"]),
            unsent=tuple(bubble_from_json(b) for b in data.get("unsent") or ()),
            reply_id=data.get("reply_id"),
            quote_id=data.get("quote_id"),
            quote_text=data.get("quote_text"),
            extra_actions=tuple(dict(a) for a in data.get("extra_actions") or ()),
        )

    def final_meta(self) -> ReplyMeta:
        """The numbers of the reply with the engine's own actions appended."""
        actions = (*self.meta.actions, *self.extra_actions)
        return ReplyMeta(
            backend=self.meta.backend,
            thinking=self.meta.thinking,
            plan=self.meta.plan,
            cost_usd=self.meta.cost_usd,
            timings_ms=self.meta.timings_ms,
            actions=actions,
        )


@dataclass(frozen=True)
class RoundData:
    """The typed form of ``conversation_state.data`` (see the module description)."""

    quiet_from: datetime | None = (
        None  # the last message arrived: the quiet window counts from here
    )
    collect_from: datetime | None = (
        None  # the round began collecting: ``max_wait_s`` counts from here
    )
    answering: tuple[str, ...] = ()  # ids of the messages the reply in work answers
    screened: tuple[str, ...] = ()  # ids that went through the crisis screen
    decision: Decision | None = None
    decided_for: int = 0  # how many waiting messages the decision has taken into account
    retries: int = 0  # failed attempts of this round
    cancelled: int = 0  # generations cancelled because the user wrote again
    continuation: bool = False  # the reply continues one that was cut short
    skip_quiet: bool = False  # the user asked for another try: nothing to wait for
    outgoing: Outgoing | None = field(default=None)

    @classmethod
    def of(cls, snapshot: ConversationSnapshot) -> RoundData:
        data = snapshot.data
        decision = data.get("decision")
        outgoing = data.get("outgoing")
        return cls(
            quiet_from=_moment(data.get("quiet_from")),
            collect_from=_moment(data.get("collect_from")),
            answering=tuple(str(i) for i in data.get("answering") or ()),
            screened=tuple(str(i) for i in data.get("screened") or ()),
            decision=Decision.from_json(decision) if decision else None,
            decided_for=int(data.get("decided_for", 0)),
            retries=int(data.get("retries", 0)),
            cancelled=int(data.get("cancelled", 0)),
            continuation=bool(data.get("continuation", False)),
            skip_quiet=bool(data.get("skip_quiet", False)),
            outgoing=Outgoing.from_json(outgoing) if outgoing else None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "quiet_from": _iso(self.quiet_from),
            "collect_from": _iso(self.collect_from),
            "answering": list(self.answering),
            "screened": list(self.screened),
            "decision": self.decision.to_json() if self.decision else None,
            "decided_for": self.decided_for,
            "retries": self.retries,
            "cancelled": self.cancelled,
            "continuation": self.continuation,
            "skip_quiet": self.skip_quiet,
            "outgoing": self.outgoing.to_json() if self.outgoing else None,
        }
