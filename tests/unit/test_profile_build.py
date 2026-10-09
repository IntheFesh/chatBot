"""Style metrics recomputed from conversations with known regularities (R-PROF-001 to 005)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.synth_chat import ChatSpec, ChatTruth, append_texts, build_chat
from twin.ingest.corpus import conversation_messages, her_messages
from twin.profile.api import load_activity_model, load_profile
from twin.profile.builder import BuildReport, rebuild
from twin.profile.diffing import diff_metrics, summarize
from twin.profile.holdout import holdout_cutoff
from twin.profile.store import VersionError, VersionStore
from twin.services import Services
from twin.storage.profile_models import ActivityModelVersion, ProfileVersion


def built(services: Services, **spec: object) -> tuple[ChatTruth, BuildReport]:
    truth = build_chat(services, ChatSpec(**spec))  # type: ignore[arg-type]
    return truth, rebuild(services, "all", reason="test")


def rate(services: Services, party: str, name: str, key: str, scope: str = "live") -> float:
    profile = load_profile(services, scope)
    assert profile is not None
    return profile.metrics.rates(party, name)[key]


def test_known_style_numbers_are_recovered(services: Services) -> None:
    truth, report = built(services)
    assert [r.status for r in report.results] == ["created", "created"]
    profile = load_profile(services, "live")
    assert profile is not None
    metrics = profile.metrics
    # comma rate 3 % (+-0.5 points), user 50 % by construction
    assert rate(services, "her", "punct_rate", "comma") == pytest.approx(0.03, abs=0.005)
    assert rate(services, "user", "punct_rate", "comma") == pytest.approx(0.5, abs=0.04)
    # four conversations a day (+-15 %); the very first message is not an initiation
    per_day = metrics.scalar("her", "initiations_per_day")
    assert per_day == pytest.approx(4.0, rel=0.15)
    assert per_day == pytest.approx(truth.her_initiations / truth.days, rel=0.02)
    length = metrics.distribution("her", "text_length")
    burst = metrics.distribution("her", "burst_size")
    gap = metrics.distribution("her", "burst_gap_s")
    assert length is not None and burst is not None and gap is not None
    assert length.median() == pytest.approx(5.0, abs=1.0)
    assert burst.median() == pytest.approx(2.0, abs=0.5)
    assert 2 <= gap.median() <= 8
    assert metrics.scalar("her", "sticker_share") == pytest.approx(0.105, abs=0.025)
    assert metrics.scalar("her", "quote_rate") == pytest.approx(1 / 16, abs=0.02)
    assert metrics.scalar("her", "emoji_code_rate") == pytest.approx(0.04, abs=0.015)
    assert metrics.scalar("her", "laugh_rate") == pytest.approx(0.05, abs=0.02)
    laugh = metrics.distribution("her", "laugh_length")
    assert laugh is not None and laugh.median() == 3
    assert metrics.rates("her", "punct_rate")["period"] == 0.0
    assert metrics.scalar("her", "emoji_code_known_share") == 1.0
    # the user writes longer messages than she does (the difference the bot has to learn)
    user_length = metrics.distribution("user", "text_length")
    assert user_length is not None and user_length.median() > 2 * length.median()
    # slow replies on workdays between 13:00 and 17:00 show up in the per-hour latency
    hourly = metrics.hourly("her", "reply_latency_by_hour")
    assert hourly is not None
    for hour in (13, 15, 16):  # busy hours with enough answers to stand on their own
        assert hourly.get(hour).median() > 10 * hourly.get(10).median()
        assert hourly.get(hour).median() == pytest.approx(1500, rel=0.4)


def test_the_rule_text_states_the_numbers(services: Services) -> None:
    built(services)
    profile = load_profile(services, "live")
    assert profile is not None
    rules = profile.version.summary_rules
    assert "几乎不用逗号，用分条代替" in rules
    assert "单条通常" in rules and re.search(r"常连发 2–[34] 条", rules)
    assert "常用微信表情代码" in rules and "表情包" in rules
    assert profile.version.rule_lines[0].startswith("几乎不用逗号")


def test_two_windows_are_kept_and_blended_by_the_recency_weight(services: Services) -> None:
    # 42 days of data with a style change in the last ten days: more commas
    truth = build_chat(services, ChatSpec(days=42))
    last = truth.last_message_at
    assert last is not None
    extra = [
        (last + timedelta(days=1, minutes=5 * i), True, "你好，今天怎么样，吃饭了没")
        for i in range(400)
    ]
    append_texts(services, extra)
    services.settings.profile.recent_days = 10
    rebuild(services, "live", reason="test")
    profile = load_profile(services, "live")
    assert profile is not None
    data = profile.metrics.data["parties"]["her"]["punct_rate"]
    full, recent, blended = (data[key]["items"]["comma"] for key in ("full", "recent", "blended"))
    assert recent > full > 0.03
    assert blended == pytest.approx(0.6 * recent + 0.4 * full, abs=1e-4)
    assert profile.metrics.window_info("recent")["days"] <= 11
    assert profile.metrics.window_info("full")["days"] >= 42


def test_the_silences_her_openings_broke_are_a_distribution_of_hours(services: Services) -> None:
    truth, _ = built(services)
    profile = load_profile(services, "live")
    assert profile is not None
    silence = profile.metrics.distribution("her", "initiation_silence_s")
    assert silence is not None and silence.n > 0
    assert silence.median() >= services.settings.profile.segment_gap_min * 60
    assert silence.n <= truth.her_initiations


def test_a_thin_recent_window_is_ignored(services: Services) -> None:
    built(services)
    services.settings.profile.recent_days = 1
    rebuild(services, "live", reason="test", force=True)
    profile = load_profile(services, "live")
    assert profile is not None
    data = profile.metrics.data["parties"]["her"]["initiations_per_day"]
    assert data["recent"]["n"] <= 2 < data["recent"]["m"]  # a day or two is too little to trust
    assert data["blended"]["v"] == data["full"]["v"]


def test_the_stored_metrics_contain_no_message_text(services: Services) -> None:
    built(services)
    profile = load_profile(services, "live")
    assert profile is not None
    document = json.dumps(profile.metrics.data, ensure_ascii=False)
    with services.db.session() as session:
        texts = {m.text for m in session.scalars(her_messages()) if m.text and len(m.text) >= 4}
    assert texts
    assert not [text for text in texts if text in document]
    # the rule text names closed vocabularies only
    assert not [text for text in texts if text in profile.version.summary_rules]


def test_frequent_phrases_are_sealed_apart_from_the_metrics(services: Services) -> None:
    truth = build_chat(services, ChatSpec(days=14))
    last = truth.last_message_at
    assert last is not None
    greeting = "晚安宝宝"
    rows = [(last + timedelta(hours=1 + i), True, greeting) for i in range(12)]
    rows += [(last + timedelta(hours=1 + i, seconds=5), True, "宝宝在吗") for i in range(9)]
    append_texts(services, rows)
    rebuild(services, "live", reason="test")
    profile = load_profile(services, "live")
    assert profile is not None
    phrases = profile.phrases()
    assert phrases is not None
    her = phrases["her"]
    assert [greeting, 12] in her["sentences"]
    terms = {item["term"]: item for item in her["address_candidates"]}
    assert "宝宝" in terms and terms["宝宝"]["start"] + terms["宝宝"]["end"] >= 12
    assert greeting not in json.dumps(profile.metrics.data, ensure_ascii=False)
    # at rest the column is ciphertext
    with services.db.session() as session:
        row = session.scalars(select(ProfileVersion)).first()
        assert row is not None
        assert greeting.encode("utf-8") not in bytes(row.phrases_ct)
        assert greeting.encode("utf-8") not in bytes(row.metrics_ct)


def test_the_address_candidates_come_from_the_data_not_from_a_word_list(
    services: Services,
) -> None:
    truth = build_chat(services, ChatSpec(days=10))
    last = truth.last_message_at
    assert last is not None
    rows = [
        (last + timedelta(hours=2 * i + 1), True, f"小熊{'吃饭' if i % 2 else '睡觉'}")
        for i in range(14)
    ]
    append_texts(services, rows)
    rebuild(services, "live", reason="test")
    profile = load_profile(services, "live")
    assert profile is not None and profile.phrases() is not None
    candidates = [c["term"] for c in profile.phrases()["her"]["address_candidates"]]  # type: ignore[index]
    assert "小熊" in candidates


# ---------------------------------------------------------------- scopes


def test_pre_holdout_reads_nothing_from_the_held_out_period(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    from twin.profile import builder

    seen: dict[str, set[str]] = {"live": set(), "pre_holdout": set()}
    original = builder._ScopePass.feed

    def recording(self: builder._ScopePass, rec: builder.Rec) -> None:
        if self.end_ts is None or rec.ts < self.end_ts:
            seen[self.scope].add(rec.id)
        original(self, rec)

    monkeypatch.setattr(builder._ScopePass, "feed", recording)
    built(services)
    cutoff = holdout_cutoff(services)
    with services.db.session() as session:
        rows = [(m.id, m.create_time_utc, m.kind) for m in session.scalars(conversation_messages())]
    after = {i for i, at, kind in rows if at >= cutoff and kind != "system"}
    before = {i for i, at, kind in rows if at < cutoff and kind != "system"}
    assert after and before
    assert seen["pre_holdout"] == before  # exactly the messages before the cutoff ...
    assert not seen["pre_holdout"] & after  # ... and none of the held-out ones
    assert seen["live"] == before | after
    with services.db.session() as session:
        her_before = len([m for m in session.scalars(her_messages()) if m.create_time_utc < cutoff])
        her_after = len([m for m in session.scalars(her_messages()) if m.create_time_utc >= cutoff])
    assert her_after > 0
    live = load_profile(services, "live")
    pre = load_profile(services, "pre_holdout")
    assert live is not None and pre is not None
    assert pre.version.her_messages == her_before
    assert live.version.her_messages == her_before + her_after
    assert pre.version.data_range["cutoff"] == cutoff.isoformat()
    stop = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert pre.metrics.window_info("full")["end"] <= stop
    # the recent window of the pre-holdout scope ends at the cutoff, not at the newest message
    assert pre.metrics.window_info("recent")["end"] <= stop
    model = load_activity_model(services, "pre_holdout")
    assert model is not None and model.her_messages == her_before


def test_the_recent_window_of_the_live_scope_ends_at_the_newest_message(
    services: Services,
) -> None:
    truth, _ = built(services, days=30)
    profile = load_profile(services, "live")
    assert profile is not None and truth.last_message_at is not None
    stamp = profile.metrics.window_info("recent")["end"]
    end = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z")
    assert abs(end - truth.last_message_at) < timedelta(seconds=2)


# ------------------------------------------------------------------ versions


def test_a_rebuild_without_new_data_changes_nothing(services: Services) -> None:
    _, first = built(services)
    store = VersionStore(services.db, services.clock)
    ids = {r.scope: r.profile_version_id for r in first.results}
    again = rebuild(services, "all", reason="test")
    assert [r.status for r in again.results] == ["unchanged", "unchanged"]
    assert {r.scope: r.profile_version_id for r in again.results} == ids
    assert len(store.history()) == 2
    forced = rebuild(services, "live", reason="test", force=True)
    assert forced.results[0].status == "created"
    assert forced.results[0].profile_version_id != ids["live"]
    assert len(store.history("live")) == 2


def test_new_messages_make_a_new_version_with_a_parent_and_a_difference(
    services: Services,
) -> None:
    truth, first = built(services)
    store = VersionStore(services.db, services.clock)
    old = store.active_profile("live")
    assert old is not None and old.parent_id is None and old.changes == ()
    last = truth.last_message_at
    assert last is not None
    append_texts(
        services,
        [
            (last + timedelta(days=1, minutes=3 * i), True, "好，嗯，行，是的，对啊")
            for i in range(300)
        ],
    )
    second = rebuild(services, "live", reason="import")
    result = second.results[0]
    assert result.status == "created" and result.changes
    new = store.active_profile("live")
    assert new is not None and new.parent_id == old.id and new.reason == "import"
    names = {c.metric for c in new.changes if c.party == "her"}
    assert any(name.startswith("punct_rate.comma") for name in names)
    comma = next(c for c in new.changes if c.metric == "punct_rate.comma")
    assert comma.after > comma.before * 1.1
    assert all(abs(c.ratio) > 0.1 for c in new.changes)
    assert summarize(new.changes)[0].startswith("她")
    # the diff between any two stored versions can be recomputed
    assert diff_metrics(store.metrics(old.id), store.metrics(new.id))
    activity = store.activity_for_profile(new.id)
    assert activity is not None and activity.profile_version_id == new.id
    assert activity.parent_id == first.results[0].activity_version_id


def test_history_resolution_and_rollback(services: Services, clock: ManualClock) -> None:
    truth, _ = built(services)
    store = VersionStore(services.db, services.clock)
    first = store.active_profile("live")
    assert first is not None
    last = truth.last_message_at
    assert last is not None
    append_texts(
        services, [(last + timedelta(days=1, minutes=3 * i), True, "好吧，行") for i in range(120)]
    )
    clock.tick(3600)
    rebuild(services, "live", reason="import")
    second = store.active_profile("live")
    assert second is not None and second.id != first.id
    ranked = store.history("live")
    assert [v.id for v in ranked] == [second.id, first.id]
    assert [v.active for v in ranked] == [True, False]
    assert (
        store.resolve("~0", "live").id == second.id and store.resolve("~1", "live").id == first.id
    )
    # ids made within the same millisecond differ in their tail only: take the shortest prefix
    # that tells the two versions apart (a longer shared prefix is ambiguous)
    split = next(i for i, (a, b) in enumerate(zip(first.id, second.id, strict=False)) if a != b)
    assert store.resolve(first.id[: split + 1], "live").id == first.id
    assert store.resolve(second.id[: split + 1], "live").id == second.id
    with pytest.raises(VersionError, match="matches 2 versions"):
        store.resolve(first.id[:split], "live")
    with pytest.raises(VersionError, match="no version"):
        store.resolve("~5", "live")
    with pytest.raises(VersionError, match="at least 4"):
        store.resolve("01")
    with pytest.raises(VersionError, match="no profile version starts"):
        store.resolve("ZZZZZZZZ")
    # roll back: the old profile and the routine model computed with it become active
    target, activity = store.rollback(first.id)
    assert target.id == first.id and activity is not None
    assert store.active_profile("live") is not None
    assert store.active_profile("live").id == first.id  # type: ignore[union-attr]
    assert store.active_activity("live").id == activity.id  # type: ignore[union-attr]
    loaded = load_profile(services, "live")
    assert loaded is not None and loaded.version.id == first.id
    with pytest.raises(VersionError):
        store.rollback("no-such-version")
    # the next recomputation continues from the version that was active
    rebuild(services, "live", reason="manual", force=True)
    newest = store.active_profile("live")
    assert newest is not None and newest.parent_id == first.id


def test_metric_rows_and_activity_rows_are_linked_and_sealed(services: Services) -> None:
    built(services)
    with services.db.session() as session:
        profiles = session.scalars(select(ProfileVersion)).all()
        models = session.scalars(select(ActivityModelVersion)).all()
        assert {p.scope for p in profiles} == {"live", "pre_holdout"}
        assert {m.profile_version_id for m in models} == {p.id for p in profiles}
        for row in profiles:
            assert row.metrics_ct and row.summary_rules_ct
            assert row.data_range["scope"] == row.scope


def test_nothing_is_stored_without_messages(services: Services) -> None:
    report = rebuild(services, "all", reason="test")
    assert [(r.status, r.note) for r in report.results] == [
        ("skipped", "no messages imported yet"),
        ("skipped", "no messages imported yet"),
    ]
    assert load_profile(services, "live") is None
    assert load_activity_model(services, "live") is None


def test_a_scope_must_be_named_correctly(services: Services) -> None:
    with pytest.raises(ValueError, match="scope must be"):
        rebuild(services, "yesterday")
