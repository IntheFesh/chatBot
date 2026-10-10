"""Outbound image allow list (R-SAFE-006) and recipient binding helpers (R-CH-007)."""

from __future__ import annotations

import pytest

from tests.support.ilink import OTHER, USER, ScriptedPrompter, SetAllowList, image_bytes
from twin.channel.base import MediaNotAllowed, RecipientNotAllowed
from twin.channel.binding import (
    RecipientGuard,
    confirm_binding,
    confirm_unbind,
    mask_user_id,
)
from twin.channel.policy import (
    CompositeMediaPolicy,
    DenyAllMediaPolicy,
    ProbeImageManifest,
    ensure_media_allowed,
    sha256_hex,
)
from twin.channel.state import ChannelStateStore
from twin.storage.db import Database

# ------------------------------------------------------------- media policy


def test_only_listed_bytes_pass_and_everything_else_raises_media_not_allowed(
    db: Database,
) -> None:
    sticker = image_bytes("GIF", (1, 2, 3))
    probe = image_bytes("PNG", (4, 5, 6))
    photo = image_bytes("JPEG", (7, 8, 9))
    manifest = ProbeImageManifest(ChannelStateStore(db))
    manifest.register_bytes(probe)
    policy = CompositeMediaPolicy([SetAllowList(sha256_hex(sticker)), manifest])
    assert ensure_media_allowed(policy, sticker) == sha256_hex(sticker)
    assert ensure_media_allowed(policy, probe) == sha256_hex(probe)
    with pytest.raises(MediaNotAllowed, match="neither a sticker"):
        ensure_media_allowed(policy, photo)
    with pytest.raises(MediaNotAllowed):
        ensure_media_allowed(policy, b"")


def test_the_probe_manifest_is_persisted_and_validated(db: Database) -> None:
    state = ChannelStateStore(db)
    first = ProbeImageManifest(state)
    digest = first.register_bytes(b"synthetic picture")
    first.register(digest)  # registering twice keeps one entry
    assert ProbeImageManifest(state).hashes() == [digest]
    assert ProbeImageManifest(state).allowed(digest)
    assert not ProbeImageManifest(state).allowed("0" * 64)
    for bad in ("", "abc", digest.upper(), "g" * 64):
        with pytest.raises(ValueError, match="lowercase hex"):
            first.register(bad)


def test_the_default_policy_allows_nothing() -> None:
    assert DenyAllMediaPolicy().allowed("a" * 64) is False
    assert CompositeMediaPolicy([]).allowed("a" * 64) is False


# --------------------------------------------------------------- binding


def test_ids_are_masked_for_the_terminal() -> None:
    assert mask_user_id("o9cq1234abcd5678@im.wechat") == "o9cq****@im.wechat"
    assert mask_user_id("abc@im.wechat") == "ab****@im.wechat"
    assert "synthetic" not in mask_user_id(USER)


def test_the_guard_resolves_only_the_bound_user() -> None:
    guard = RecipientGuard(lambda: USER)
    assert guard.resolve() == USER
    assert guard.resolve(USER) == USER
    with pytest.raises(RecipientNotAllowed, match="only sends to the bound user"):
        guard.resolve(OTHER)


def test_the_guard_refuses_everything_while_nobody_is_bound() -> None:
    guard = RecipientGuard(lambda: None)
    with pytest.raises(RecipientNotAllowed, match="no user is bound"):
        guard.resolve()
    with pytest.raises(RecipientNotAllowed):
        guard.resolve(USER)


def test_binding_needs_an_explicit_yes() -> None:
    yes = ScriptedPrompter(True)
    assert confirm_binding(yes, USER, matches_expected=True) is True
    assert mask_user_id(USER) in yes.transcript
    assert USER not in yes.transcript  # the full id is never printed
    assert confirm_binding(ScriptedPrompter(False), USER, matches_expected=True) is False


def test_binding_an_account_that_did_not_scan_needs_a_typed_word() -> None:
    refused = ScriptedPrompter("no")
    assert confirm_binding(refused, OTHER, matches_expected=False) is False
    assert "NOT the account that scanned" in refused.transcript
    assert confirm_binding(ScriptedPrompter("bind"), OTHER, matches_expected=False) is True
    # a plain yes/no key press is not enough in this case
    with pytest.raises(AssertionError, match="expected text"):
        confirm_binding(ScriptedPrompter(True), OTHER, matches_expected=False)


def test_binding_without_a_comparison_says_so() -> None:
    prompter = ScriptedPrompter(True)
    assert confirm_binding(prompter, USER, matches_expected=None) is True
    assert "cannot compare" in prompter.transcript


def test_unbinding_asks_twice() -> None:
    assert confirm_unbind(ScriptedPrompter(True, "unbind"), USER) is True
    assert confirm_unbind(ScriptedPrompter(False), USER) is False
    second = ScriptedPrompter(True, "nope")
    assert confirm_unbind(second, USER) is False
    assert second.remaining() == 0
