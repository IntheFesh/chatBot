"""R-SCOPE-008: the non-goals of version 1 left no half-finished product behind (round 16).

The non-goals: voice or video calls, sending voice messages, generating her photos, sending a
message to anyone but the user, group chats.  "Not done" has to mean *not there*: no command, no
chat command, no channel method, no library, no function with such a name - a half-finished
feature is worse than none because it looks like one.  The scans below fail if one appears; what
the program does about each non-goal on purpose (turning down a request in her voice, skipping
group chats on import, refusing foreign recipients and pictures that are not stickers) is tested
where it lives and cited here.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.support.cli_tree import command_tree, group_paths
from tests.support.clock import ManualClock
from tests.support.commands_world import CommandWorld, open_world
from tests.support.embedding import HashingBackend
from tests.support.nodes import missing_nodes
from twin.channel.base import Channel
from twin.commands.rating import rating_command
from twin.schedule.proactive.store import RatingStore
from twin.services import Services

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "twin"

# words that name a feature of a non-goal (not the reading of such records in an import, which
# is ingest.normalize turning a call into the words "[通话 37 分钟]")
ENGLISH_WORDS = {
    "voice",
    "speech",
    "tts",
    "asr",
    "whisper",
    "dial",
    "phone",
    "video",
    "videos",
    "moments",
    "timeline",
    "group",
    "groups",
    "chatroom",
    "broadcast",
    "friend",
    "friends",
    "photo",
    "photos",
    "selfie",
    "portrait",
    "avatar",
    "draw",
    "paint",
    "diffusion",
}
CHINESE_WORDS = re.compile(
    "语音|通话|电话|视频|朋友圈|群聊|拍照|自拍|头像|画图|绘图|生成.{0,4}(?:图|照片)"
)
_HUMP = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def names_a_non_goal(text: str) -> bool:
    """Does ``text`` (a command, an option, a sentence) name a feature of a non-goal?

    English is compared word by word (``llm_min_calls`` and ``recall_facts`` are not features),
    Chinese by substring.  ``call`` alone is not in the list because "calls of the model" are
    everywhere; a command or option named ``call`` is looked for separately.
    """
    words = re.split(r"[^A-Za-z]+", _HUMP.sub(" ", text).lower())
    return bool(ENGLISH_WORDS & set(words)) or CHINESE_WORDS.search(text) is not None


NON_GOAL_LIBRARIES = {
    "whisper",
    "faster_whisper",
    "speech_recognition",
    "vosk",
    "pyttsx3",
    "gtts",
    "edge_tts",
    "TTS",
    "diffusers",
    "stable_diffusion",
    "pyaudio",
    "sounddevice",
    "cv2",
    "moviepy",
    "pyautogui",
    "itchat",
    "wxpy",
    "pywechat",
}
NON_GOAL_FUNCTIONS = re.compile(
    r"^(send|make|start|place|answer|join|create|post|generate|render|synthes[ie]ze|dial|record)"
    r"_(voice|audio|video|call|group|moment|moments|photo|selfie|portrait|speech|tts)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------- the command line


def test_no_command_or_option_of_twin_belongs_to_a_non_goal() -> None:
    offenders = []
    for path, info in command_tree().items():
        words = [*path, *(option.lstrip("-") for option in info.options)]
        for word in words:
            if names_a_non_goal(word) or word in {"call", "calls"}:
                offenders.append(f"twin {info.name}: {word}")
    for group in group_paths():
        if any(names_a_non_goal(part) for part in group):
            offenders.append(f"twin {' '.join(group)}")
    assert offenders == []


def test_the_help_text_of_no_command_promises_a_non_goal() -> None:
    offenders = [
        f"twin {info.name}: {info.help}"
        for info in command_tree().values()
        if re.search(
            r"voice call|video call|group chat|moments|语音通话|视频通话|群聊|朋友圈", info.help
        )
    ]
    assert offenders == []


# ------------------------------------------------------------------- the chat commands


@pytest.fixture
async def world(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> AsyncIterator[CommandWorld]:
    async with open_world(services, clock, start=datetime(2026, 10, 9, 12, 0, tzinfo=UTC)) as built:
        built.router.register(
            rating_command(
                RatingStore(services.db, services.clock), services.clock, built.rig.kit.time
            )
        )
        yield built


async def test_no_chat_command_belongs_to_a_non_goal(world: CommandWorld) -> None:
    """The table of R-CMD-002 has no command to call, send a voice message, or add a group."""
    offenders = []
    for spec in world.router.registry.specs():
        for text in (spec.name, *spec.aliases, spec.summary, spec.syntax, spec.example):
            if names_a_non_goal(text):
                offenders.append(f"/{spec.name}: {text}")
    assert offenders == []


async def test_the_help_lists_only_the_commands_of_the_table(world: CommandWorld) -> None:
    reply = await world.reply("/帮助")
    assert not names_a_non_goal(reply), reply


# ------------------------------------------------------------------------ the channel


def test_a_channel_can_send_text_pictures_and_the_typing_mark_and_nothing_else() -> None:
    sending = {
        name
        for name, member in vars(Channel).items()
        if getattr(member, "__isabstractmethod__", False) and name.startswith("send")
    }
    assert sending == {"send_text", "send_image", "send_typing"}
    abstract = {
        name
        for name, member in vars(Channel).items()
        if getattr(member, "__isabstractmethod__", False)
    }
    assert abstract == {
        "start",
        "stop",
        "incoming",
        "send_text",
        "send_image",
        "send_typing",
        "capabilities",
        "session_state",
    }


def test_the_wechat_endpoints_the_program_may_call_have_no_voice_video_or_group_function() -> None:
    """The list of the endpoints in ``test_channel_rules`` (the program calls no others)."""
    tree = ast.parse((ROOT / "tests" / "unit" / "test_channel_rules.py").read_text("utf-8"))
    endpoints: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ALLOWED_ENDPOINTS" for t in node.targets
        ):
            endpoints = set(ast.literal_eval(node.value))
    assert "sendmessage" in endpoints
    assert not any(names_a_non_goal(endpoint) for endpoint in endpoints)


# -------------------------------------------------------------------------- the source


def test_no_function_or_class_of_the_program_implements_a_non_goal() -> None:
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and (
                NON_GOAL_FUNCTIONS.match(node.name)
                or NON_GOAL_FUNCTIONS.match(re.sub(r"(?<!^)(?=[A-Z])", "_", node.name))
            ):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno} {node.name}")
    assert offenders == []


def test_no_library_for_speech_video_calls_or_picture_generation_is_used_or_required() -> None:
    imported = set()
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imported.add(node.module.split(".")[0])
    assert imported & NON_GOAL_LIBRARIES == set()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    required = {
        re.split(r"[<>=!~\[ ;]", dependency, maxsplit=1)[0].lower().replace("-", "_")
        for dependency in pyproject["project"]["dependencies"]
    }
    assert required & {name.lower().replace("-", "_") for name in NON_GOAL_LIBRARIES} == set()


def test_the_default_model_settings_do_not_ask_for_audio_or_images_to_be_made() -> None:
    from twin.config.settings import Settings

    names = " ".join(Settings.model_fields)
    assert not names_a_non_goal(names)
    text = (ROOT / "config" / "config.example.yaml").read_text(encoding="utf-8")
    keys = " ".join(re.findall(r"^\s*([A-Za-z_]+):", text, re.MULTILINE))
    assert not names_a_non_goal(keys)


# ------------------------------------------- what the program does about each non-goal


NON_GOAL_TESTS = {
    "voice and video calls, voice messages, photos: a promise is turned down in her voice": [
        "tests/unit/test_wordlists.py::test_commitment_patterns_catch_promises_of_real_world_actions",
        "tests/unit/test_engine_postprocess.py::test_a_promise_of_a_real_world_action_is_a_violation",
        "tests/unit/test_engine_pipeline.py::test_a_promise_alone_leaves_nothing_and_ends_in_the_fallback",
    ],
    "pictures out: only stickers of the library, never a photo of hers": [
        "tests/unit/test_channel_policy_binding.py::test_only_listed_bytes_pass_and_everything_else_raises_media_not_allowed",
        "tests/unit/test_engine_safety.py::test_nothing_in_the_engine_package_sends_a_picture_by_itself",
    ],
    "nobody but the user: the recipient is always the bound user": [
        "tests/unit/test_ilink_outbound.py::test_only_the_bound_user_can_be_named_as_the_recipient",
        "tests/unit/test_ilink_poller.py::test_echoes_and_strangers_are_dropped_but_marked_as_seen",
    ],
    "group chats: skipped on import, refused as the target": [
        "tests/unit/test_ingest_importer.py::test_group_chats_and_other_conversations_are_ignored",
        "tests/unit/test_ingest_files.py::test_groups_are_recognised_by_flag_and_by_name",
    ],
}


@pytest.mark.parametrize("non_goal", sorted(NON_GOAL_TESTS))
def test_what_the_program_does_about_each_non_goal_is_tested(non_goal: str) -> None:
    assert missing_nodes(NON_GOAL_TESTS[non_goal]) == [], non_goal
