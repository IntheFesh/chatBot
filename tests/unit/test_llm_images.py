"""Image input for vision requests (R-LLM-004)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from twin.llm import images as images_module
from twin.llm.capabilities import DOCUMENTED, LlmCapabilities
from twin.llm.errors import ImagePlacementError, ImageTooLargeError, UnsupportedImageError
from twin.llm.images import (
    DETAIL_PHOTO,
    DETAIL_STICKER,
    ImageInput,
    PreparedImage,
    attach_images,
    count_images,
    estimate_encoded_bytes,
    prepare_image,
    sniff_mime,
    user_message,
    validate_image_placement,
)
from twin.llm.synth_images import draw_gif, draw_jpeg, draw_png
from twin.llm.types import ChatMessage
from twin.storage.crypto import KeyRing, generate_key, use_keyring
from twin.storage.media import MediaKind, MediaStore


def webp_bytes(width: int = 64, height: int = 48) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 200, 10)).save(buffer, format="WEBP")
    return buffer.getvalue()


# ----------------------------------------------------------------- recognising


def test_formats_are_recognised_from_the_header() -> None:
    assert sniff_mime(draw_jpeg()) == "image/jpeg"
    assert sniff_mime(draw_png()) == "image/png"
    assert sniff_mime(draw_gif()) == "image/gif"
    assert sniff_mime(webp_bytes()) == "image/webp"
    for junk in (b"", b"hello world", b"RIFF\x00\x00\x00\x00WAVE", b"GIF90a"):
        with pytest.raises(UnsupportedImageError, match="header"):
            sniff_mime(junk)


def test_the_file_name_does_not_decide_the_type(tmp_path: Path) -> None:
    path = tmp_path / "looks_like.jpg"
    path.write_bytes(draw_png())
    prepared = prepare_image(ImageInput.from_path(path).read())
    assert prepared.mime == "image/png"
    assert prepared.data_url().startswith("data:image/png;base64,")


def test_garbage_with_a_valid_header_is_rejected() -> None:
    with pytest.raises(UnsupportedImageError, match="decode"):
        prepare_image(b"\xff\xd8\xff" + b"not really a jpeg" * 20)


def test_truncated_images_are_rejected() -> None:
    data = draw_png(200, 200)
    with pytest.raises(UnsupportedImageError):
        prepare_image(data[: len(data) // 2])


# ----------------------------------------------------------------------- inputs


def test_inputs_read_from_bytes_paths_and_the_encrypted_media_store(tmp_path: Path) -> None:
    data = draw_jpeg(64, 64)
    assert ImageInput.from_bytes(data).read() == data
    path = tmp_path / "a.bin"
    path.write_bytes(data)
    assert ImageInput.from_path(str(path)).read() == data
    ring = KeyRing({1: generate_key()}, 1)
    with use_keyring(ring):
        store = MediaStore(tmp_path / "media", tmp_path / "tmp", chunk_size=4096)
        stored = store.put(data, MediaKind.IMAGE)
        assert ImageInput.from_media(store, stored.sha256).read() == data


def test_detail_values_are_checked() -> None:
    assert ImageInput.from_bytes(b"x", DETAIL_STICKER).detail == "low"
    assert ImageInput.from_bytes(b"x", DETAIL_PHOTO).detail == "auto"
    with pytest.raises(ValueError, match="detail"):
        ImageInput.from_bytes(b"x", "ultra")


# ------------------------------------------------------------------ preparation


def test_images_within_the_limits_are_sent_unchanged() -> None:
    for data in (draw_jpeg(300, 200), draw_png(100, 100), draw_gif(), webp_bytes()):
        prepared = prepare_image(data, detail="low")
        assert prepared.data == data and prepared.note is None
        assert prepared.detail == "low"


def test_animated_gifs_keep_their_frames() -> None:
    prepared = prepare_image(draw_gif(96, 96, frames=5))
    assert prepared.mime == "image/gif" and prepared.frames == 5
    assert (prepared.width, prepared.height) == (96, 96)


def test_the_detail_parameter_is_dropped_when_the_probe_found_it_unsupported() -> None:
    caps = LlmCapabilities(detail_supported=False)
    prepared = prepare_image(draw_png(), detail="low", capabilities=caps)
    assert prepared.detail is None
    assert "detail" not in prepared.part()["image_url"]
    assert prepare_image(draw_png(), detail="low").part()["image_url"]["detail"] == "low"


def test_a_gif_becomes_a_png_of_its_first_frame_when_gifs_are_unsupported() -> None:
    caps = LlmCapabilities(gif_supported=False)
    prepared = prepare_image(draw_gif(), capabilities=caps)
    assert prepared.mime == "image/png" and prepared.frames == 1
    assert prepared.note is not None and "first frame" in prepared.note
    # a still GIF is not touched (it is not animated)
    still = io.BytesIO()
    Image.new("P", (16, 16)).save(still, format="GIF")
    assert prepare_image(still.getvalue(), capabilities=caps).mime == "image/gif"


def test_oversized_dimensions_are_scaled_to_the_side_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images_module, "MAX_IMAGE_SIDE_PX", 200)
    prepared = prepare_image(draw_jpeg(400, 300))
    assert prepared.mime == "image/jpeg"
    assert (prepared.width, prepared.height) == (200, 150)
    assert prepared.note is not None and "scaled down" in prepared.note
    decoded = Image.open(io.BytesIO(prepared.data))
    assert decoded.size == (200, 150) and decoded.format == "JPEG"


def test_the_lower_side_limit_applies_with_fifteen_or_more_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(images_module, "MAX_IMAGE_SIDE_PX", 400)
    monkeypatch.setattr(images_module, "MAX_IMAGE_SIDE_PX_MANY", 100)
    big = draw_png(300, 200)
    assert prepare_image(big).note is None
    many = prepare_image(big, many_images=True)
    assert max(many.width, many.height) == 100 and many.mime == "image/png"


def test_oversized_animated_gifs_are_scaled_frame_by_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images_module, "MAX_IMAGE_SIDE_PX", 64)
    prepared = prepare_image(draw_gif(128, 128, frames=3))
    assert prepared.mime == "image/gif" and prepared.frames == 3
    assert (prepared.width, prepared.height) == (64, 64)


def test_oversized_webp_is_scaled_and_stays_webp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images_module, "MAX_IMAGE_SIDE_PX", 32)
    prepared = prepare_image(webp_bytes(64, 48))
    assert prepared.mime == "image/webp" and (prepared.width, prepared.height) == (32, 24)


def test_files_over_the_byte_limit_are_shrunk_until_they_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    noisy = Image.effect_noise((256, 256), 80).convert("RGB")
    buffer = io.BytesIO()
    noisy.save(buffer, format="PNG")
    raw = buffer.getvalue()
    monkeypatch.setattr(images_module, "MAX_RAW_IMAGE_BYTES", len(raw) // 3)
    prepared = prepare_image(raw)
    assert len(prepared.data) <= len(raw) // 3
    assert prepared.width < 256 and prepared.mime == "image/png"


def test_a_huge_animation_falls_back_to_its_first_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    gif = draw_gif(160, 160, frames=4)
    monkeypatch.setattr(images_module, "MAX_RAW_IMAGE_BYTES", 20_000)
    monkeypatch.setattr(images_module, "MAX_IMAGE_SIDE_PX", 100)  # forces the shrinking path
    monkeypatch.setattr(images_module, "_encode_animated", lambda *_: b"x" * 30_000)
    result = prepare_image(gif)
    assert result.mime == "image/png" and result.frames == 1
    assert result.note is not None and "first frame" in result.note
    assert max(result.width, result.height) <= 100


def test_an_image_that_cannot_be_made_to_fit_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images_module, "MAX_RAW_IMAGE_BYTES", 10)
    with pytest.raises(ImageTooLargeError, match="exceeds"):
        prepare_image(draw_jpeg(300, 300))


# ------------------------------------------------------------- message building


def prepared() -> PreparedImage:
    return prepare_image(draw_png(32, 32), detail="low")


def test_user_message_puts_the_text_before_the_images() -> None:
    message = user_message("这是什么", [prepared(), prepared()])
    parts = message["content"]
    assert isinstance(parts, list)
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")  # type: ignore[typeddict-item]
    assert user_message("", [prepared()])["content"][0]["type"] == "image_url"  # type: ignore[index]


def test_attach_images_goes_into_the_last_user_message() -> None:
    history: list[ChatMessage] = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "look at this"},
        {"role": "assistant", "content": "trailing"},
    ]
    result = attach_images(history, [prepared()])
    assert result[:3] == history[:3] and result[4] == history[4]
    content = result[3]["content"]
    assert isinstance(content, list) and [p["type"] for p in content] == ["text", "image_url"]
    assert history[3]["content"] == "look at this"  # the input list is not modified
    assert attach_images(history, []) == history
    with pytest.raises(ImagePlacementError, match="user message"):
        attach_images([{"role": "system", "content": "x"}], [prepared()])


def test_attach_images_extends_a_message_that_already_has_parts() -> None:
    first = user_message("a", [prepared()])
    result = attach_images([first], [prepared()])
    assert count_images(result) == 2


def test_images_are_only_allowed_in_user_messages() -> None:
    image_part = prepared().part()
    ok: list[ChatMessage] = [{"role": "user", "content": [image_part]}]
    validate_image_placement(ok)
    for role in ("system", "assistant"):
        bad: list[ChatMessage] = [
            {"role": "user", "content": "hi"},
            {"role": role, "content": [image_part]},  # type: ignore[typeddict-item]
        ]
        with pytest.raises(ImagePlacementError, match=role):
            validate_image_placement(bad)
    validate_image_placement([{"role": "system", "content": "plain text is fine"}])


def test_counting_and_sizing_helpers() -> None:
    one = prepared()
    messages = [user_message("x", [one, one]), {"role": "assistant", "content": "ok"}]
    assert count_images(messages) == 2  # type: ignore[arg-type]
    assert estimate_encoded_bytes([one, one]) == 2 * len(base64.b64encode(one.data))
    assert DOCUMENTED.detail_supported
