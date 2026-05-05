import base64
from pathlib import Path

import pytest

from voxtream.server import (
    MAX_PROMPT_AUDIO_BYTES,
    _b64_to_bytes,
    _ensure_prompt_audio_file,
    _validate_prompt_audio_path,
)


def test_validate_prompt_audio_path_rejects_paths_outside_root(tmp_path):
    prompt_root = tmp_path / "allowed"
    prompt_root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")

    with pytest.raises(ValueError, match="inside"):
        _validate_prompt_audio_path(str(outside), prompt_root)


def test_validate_prompt_audio_path_accepts_file_inside_root(tmp_path):
    prompt_root = tmp_path / "allowed"
    prompt_root.mkdir()
    prompt = prompt_root / "voice.wav"
    prompt.write_bytes(b"RIFF")

    assert _validate_prompt_audio_path(str(prompt), prompt_root) == prompt.resolve()


def test_b64_to_bytes_rejects_invalid_base64():
    with pytest.raises(ValueError, match="valid base64"):
        _b64_to_bytes("not base64?")


def test_b64_to_bytes_rejects_large_payload():
    payload = base64.b64encode(b"x" * (MAX_PROMPT_AUDIO_BYTES + 1)).decode()

    with pytest.raises(ValueError, match="exceeds"):
        _b64_to_bytes(payload)


def test_ensure_prompt_audio_file_creates_temp_for_base64_and_marks_temp(tmp_path):
    prompt_path, is_temp = _ensure_prompt_audio_file(
        None,
        "data:audio/wav;base64," + base64.b64encode(b"RIFF").decode(),
        prompt_root=tmp_path,
    )

    try:
        assert is_temp is True
        assert isinstance(prompt_path, Path)
        assert prompt_path.exists()
        assert prompt_path.read_bytes() == b"RIFF"
    finally:
        prompt_path.unlink(missing_ok=True)
