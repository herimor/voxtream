import json
import sys
from pathlib import Path

import numpy as np

import voxtream.run as run


def test_run_main_output_matches_reference(monkeypatch, tmp_path):

    repo_root = Path(__file__).resolve().parents[1]
    fixture_path = repo_root / "assets" / "test" / "reference_frames.npy"
    config_path = repo_root / "configs" / "generator.json"

    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Missing regression fixture: {fixture_path}. "
            "Generate it by running `python scripts/generate_run_reference.py`."
        )

    expected_audio = np.load(fixture_path)
    with config_path.open() as f:
        config = json.load(f)

    captured_write = {}

    class FakeSoundFile:
        def __init__(self, output_path, mode, samplerate, channels):
            captured_write["path"] = str(output_path)
            captured_write["mode"] = mode
            captured_write["samplerate"] = samplerate
            captured_write["channels"] = channels
            captured_write["chunks"] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            captured_write["data"] = np.concatenate(captured_write["chunks"])

        def write(self, data):
            captured_write["chunks"].append(np.asarray(data))

    monkeypatch.setattr(run.sf, "SoundFile", FakeSoundFile)

    output_path = tmp_path / "voxtream_run_ref.wav"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxtream",
            "-c",
            str(config_path),
            "-pa",
            str(repo_root / "assets" / "test" / "english_male.wav"),
            "-t",
            "reference text",
            "-o",
            str(output_path),
            "--prompt-enhancement",
        ],
    )

    run.main()

    assert captured_write["path"] == str(output_path)
    assert captured_write["mode"] == "w"
    assert captured_write["channels"] == 1
    assert captured_write["data"].shape == expected_audio.shape
    np.testing.assert_allclose(
        captured_write["data"], expected_audio, rtol=1e-5, atol=1e-6
    )
    assert captured_write["samplerate"] == config["mimi_sr"]
