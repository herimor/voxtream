import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import voxtream.run as run


def test_run_main_output_matches_reference(monkeypatch, tmp_path):

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    fixture_name = (
        "reference_frames.npy"
        if device_name == "cuda"
        else f"reference_frames_{device_name}.npy"
    )
    fixture_path = REPO_ROOT / "assets" / "test" / fixture_name
    config_path = REPO_ROOT / "configs" / "generator.json"

    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Missing regression fixture: {fixture_path}. "
            f"Generate it on a {device_name.upper()} device using the run.py "
            "regression invocation."
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
            str(REPO_ROOT / "assets" / "test" / "english_male.wav"),
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
