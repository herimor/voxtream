import json
import os
import platform
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
    # Sampling makes outputs diverge across CPU architectures / BLAS backends,
    # so CPU fixtures are keyed by OS and architecture.
    fixture_name = (
        "reference_frames.npy"
        if device_name == "cuda"
        else f"reference_frames_cpu_{sys.platform}_{platform.machine().lower()}.npy"
    )
    fixture_path = REPO_ROOT / "assets" / "test" / fixture_name
    config_path = REPO_ROOT / "configs" / "generator.json"
    update_fixture = os.environ.get("UPDATE_REGRESSION_FIXTURES") == "1"

    if not update_fixture and not fixture_path.exists():
        raise FileNotFoundError(
            f"Missing regression fixture: {fixture_path}. "
            "Generate it on this machine with: "
            "UPDATE_REGRESSION_FIXTURES=1 pytest tests/test_run_output_regression.py"
        )

    with config_path.open() as f:
        config = json.load(f)

    captured_write = {}

    def fake_sf_write(output_path, data, samplerate):
        captured_write["path"] = str(output_path)
        captured_write["data"] = np.asarray(data)
        captured_write["samplerate"] = samplerate

    monkeypatch.setattr(run.sf, "write", fake_sf_write)

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
    if update_fixture:
        np.save(fixture_path, captured_write["data"])
    expected_audio = np.load(fixture_path)
    assert captured_write["data"].shape == expected_audio.shape
    np.testing.assert_allclose(
        captured_write["data"], expected_audio, rtol=1e-5, atol=1e-6
    )
    assert captured_write["samplerate"] == config["mimi_sr"]
