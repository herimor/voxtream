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
            str(repo_root / "assets" / "audio" / "english_male.wav"),
            "-t",
            "reference text",
            "-o",
            str(output_path),
        ],
    )

    run.main()

    assert captured_write["path"] == str(output_path)
    assert captured_write["data"].shape == expected_audio.shape
    np.testing.assert_allclose(
        captured_write["data"], expected_audio, rtol=1e-5, atol=1e-6
    )
    assert captured_write["samplerate"] == config["mimi_sr"]
