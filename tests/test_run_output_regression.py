import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import voxtream.run as run

# model: generator config, fixture directory and the extra arguments of `voxtream`
MODELS = {
    "voxtream2.5": ("configs/generator.json", "assets/test", []),
    # as the regression test of the release ran it
    "voxtream2.1": (
        "configs/generator_voxtream2.1.json",
        "assets/test/voxtream2.1",
        ["--prompt-enhancement"],
    ),
}


@pytest.mark.parametrize("model", list(MODELS))
def test_run_main_output_matches_reference(model, monkeypatch, tmp_path):

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    # Sampling makes outputs diverge across CPU architectures / BLAS backends,
    # so CPU fixtures are keyed by OS and architecture.
    fixture_name = (
        "reference_frames.npy"
        if device_name == "cuda"
        else f"reference_frames_cpu_{sys.platform}_{platform.machine().lower()}.npy"
    )
    config_name, fixture_dir, extra_args = MODELS[model]
    fixture_path = REPO_ROOT / fixture_dir / fixture_name
    config_path = REPO_ROOT / config_name
    update_fixture = os.environ.get("UPDATE_REGRESSION_FIXTURES") == "1"

    if not update_fixture and not fixture_path.exists():
        raise FileNotFoundError(
            f"Missing regression fixture: {fixture_path}. "
            "Generate it on this machine with: "
            "UPDATE_REGRESSION_FIXTURES=1 pytest tests/test_run_output_regression.py"
        )

    with config_path.open() as f:
        config = json.load(f)
    # the CPU fixtures were made with the PyTorch stacks
    config["gguf"] = False
    config_path = tmp_path / config_path.name
    config_path.write_text(json.dumps(config))

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
            *extra_args,
        ],
    )

    run.main()

    assert captured_write["path"] == str(output_path)
    if update_fixture:
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(fixture_path, captured_write["data"])
    expected_audio = np.load(fixture_path)
    assert captured_write["data"].shape == expected_audio.shape
    np.testing.assert_allclose(
        captured_write["data"], expected_audio, rtol=1e-5, atol=1e-6
    )
    assert captured_write["samplerate"] == config["mimi_sr"]
