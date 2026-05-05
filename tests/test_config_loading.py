import json

import pytest

from voxtream.config import load_generator_config, load_speaking_rate_config


def test_load_generator_config_from_default_package_or_repo_path():
    config = load_generator_config()

    assert config.mimi_sr == 24000
    assert config.model_repo == "herimor/voxtream2"


def test_load_generator_config_rejects_unknown_field(tmp_path):
    config_path = tmp_path / "generator.json"
    config = load_generator_config()
    payload = config.__dict__.copy()
    payload["unexpected"] = True
    config_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unknown fields"):
        load_generator_config(config_path)


def test_load_generator_config_rejects_invalid_ranges(tmp_path):
    config_path = tmp_path / "generator.json"
    config = load_generator_config()
    payload = config.__dict__.copy()
    payload["mimi_frame_ms"] = 0
    config_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="mimi_frame_ms"):
        load_generator_config(config_path)


def test_load_speaking_rate_config_validates_shape(tmp_path):
    config_path = tmp_path / "speaking_rate.json"
    config_path.write_text(json.dumps({"1": {"duration_state": [], "weight": 1.0, "cfg_gamma": 1.0}}))

    with pytest.raises(ValueError, match="duration_state"):
        load_speaking_rate_config(config_path)
