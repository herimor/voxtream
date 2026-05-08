import numpy as np
import pytest

from voxtream.config import load_generator_config
from voxtream.utils.generator.prompt import (
    _load_prompt_cache,
    _prompt_cache_metadata,
    _prompt_cache_path,
    _save_prompt_cache,
)


def test_prompt_cache_uses_npz_without_pickle(tmp_path):
    torch = pytest.importorskip("torch")
    prompt = tmp_path / "voice.wav"
    prompt.write_bytes(b"RIFF")
    config = load_generator_config()
    cache_path = _prompt_cache_path(prompt)
    metadata = _prompt_cache_metadata(prompt, config, enhance_prompt=False, apply_vad=False)
    audio_tokens = torch.zeros((1, 2, 3), dtype=torch.int64)
    spk_embedding = torch.ones((1, 4), dtype=torch.float32)

    _save_prompt_cache(cache_path, metadata, audio_tokens, spk_embedding)

    with np.load(cache_path, allow_pickle=False) as data:
        assert set(data.files) == {"metadata", "audio_tokens", "spk_embedding"}


def test_prompt_cache_rejects_stale_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    prompt = tmp_path / "voice.wav"
    prompt.write_bytes(b"RIFF")
    config = load_generator_config()
    cache_path = _prompt_cache_path(prompt)
    metadata = _prompt_cache_metadata(prompt, config, enhance_prompt=False, apply_vad=False)
    audio_tokens = torch.zeros((1, 2, 3), dtype=torch.int64)
    spk_embedding = torch.ones((1, 4), dtype=torch.float32)
    _save_prompt_cache(cache_path, metadata, audio_tokens, spk_embedding)

    stale_metadata = dict(metadata)
    stale_metadata["cache_key"] = "stale"

    assert _load_prompt_cache(cache_path, stale_metadata) is None
