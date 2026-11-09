"""The demo's cache of encoded prompts (`PromptCache`) and the stage reporter."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxtream.config import SpeechGeneratorConfig  # noqa: E402
from voxtream.utils.app import PromptCache, StageReporter  # noqa: E402

OPTIONS = dict(enhance_prompt=False, apply_vad=False, normalize_loudness=True)


@pytest.fixture(scope="module")
def config() -> SpeechGeneratorConfig:
    with open(REPO_ROOT / "configs" / "generator.json") as f:
        return SpeechGeneratorConfig(**json.load(f))


def write_wav(path: Path, seed: int, sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(sample_rate) * 3000).astype(np.int16)
    sf.write(path, samples, sample_rate, subtype="PCM_16")
    return path


def test_entry_survives_the_copy_gradio_makes(tmp_path, config):
    cache = PromptCache(tmp_path / "cache", config)
    src = write_wav(tmp_path / "voice.wav", seed=1)
    assert cache.load(src, **OPTIONS) is None
    tokens = np.random.default_rng(0).integers(0, 2048, (1, 32, 40))
    emb = np.random.default_rng(0).standard_normal((1, 192)).astype(np.float32)
    path = cache.save(src, tokens, emb, **OPTIONS)
    assert path.name.startswith("voice.") and path.suffix == ".npz"
    # gradio decodes the prompt and writes it again under another path
    samples, sr = sf.read(src, dtype="int16")
    copy = tmp_path / "gradio" / "abc" / "voice.wav"
    copy.parent.mkdir(parents=True)
    sf.write(copy, samples, sr, subtype="PCM_16")
    loaded = cache.load(copy, **OPTIONS)
    assert loaded is not None
    np.testing.assert_array_equal(loaded[0], tokens)
    np.testing.assert_array_equal(loaded[1], emb)
    assert len(cache.entries()) == 1


def test_other_audio_or_settings_are_other_entries(tmp_path, config):
    cache = PromptCache(tmp_path / "cache", config)
    a = write_wav(tmp_path / "a" / "voice.wav", seed=1)
    b = write_wav(tmp_path / "b" / "voice.wav", seed=2)
    cache.save(a, np.zeros((1, 32, 3)), np.zeros((1, 192)), **OPTIONS)
    assert cache.load(b, **OPTIONS) is None
    assert cache.load(a, **{**OPTIONS, "apply_vad": True}) is None
    cache.save(b, np.ones((1, 32, 3)), np.ones((1, 192)), **OPTIONS)
    assert len(cache.entries()) == 2
    assert cache.load(a, **OPTIONS)[0].sum() == 0
    assert cache.load(b, **OPTIONS)[0].sum() == 96


def test_stage_reporter_hands_out_stages_and_the_result():
    reporter = StageReporter()

    def call():
        reporter("mimi")
        reporter("speaker")
        return 42

    stages = reporter.run(call)
    seen = []
    while True:
        try:
            seen.append(next(stages))
        except StopIteration as done:
            assert done.value == 42
            break
    assert seen == ["mimi", "speaker"]

    def failing():
        reporter("mimi")
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        list(StageReporter().run(failing))
