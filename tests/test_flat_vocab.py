"""The flat phoneme vocabulary of VoXtream2.1: frontend, model and training data."""

import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxtream.utils.generator import flat_text
from voxtream.utils.generator.flat_text import FlatPhonemeVocab
from voxtream.utils.model import remove_punctuation

# a vocabulary laid out as phoneme_to_token.json: phonemes, punctuation, sil, unk
PHONES = ["h", "ə", "l", "oʊ", "w", "ɜː", "d"]
VOCAB = {
    **{ph: i for i, ph in enumerate(PHONES)},
    ".": 7,
    ",": 8,
    "?": 9,
    "sil": 10,
    "!": 11,
    "unk": 12,
}
VOCAB_SIZE = 15  # BOS 13, EOS 14


def _vocab() -> FlatPhonemeVocab:
    return FlatPhonemeVocab(phone_to_token=VOCAB, vocab_size=VOCAB_SIZE)


def test_flat_vocab_specials():
    vocab = _vocab()
    assert (vocab.bos, vocab.eos, vocab.unk, vocab.sil) == (13, 14, 12, 10)
    assert vocab.prompt_tokens(4).tolist() == [13, 12, 12, 12]

    # punctuation rows are found in the conditional row and removed from both
    tokens = torch.tensor([[13, 12, 0, 1, 8, 2, 7, 10, 14], [13, 12] + [12] * 7])
    rows = vocab.punct_del_indices(tokens)
    assert rows.tolist() == [[4, 6], [4, 6]]


def test_flat_text_to_phone_tokens():
    class Phonemizer:
        def phonemize(self, text, language):
            return "h|ə|l|oʊ, w|ɜː|l|d|.|"

    config = SimpleNamespace(text_context=" context", text_context_length=18)
    tokens, ins, punct = flat_text.text_to_phone_tokens(
        "hello, world.", config, _vocab(), Phonemizer(), normalize=False
    )
    # punctuation leaves the phoneme sequence and is inserted back where it was
    assert tokens == [0, 1, 2, 3, 4, 5, 2, 6]
    assert ins == [4, 8] and punct == [8, 7]
    assert np.insert(tokens, ins, punct).tolist() == [0, 1, 2, 3, 8, 4, 5, 2, 6, 7]


def test_remove_punctuation():
    data = torch.arange(12, dtype=torch.float32).view(1, 6, 2).repeat(2, 1, 1)
    out = remove_punctuation(data, torch.tensor([[1, 4], [2, -1]]))
    # kept rows in order, the tail padded with the last row
    assert out[0, :, 0].tolist() == [0, 4, 6, 10, 10, 10]
    assert out[1, :, 0].tolist() == [0, 2, 6, 8, 10, 10]
    # no rows to remove leaves the data as it is
    assert torch.equal(remove_punctuation(data, torch.zeros(2, 0).long()), data)


def _flat_config(**config):
    from voxtream.model import ModelConfig
    from voxtream.utils.model import MODEL_POOL, get_llama3_2

    MODEL_POOL["test_flat_former"] = lambda: get_llama3_2(
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        embed_dim=32,
        intermediate_dim=64,
        max_seq_len=128,
    )
    params = dict(
        phone_former="test_flat_former",
        temp_former="test_flat_former",
        dep_former="test_flat_former",
        phone_vocab_size=VOCAB_SIZE,
        audio_vocab_size=18,
        audio_pad_size=0,
        embedding_dim=32,
        spk_embedding_dim=8,
        num_codebooks=4,
        num_phone_states=6,
        amortization_divisor=2,
        max_look_ahead=2,
        audio_window_size=16,
        phone_window_size=64,
    )
    return ModelConfig(**{**params, **config})


def test_flat_model():
    from voxtream.model import Model

    with pytest.raises(ValueError):  # one kind of vocabulary
        Model(_flat_config(phone_root_vocab_size=20, phone_attr_vocab_sizes=[3]))
    torch.manual_seed(0)
    model = Model(_flat_config()).eval()
    assert model.phone_sink_attention is None
    assert not hasattr(model, "phone_attr_embeddings")

    tokens = torch.tensor([[13, 12, 0, 1, 8, 2, 7, 10, 14]])
    punct = torch.tensor([[4, 6]])
    emb = model.extract_phoneme_embeddings(
        tokens, prompt_len=2, punct_del_indices=punct
    )
    full = model.extract_phoneme_embeddings(tokens, prompt_len=2)
    assert torch.equal(emb, remove_punctuation(full, punct))

    # the mask is picked from the text after the prompt, as released: 7 rows over a
    # look-ahead of 2 is the largest mask
    from voxtream.utils.model import get_mask

    calls = []
    real = get_mask

    def spy(mask, input_pos):
        calls.append(mask)
        return real(mask, input_pos)

    import voxtream.model as model_module

    model_module.get_mask, saved = spy, model_module.get_mask
    try:
        model.extract_phoneme_embeddings(tokens, prompt_len=2)
        model.extract_phoneme_embeddings(tokens[:, :3], prompt_len=2)
    finally:
        model_module.get_mask = saved
    assert torch.equal(calls[0], model.phone_former_mask[1])
    assert torch.equal(calls[1], model.phone_former_mask[0])


def test_flat_pointer_steps_off_the_end():
    """A state held `final_phone_hold_frames` times from the second-to-last phoneme on
    is left on the model's behalf, and generation ends on `sil`."""
    from voxtream.utils.generator.context import FrameState

    config = SimpleNamespace(
        final_phone_hold_frames=3, frame_repeat_counter=25, num_phones_per_frame=2
    )
    phoneme_index_map = {"0": [0, 1], "2": [1, 1]}
    ctx = SimpleNamespace(
        config=config, phoneme_index_map=phoneme_index_map, batch_size=1
    )
    state = FrameState(
        phone_emb_indices=torch.zeros(1, 1, 2, dtype=torch.long),
        phone_emb_max_idx=8,
        eos_idx=100,
        state_counter={},
    )
    frame = torch.zeros(1, 4, dtype=torch.long)
    hold = torch.tensor(0)
    for idx in range(3):  # held on phoneme 8 of 10 (sil at 10, EOS at 11)
        _, state = flat_text.update_indices_and_tokens(
            hold, frame, idx, 10, 0, state, ctx
        )
        assert state.phone_emb_max_idx == 8
    _, state = flat_text.update_indices_and_tokens(hold, frame, 3, 10, 0, state, ctx)
    assert state.phone_emb_max_idx == 9 and state.eos_idx == 100
    _, state = flat_text.update_indices_and_tokens(
        torch.tensor(2), frame, 4, 10, 0, state, ctx
    )
    assert state.phone_emb_max_idx == 10 and state.eos_idx == 4


def _write_flat_arrays(root: Path, num_items=6, num_frames=40, num_phones=30):
    """Arrays laid out as herimor/voxtream2-train."""
    rng = np.random.default_rng(0)

    def objects(items):
        out = np.empty(len(items), dtype=object)
        out[:] = items
        return out

    codes, emb, shifts, phones, punct = [], [], [], [], []
    for _ in range(num_items):
        codes.append(rng.integers(0, 16, (4, num_frames)))
        rows = np.minimum(np.arange(num_frames) * num_phones // num_frames, 28)
        emb.append(np.stack([rows, rows + 1], 1))
        shifts.append(rng.integers(0, 6, num_frames))
        phones.append(rng.integers(0, 7, num_phones))
        # punctuation inserted before phonemes 10 and 20
        punct.append(([10, 20], [10, 21], [8, 7]))
    root.mkdir(parents=True)
    np.save(root / "mimi_codes.npy", np.stack(codes))
    np.save(root / "phone_emb_indices.npy", objects(emb), allow_pickle=True)
    np.save(root / "sem_label_shifts.npy", objects(shifts), allow_pickle=True)
    np.save(root / "phone_tokens.npy", objects(phones), allow_pickle=True)
    np.save(root / "punctuation.npy", objects(punct), allow_pickle=True)
    np.save(root / "spk_templates.npy", rng.standard_normal((num_items, 8), np.float32))


@pytest.mark.filterwarnings(r"ignore:You are trying to `self.log\(\)`")
def test_flat_dataset_trains(tmp_path):
    from omegaconf import OmegaConf

    from voxtream.dataset import DATASET_POOL
    from voxtream.trainer import Trainer

    _write_flat_arrays(tmp_path / "toy")
    files = dict(
        audio_codes="mimi_codes.npy",
        phoneme_sequence_map="phone_tokens.npy",
        phoneme_embedding_indices="phone_emb_indices.npy",
        semantic_label_shifts="sem_label_shifts.npy",
        spk_templates="spk_templates.npy",
        punctuation="punctuation.npy",
    )
    model_config = asdict(_flat_config(dep_former_head="ar"))
    dataset = DATASET_POOL["flat"](
        base_dir=tmp_path,
        model=model_config,
        datasets={"toy": files},
        audio_delay_frames=1,
        dtype="int64",
        pad_len=1,
        semantic_label_pad=2,
        num_phones_per_frame=2,
        phoneme_index_map={str(i): 2 + i % 2 for i in range(6)},
        cfg_prob=0.1,
        prompt_length_sec=[1, 2],
        mimi_tps=4,
    )
    torch.manual_seed(0)
    np.random.seed(0)
    batch = dataset.collate_fn([dataset[i] for i in range(4)])
    assert batch["phone_tokens"].dim() == 2
    assert batch["audio_tokens"].shape == (4, 4, 18)
    # the punctuation rows hold punctuation tokens
    rows = batch["punct_del_indices"]
    picked = batch["phone_tokens"].gather(1, rows.clamp_min(0))[rows >= 0]
    assert set(picked.tolist()) <= {7, 8}

    config = OmegaConf.create(
        dict(
            model=model_config,
            compile_forward=False,
            dep_former_weight_path=None,
            dep_former_name=None,
            model_repo=None,
            model_weight_path=None,
            freeze=[],
        )
    )
    module = Trainer(config)
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
