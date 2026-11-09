"""
The KV-cached phoneme transformer (`PhoneSinkAttention`) against the full-sequence pass.

A tiny model stands in for the real stack: the equivalence does not depend on the size,
only on the window, the look-ahead and the number of layers, and a small cache forces
the keys to be dropped and re-based several times within a short stream.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxtream.model import Model, ModelConfig  # noqa: E402
from voxtream.utils.model import (  # noqa: E402
    MODEL_POOL,
    create_mask,
    get_llama3_2,
)

NUM_LAYERS, LOOK_AHEAD, WINDOW, CACHE_LEN = 3, 2, 16, 64
TINY = dict(
    num_heads=4, num_kv_heads=2, embed_dim=64, intermediate_dim=128, max_seq_len=256
)
MODEL_POOL["test_phone_former"] = lambda: get_llama3_2(num_layers=NUM_LAYERS, **TINY)
MODEL_POOL["test_former"] = lambda: get_llama3_2(num_layers=1, **TINY)


def build_model(max_phone_rows: int = None) -> Model:
    torch.manual_seed(0)
    config = ModelConfig(
        phone_former="test_phone_former",
        temp_former="test_former",
        dep_former="test_former",
        phone_root_vocab_size=20,
        phone_attr_vocab_sizes=[3, 2, 5, 2],
        audio_vocab_size=16,
        audio_pad_size=0,
        embedding_dim=64,
        spk_embedding_dim=8,
        num_codebooks=4,
        num_phone_states=6,
        amortization_divisor=1,
        max_look_ahead=LOOK_AHEAD,
        audio_window_size=32,
        phone_window_size=WINDOW,
    )
    model = Model(config).eval()
    # the attribute embeddings start at zero; give them weight so every stream matters
    with torch.no_grad():
        model.phone_attr_embeddings.weight.normal_()
    model.setup_caches(
        max_batch_size=2,
        dtype=torch.float32,
        phone_cache_len=CACHE_LEN,
        max_phone_rows=max_phone_rows,
    )
    return model


@pytest.fixture(scope="module")
def model() -> Model:
    return build_model()


def random_tokens(seq_len: int, batch_size: int = 2, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    sizes = torch.tensor([20, 3, 2, 5, 2])
    return torch.stack(
        [torch.randint(0, int(n), (batch_size, seq_len), generator=g) for n in sizes],
        dim=-1,
    )


def full_pass(model: Model, tokens: torch.Tensor) -> torch.Tensor:
    """Reference: the whole sequence, one pass, under the mask of the window."""
    emb = model._embed_phone_tokens(tokens)
    seq_len = tokens.size(1)
    mask = create_mask(seq_len, WINDOW, LOOK_AHEAD)
    pos = torch.arange(seq_len).unsqueeze(0).expand(tokens.size(0), -1)
    with torch.no_grad(), model.phone_sink_attention.disabled():
        return model.phone_former(emb, input_pos=pos, mask=mask[None].expand(2, -1, -1))


def stream(model: Model, tokens: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    out, sent = None, 0
    for n in chunks:
        sent += n
        out = model.encode_phonemes(tokens[:, :sent])
        assert out.shape == (tokens.size(0), sent, TINY["embed_dim"])
    return out


@pytest.mark.parametrize("seed", [1, 2])
def test_stream_matches_full_pass_at_every_chunk(model, seed):
    """Word by word, through several re-bases of the cache, the buffer always equals
    the full pass over the text sent so far."""
    torch.manual_seed(seed)
    tokens = random_tokens(130, seed=seed)
    model.reset_caches()
    sent = 0
    while sent < tokens.size(1):
        n = min(int(torch.randint(1, 8, (1,))), tokens.size(1) - sent)
        sent += n
        out = model.encode_phonemes(tokens[:, :sent])
        ref = full_pass(model, tokens[:, :sent])
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    assert model.phone_sink_attention.offset > 0  # the cache did drop keys


def test_whole_text_at_once_is_chunked_exactly(model):
    """A long sequence in one call runs in chunks the cache can take."""
    tokens = random_tokens(150, seed=3)
    model.reset_caches()
    assert tokens.size(1) > model.phone_sink_attention.max_chunk
    out = model.encode_phonemes(tokens)
    torch.testing.assert_close(out, full_pass(model, tokens), atol=1e-4, rtol=1e-4)


def test_prompt_rows_first_then_text(model):
    """The generator encodes the prompt rows before any text arrives."""
    tokens = random_tokens(60, seed=4)
    model.reset_caches()
    model.encode_phonemes(tokens[:, :25])
    out = stream(model, tokens, [25, 3, 1, 10, 21])
    torch.testing.assert_close(out, full_pass(model, tokens), atol=1e-4, rtol=1e-4)


def test_changed_last_row_is_re_encoded(model):
    """A chunk of punctuation alone changes the last row already sent."""
    tokens = random_tokens(90, seed=5)
    model.reset_caches()
    stream(model, tokens, [40, 30, 20])
    changed = tokens.clone()
    changed[:, -1, 3] = (changed[:, -1, 3] + 1) % 5
    out = model.encode_phonemes(changed)
    torch.testing.assert_close(out, full_pass(model, changed), atol=1e-4, rtol=1e-4)
    # and the stream goes on from there
    more = torch.cat([changed, random_tokens(7, seed=6)], dim=1)
    out = model.encode_phonemes(more)
    torch.testing.assert_close(out, full_pass(model, more), atol=1e-4, rtol=1e-4)


def test_change_behind_dropped_keys_is_refused(model):
    tokens = random_tokens(120, seed=7)
    model.reset_caches()
    stream(model, tokens, [60, 60])
    assert model.phone_sink_attention.offset > 0
    changed = tokens.clone()
    changed[:, 0, 0] = (changed[:, 0, 0] + 1) % 20
    with pytest.raises(ValueError, match="left the cache"):
        model.encode_phonemes(changed)
    with pytest.raises(ValueError, match="shrank"):
        model.encode_phonemes(tokens[:, :100])


def test_unchanged_call_is_a_no_op(model):
    tokens = random_tokens(30, seed=8)
    model.reset_caches()
    out = model.encode_phonemes(tokens)
    before = out.clone()
    again = model.encode_phonemes(tokens)
    assert again.data_ptr() == out.data_ptr()
    torch.testing.assert_close(again, before)


def test_full_sequence_pass_is_unaffected_by_the_caches(model):
    tokens = random_tokens(50, seed=11)
    model.reset_caches()
    model.encode_phonemes(tokens[:, :20])  # leaves keys in the caches
    with torch.no_grad():
        legacy = model.extract_phoneme_embeddings(tokens, look_ahead_phones=10**6)
    torch.testing.assert_close(legacy, full_pass(model, tokens), atol=1e-5, rtol=1e-5)
    # and did not disturb the stream
    out = model.encode_phonemes(tokens)
    torch.testing.assert_close(out, full_pass(model, tokens), atol=1e-4, rtol=1e-4)


def test_fixed_size_step_matches_full_pass(model):
    """With a captured step every small forward is widened to a fixed number of rows;
    the recomputed rows and the re-bases around them leave the result unchanged."""
    encoder = model.phone_sink_attention
    rows = 2 * NUM_LAYERS * LOOK_AHEAD + 4
    calls = []

    def counting_step(tokens, pos, slots):
        calls.append(tokens.size(1))
        return encoder._step(tokens, pos, slots)

    encoder.enable_graph(counting_step, rows=rows)
    try:
        tokens = random_tokens(140, seed=12)
        model.reset_caches()
        model.encode_phonemes(tokens[:, :30])  # the prompt: more rows than the step
        sent = 30
        while sent < tokens.size(1):
            sent = min(sent + int(torch.randint(1, 6, (1,))), tokens.size(1))
            out = model.encode_phonemes(tokens[:, :sent])
            ref = full_pass(model, tokens[:, :sent])
            torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        assert calls and all(n == rows for n in calls)
        assert encoder.offset > 0
        with pytest.raises(ValueError, match="receptive field"):
            encoder.enable_graph(counting_step, rows=NUM_LAYERS * LOOK_AHEAD)
        with pytest.raises(ValueError, match="does not fit"):
            encoder.enable_graph(counting_step, rows=CACHE_LEN - WINDOW + 2)
    finally:
        encoder.step_fn, encoder.graph_rows = None, 0


def test_stream_past_the_trained_positions():
    """
    Set up for more rows than the rotary embedding has positions, the encoder extends
    the rotary cache: the trained range is unchanged, and a stream runs past it.
    """
    model = build_model(max_phone_rows=400)
    rope = model.phone_former.layers[0].attn.pos_embeddings
    assert rope.max_seq_len == 400 and rope.cache.size(0) == 400
    ref = build_model()  # same seed, same weights, the constructor's rotary cache
    ref_rope = ref.phone_former.layers[0].attn.pos_embeddings
    assert torch.equal(rope.cache[: ref_rope.cache.size(0)], ref_rope.cache)
    tokens = random_tokens(400, seed=8)
    model.reset_caches()
    out = model.encode_phonemes(tokens)
    assert out.shape == (2, 400, TINY["embed_dim"])
    # a row's output depends on the rows up to its receptive field, so the rows before
    # the trained range match the unextended model's pass over that range
    n = TINY["max_seq_len"] - model.phone_sink_attention.config.receptive_field
    torch.testing.assert_close(
        out[:, :n],
        full_pass(ref, tokens[:, : TINY["max_seq_len"]])[:, :n],
        atol=1e-4,
        rtol=1e-4,
    )
