"""Round trip of the GGUF converter and the llama.cpp backend on tiny random stacks."""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from voxtream.utils.gguf import backend, convert  # noqa: E402
from voxtream.utils.model import get_llama3_2, prepare_transformer  # noqa: E402
from voxtream.utils.mog import MoGHead  # noqa: E402


def rel_err(ref: torch.Tensor, out: torch.Tensor) -> float:
    return ((ref - out).abs().max() / ref.abs().max().clamp_min(1e-8)).item()


@pytest.fixture(scope="module")
def tiny_model():
    """A `Model`-like holder of a tiny temporal transformer, semantic head, depth
    transformer and MoG MLP."""
    torch.manual_seed(0)
    temp_former, dim = prepare_transformer(
        get_llama3_2(
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            embed_dim=64,
            intermediate_dim=128,
        )
    )
    # the depth transformer is the stack without a folded head: llama.cpp must hand
    # its hidden state back through the identity block alone
    dep_former, _ = prepare_transformer(
        get_llama3_2(
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            embed_dim=64,
            intermediate_dim=128,
        )
    )
    mog_head = MoGHead(
        hidden_dim=dim,
        latent_dim=16,
        num_components=8,
        low_rank=4,
        num_layers=2,
        intermediate_dim=128,
    )
    # norms off their initial values, so that the (1 + w) layout is exercised
    for name, p in mog_head.mlp.named_parameters():
        if name.endswith("weight") and p.ndim == 1:
            p.data.uniform_(-0.3, 0.3)

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.temp_former = temp_former
            self.dep_former = dep_former
            self.sem_head = nn.Linear(dim, 24, bias=False)
            self.mog_head = mog_head

    return Holder().eval()


@pytest.fixture(scope="module")
def converted(tiny_model, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("gguf")
    state_dict = {k: v.detach() for k, v in tiny_model.state_dict().items()}
    convert.convert_model(tiny_model, state_dict, out_dir)
    return out_dir


def test_stacks_match_pytorch(tiny_model, converted):
    lines = backend.verify_stacks(tiny_model, converted)
    assert lines
    for line in lines:
        if "effect of the bias" in line:
            # the prompt bias must change the reference, so that a backend ignoring
            # it could not pass the comparison below
            assert float(line.split("rel ")[-1]) > 0.02, line
            continue
        assert float(line.split("rel ")[-1]) < 5e-2, line


def test_prompt_bias_copies():
    assert backend.bias_copies(None) == 0
    assert backend.bias_copies(0.0) == 0
    assert backend.bias_copies(3.0) == 19  # e^3 = 20.09 -> 20 keys
    assert math.isclose(backend.effective_bias(3.0), math.log(20))


def test_manifest_and_gguf_dir(converted, tmp_path):
    """The GGUF files sit next to the model weights: the manifest lists the q8_0 files
    and the stacks and nothing else, and the files are found in `gguf_dir`, else in a
    local `model_repo`."""
    from types import SimpleNamespace

    manifest = convert.load_manifest(converted)
    assert set(manifest) == {convert.FILES_KEY, convert.STACKS_KEY}
    files = manifest[convert.FILES_KEY]
    assert all(name.endswith("-q8_0.gguf") for name in files.values())
    assert all((converted / name).exists() for name in files.values())

    config = SimpleNamespace(gguf_dir=None, model_repo=str(converted), hf_token=None)
    assert backend.resolve_gguf_dir(config) == converted
    config.gguf_dir = str(tmp_path)
    assert backend.resolve_gguf_dir(config) == tmp_path
    assert backend.resolve_gguf_dir(config, gguf_dir="x") == Path("x")


def test_mimi_transformer_round_trip(tmp_path):
    """A tiny moshi transformer with the shape of Mimi's decoder transformer: pre-LayerNorm,
    RoPE, GELU feed-forward, layer scales, a context window; streamed two frames at a
    time past the window."""
    from moshi.modules.transformer import ProjectedTransformer

    torch.manual_seed(0)
    transformer = ProjectedTransformer(
        input_dimension=64,
        output_dimensions=[64],
        d_model=64,
        num_heads=2,
        num_layers=2,
        causal=True,
        layer_scale=0.1,
        context=12,
        conv_layout=True,
        max_period=10000,
        gating="none",
        norm="layer_norm",
        positional_embedding="rope",
        dim_feedforward=128,
    ).eval()
    for name, p in transformer.named_parameters():  # random norms and scales
        if p.ndim == 1:
            p.data.uniform_(-0.5, 0.5) if "bias" in name else p.data.uniform_(0.5, 1.5)
        if "layer_scale" in name:
            p.data.uniform_(0.05, 0.2)

    class Mimi(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_transformer = transformer

    mimi = Mimi()
    spec = convert.mimi_transformer_spec(mimi)
    assert spec.context == 12 and spec.padded_dim == 96
    paths = convert.write_stacks(
        {spec.name: spec}, {"mimi": mimi.state_dict()}, tmp_path
    )
    convert.write_manifest(tmp_path, {spec.name: spec}, paths)

    lines = backend.verify_mimi_transformer(mimi, tmp_path, steps=20)
    err = float(lines[0].split("max rel err ")[1].split()[0])
    assert err < 8e-2, lines[0]


def test_mimi_backend_resets_with_streaming_state(tmp_path):
    """The llama.cpp cache of the Mimi decoder transformer is not part of moshi's
    streaming state: the patch resets it when that state is reset and when a
    streaming context is entered, so an utterance never sees the keys of the one
    before it."""
    from moshi.modules.transformer import ProjectedTransformer

    torch.manual_seed(0)
    transformer = ProjectedTransformer(
        input_dimension=64,
        output_dimensions=[64],
        d_model=64,
        num_heads=2,
        num_layers=1,
        causal=True,
        layer_scale=0.1,
        context=12,
        conv_layout=True,
        max_period=10000,
        gating="none",
        norm="layer_norm",
        positional_embedding="rope",
        dim_feedforward=128,
    ).eval()

    class Mimi(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_transformer = transformer
            self.calls = []

        def reset_streaming(self, reset_mask=None):
            self.calls.append("reset")

        def _start_streaming(self, batch_size, exit_stack):
            self.calls.append("start")

    mimi = Mimi()
    spec = convert.mimi_transformer_spec(mimi)
    paths = convert.write_stacks(
        {spec.name: spec}, {"mimi": mimi.state_dict()}, tmp_path
    )
    convert.write_manifest(tmp_path, {spec.name: spec}, paths)

    stack = backend.patch_mimi_with_gguf(mimi, tmp_path)
    assert stack is not None and mimi.decoder_transformer is stack

    stack(torch.randn(1, 64, 3))
    assert stack.pos == 3
    mimi._start_streaming(1, None)  # what `mimi.streaming(1)` enters through
    assert stack.pos == 0 and mimi.calls == ["start"]
    stack(torch.randn(1, 64, 2))
    mimi.reset_streaming()
    assert stack.pos == 0 and mimi.calls == ["start", "reset"]
