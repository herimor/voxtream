"""Convert the transformer stacks of a VoXtream2 checkpoint into GGUF files.

What is converted:

* ``temp_former`` (and ``dep_former`` when the model has one): torchtune
  ``llama3_2`` decoders with ``tok_embeddings`` and ``output`` replaced by
  ``nn.Identity``, so llama.cpp runs them embeddings-in / embeddings-out under
  ``arch=llama``. The semantic head of the temporal transformer is folded into
  the GGUF ``output.weight`` together with an identity block, so one
  ``llama_decode`` returns both the semantic logits and the final hidden state
  (see ``voxtream.utils.gguf.backend``).
* the MLP stack of the Mixture of Gaussians head (``mog_head.mlp``): residual
  gated-GELU layers with pre- and post-RMSNorm, exactly the block of
  ``arch=gemma2`` once its attention is neutralized (zero value projection), so
  llama.cpp runs it as a Gemma2 model.

Everything else (phone_former, embeddings, projections, codebooks, Mimi, the
speaker encoder) stays in PyTorch and is loaded from the model weights
(``model.safetensors``), which the GGUF files sit next to in the model repo; the
PyTorch copies of the converted stacks are dropped once they are swapped out.

The projection matrices are written in ``q8_0`` (int8: blocks of 32 weights sharing
one fp16 scale); norm scales and RoPE factors stay fp32.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np

# weight format of the projection matrices, the suffix of the file names
QUANT = "q8_0"

MANIFEST_NAME = "voxtream_gguf.json"
FILES_KEY = "files"
STACKS_KEY = "stacks"

# A vocab is structurally required by llama.cpp even though the stacks have no
# embedding layer: embeddings go in and come out.
STUB_VOCAB_SIZE = 8


@dataclass(frozen=True)
class TransformerSpec:
    """Shape of one torchtune llama3_2 stack of the checkpoint."""

    prefix: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    embed_dim: int
    intermediate_dim: int
    head_dim: int
    max_seq_len: int
    norm_eps: float
    rope_base: float
    rope_scale_factor: float
    rope_low_freq_factor: float
    rope_high_freq_factor: float
    rope_old_context_len: int
    # name of a linear head (``<name>.weight`` in the state dict, no bias) folded
    # into ``output.weight`` ahead of the identity block; None (the depth
    # transformer) leaves the identity block alone
    output_head: str = None
    n_logits: int = 0
    kind: str = "llama"
    source: str = "model"

    @property
    def n_vocab(self) -> int:
        """Rows of ``output.weight``: the head's logits, then the hidden state."""
        return self.n_logits + self.embed_dim

    def filename(self) -> str:
        return f"{self.prefix}-{QUANT}.gguf"


@dataclass(frozen=True)
class MoGMLPSpec:
    """Shape of the MLP stack of the MoG head (``MoGHead.mlp``)."""

    prefix: str
    num_layers: int
    embed_dim: int
    intermediate_dim: int
    norm_eps: float
    # llama.cpp scales the input of a Gemma2 model by sqrt(embed_dim); the
    # backend pre-scales by this so the stack sees the input unchanged
    input_scale: float
    # dimension of the (zero) attention projections that make a Gemma2 layer a
    # plain residual MLP layer
    attn_dim: int = 32
    kind: str = "gemma2"
    source: str = "model"

    @property
    def name(self) -> str:
        return "mog_head"

    def filename(self) -> str:
        return f"{self.name}-{QUANT}.gguf"


@dataclass(frozen=True)
class MimiTransformerSpec:
    """Shape of the decoder transformer of Mimi (moshi ``StreamingTransformer``).

    A pre-LayerNorm transformer with RoPE, a GELU feed-forward and a learned
    per-channel scale on both residual branches, attending to the last
    ``context`` positions. llama.cpp runs it as ``arch=gptneox`` (sequential
    residual): the layer scales are folded into the output projections and the
    query/key rows permuted from torch's interleaved RoPE pairs to the NEOX
    layout. The context window is kept by the backend.
    """

    prefix: str
    num_layers: int
    num_heads: int
    embed_dim: int
    intermediate_dim: int
    head_dim: int
    norm_eps: float
    rope_base: float
    context: int
    # scale of the channel carrying the per-frame mean of the stream, and the
    # constant put into two channels before the final norm; see
    # `write_mimi_transformer_gguf`
    mean_scale: float = 2.0
    shift: float = 8.0
    kind: str = "gptneox"
    source: str = "mimi"

    @property
    def name(self) -> str:
        return "mimi_decoder_transformer"

    @property
    def padded_dim(self) -> int:
        """Channels of the residual stream in llama.cpp: the model's and one zero head
        holding the mean channel, the two shift channels and zeros."""
        return self.embed_dim + self.head_dim

    @property
    def mean_channel(self) -> int:
        return self.embed_dim

    @property
    def shift_channels(self) -> tuple:
        return self.embed_dim + 1, self.embed_dim + 2

    def filename(self) -> str:
        return f"{self.name}-{QUANT}.gguf"


SPEC_KINDS = {
    "llama": TransformerSpec,
    "gemma2": MoGMLPSpec,
    "gptneox": MimiTransformerSpec,
}


def transformer_spec(
    prefix: str, module, output_head: str = None, n_logits: int = 0
) -> TransformerSpec:
    """Spec of a torchtune ``TransformerDecoder`` built by ``MODEL_POOL``."""
    attn = module.layers[0].attn
    rope = attn.pos_embeddings
    return TransformerSpec(
        prefix=prefix,
        num_layers=len(module.layers),
        num_heads=attn.num_heads,
        num_kv_heads=attn.num_kv_heads,
        embed_dim=attn.embed_dim,
        intermediate_dim=module.layers[0].mlp.w1.out_features,
        head_dim=attn.head_dim,
        max_seq_len=module.max_seq_len,
        norm_eps=float(module.layers[0].sa_norm.eps),
        rope_base=float(rope.base),
        rope_scale_factor=float(rope.scale_factor),
        rope_low_freq_factor=float(rope.low_freq_factor),
        rope_high_freq_factor=float(rope.high_freq_factor),
        rope_old_context_len=int(rope.old_context_len),
        output_head=output_head,
        n_logits=n_logits,
    )


def mog_mlp_spec(prefix: str, mog_head) -> MoGMLPSpec:
    """Spec of ``MoGHead.mlp``: the residual MLP layers and the final RMSNorm."""
    layers = list(mog_head.mlp)[:-1]
    return MoGMLPSpec(
        prefix=prefix,
        num_layers=len(layers),
        embed_dim=layers[0].gate_proj.in_features,
        intermediate_dim=layers[0].gate_proj.out_features,
        norm_eps=float(layers[0].pre_norm.eps),
        input_scale=1.0 / math.sqrt(layers[0].gate_proj.in_features),
    )


def mimi_transformer_spec(mimi) -> MimiTransformerSpec:
    """Spec of ``mimi.decoder_transformer`` of a moshi ``MimiModel``."""
    transformer = mimi.decoder_transformer.transformer
    layer = transformer.layers[0]
    attn = layer.self_attn
    assert mimi.decoder_transformer.input_proj is None, "unexpected input projection"
    assert attn.rope is not None and attn.rope.interleave, "expected interleaved RoPE"
    assert attn.causal and attn.context, "expected a causal context"
    assert layer.gating is None, "expected a plain GELU feed-forward"
    return MimiTransformerSpec(
        prefix="decoder_transformer.transformer",
        num_layers=len(transformer.layers),
        num_heads=attn.num_heads,
        embed_dim=attn.embed_dim,
        intermediate_dim=layer.linear1.out_features,
        head_dim=attn.embed_dim // attn.num_heads,
        norm_eps=float(layer.norm1.eps),
        rope_base=float(attn.rope.max_period),
        context=int(attn.context),
    )


def model_specs(model) -> dict:
    """The stacks of a ``voxtream.model.Model`` llama.cpp can take over."""
    specs = {
        "temp_former": transformer_spec(
            "temp_former",
            model.temp_former,
            output_head="sem_head",
            n_logits=model.sem_head.out_features,
        )
    }
    if model.dep_former is not None:
        specs["dep_former"] = transformer_spec("dep_former", model.dep_former)
    if getattr(model, "mog_head", None) is not None:
        specs["mog_head"] = mog_mlp_spec("mog_head.mlp", model.mog_head)
    return specs


def rope_freq_factors(spec: TransformerSpec) -> np.ndarray:
    """Per-frequency RoPE divisors equivalent to ``Llama3ScaledRoPE.apply_scaling``.

    torchtune rescales the frequencies directly; ggml instead divides the base
    angle by a per-frequency factor supplied as ``rope_freqs.weight``.
    """
    head_dim = spec.head_dim
    freqs = 1.0 / (
        spec.rope_base
        ** (np.arange(0, head_dim, 2, dtype=np.float64)[: head_dim // 2] / head_dim)
    )
    low_freq_wavelen = spec.rope_old_context_len / spec.rope_low_freq_factor
    high_freq_wavelen = spec.rope_old_context_len / spec.rope_high_freq_factor

    factors = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            factors.append(1.0)
        elif wavelen > low_freq_wavelen:
            factors.append(float(spec.rope_scale_factor))
        else:
            smooth = (
                spec.rope_old_context_len / wavelen - spec.rope_low_freq_factor
            ) / (spec.rope_high_freq_factor - spec.rope_low_freq_factor)
            factors.append(1.0 / ((1 - smooth) / spec.rope_scale_factor + smooth))

    return np.asarray(factors, dtype=np.float32)


class _Writer:
    """gguf writer with the projection matrices in q8_0."""

    def __init__(self, out_path: Path, arch: str, state_dict):
        import gguf

        self.gguf = gguf
        self.writer = gguf.GGUFWriter(str(out_path), arch=arch)
        self.state_dict = state_dict

    def get(self, name: str) -> np.ndarray:
        return self.state_dict[name].detach().float().cpu().numpy()

    def add_weight(self, gguf_name: str, data: np.ndarray) -> None:
        """One projection matrix, in q8_0."""
        data = np.ascontiguousarray(data, dtype=np.float32)
        q8_0 = self.gguf.GGMLQuantizationType.Q8_0
        # gguf-py derives the logical shape from the packed byte shape, so the
        # quantized array is passed through as-is.
        self.writer.add_tensor(
            gguf_name, self.gguf.quants.quantize(data, q8_0), raw_dtype=q8_0
        )

    def add_f32(self, gguf_name: str, data: np.ndarray) -> None:
        self.writer.add_tensor(gguf_name, np.ascontiguousarray(data, dtype=np.float32))

    def close(self) -> None:
        self.writer.add_file_type(self.gguf.LlamaFileType.MOSTLY_Q8_0)
        self.writer.write_header_to_file()
        self.writer.write_kv_data_to_file()
        self.writer.write_tensors_to_file()
        self.writer.close()


def write_transformer_gguf(spec: TransformerSpec, state_dict, out_path: Path) -> None:
    """One torchtune llama3_2 stack as an ``arch=llama`` GGUF."""
    w = _Writer(out_path, "llama", state_dict)
    writer = w.writer
    writer.add_name(f"voxtream2-{spec.prefix}")
    writer.add_block_count(spec.num_layers)
    writer.add_context_length(spec.max_seq_len)
    writer.add_embedding_length(spec.embed_dim)
    writer.add_feed_forward_length(spec.intermediate_dim)
    writer.add_head_count(spec.num_heads)
    writer.add_head_count_kv(spec.num_kv_heads)
    writer.add_key_length(spec.head_dim)
    writer.add_value_length(spec.head_dim)
    writer.add_layer_norm_rms_eps(spec.norm_eps)
    writer.add_rope_dimension_count(spec.head_dim)
    writer.add_rope_freq_base(spec.rope_base)
    writer.add_vocab_size(spec.n_vocab)
    # no tokenizer: embeddings go in and come out
    writer.add_tokenizer_model("none")

    p = spec.prefix
    # torchtune's RoPE pairs adjacent channels (x[2i], x[2i+1]), which is ggml's
    # GGML_ROPE_TYPE_NORM layout used by arch=llama: no head permutation needed.
    for i in range(spec.num_layers):
        w.add_weight(
            f"blk.{i}.attn_q.weight", w.get(f"{p}.layers.{i}.attn.q_proj.weight")
        )
        w.add_weight(
            f"blk.{i}.attn_k.weight", w.get(f"{p}.layers.{i}.attn.k_proj.weight")
        )
        w.add_weight(
            f"blk.{i}.attn_v.weight", w.get(f"{p}.layers.{i}.attn.v_proj.weight")
        )
        w.add_weight(
            f"blk.{i}.attn_output.weight",
            w.get(f"{p}.layers.{i}.attn.output_proj.weight"),
        )
        w.add_weight(f"blk.{i}.ffn_gate.weight", w.get(f"{p}.layers.{i}.mlp.w1.weight"))
        w.add_weight(f"blk.{i}.ffn_down.weight", w.get(f"{p}.layers.{i}.mlp.w2.weight"))
        w.add_weight(f"blk.{i}.ffn_up.weight", w.get(f"{p}.layers.{i}.mlp.w3.weight"))
        # RMSNorm scales stay f32, as llama.cpp expects
        w.add_f32(f"blk.{i}.attn_norm.weight", w.get(f"{p}.layers.{i}.sa_norm.scale"))
        w.add_f32(f"blk.{i}.ffn_norm.weight", w.get(f"{p}.layers.{i}.mlp_norm.scale"))

    w.add_f32("output_norm.weight", w.get(f"{p}.norm.scale"))
    w.add_f32("rope_freqs.weight", rope_freq_factors(spec))

    # The embedding matrix is structurally required and never evaluated. The
    # output matrix is the folded head: its logits, then the identity, so the
    # logits of a token end with its final hidden state. Without a head (the
    # depth transformer) it is the identity alone.
    w.add_weight(
        "token_embd.weight", np.zeros((spec.n_vocab, spec.embed_dim), dtype=np.float32)
    )
    output = np.eye(spec.embed_dim, dtype=np.float32)
    if spec.output_head:
        head = w.get(f"{spec.output_head}.weight")
        assert head.shape == (spec.n_logits, spec.embed_dim), head.shape
        output = np.concatenate([head, output])
    w.add_weight("output.weight", output)
    w.close()


def write_mog_mlp_gguf(spec: MoGMLPSpec, state_dict, out_path: Path) -> None:
    """The MLP stack of the MoG head as an ``arch=gemma2`` GGUF.

    A Gemma2 layer is ``x + post_attn_norm(attn(attn_norm(x)))`` followed by
    ``x + post_ffn_norm(ffn(ffn_norm(x)))`` with a gated GELU (tanh) feed-forward
    and RMSNorms that scale by ``1 + w``, which is ``MLPLayer`` of
    ``voxtream.utils.mog`` once the attention contributes nothing: a zero value
    projection makes the attention output zero, and the RMSNorm of zero is zero.
    """
    w = _Writer(out_path, "gemma2", state_dict)
    writer = w.writer
    d, a = spec.embed_dim, spec.attn_dim
    writer.add_name("voxtream2-mog-head-mlp")
    writer.add_block_count(spec.num_layers)
    writer.add_context_length(64)
    writer.add_embedding_length(d)
    writer.add_feed_forward_length(spec.intermediate_dim)
    writer.add_head_count(1)
    writer.add_head_count_kv(1)
    writer.add_key_length(a)
    writer.add_value_length(a)
    writer.add_layer_norm_rms_eps(spec.norm_eps)
    writer.add_rope_dimension_count(a)
    writer.add_rope_freq_base(10000.0)
    writer.add_sliding_window(64)
    writer.add_attn_logit_softcapping(50.0)
    writer.add_final_logit_softcapping(30.0)
    writer.add_vocab_size(STUB_VOCAB_SIZE)
    writer.add_tokenizer_model("none")

    p = spec.prefix
    zeros_qkv = np.zeros((a, d), dtype=np.float32)
    zeros_o = np.zeros((d, a), dtype=np.float32)
    ones = np.ones(d, dtype=np.float32)
    for i in range(spec.num_layers):
        w.add_f32(f"blk.{i}.attn_norm.weight", ones)
        w.add_weight(f"blk.{i}.attn_q.weight", zeros_qkv)
        w.add_weight(f"blk.{i}.attn_k.weight", zeros_qkv)
        w.add_weight(f"blk.{i}.attn_v.weight", zeros_qkv)
        w.add_weight(f"blk.{i}.attn_output.weight", zeros_o)
        w.add_f32(f"blk.{i}.post_attention_norm.weight", ones)
        # gemma2 stores every RMSNorm scale as 1 + w
        w.add_f32(f"blk.{i}.ffn_norm.weight", 1.0 + w.get(f"{p}.{i}.pre_norm.weight"))
        w.add_weight(f"blk.{i}.ffn_gate.weight", w.get(f"{p}.{i}.gate_proj.weight"))
        w.add_weight(f"blk.{i}.ffn_up.weight", w.get(f"{p}.{i}.up_proj.weight"))
        w.add_weight(f"blk.{i}.ffn_down.weight", w.get(f"{p}.{i}.down_proj.weight"))
        w.add_f32(
            f"blk.{i}.post_ffw_norm.weight", 1.0 + w.get(f"{p}.{i}.post_norm.weight")
        )
    w.add_f32("output_norm.weight", 1.0 + w.get(f"{p}.{spec.num_layers}.weight"))
    w.add_weight("token_embd.weight", np.zeros((STUB_VOCAB_SIZE, d), dtype=np.float32))
    w.close()


def write_mimi_transformer_gguf(
    spec: MimiTransformerSpec, state_dict, out_path: Path
) -> None:
    """The decoder transformer of Mimi as an ``arch=gptneox`` GGUF.

    Mimi's stack has no final norm, while every llama.cpp architecture ends with
    one, and a LayerNorm output cannot be undone: the per-frame mean and scale
    of the residual stream are gone. They are carried through the network
    instead, in channels of a widened residual stream (one extra, all-zero
    attention head):

    * the model's channels hold the stream less its per-frame mean: every
      branch output is projected onto zero mean (its channel mean subtracted),
      which the branch inputs cannot see, since a LayerNorm is shift-invariant;
    * one channel accumulates ``mean_scale`` times that mean, from the input
      and from the channel mean of every branch output, which is a row of
      the branch's output projection;
    * the remaining channels are zero, so that the per-layer norms see a mean
      of zero and a variance scaled by a known factor, which the norm scales
      and epsilon absorb exactly;
    * the last feed-forward bias puts ``+shift`` and ``-shift`` into two zero
      channels: their outputs after the final norm give its scale, from which
      the backend restores the stream and its mean.
    """
    w = _Writer(out_path, "gptneox", state_dict)
    writer = w.writer
    d, e, h, hd = spec.embed_dim, spec.padded_dim, spec.num_heads, spec.head_dim
    ff = spec.intermediate_dim
    writer.add_name("mimi-decoder-transformer")
    writer.add_block_count(spec.num_layers)
    writer.add_context_length(4096)
    writer.add_embedding_length(e)
    writer.add_feed_forward_length(ff)
    writer.add_head_count(h + 1)
    writer.add_head_count_kv(h + 1)
    writer.add_key_length(hd)
    writer.add_value_length(hd)
    # the norms see d live channels of e: their variance is scaled by d / e
    writer.add_layer_norm_eps(spec.norm_eps * d / e)
    norm_scale = math.sqrt(d / e)
    writer.add_rope_dimension_count(hd)
    writer.add_rope_freq_base(spec.rope_base)
    writer.add_parallel_residual(False)
    writer.add_vocab_size(STUB_VOCAB_SIZE)
    writer.add_tokenizer_model("none")

    def to_neox(rows: np.ndarray) -> np.ndarray:
        """Rows of a projection permuted from interleaved RoPE pairs (2j, 2j+1)
        to the NEOX pairs (j, j + head_dim / 2), head by head."""
        return rows.reshape(h, hd // 2, 2, d).transpose(0, 2, 1, 3).reshape(h * hd, d)

    def pad_rows(m: np.ndarray) -> np.ndarray:
        return np.concatenate([m, np.zeros((e - d, m.shape[1]), dtype=np.float32)])

    def pad_cols(m: np.ndarray) -> np.ndarray:
        return np.concatenate([m, np.zeros((m.shape[0], e - d), dtype=np.float32)], 1)

    def stream_rows(m: np.ndarray) -> np.ndarray:
        """A branch output projection (d, n) onto the widened stream: zero-mean
        rows, then the row accumulating the mean."""
        out = np.zeros((e, m.shape[1]), dtype=np.float32)
        out[:d] = m - m.mean(0, keepdims=True)
        out[d] = spec.mean_scale * m.mean(0)
        return out

    def norm_params(name: str) -> tuple:
        weight = np.zeros(e, dtype=np.float32)
        bias = np.zeros(e, dtype=np.float32)
        weight[:d] = w.get(f"{name}.weight") * norm_scale
        bias[:d] = w.get(f"{name}.bias")
        return weight, bias

    p = spec.prefix
    for i in range(spec.num_layers):
        q = f"{p}.layers.{i}"
        qkv = w.get(f"{q}.self_attn.in_projs.0.weight")
        qkv = np.concatenate(
            [
                pad_rows(to_neox(qkv[:d])),
                pad_rows(to_neox(qkv[d : 2 * d])),
                pad_rows(qkv[2 * d :]),
            ]
        )
        w.add_weight(f"blk.{i}.attn_qkv.weight", pad_cols(qkv))
        w.add_f32(f"blk.{i}.attn_qkv.bias", np.zeros(3 * e, dtype=np.float32))
        # the layer scale of a residual branch is a row scale of its last projection
        ls1 = w.get(f"{q}.layer_scale_1.scale")[:, None]
        ls2 = w.get(f"{q}.layer_scale_2.scale")[:, None]
        attn_out = stream_rows(ls1 * w.get(f"{q}.self_attn.out_projs.0.weight"))
        w.add_weight(f"blk.{i}.attn_output.weight", pad_cols(attn_out))
        w.add_f32(f"blk.{i}.attn_output.bias", np.zeros(e, dtype=np.float32))
        weight, bias = norm_params(f"{q}.norm1")
        w.add_f32(f"blk.{i}.attn_norm.weight", weight)
        w.add_f32(f"blk.{i}.attn_norm.bias", bias)
        weight, bias = norm_params(f"{q}.norm2")
        w.add_f32(f"blk.{i}.ffn_norm.weight", weight)
        w.add_f32(f"blk.{i}.ffn_norm.bias", bias)
        w.add_weight(f"blk.{i}.ffn_up.weight", pad_cols(w.get(f"{q}.linear1.weight")))
        w.add_f32(f"blk.{i}.ffn_up.bias", np.zeros(ff, dtype=np.float32))
        w.add_weight(
            f"blk.{i}.ffn_down.weight", stream_rows(ls2 * w.get(f"{q}.linear2.weight"))
        )
        down_bias = np.zeros(e, dtype=np.float32)
        if i == spec.num_layers - 1:
            plus, minus = spec.shift_channels
            down_bias[plus], down_bias[minus] = spec.shift, -spec.shift
        w.add_f32(f"blk.{i}.ffn_down.bias", down_bias)
    # the final norm llama.cpp insists on; the backend undoes it
    w.add_f32("output_norm.weight", np.ones(e, dtype=np.float32))
    w.add_f32("output_norm.bias", np.zeros(e, dtype=np.float32))
    stub = np.zeros((STUB_VOCAB_SIZE, e), dtype=np.float32)
    w.add_weight("token_embd.weight", stub)
    w.add_weight("output.weight", stub)
    w.close()


def write_stacks(
    specs: Dict[str, object], state_dicts: Dict[str, dict], out_dir: Path
) -> Dict[str, Path]:
    """One GGUF per stack. Returns {stack name: path}.

    Args:
        state_dicts: the weights of every source ("model", "mimi")
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, spec in specs.items():
        out_path = out_dir / spec.filename()
        state_dict = state_dicts[spec.source]
        if isinstance(spec, MoGMLPSpec):
            write_mog_mlp_gguf(spec, state_dict, out_path)
        elif isinstance(spec, MimiTransformerSpec):
            write_mimi_transformer_gguf(spec, state_dict, out_path)
        else:
            write_transformer_gguf(spec, state_dict, out_path)
        paths[name] = out_path
    return paths


def write_manifest(out_dir: Path, specs: Dict[str, object], paths) -> Path:
    manifest_path = out_dir / MANIFEST_NAME
    manifest = {
        FILES_KEY: {name: path.name for name, path in paths.items()},
        STACKS_KEY: {name: asdict(spec) for name, spec in specs.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def load_manifest(gguf_dir: Path) -> dict:
    manifest_path = Path(gguf_dir) / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found; convert the checkpoint with "
            "`voxtream-convert-gguf` first"
        )
    manifest = json.loads(manifest_path.read_text())
    stacks = {}
    for name, fields in manifest.get(STACKS_KEY, {}).items():
        stacks[name] = SPEC_KINDS[fields["kind"]](**fields)
    manifest[STACKS_KEY] = stacks
    return manifest


def convert_model(model, state_dict, out_dir: Path, mimi=None) -> Dict[str, Path]:
    """Write the GGUF files of every stack of ``model``.

    Args:
        model: the ``voxtream.model.Model`` the state dict belongs to, for the
            shapes of its stacks
        state_dict: its weights (``load_model_weights``)
        mimi: a loaded moshi ``MimiModel`` whose decoder transformer is converted
            as well (its other parts stay in PyTorch and load from the Hub)

    Returns:
        {stack name: path}
    """
    specs = model_specs(model)
    state_dicts = {"model": state_dict}
    if mimi is not None:
        spec = mimi_transformer_spec(mimi)
        specs[spec.name] = spec
        state_dicts["mimi"] = {k: v for k, v in mimi.state_dict().items()}
    out_dir = Path(out_dir)
    paths = write_stacks(specs, state_dicts, out_dir)
    write_manifest(out_dir, specs, paths)
    return paths


def main() -> None:
    import argparse

    import torch

    from voxtream.config import SpeechGeneratorConfig
    from voxtream.utils.generator import existing_file
    from voxtream.utils.generator.setup import build_generator_model
    from voxtream.utils.model import load_model_weights

    parser = argparse.ArgumentParser(description="Convert VoXtream2 to GGUF")
    parser.add_argument(
        "-cfg",
        "--config",
        type=existing_file,
        default="configs/generator.json",
        help="Generator config naming the checkpoint (model_weight_path or the Hub)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for the GGUF files (default: gguf_dir of the config, else "
        "model_repo when it is a local directory, so the files sit next to the weights)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the converted stacks through llama.cpp on random inputs and "
        "compare them with the PyTorch stacks",
    )
    parser.add_argument(
        "--no-mimi",
        action="store_true",
        help="Leave the decoder transformer of Mimi in PyTorch",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="CPU threads of PyTorch and of llama.cpp (--verify)",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    with open(args.config) as f:
        config = SpeechGeneratorConfig(**json.load(f))

    model, _, weight_path = build_generator_model(config)
    state_dict = load_model_weights(weight_path)
    out_dir = args.out_dir or config.gguf_dir
    if out_dir is None:
        if not Path(config.model_repo).is_dir():
            parser.error("give --out-dir: model_repo is not a local directory")
        out_dir = config.model_repo
    out_dir = Path(out_dir)
    print(f"Converting {weight_path} -> {out_dir}")

    mimi = None
    if not args.no_mimi:
        from voxtream.utils.generator.setup import load_mimi_model

        mimi = load_mimi_model(config, "cpu", torch.float32, config.num_codebooks)

    paths = convert_model(model, state_dict, out_dir, mimi=mimi)
    for name, path in paths.items():
        print(f"{name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    if args.verify:
        from voxtream.utils.gguf.backend import verify_stacks

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        for line in verify_stacks(model, out_dir, mimi=mimi, n_threads=args.threads):
            print(line)


if __name__ == "__main__":
    main()
