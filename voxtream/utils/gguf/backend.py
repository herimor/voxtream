"""llama.cpp-backed drop-in replacements for the VoXtream2 transformer stacks.

``LlamaCppTransformer`` mimics the slice of the torchtune ``TransformerDecoder``
API that ``voxtream.model.Model`` uses at inference time:

    transformer(h, input_pos=..., mask=...) -> torch.Tensor
    transformer.reset_caches() / caches_are_enabled() / setup_caches(...)

``LlamaCppMoGMLP`` replaces ``MoGHead.mlp`` (``mlp(x) -> features``).

The temporal transformer mask
-----------------------------
``SinkAttention`` hands the temporal transformer a mask that is causal over the
live context (the recent window never masks anything: the cache is rebuilt
before a position could fall out of it) with one addition llama.cpp cannot
express directly: a constant ``prompt_attention_bias`` added to the attention
logit of every acoustic-prompt key. A bias ``b`` on a key is the same softmax as
``e^b`` copies of that key, so the backend emulates it exactly, up to rounding
``e^b`` to an integer: after the prompt is prefilled its KV entries are saved
(``llama_state_seq_get_data``), restored ``round(e^b) - 1`` times into auxiliary
sequences, and those cells are shared back with the query's sequence
(``llama_memory_seq_cp``, which in a unified KV cache adds the sequence id to
the cells instead of copying them). The effective bias is ``ln(round(e^b))``:
``3.0`` becomes ``ln 20 = 2.996``.

The CFG batch of two (conditional + unconditional) maps onto two llama.cpp
sequences in one unified context.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from voxtream.utils.gguf.convert import (
    FILES_KEY,
    MANIFEST_NAME,
    QUANT,
    STACKS_KEY,
    MimiTransformerSpec,
    MoGMLPSpec,
    TransformerSpec,
    load_manifest,
)

logger = logging.getLogger("voxtream")

# sequences one llama.cpp context may hold (LLAMA_MAX_SEQ)
MAX_SEQ = 64


def default_threads() -> int:
    """Performance cores only.

    The stacks decode one or two tokens at a time, so the per-matmul barrier
    dominates: spilling onto Apple's efficiency cores makes every thread wait for
    the slowest one.
    """
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True,
            text=True,
            check=True,
        )
        return max(1, int(out.stdout.strip()))
    except Exception:
        return max(1, os.cpu_count() or 4)


def bias_copies(bias: float) -> int:
    """Copies of every prompt key that emulate an attention bias of ``bias``."""
    if bias is None or bias <= 0:
        return 0
    return max(0, int(round(math.exp(bias))) - 1)


def effective_bias(bias: float) -> float:
    return math.log(bias_copies(bias) + 1)


_LOG_SILENCED = False
_NULL_LOG_CB = None


def _silence_llama_log(llama_cpp) -> None:
    global _LOG_SILENCED, _NULL_LOG_CB
    if _LOG_SILENCED:
        return

    @llama_cpp.llama_log_callback
    def _cb(level, text, user_data):  # noqa: ARG001
        pass

    _NULL_LOG_CB = _cb
    llama_cpp.llama_log_set(_cb, ctypes.c_void_p(0))
    _LOG_SILENCED = True


def _load_llama_cpp():
    import llama_cpp

    return llama_cpp


class _LlamaCppStack:
    """A GGUF model and one context, fed embeddings through a fixed-layout batch."""

    def __init__(
        self,
        gguf_path: Path,
        embed_dim: int,
        n_ctx: int,
        n_batch: int,
        n_seq_max: int,
        n_threads: int = None,
        n_gpu_layers: int = 0,
        embeddings: bool = False,
        flash_attn: bool = True,
        verbose: bool = False,
    ) -> None:
        self._llama_cpp = llama_cpp = _load_llama_cpp()
        self.embed_dim = embed_dim
        self.n_threads = n_threads or default_threads()
        if not verbose:
            _silence_llama_log(llama_cpp)
        llama_cpp.llama_backend_init()

        model_params = llama_cpp.llama_model_default_params()
        model_params.n_gpu_layers = n_gpu_layers
        model_params.use_mmap = True
        if n_gpu_layers == 0:
            # an empty (NULL-terminated) device list keeps ggml off Metal entirely
            self._devices = (ctypes.c_void_p * 1)(None)
            model_params.devices = ctypes.cast(self._devices, ctypes.c_void_p)
        self._model = llama_cpp.llama_model_load_from_file(
            str(gguf_path).encode("utf-8"), model_params
        )
        if not self._model:
            raise RuntimeError(f"llama.cpp failed to load {gguf_path}")
        assert (
            llama_cpp.llama_model_n_embd(self._model) == embed_dim
        ), "embed_dim mismatch"

        ctx_params = llama_cpp.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_batch
        ctx_params.n_ubatch = n_batch
        ctx_params.n_seq_max = n_seq_max
        ctx_params.n_outputs_max = n_batch
        ctx_params.n_threads = self.n_threads
        ctx_params.n_threads_batch = self.n_threads
        ctx_params.embeddings = embeddings
        ctx_params.pooling_type = llama_cpp.LLAMA_POOLING_TYPE_NONE
        ctx_params.type_k = llama_cpp.GGML_TYPE_F16
        ctx_params.type_v = llama_cpp.GGML_TYPE_F16
        ctx_params.flash_attn_type = (
            llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
            if flash_attn
            else llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED
        )
        # one KV cache shared by all sequences: a cell may belong to several of
        # them, which the prompt-key duplication relies on
        ctx_params.kv_unified = True
        ctx_params.no_perf = True
        self._ctx = llama_cpp.llama_init_from_model(self._model, ctx_params)
        if not self._ctx:
            raise RuntimeError(f"llama.cpp failed to create a context for {gguf_path}")
        self._memory = llama_cpp.llama_get_memory(self._ctx)

        self._n_batch = n_batch
        self._batch = llama_cpp.llama_batch_init(n_batch, embed_dim, 1)
        self._flat_embd = np.ctypeslib.as_array(
            self._batch.embd, shape=(n_batch * embed_dim,)
        )
        self._flat_pos = np.ctypeslib.as_array(self._batch.pos, shape=(n_batch,))
        self._flat_logits = np.ctypeslib.as_array(self._batch.logits, shape=(n_batch,))
        np.ctypeslib.as_array(self._batch.n_seq_id, shape=(n_batch,))[:] = 1

    def _decode(
        self,
        embd: np.ndarray,
        pos: np.ndarray,
        seq_ids: np.ndarray,
        outputs: np.ndarray,
    ) -> int:
        """
        Args:
            embd: (n_tokens, embed_dim) float32
            pos, seq_ids, outputs: (n_tokens,) position, sequence and output flag

        Returns:
            number of output tokens
        """
        n_tokens = embd.shape[0]
        assert n_tokens <= self._n_batch, "batch larger than the llama.cpp batch"
        self._batch.n_tokens = n_tokens
        self._flat_embd[: n_tokens * self.embed_dim] = embd.reshape(-1)
        self._flat_pos[:n_tokens] = pos
        self._flat_logits[:n_tokens] = outputs
        for i in range(n_tokens):
            self._batch.seq_id[i][0] = int(seq_ids[i])
        rc = self._llama_cpp.llama_decode(self._ctx, self._batch)
        if rc != 0:
            raise RuntimeError(f"llama_decode failed with code {rc}")
        return int(outputs.sum())

    def _read(self, ptr, n_rows: int, n_cols: int) -> np.ndarray:
        if not ptr:
            raise RuntimeError("llama.cpp returned no output")
        out = np.ctypeslib.as_array(ptr, shape=(n_rows, n_cols))
        # always copy: the buffer is llama.cpp's own and the next decode overwrites it
        return np.array(out, dtype=np.float32, copy=True)

    def clear(self) -> None:
        self._llama_cpp.llama_memory_clear(self._memory, True)

    def set_threads(self, n_threads: int) -> None:
        self.n_threads = int(n_threads)
        self._llama_cpp.llama_set_n_threads(self._ctx, n_threads, n_threads)

    def close(self) -> None:
        llama_cpp = self._llama_cpp
        if getattr(self, "_batch", None) is not None:
            llama_cpp.llama_batch_free(self._batch)
            self._batch = None
        if getattr(self, "_ctx", None) is not None:
            llama_cpp.llama_free(self._ctx)
            self._ctx = None
        if getattr(self, "_model", None) is not None:
            llama_cpp.llama_model_free(self._model)
            self._model = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


class LlamaCppTransformer(_LlamaCppStack):
    """One GGUF transformer stack executed by llama.cpp, embeddings in and out.

    The output of a call is the final hidden state of the last position of every
    sequence, ``(batch_size, 1, embed_dim)``; the model only ever reads that
    row. When the GGUF carries a folded head (the semantic head of the temporal
    transformer), its logits of the same rows are kept in ``logits`` and served
    by ``sem_head``.
    """

    def __init__(
        self,
        gguf_path: Path,
        spec: TransformerSpec,
        batch_size: int,
        n_threads: int = None,
        n_gpu_layers: int = 0,
        max_positions: int = None,
        max_prompt_len: int = 0,
        prompt_bias: float = None,
        flash_attn: bool = True,
        verbose: bool = False,
        check_mask: bool = False,
    ) -> None:
        self.spec = spec
        self.batch_size = batch_size
        self.max_seq_len = spec.max_seq_len
        self.check_mask = check_mask
        self.n_logits = spec.n_logits if spec.output_head else 0
        self.logits: Optional[torch.Tensor] = None
        self._caches_enabled = False
        self._bias_warned = set()

        max_positions = max_positions or spec.max_seq_len
        self.max_copies = bias_copies(prompt_bias)
        self.max_prompt_len = max_prompt_len if self.max_copies else 0
        n_seq_max = batch_size * (1 + self.max_copies)
        if n_seq_max > MAX_SEQ:
            raise ValueError(
                f"a prompt attention bias of {prompt_bias} needs {self.max_copies} "
                f"copies of the prompt keys, {n_seq_max} llama.cpp sequences for a "
                f"batch of {batch_size}; at most {MAX_SEQ} are available, so the bias "
                f"must stay below {math.log(MAX_SEQ // batch_size + 0.5):.2f}"
            )
        n_batch = batch_size * max_positions
        n_ctx = batch_size * (max_positions + self.max_copies * self.max_prompt_len)
        super().__init__(
            gguf_path,
            spec.embed_dim,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_seq_max=n_seq_max,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            embeddings=False,
            flash_attn=flash_attn,
            verbose=verbose,
        )
        self.n_vocab = spec.n_vocab
        if hasattr(self._llama_cpp, "llama_model_get_vocab"):
            vocab = self._llama_cpp.llama_model_get_vocab(self._model)
            n_vocab = self._llama_cpp.llama_vocab_n_tokens(vocab)
            if n_vocab:
                assert n_vocab == spec.n_vocab, f"vocab {n_vocab} != {spec.n_vocab}"

    # -- torchtune TransformerDecoder surface ----------------------------------

    def setup_caches(self, max_batch_size: int, dtype: torch.dtype, **kwargs) -> None:
        self._caches_enabled = True

    def caches_are_enabled(self) -> bool:
        return self._caches_enabled

    def reset_caches(self) -> None:
        self.clear()

    def sem_head(self, last_h: torch.Tensor) -> torch.Tensor:
        """The logits of the folded head for the rows of the last call."""
        if self.logits is None or self.logits.shape[0] != last_h.shape[0]:
            raise RuntimeError("no logits of the folded head for this batch")
        return self.logits.to(device=last_h.device, dtype=last_h.dtype)

    def __call__(
        self,
        h: torch.Tensor,
        input_pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run ``h`` (batch, seq_len, embed_dim) at ``input_pos`` through the stack."""
        bs, seq_len, dim = h.shape
        assert dim == self.embed_dim, f"expected embed_dim {self.embed_dim}, got {dim}"
        assert bs == self.batch_size, f"expected batch {self.batch_size}, got {bs}"

        positions = input_pos
        if positions.shape[0] == 1 and bs > 1:
            positions = positions.expand(bs, -1)
        prompt_len, copies = self._parse_mask(mask, positions)

        embd = h.detach().to(dtype=torch.float32).contiguous().cpu().numpy()
        pos = positions.detach().to(dtype=torch.int32).contiguous().cpu().numpy()

        if seq_len > 1 and copies > 0:
            # a prefill that holds the prompt: its rows first, so that their keys
            # can be duplicated before any later row attends to them
            n_prompt = int((pos[0] < prompt_len).sum())
            if n_prompt > 0:
                self._run(embd[:, :n_prompt], pos[:, :n_prompt], n_prompt == seq_len)
                self._duplicate_prompt(prompt_len, copies)
            if n_prompt < seq_len:
                self._run(embd[:, n_prompt:], pos[:, n_prompt:], True)
        else:
            self._run(embd, pos, True)

        return self._read_outputs(h.dtype, h.device)

    # -- internals -------------------------------------------------------------

    def _run(self, embd: np.ndarray, pos: np.ndarray, want_outputs: bool) -> None:
        """Decode a (batch, seq_len, dim) slice, position-major in the llama.cpp batch,
        the last token of every sequence an output."""
        bs, seq_len = pos.shape
        n_tokens = bs * seq_len
        seq_ids = np.tile(np.arange(bs, dtype=np.int32), seq_len)
        outputs = np.zeros(n_tokens, dtype=np.int8)
        if want_outputs:
            outputs[-bs:] = 1
        self._decode(
            embd.transpose(1, 0, 2).reshape(n_tokens, -1),
            pos.transpose(1, 0).reshape(-1),
            seq_ids,
            outputs,
        )

    def _duplicate_prompt(self, prompt_len: int, copies: int) -> None:
        """Share ``copies`` more copies of the prompt keys with every sequence."""
        llama_cpp = self._llama_cpp
        if prompt_len > self.max_prompt_len or copies > self.max_copies:
            raise ValueError(
                f"a prompt of {prompt_len} frames with {copies} copies does not fit "
                f"the llama.cpp context sized for {self.max_prompt_len} frames and "
                f"{self.max_copies} copies"
            )
        for s in range(self.batch_size):
            size = llama_cpp.llama_state_seq_get_size(self._ctx, s)
            buf = (ctypes.c_uint8 * size)()
            n = llama_cpp.llama_state_seq_get_data(self._ctx, buf, size, s)
            if n == 0:
                raise RuntimeError("llama.cpp could not save the prompt keys")
            for c in range(copies):
                aux = self.batch_size + s * copies + c
                if llama_cpp.llama_state_seq_set_data(self._ctx, buf, size, aux) == 0:
                    raise RuntimeError("llama.cpp could not restore the prompt keys")
                llama_cpp.llama_memory_seq_cp(self._memory, aux, s, 0, prompt_len)
                llama_cpp.llama_memory_seq_rm(self._memory, aux, prompt_len, -1)

    def _parse_mask(
        self, mask: Optional[torch.Tensor], positions: torch.Tensor
    ) -> tuple[int, int]:
        """
        Returns:
            prompt keys carrying a bias and the copies of them that emulate it;
            (0, 0) for a plain causal mask
        """
        if mask is None or mask.dtype == torch.bool:
            if self.check_mask and mask is not None:
                _assert_causal(mask, positions)
            return 0, 0
        row = mask[0, -1].float()
        bias = float(row.max().item())
        if bias <= 0:
            if self.check_mask:
                _assert_causal(mask == 0, positions)
            return 0, 0
        prompt_len = int((row == bias).sum().item())
        if self.check_mask:
            _assert_sink(mask, positions, prompt_len, bias)
        copies = bias_copies(bias)
        if bias not in self._bias_warned:
            self._bias_warned.add(bias)
            eff = math.log(copies + 1)
            if abs(eff - bias) > 1e-3:
                logger.info(
                    "llama.cpp emulates the prompt attention bias %.3f with %d copies "
                    "of the prompt keys, an effective bias of %.3f",
                    bias,
                    copies,
                    eff,
                )
        return prompt_len, copies

    def _read_outputs(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        bs = self.batch_size
        ptr = self._llama_cpp.llama_get_logits(self._ctx)
        out = self._read(ptr, bs, self.n_vocab)
        if self.n_logits:
            self.logits = torch.from_numpy(out[:, : self.n_logits])
            hidden = out[:, self.n_logits :]
        else:
            hidden = out[:, -self.embed_dim :]
        return (
            torch.from_numpy(np.ascontiguousarray(hidden))
            .view(bs, 1, -1)
            .to(device=device, dtype=dtype)
        )


class LlamaCppMoGMLP(_LlamaCppStack):
    """The MLP stack of the MoG head as a Gemma2 model: ``mlp(x) -> features``."""

    def __init__(
        self,
        gguf_path: Path,
        spec: MoGMLPSpec,
        max_tokens: int = 8,
        n_threads: int = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        self.max_tokens = max_tokens
        super().__init__(
            gguf_path,
            spec.embed_dim,
            n_ctx=max(64, max_tokens),
            n_batch=max_tokens,
            n_seq_max=1,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            embeddings=True,
            flash_attn=False,
            verbose=verbose,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.detach().reshape(-1, self.embed_dim)
        n = x2.shape[0]
        assert (
            n <= self.max_tokens
        ), f"{n} rows, the MoG MLP batch holds {self.max_tokens}"
        embd = (x2.to(torch.float32) * self.spec.input_scale).contiguous().cpu().numpy()
        # the attention of the stack is neutralized, the cells only take room
        self.clear()
        self._decode(
            embd,
            np.arange(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
            np.ones(n, dtype=np.int8),
        )
        out = self._read(
            self._llama_cpp.llama_get_embeddings(self._ctx), n, self.embed_dim
        )
        return torch.from_numpy(out).view(shape).to(device=x.device, dtype=x.dtype)


class LlamaCppMimiTransformer(_LlamaCppStack):
    """The decoder transformer of Mimi, streaming: ``transformer(x) -> [y]`` for
    ``x`` of shape (1, channels, frames), the positions running on from call to
    call until ``reset``. Keys older than the context are dropped before every
    call, as moshi's ring cache does."""

    def __init__(
        self,
        gguf_path: Path,
        spec: MimiTransformerSpec,
        max_frames: int = 64,
        n_threads: int = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        self.max_frames = max_frames
        self.pos = 0
        super().__init__(
            gguf_path,
            spec.padded_dim,
            n_ctx=spec.context + 2 * max_frames,
            n_batch=max_frames,
            n_seq_max=1,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            embeddings=True,
            flash_attn=True,
            verbose=verbose,
        )

    def reset(self) -> None:
        self.clear()
        self.pos = 0

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        spec = self.spec
        bs, dim, n = x.shape
        assert bs == 1 and dim == spec.embed_dim, f"unexpected input {tuple(x.shape)}"
        assert n <= self.max_frames, f"{n} frames, the batch holds {self.max_frames}"
        # the last query of this call sees `context` keys, itself included
        oldest = self.pos + n - spec.context
        if oldest > 0:
            self._llama_cpp.llama_memory_seq_rm(self._memory, 0, -1, oldest)

        # the widened stream of `write_mimi_transformer_gguf`: the frames less their
        # mean, the scaled mean in its channel, zeros elsewhere
        frames = x[0].T.detach().to(torch.float32).cpu().numpy()
        mean = frames.mean(-1, keepdims=True)
        embd = np.zeros((n, spec.padded_dim), dtype=np.float32)
        embd[:, :dim] = frames - mean
        embd[:, spec.mean_channel] = spec.mean_scale * mean[:, 0]
        self._decode(
            embd,
            np.arange(self.pos, self.pos + n, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
            np.ones(n, dtype=np.int8),
        )
        self.pos += n
        y = self._read(
            self._llama_cpp.llama_get_embeddings(self._ctx), n, spec.padded_dim
        )

        # undo the final norm: its scale from the two shift channels, the stream mean
        # it subtracted (the mean channel over all channels) and the frame mean
        plus, minus = spec.shift_channels
        scale = 2 * spec.shift / (y[:, plus] - y[:, minus])
        e = spec.padded_dim
        mean = y[:, spec.mean_channel] * scale / (spec.mean_scale * (1 - 1 / e))
        out = y[:, :dim] * scale[:, None] + (spec.mean_scale * mean / e + mean)[:, None]
        return [torch.from_numpy(out).T.unsqueeze(0).to(device=x.device, dtype=x.dtype)]


def _assert_causal(mask: torch.Tensor, positions: torch.Tensor) -> None:
    """``mask`` (batch, seq_len, max_seq_len) boolean must attend to exactly ``0..pos``."""
    if positions.shape[0] == 1 and mask.shape[0] > 1:
        positions = positions.expand(mask.shape[0], -1)
    keys = torch.arange(mask.shape[-1], device=mask.device).view(1, 1, -1)
    expected = keys <= positions.unsqueeze(-1)
    diff = int((mask.bool() ^ expected).sum().item())
    if diff:
        raise AssertionError(
            f"non-causal mask reached the GGUF backend ({diff} differing entries)"
        )


def _assert_sink(
    mask: torch.Tensor, positions: torch.Tensor, prompt_len: int, bias: float
) -> None:
    """A causal float mask whose keys ``0..prompt_len`` carry ``bias``, nothing else."""
    if positions.shape[0] == 1 and mask.shape[0] > 1:
        positions = positions.expand(mask.shape[0], -1)
    keys = torch.arange(mask.shape[-1], device=mask.device).view(1, 1, -1)
    visible = keys <= positions.unsqueeze(-1)
    expected = torch.full(mask.shape, float("-inf"), dtype=torch.float32)
    expected = expected.masked_fill(visible, 0.0)
    expected = expected.masked_fill(visible & (keys < prompt_len), bias)
    diff = int((mask.float().cpu() != expected).sum().item())
    if diff:
        raise AssertionError(
            f"a mask other than causal + prompt bias reached the GGUF backend "
            f"({diff} differing entries)"
        )


def resolve_gguf_dir(config, gguf_dir=None) -> Path:
    """
    The directory of the GGUF files: an explicit one, else ``config.gguf_dir``, else
    the model repo (``config.model_repo``), whose manifest and GGUF files are
    downloaded unless it is a local directory.
    """
    gguf_dir = gguf_dir or config.gguf_dir
    if gguf_dir is not None:
        return Path(gguf_dir)
    if Path(config.model_repo).is_dir():
        return Path(config.model_repo)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            config.model_repo,
            allow_patterns=[MANIFEST_NAME, f"*-{QUANT}.gguf"],
            token=config.hf_token,
        )
    )


def _replace_module(parent, name: str, backend) -> None:
    # nn.Module.__setattr__ refuses non-Module values for registered submodules
    if name in parent._modules:
        del parent._modules[name]
    object.__setattr__(parent, name, backend)


def patch_model_with_gguf(
    model,
    batch_size: int,
    gguf_dir: Path,
    n_threads: int = None,
    n_gpu_layers: int = 0,
    max_positions: int = None,
    max_prompt_len: int = 0,
    prompt_bias: float = None,
    check_mask: bool = False,
    verbose: bool = False,
) -> Dict[str, _LlamaCppStack]:
    """Swap the converted stacks of ``model`` for their llama.cpp counterparts.

    Args:
        max_positions: positions a sequence of the temporal transformer holds
            (the audio window), defaults to the stack's trained range
        max_prompt_len, prompt_bias: frames of the acoustic prompt and the bias on
            their keys the context must have room for (see the module docstring)
    """
    gguf_dir = Path(gguf_dir)
    manifest = load_manifest(gguf_dir)
    specs = manifest[STACKS_KEY]
    backends: Dict[str, _LlamaCppStack] = {}
    for name, filename in manifest[FILES_KEY].items():
        spec = specs[name]
        path = gguf_dir / filename
        if spec.source != "model":
            continue
        if isinstance(spec, MoGMLPSpec):
            backend = LlamaCppMoGMLP(
                path,
                spec,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
            _replace_module(model.mog_head, "mlp", backend)
        else:
            temporal = name == "temp_former"
            backend = LlamaCppTransformer(
                path,
                spec,
                batch_size=batch_size,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                max_positions=max_positions if temporal else None,
                max_prompt_len=max_prompt_len if temporal else 0,
                prompt_bias=prompt_bias if temporal else None,
                check_mask=check_mask,
                verbose=verbose,
            )
            _replace_module(model, name, backend)
            if spec.output_head:
                _replace_module(model, spec.output_head, backend.sem_head)
            backend.setup_caches(batch_size, torch.float32)
        backends[name] = backend
    return backends


def patch_mimi_with_gguf(
    mimi,
    gguf_dir: Path,
    n_threads: int = None,
    n_gpu_layers: int = 0,
    verbose: bool = False,
) -> Optional[LlamaCppMimiTransformer]:
    """Swap the decoder transformer of a moshi ``MimiModel`` for llama.cpp, when
    the converted files hold it. The backend's cache is not part of moshi's
    streaming state, so it is reset here whenever that state is: on
    ``mimi.reset_streaming`` and on entering ``mimi.streaming``."""
    gguf_dir = Path(gguf_dir)
    manifest = load_manifest(gguf_dir)
    specs = manifest[STACKS_KEY]
    for name, filename in manifest[FILES_KEY].items():
        spec = specs[name]
        if not isinstance(spec, MimiTransformerSpec):
            continue
        backend = LlamaCppMimiTransformer(
            gguf_dir / filename,
            spec,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        _replace_module(mimi, "decoder_transformer", backend)
        reset_streaming = mimi.reset_streaming

        def reset(*args, _reset=reset_streaming, _backend=backend, **kwargs):
            _reset(*args, **kwargs)
            _backend.reset()

        mimi.reset_streaming = reset
        # moshi's `streaming` context enters through `_start_streaming`
        start_streaming = getattr(mimi, "_start_streaming", None)
        if start_streaming is not None:

            def start(*args, _start=start_streaming, _backend=backend, **kwargs):
                _backend.reset()
                return _start(*args, **kwargs)

            mimi._start_streaming = start
        return backend
    return None


@torch.no_grad()
def verify_mimi_transformer(
    mimi, gguf_dir: Path, steps: int = 140, seed: int = 0, n_threads: int = None
) -> list[str]:
    """Compare the llama.cpp decoder transformer of Mimi with moshi's in streaming
    mode, two frames per call, past the context window.

    Args:
        n_threads: threads of llama.cpp, see `default_threads`
    """
    gguf_dir = Path(gguf_dir)
    manifest = load_manifest(gguf_dir)
    spec = next(
        s for s in manifest[STACKS_KEY].values() if isinstance(s, MimiTransformerSpec)
    )
    backend = LlamaCppMimiTransformer(
        gguf_dir / manifest[FILES_KEY][spec.name], spec, n_threads=n_threads
    )
    gen = torch.Generator().manual_seed(seed)
    transformer = mimi.decoder_transformer
    errs = []
    with transformer.streaming(1):
        for _ in range(steps):
            x = torch.randn(1, spec.embed_dim, 2, generator=gen)
            (ref,) = transformer(x)
            (out,) = backend(x)
            errs.append(((ref - out).abs().max() / ref.abs().max()).item())
    backend.close()
    return [
        f"mimi_decoder_transformer: max rel err {max(errs):.3e} over {steps} "
        f"steps of 2 frames (context {spec.context}), last step {errs[-1]:.3e}"
    ]


@torch.no_grad()
def verify_stacks(
    model, gguf_dir: Path, seed: int = 0, mimi=None, n_threads: int = None
) -> list[str]:
    """Compare every converted stack with its PyTorch original on random inputs.

    Args:
        n_threads: threads of llama.cpp, see `default_threads`

    Returns:
        one line per check with the maximum absolute and relative errors
    """
    gguf_dir = Path(gguf_dir)
    manifest = load_manifest(gguf_dir)
    specs = manifest[STACKS_KEY]
    gen = torch.Generator().manual_seed(seed)
    lines = []
    bs, prompt_len, tail, bias = 2, 12, 20, 3.0

    def report(name: str, ref: torch.Tensor, out: torch.Tensor) -> None:
        ref, out = ref.float().flatten(), out.float().flatten()
        err = (ref - out).abs().max().item()
        rel = err / ref.abs().max().clamp_min(1e-8).item()
        lines.append(f"{name}: max abs err {err:.3e}, rel {rel:.3e}")

    for name, filename in manifest[FILES_KEY].items():
        spec, path = specs[name], gguf_dir / filename
        if isinstance(spec, MimiTransformerSpec):
            if mimi is not None:
                lines.extend(
                    verify_mimi_transformer(
                        mimi, gguf_dir, seed=seed, n_threads=n_threads
                    )
                )
            continue
        if isinstance(spec, MoGMLPSpec):
            mlp = LlamaCppMoGMLP(path, spec, n_threads=n_threads)
            x = torch.randn(2, spec.embed_dim, generator=gen)
            report(name, model.mog_head.mlp(x), mlp(x))
            mlp.close()
            continue

        stack = getattr(model, spec.prefix)
        stack.setup_caches(bs, torch.float32, decoder_max_seq_len=spec.max_seq_len)
        backend = LlamaCppTransformer(
            path,
            spec,
            batch_size=bs,
            max_positions=64,
            max_prompt_len=prompt_len,
            prompt_bias=bias,
            check_mask=True,
            n_threads=n_threads,
        )
        keys = torch.arange(spec.max_seq_len).view(1, 1, -1)

        def masks(
            positions: torch.Tensor, with_bias: bool, keys: torch.Tensor = keys
        ) -> torch.Tensor:
            visible = keys <= positions.unsqueeze(-1)
            if not with_bias:
                return visible.expand(bs, -1, -1)
            m = torch.full(visible.shape, float("-inf")).masked_fill(visible, 0.0)
            m = m.masked_fill(
                visible & (keys < prompt_len), math.log(bias_copies(bias) + 1)
            )
            return m.expand(bs, -1, -1)

        for with_bias in (False, True):
            stack.reset_caches()
            backend.reset_caches()
            tag = f"{name} ({'prompt bias' if with_bias else 'causal'})"
            # prefill of prompt + tail, then two single-row steps
            pos = torch.arange(prompt_len + tail).unsqueeze(0)
            h = torch.randn(bs, pos.shape[1], spec.embed_dim, generator=gen)
            ref = stack(h, input_pos=pos, mask=masks(pos, with_bias))[:, -1:]
            if with_bias:
                # the bias must move the reference by far more than the error above
                stack.reset_caches()
                ref_plain = stack(h, input_pos=pos, mask=masks(pos, False))[:, -1:]
                stack.reset_caches()
                stack(h, input_pos=pos, mask=masks(pos, True))
                report(f"{tag} prefill, effect of the bias on PyTorch", ref, ref_plain)
            out = backend(h, pos, masks(pos, with_bias))
            report(f"{tag} prefill", ref, out)
            if spec.output_head:
                head = getattr(model, spec.output_head)
                report(f"{tag} prefill logits", head(ref[:, -1]), backend.logits)
            for step in range(2):
                pos = torch.tensor([[prompt_len + tail + step]])
                h = torch.randn(bs, 1, spec.embed_dim, generator=gen)
                ref = stack(h, input_pos=pos, mask=masks(pos, with_bias))
                out = backend(h, pos, masks(pos, with_bias))
                report(f"{tag} step {step}", ref, out)
        backend.close()
    return lines
