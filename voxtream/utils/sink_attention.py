from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn as nn


@dataclass
class SinkAttentionConfig:
    audio_window_size: int
    min_recent_window: int = 1
    tail_budget_padding: int = 1
    tail_window_divisor: int = 2
    # Added to the attention logit of every prompt key (`SpeechGeneratorConfig.
    # prompt_attention_bias`). The mask is then additive rather than boolean: 0 on a
    # visible key, the bias on a visible prompt key, -inf elsewhere. SDPA takes both.
    prompt_bias: float = None
    prompt_bias_dtype: torch.dtype = torch.bfloat16
    # Frames of generated audio before the bias applies (`SpeechGeneratorConfig.
    # prompt_attention_bias_start_sec`): a row generated earlier takes no bias, also
    # when it is re-encoded at a cache rebuild. 0 applies the bias from the first frame.
    prompt_bias_start: int = 0


class SinkAttention:
    def __init__(self, config: SinkAttentionConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._sink_prompt_len = None
        self._sink_prompt_phone_emb = None
        self._sink_prompt_audio_tokens = None
        self._sink_phone_emb_buf = []
        self._sink_audio_tokens_buf = []
        self._sink_pos_offset = 0
        self._restart_pending = False

    def restart(self) -> None:
        """Drop the generated context: the next step attends to the prompt alone, as the
        first step of the utterance did. Frames generated from there fill the window
        again as usual."""
        self._sink_phone_emb_buf = []
        self._sink_audio_tokens_buf = []
        self._restart_pending = True

    def get_context(
        self,
        input_pos: torch.Tensor,
        phone_emb: torch.Tensor,
        audio_tokens: torch.Tensor,
        temp_former,
        temp_former_causal_mask: torch.Tensor,
        embed_audio_tokens: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._sink_prompt_len is None:
            self._sink_prompt_len = int(input_pos.shape[1])
            self._sink_prompt_phone_emb = phone_emb.detach()
            self._sink_prompt_audio_tokens = audio_tokens.detach()
            self._sink_phone_emb_buf = []
            self._sink_audio_tokens_buf = []
            self._sink_pos_offset = 0

        local_input_pos = input_pos
        if input_pos.shape[1] == 1:
            global_pos = int(input_pos.max().item())
            local_pos = global_pos - self._sink_pos_offset
            if local_pos >= self.config.audio_window_size or self._restart_pending:
                self._restart_pending = False
                self._rebuild_temp_former_cache(
                    global_pos,
                    temp_former,
                    temp_former_causal_mask,
                    embed_audio_tokens,
                )
            local_input_pos = input_pos - self._sink_pos_offset

        curr_temp_former_mask = self._get_temp_former_sink_mask(
            local_input_pos, temp_former_causal_mask, global_pos=input_pos
        )

        return local_input_pos, curr_temp_former_mask

    def append_step(self, phone_emb: torch.Tensor, audio_tokens: torch.Tensor) -> None:
        if self._sink_prompt_len is None:
            return

        self._sink_phone_emb_buf.append(phone_emb.detach())
        self._sink_audio_tokens_buf.append(audio_tokens.detach())
        self._trim_sink_buffers()

    def _get_temp_former_sink_mask(
        self,
        input_pos: torch.Tensor,
        causal_mask: torch.Tensor,
        global_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_pos: (batch_size, seq_len) positions in the cache
            causal_mask: (max_seq_len, max_seq_len)
            global_pos: (batch_size, seq_len) positions in the stream, the prompt at
                0..prompt_len - 1; defaults to the cache positions shifted by the offset
                of the current window

        Returns:
            (batch_size, seq_len, max_seq_len) boolean mask, or additive (0 on a visible
            key, `prompt_bias` on a visible prompt key, -inf elsewhere) with a prompt bias
        """
        if self._sink_prompt_len is None:
            return causal_mask[input_pos, :]

        prompt_len = int(self._sink_prompt_len)
        max_seq_len = causal_mask.size(0)
        positions = torch.arange(max_seq_len, device=input_pos.device).view(1, 1, -1)
        pos = input_pos.unsqueeze(-1)
        causal = positions <= pos

        window = max(
            self.config.min_recent_window, self.config.audio_window_size - prompt_len
        )
        start = pos - (window - 1)
        start = torch.maximum(start, input_pos.new_full((), prompt_len))
        prompt_mask = positions < prompt_len
        recent_mask = positions >= start

        mask = causal & (prompt_mask | recent_mask)
        if self.config.prompt_bias is None:
            return mask
        # the prompt rows themselves see nothing but prompt keys, so the bias is a
        # constant over their row and leaves their softmax unchanged
        biased = mask & prompt_mask
        if self.config.prompt_bias_start > 0:
            # by the row's own time in the stream, so that a cache rebuild encodes a
            # row as it was first generated
            if global_pos is None:
                global_pos = input_pos + self._sink_pos_offset
            generated = global_pos.unsqueeze(-1) - prompt_len
            biased = biased & (generated >= self.config.prompt_bias_start)
        bias = torch.full(
            mask.shape,
            float("-inf"),
            dtype=self.config.prompt_bias_dtype,
            device=mask.device,
        )
        bias = bias.masked_fill(mask, 0.0)
        return bias.masked_fill(biased, float(self.config.prompt_bias))

    def _trim_sink_buffers(self) -> None:
        tail_budget = max(
            0,
            self.config.audio_window_size
            - int(self._sink_prompt_len)
            - self.config.tail_budget_padding,
        )
        tail_limit = min(
            self.config.audio_window_size // self.config.tail_window_divisor,
            tail_budget,
        )
        if tail_limit <= 0:
            self._sink_phone_emb_buf = []
            self._sink_audio_tokens_buf = []
            return
        if len(self._sink_phone_emb_buf) > tail_limit:
            self._sink_phone_emb_buf = self._sink_phone_emb_buf[-tail_limit:]
        if len(self._sink_audio_tokens_buf) > tail_limit:
            self._sink_audio_tokens_buf = self._sink_audio_tokens_buf[-tail_limit:]

    def _rebuild_temp_former_cache(
        self,
        global_pos: int,
        temp_former,
        temp_former_causal_mask: torch.Tensor,
        embed_audio_tokens: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        if (
            self._sink_prompt_phone_emb is None
            or self._sink_prompt_audio_tokens is None
        ):
            return

        self._trim_sink_buffers()
        if self._sink_phone_emb_buf:
            tail_phone_emb = torch.cat(self._sink_phone_emb_buf, dim=1)
            tail_audio_tokens = torch.cat(self._sink_audio_tokens_buf, dim=2)
            phone_emb = torch.cat([self._sink_prompt_phone_emb, tail_phone_emb], dim=1)
            audio_tokens = torch.cat(
                [self._sink_prompt_audio_tokens, tail_audio_tokens], dim=2
            )
        else:
            phone_emb = self._sink_prompt_phone_emb
            audio_tokens = self._sink_prompt_audio_tokens

        seq_len = phone_emb.shape[1]
        input_pos = (
            torch.arange(0, seq_len, device=phone_emb.device)
            .unsqueeze(0)
            .repeat(phone_emb.size(0), 1)
        )

        audio_emb = embed_audio_tokens(audio_tokens)
        emb = torch.cat([phone_emb.unsqueeze(1), audio_emb], dim=1)
        h = emb.sum(dim=1, dtype=audio_emb.dtype)

        temp_former.reset_caches()
        mask = self._get_temp_former_sink_mask(
            input_pos,
            temp_former_causal_mask,
            global_pos=input_pos + (global_pos - seq_len),
        )
        temp_former(h, input_pos=input_pos, mask=mask).to(dtype=audio_emb.dtype)

        self._sink_pos_offset = global_pos - seq_len


@dataclass
class PhoneSinkAttentionConfig:
    # keys a row may see behind it, the sink rows included: the mask the phoneme
    # transformer was trained with (`phone_window_size`)
    phone_window_size: int
    # rows a row may see ahead of it, fixed for the whole stream; the look-ahead masks of
    # training compound through the stack into a receptive field of
    # `num_layers * look_ahead` rows, which is how far behind the newest row an output
    # can still change
    look_ahead: int
    num_layers: int
    # slots per layer. Holds the window, the receptive field and the chunk being
    # encoded; the positions the model sees stay below it however long the stream runs
    cache_len: int = 1024
    # smallest chunk the cache must take in one forward, checked at setup
    min_chunk: int = 32

    @property
    def receptive_field(self) -> int:
        return self.num_layers * self.look_ahead


class PositionalKVCache(nn.Module):
    """
    Key-value cache of one attention layer, written at explicit slots.

    torchtune's `KVCache` appends: an update fills the slots after the last one written.
    The streaming phoneme transformer rewrites recent rows whose keys went stale when
    more text arrived, so here the slots of an update are set beforehand (`write_pos`)
    and may lie anywhere. `MultiHeadAttention` only relies on `update` and the buffers.
    """

    def __init__(
        self,
        batch_size: int,
        cache_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        super().__init__()
        shape = (batch_size, num_heads, cache_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(shape, dtype=dtype, device=device), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(shape, dtype=dtype, device=device), persistent=False
        )
        self.batch_size = batch_size
        self.write_pos: torch.Tensor | None = None

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.write_pos = None

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            k_val, v_val: (batch_size, num_heads, seq_len, head_dim), one row per slot
                of `write_pos`

        Returns:
            the whole key and value caches, (batch_size, num_heads, cache_len, head_dim)
        """
        if self.write_pos is None:
            raise RuntimeError("write_pos must be set before the forward")
        bsz = k_val.shape[0]
        if bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"the cache was set up for a batch of {self.k_cache.shape[0]}, "
                f"got keys for a batch of {bsz}"
            )
        self.k_cache[:bsz, :, self.write_pos] = k_val
        self.v_cache[:bsz, :, self.write_pos] = v_val
        return self.k_cache[:bsz], self.v_cache[:bsz]

    def shift(self, delta: int, last: int) -> None:
        """
        Move the slots `[delta, last)` down to `[0, last - delta)`.

        The keys carry the rotary embedding of their position in the stream, which the
        move does not change, so they move as they are.
        """
        k = self.k_cache[:, :, delta:last].clone()
        v = self.v_cache[:, :, delta:last].clone()
        self.k_cache[:, :, : last - delta] = k
        self.v_cache[:, :, : last - delta] = v


@contextmanager
def kv_cache_disabled(transformer: nn.Module) -> Iterator[None]:
    """Run `transformer` as a plain full-sequence model, leaving its caches untouched."""
    layers = [layer for layer in transformer.layers if layer.attn.kv_cache is not None]
    enabled = [layer.attn.cache_enabled for layer in layers]
    for layer in layers:
        layer.attn.cache_enabled = False
    try:
        yield
    finally:
        for layer, was_enabled in zip(layers, enabled):
            layer.attn.cache_enabled = was_enabled


class PhoneSinkAttention:
    """
    Streaming phoneme transformer over a bounded KV cache.

    The phoneme transformer attends through a sliding window with a look-ahead: row `i`
    sees rows `[i - W + 1, i + la]`. The look-ahead compounds through the `L` layers, so
    the output of row `i` is final once row `i + L * la` exists, and the outputs of the
    last `L * la` rows still change when more text arrives. A chunk of text therefore
    does not re-encode the sequence, as the full-sequence pass of
    `Model.extract_phoneme_embeddings` does: it runs the stack over the new rows together
    with the `L * la` rows before them, rewriting the keys and values of those rows in
    the cache, and leaves every earlier row alone. The result is the full-sequence pass
    with the look-ahead mask `la`, computed once per row instead of once per chunk.

    The cache holds `cache_len` slots per layer; a row's slot is its position in the
    stream less the rows dropped so far. When a chunk does not fit, the keys no future
    row can see any more - everything further than the window behind the rows that may
    still be recomputed - are dropped, and the rest moved down. The positions the model
    sees are those of the stream, as in the full-sequence pass, so the attention is
    unchanged by the move and nothing is re-encoded: the kept keys were computed with
    their full context. This is what makes it exact where the audio `SinkAttention` is
    not, which re-encodes the prompt and a tail of recent frames from scratch at every
    rebuild. A stream is bounded by the `max_rows` of `setup`, the `max_phone_tokens` of
    the generator; the rotary table is extended past its trained range to reach it.
    """

    def __init__(self, config: PhoneSinkAttentionConfig):
        self.config = config
        self.phone_former: nn.Module | None = None
        self.emb: torch.Tensor | None = None
        self.tokens: torch.Tensor | None = None
        self.embed: Callable[[torch.Tensor], torch.Tensor] | None = None
        # a captured `_step` of exactly `graph_rows` rows, see `enable_graph`
        self.step_fn: Callable | None = None
        self.graph_rows = 0
        self.reset()

    @property
    def lookback(self) -> int:
        """Rows behind the newest one a later forward may start from."""
        return max(self.config.receptive_field, self.graph_rows - 1)

    @property
    def is_setup(self) -> bool:
        return self.phone_former is not None

    @property
    def max_chunk(self) -> int:
        """Rows one forward may hold: the cache less the window and the receptive field."""
        cfg = self.config
        return cfg.cache_len - cfg.phone_window_size + 1 - cfg.receptive_field

    def setup(
        self,
        phone_former: nn.Module,
        batch_size: int,
        dtype: torch.dtype,
        cache_len: int = None,
        max_rows: int = None,
        num_streams: int = 1,
    ) -> None:
        """
        Install a positional KV cache in every layer of `phone_former`.

        Args:
            cache_len: slots per layer, defaults to the config
            max_rows: rows of a stream, the size of the output buffer; defaults to the
                position range of the transformer
            num_streams: token streams of a row, for the copy of the stream kept to
                tell changed rows from new ones
        """
        if cache_len is not None:
            self.config.cache_len = int(cache_len)
        cfg = self.config
        if self.max_chunk < cfg.min_chunk:
            raise ValueError(
                f"phone cache of {cfg.cache_len} slots is too small: it must hold the "
                f"window ({cfg.phone_window_size}), the receptive field "
                f"({cfg.receptive_field}) and a chunk of at least {cfg.min_chunk} rows"
            )
        device = next(phone_former.parameters()).device
        for layer in phone_former.layers:
            attn = layer.attn
            attn.kv_cache = PositionalKVCache(
                batch_size, cfg.cache_len, attn.num_heads, attn.head_dim, dtype, device
            )
            attn.cache_enabled = True
        self.phone_former = phone_former
        max_rows = max(max_rows or 0, phone_former.max_seq_len)
        if max_rows > phone_former.max_seq_len:
            # the rotary embedding sees stream positions, so its cache must reach the
            # end of the longest stream. Attention scores only depend on the distance
            # between positions, bounded by the window, so positions past the trained
            # range are as valid as those within it. The rows of the trained range are
            # kept as the constructor built them; the new rows are computed as it did,
            # in float32 before the cast to the model dtype: the model is already in
            # bfloat16 here, which cannot count past 256
            for rope in {
                id(layer.attn.pos_embeddings): layer.attn.pos_embeddings
                for layer in phone_former.layers
            }.values():
                cache, theta = rope.cache, rope.theta
                rope.max_seq_len = max_rows
                rope.rope_init()
                tail = rope.cache[cache.size(0) :].to(cache.device, cache.dtype)
                rope.cache = torch.cat([cache, tail])
                rope.theta = rope.theta.to(theta.device, theta.dtype)
            phone_former.max_seq_len = max_rows
        dim = phone_former.layers[0].attn.embed_dim
        self.emb = torch.zeros(batch_size, max_rows, dim, dtype=dtype, device=device)
        # allocated here rather than on first use, which may be under inference mode
        self.tokens = torch.zeros(
            batch_size, max_rows, num_streams, dtype=torch.int64, device=device
        )
        self.reset()

    def reset(self) -> None:
        """Start a new stream."""
        self.length = 0  # rows of the stream encoded so far
        self.offset = 0  # position of slot 0, before the shift of any key
        if self.is_setup:
            for layer in self.phone_former.layers:
                layer.attn.kv_cache.reset()

    def enable_graph(self, step_fn: Callable, rows: int = 64) -> None:
        """
        Run every forward of at most `rows` rows through `step_fn`, a captured or
        compiled `_step` of exactly that many rows.

        The eager stack costs several milliseconds of kernel launches per forward
        however few rows it holds, which is most of a word's cost. A fixed row count
        lets the step be captured once: a forward of fewer rows is widened backwards to
        `rows`, and the extra rows, whose inputs and attention did not change, recompute
        to what they already were. A forward of more rows, such as a prompt or a whole
        text, runs eagerly.

        Args:
            step_fn: wraps `self._step`, e.g. `CUDAGraphed(self._step)`
            rows: rows per captured forward; must leave room for the window
        """
        cfg = self.config
        if rows <= cfg.receptive_field:
            raise ValueError(
                f"a captured step of {rows} rows cannot hold the receptive field "
                f"({cfg.receptive_field}) and a new row"
            )
        if rows > cfg.cache_len - cfg.phone_window_size + 1:
            raise ValueError(
                f"a captured step of {rows} rows does not fit the cache of "
                f"{cfg.cache_len} slots next to the window ({cfg.phone_window_size})"
            )
        self.step_fn = step_fn
        self.graph_rows = int(rows)

    @contextmanager
    def disabled(self) -> Iterator[None]:
        """Run the phoneme transformer as a plain full-sequence model."""
        if not self.is_setup:
            yield
            return
        with kv_cache_disabled(self.phone_former):
            yield

    def extend(
        self,
        phone_tokens: torch.Tensor,
        embed: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode the rows of `phone_tokens` not encoded yet, and those still able to
        change, and return the embeddings of the whole stream.

        Args:
            phone_tokens: (batch_size, seq_len, num_streams) the stream so far: the rows
                already sent, with new ones behind them. A row already sent may change -
                a chunk of punctuation alone lands on the last phoneme sent - and is
                re-encoded with the rows after it; a change further back than the
                receptive field of the newest key dropped from the cache is an error.
            embed: token embedding of the phoneme transformer

        Returns:
            (batch_size, seq_len, dim) view of the output buffer, updated in place by
            later calls
        """
        if not self.is_setup:
            raise RuntimeError("call setup() before encoding a stream")
        total = phone_tokens.size(1)
        if total > self.emb.size(1):
            raise ValueError(
                f"stream of {total} rows exceeds the {self.emb.size(1)} the encoder holds"
            )
        if total < self.length:
            raise ValueError(
                f"stream shrank from {self.length} to {total} rows; reset() starts a new one"
            )
        if (
            self.tokens.shape[0] != phone_tokens.shape[0]
            or self.tokens.shape[2] != phone_tokens.shape[2]
        ):
            self.tokens = torch.zeros(
                phone_tokens.shape[0],
                self.emb.size(1),
                phone_tokens.shape[2],
                dtype=torch.int64,
                device=phone_tokens.device,
            )

        # the first row that is new or has changed since it was encoded
        changed = self.length
        if self.length > 0:
            stale = (
                (phone_tokens[:, : self.length] != self.tokens[:, : self.length])
                .any(dim=-1)
                .any(dim=0)
                .nonzero()
            )
            if stale.numel() > 0:
                changed = int(stale[0].item())
        self.tokens[:, :total] = phone_tokens
        if total == 0:
            return self.emb[:, :0]

        self.embed = embed
        start = max(0, changed - self.config.receptive_field)
        if start < total:
            self._encode(start, total)
        self.length = total
        return self.emb[:, :total]

    def _encode(self, start: int, total: int) -> None:
        """Run the stack over rows `[start, total)`, in chunks the cache can take."""
        rf, window = self.config.receptive_field, self.config.phone_window_size
        a, first = start, True
        while True:
            b = min(total, a + self.max_chunk)
            if self.graph_rows and b - a <= self.graph_rows <= b:
                a = b - self.graph_rows  # the fixed rows of the captured step
            if first and self.offset > 0:
                # the keys the first row attends to must still be in the cache
                if a - window + 1 < self.offset:
                    raise ValueError(
                        f"row {start + rf} changed after keys it attends to left the "
                        "cache"
                    )
                first = False
            # the earliest row a later forward may start from: the next chunk, or as
            # far behind the last row as a change to it (a chunk of punctuation) or a
            # captured step reaches
            future_start = b - rf if b < total else total - 1 - self.lookback
            keep = min(a, future_start) - window + 1
            self._fit(b, keep)
            self._forward(a, b)
            if b == total:
                return
            a = b - rf

    def _fit(self, end: int, keep: int) -> None:
        """
        Make room for the rows up to `end`, dropping the keys before row `keep`.

        The keys of rows `[keep, encoded)` move down to slot 0, and every later row
        lands `delta` slots lower than it would have.
        """
        cache_len = self.config.cache_len
        if end - self.offset <= cache_len:
            return
        new_offset = max(keep, 0)
        if new_offset <= self.offset or end - new_offset > cache_len:
            raise ValueError(
                f"phone cache of {cache_len} slots cannot hold rows "
                f"{keep}..{end} of the stream; raise phone_cache_len"
            )
        delta = new_offset - self.offset
        last = self.length - self.offset  # slots written so far
        if last > delta:
            for layer in self.phone_former.layers:
                layer.attn.kv_cache.shift(delta, last)
        self.offset = new_offset

    def _forward(self, a: int, b: int) -> None:
        """Encode rows `[a, b)`: their keys and values go into the cache, their outputs
        into the buffer."""
        pos = torch.arange(a, b, device=self.emb.device)
        step = self._step
        if self.step_fn is not None and b - a == self.graph_rows:
            step = self.step_fn
        out = step(self.tokens[:, a:b], pos, pos - self.offset)
        self.emb[: out.size(0), a:b] = out.to(self.emb.dtype)
        # rows the next forward of this call will see: `length` is only final after it
        self.length = max(self.length, b)

    def _step(
        self, tokens: torch.Tensor, pos: torch.Tensor, slots: torch.Tensor
    ) -> torch.Tensor:
        """
        One forward of the stack, the newest row of the stream last.

        Args:
            tokens: (batch_size, seq_len, num_streams) rows being encoded
            pos: (seq_len,) their positions in the stream, which the rotary embedding
                sees, as in the full-sequence pass
            slots: (seq_len,) the cache slots they are written to

        Returns:
            (batch_size, seq_len, dim) outputs
        """
        emb = self.embed(tokens)
        batch_size = emb.size(0)
        mask = self._mask(slots)
        for layer in self.phone_former.layers:
            layer.attn.kv_cache.write_pos = slots
        return self.phone_former(
            emb,
            input_pos=pos.unsqueeze(0).expand(batch_size, -1),
            mask=mask.expand(batch_size, -1, -1),
        )

    def _mask(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: (seq_len,) cache slots of the rows being encoded, the last of them the
                newest row of the stream

        Returns:
            (1, seq_len, cache_len) attention mask: the rows of the window behind each
            query and the look-ahead ahead of it, as far as the stream goes
        """
        slots = torch.arange(self.config.cache_len, device=pos.device).view(1, -1)
        query = pos.view(-1, 1)
        # no row sees past its look-ahead, nor a slot not written: the stream ends at
        # the last row of this forward
        visible = (slots <= query + self.config.look_ahead) & (slots <= pos[-1])
        first = query - (self.config.phone_window_size - 1)
        return (visible & (slots >= first)).unsqueeze(0)
