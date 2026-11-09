"""
Text frontend of the models with a flat phoneme vocabulary (VoXtream2, VoXtream2.1).

A phoneme is a single token of `phoneme_to_token.json`, its stress and length included.
Punctuation is a token of its own: it is inserted into the sequence after the phoneme it
follows, seen by the phoneme transformer, and its row is removed from the transformer
output, so the frame pointer indexes phonemes only. The text closes on `sil` and `[EOS]`
and the prompt is masked as `[BOS]`, `[UNK]`... and a closing `[UNK]`.

The functions mirror those of `text.py` and `runtime.update_indices_and_tokens` for the
factored vocabulary and reproduce the generation of the released models.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List

import numpy as np
import torch

from voxtream.config import SpeechGeneratorConfig
from voxtream.utils.generator.context import FrameState, GenerationContext
from voxtream.utils.text.normalizer import english_normalizer

PUNCT_SYMBOLS = (".", ",", "?", "!")


@dataclass
class FlatPhonemeVocab:
    """`phoneme_to_token.json`; the specials past it sit at the end of the vocabulary."""

    phone_to_token: Dict[str, int]
    vocab_size: int
    punct_ids: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.punct_ids = torch.tensor([self.phone_to_token[s] for s in PUNCT_SYMBOLS])

    @classmethod
    def load(cls, path: Path, vocab_size: int) -> "FlatPhonemeVocab":
        with open(path) as f:
            return cls(phone_to_token=json.load(f), vocab_size=vocab_size)

    @property
    def bos(self) -> int:
        return self.vocab_size - 2

    @property
    def eos(self) -> int:
        return self.vocab_size - 1

    @property
    def unk(self) -> int:
        return self.phone_to_token["unk"]

    @property
    def sil(self) -> int:
        return self.phone_to_token["sil"]

    def prompt_tokens(self, num_rows: int) -> np.ndarray:
        """`[num_rows]` tokens of the masked prompt: `[BOS]`, then `[UNK]` per frame."""
        tokens = np.full((num_rows,), self.unk, dtype=np.int64)
        tokens[0] = self.bos
        return tokens

    def punct_del_indices(self, phone_tokens: torch.Tensor) -> torch.Tensor:
        """
        Rows of the punctuation tokens, removed from the phoneme transformer output.

        They are found in the conditional row and removed from the unconditional one
        alike, whose text is all `[UNK]`.

        Args:
            phone_tokens: (batch_size, phone_seq_len)

        Returns:
            (batch_size, num_punct)
        """
        punct = torch.isin(phone_tokens[0], self.punct_ids.to(phone_tokens.device))
        rows = punct.nonzero().squeeze(-1)
        return rows.unsqueeze(0).repeat(phone_tokens.size(0), 1)


def text_to_phone_tokens(
    text: str,
    config: SpeechGeneratorConfig,
    vocab: FlatPhonemeVocab,
    phonemizer,
    normalize: bool = True,
    language: str = "en-us",
    full_streaming: bool = False,
) -> tuple[List[int], List[int], List[int]]:
    """
    Returns:
        phone_tokens: phoneme tokens, punctuation excluded
        punct_ins_indices: where the punctuation tokens go into `phone_tokens`
        punct_tokens: the punctuation tokens
    """
    phone_tokens: List[int] = []
    punct_ins_indices: List[int] = []
    punct_tokens: List[int] = []

    if normalize:
        # a stream chunk keeps the pauses at its edges ("Wait" / "-" / "what")
        text = english_normalizer(text, strip_edges=not full_streaming)

    if full_streaming:
        text += config.text_context

    phonemes = phonemizer.phonemize(text, language=language)

    if full_streaming:
        phonemes = phonemes[: -config.text_context_length]

    phones = []
    for word in phonemes.split():
        for ph in word.split("|"):
            if ph == "":
                continue
            phones.append(ph)

        if phones and phones[-1].endswith(PUNCT_SYMBOLS):
            phone = phones.pop()
            symb = phone[-1]
            if phone[:-1] != "":
                phones.append(phone[:-1])
            punct_tokens.append(vocab.phone_to_token[symb])
            punct_ins_indices.append(len(phones))

    for ph in phones:
        phone_tokens.append(vocab.phone_to_token.get(ph, vocab.unk))

    return phone_tokens, punct_ins_indices, punct_tokens


def _with_unconditional(tokens: List[int], ctx: GenerationContext) -> torch.Tensor:
    """`[1, L]`, with an all-UNK copy alongside it when CFG is on."""
    batch = torch.tensor([tokens], device=ctx.device, dtype=torch.int64)
    if ctx.config.cfg_gamma is not None:
        batch = torch.cat([batch, torch.full_like(batch, ctx.vocab.unk)])
    return batch


def prepare_non_streaming_text(
    text: str,
    prompt_phone_tokens: torch.Tensor,
    prompt_len: int,
    ctx: GenerationContext,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Prepare phone embeddings for non-streaming mode."""
    phone_tokens_to_gen, punct_ins_indices, punct_tokens = text_to_phone_tokens(
        text=text,
        config=ctx.config,
        vocab=ctx.vocab,
        phonemizer=ctx.phonemizer,
    )

    max_len = ctx.config.max_phone_tokens - prompt_phone_tokens.shape[1]
    if len(phone_tokens_to_gen) > max_len:
        ctx.logger.warning(
            f"Input text exceeds max_phone_tokens ({ctx.config.max_phone_tokens}); trimming."
        )
        phone_tokens_to_gen = phone_tokens_to_gen[:max_len]
        punct_tokens = [
            token
            for idx, token in zip(punct_ins_indices, punct_tokens, strict=True)
            if idx < max_len
        ]
        punct_ins_indices = [idx for idx in punct_ins_indices if idx < max_len]

    phone_tokens_to_gen = np.insert(
        phone_tokens_to_gen, punct_ins_indices, punct_tokens
    ).tolist()
    phone_tokens_to_gen.extend([ctx.vocab.sil, ctx.vocab.eos])

    phone_tokens = torch.cat(
        [prompt_phone_tokens, _with_unconditional(phone_tokens_to_gen, ctx)], dim=1
    )
    punct_del_indices = ctx.vocab.punct_del_indices(phone_tokens)
    phone_emb = ctx.extract_phone_embeddings(
        phone_tokens, prompt_len=prompt_len, punct_del_indices=punct_del_indices
    )
    # the pointer stops before `sil` and `[EOS]`
    phone_seq_len = phone_emb.shape[1] - 2 - punct_del_indices.shape[1]

    return phone_tokens, phone_emb, phone_seq_len


def prepare_streaming_text(
    text_gen: Generator[str, None, None],
    phone_tokens: torch.Tensor,
    empty_text_stream: bool,
    prompt_len: int,
    pointer: int,
    ctx: GenerationContext,
) -> tuple[torch.Tensor, torch.Tensor, int, bool] | None:
    """Fetch next text chunk from generator and update phone tokens/embeddings."""
    try:
        text_chunk = next(text_gen)
        if text_chunk is None:
            return None
    except StopIteration:
        empty_text_stream = True

    if empty_text_stream:
        phone_tokens_to_gen = [ctx.vocab.sil, ctx.vocab.eos]
        punct_ins_indices, punct_tokens = [], []
    else:
        phone_tokens_to_gen, punct_ins_indices, punct_tokens = text_to_phone_tokens(
            text_chunk,
            config=ctx.config,
            vocab=ctx.vocab,
            phonemizer=ctx.phonemizer,
            full_streaming=True,
        )

    max_len = ctx.config.max_phone_tokens - phone_tokens.shape[1]
    if len(phone_tokens_to_gen) > max_len and not empty_text_stream:
        ctx.logger.warning(
            f"Input text stream is too long; trimming to fit max_phone_tokens ({ctx.config.max_phone_tokens})."
        )
        phone_tokens_to_gen = phone_tokens_to_gen[:max_len]
        punct_tokens = [
            token
            for idx, token in zip(punct_ins_indices, punct_tokens, strict=True)
            if idx < max_len
        ]
        punct_ins_indices = [idx for idx in punct_ins_indices if idx < max_len]
        phone_tokens_to_gen.extend([ctx.vocab.sil, ctx.vocab.eos])
        empty_text_stream = True

    phone_tokens_to_gen = np.insert(
        np.asarray(phone_tokens_to_gen, dtype=np.int64), punct_ins_indices, punct_tokens
    ).tolist()
    phone_tokens = torch.cat(
        [phone_tokens, _with_unconditional(phone_tokens_to_gen, ctx)], dim=1
    )

    punct_del_indices = ctx.vocab.punct_del_indices(phone_tokens)
    num_punct = punct_del_indices.shape[1]
    phone_emb = ctx.extract_phone_embeddings(
        phone_tokens,
        prompt_len=prompt_len,
        punct_del_indices=punct_del_indices if num_punct else None,
    )
    phone_seq_len = phone_emb.shape[1] - num_punct
    if empty_text_stream:
        phone_seq_len -= 2

    return phone_tokens, phone_emb, phone_seq_len, empty_text_stream


def update_indices_and_tokens(
    pred_shift: torch.Tensor,
    frame: torch.Tensor,
    idx: int,
    phone_seq_len: int,
    max_phone_idx: int,
    frame_state: FrameState,
    ctx: GenerationContext,
) -> tuple[torch.Tensor, FrameState]:
    """
    Update phone embedding indices, audio tokens, and EOS logic.

    The pointer is stepped on the model's behalf after `final_phone_hold_frames` repeats
    of a state from the second-to-last phoneme on, and generation ends once it reaches
    `sil`. `max_phone_idx` is accepted for the signature of the factored frontend.
    """
    pred_shift_int = int(pred_shift.item())
    shift, num_tokens = ctx.phoneme_index_map[str(pred_shift_int)]

    start = frame_state.phone_emb_max_idx + shift
    state = (start, start + num_tokens)
    if state in frame_state.state_counter:
        if (
            start >= phone_seq_len - 2
            and frame_state.state_counter[state] == ctx.config.final_phone_hold_frames
        ):
            start += 1
            state = (start, start + num_tokens)
        # Push the model to move forward if it's stuck on the same state for too long
        elif frame_state.state_counter[state] > ctx.config.frame_repeat_counter:
            start += 1
            state = (start, start + num_tokens)

    eos_idx = frame_state.eos_idx

    if start >= phone_seq_len:
        end_token = min(start, phone_seq_len + 1)
        val = [end_token] * ctx.config.num_phones_per_frame
        eos_idx = idx
    else:
        val = list(range(int(start), int(start + num_tokens)))
        while len(val) < ctx.config.num_phones_per_frame:
            val.append(val[-1])

    phone_emb_max_idx = val[-1]
    phone_emb_indices = torch.tensor(
        [[val]] * ctx.batch_size,
        device=frame_state.phone_emb_indices.device,
        dtype=torch.int64,
    )
    mimi_codes = frame.unsqueeze(dim=2).repeat((ctx.batch_size, 1, 1))

    if state not in frame_state.state_counter:
        frame_state.state_counter[state] = 1
    else:
        frame_state.state_counter[state] += 1

    return (
        mimi_codes,
        FrameState(
            phone_emb_indices=phone_emb_indices,
            phone_emb_max_idx=phone_emb_max_idx,
            eos_idx=eos_idx,
            state_counter=frame_state.state_counter,
        ),
    )
