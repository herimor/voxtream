from collections.abc import Iterator
from dataclasses import dataclass
from typing import Dict, Generator, List

import numpy as np
import torch

from voxtream.config import SpeechGeneratorConfig
from voxtream.utils.generator.context import GenerationContext
from voxtream.utils.model import remove_punctuation
from voxtream.utils.text.normalizer import english_normalizer

VOWEL_PHONEME_SYMBOLS = set("aeiouæɑɒɔəɚɛɜɝɪɨʊʌøœɐɞɘɵɤɯyʉ")


@dataclass
class WordPhoneSpan:
    text: str
    start: int
    end: int


@dataclass
class TextProgressMetadata:
    words: List[WordPhoneSpan]
    phones: List[str]
    vowel_prefix: List[int]


def is_vowel_phone(phone: str) -> bool:
    """Approximate whether an IPA phoneme contains a vowel nucleus."""
    return any(ch in VOWEL_PHONEME_SYMBOLS for ch in phone.lower())


def build_text_progress_metadata(
    text: str,
    config: SpeechGeneratorConfig,
    phone_to_token: Dict,
    phonemizer,
    normalize: bool = True,
    language: str = "en-us",
    max_phone_tokens: int | None = None,
) -> TextProgressMetadata:
    """Build word spans and vowel counts in phoneme-token coordinates."""
    if normalize:
        text = english_normalizer(text)

    display_words = text.split()
    phonemes = phonemizer.phonemize(text, language=language)
    phoneme_words = phonemes.split()
    punct_symbols = tuple(config.punct_map.keys())

    phones: List[str] = []
    words: List[WordPhoneSpan] = []
    truncated = False

    for idx, phoneme_word in enumerate(phoneme_words):
        word_phones = [ph for ph in phoneme_word.split("|") if ph]
        if word_phones and word_phones[-1].endswith(punct_symbols):
            phone = word_phones.pop()
            if phone[:-1]:
                word_phones.append(phone[:-1])

        start = len(phones)
        for ph in word_phones:
            phones.append(ph)
            if max_phone_tokens is not None and len(phones) >= max_phone_tokens:
                truncated = True
                break

        end = len(phones)
        display_word = display_words[idx] if idx < len(display_words) else phoneme_word
        words.append(WordPhoneSpan(text=display_word, start=start, end=end))

        if max_phone_tokens is not None and len(phones) >= max_phone_tokens:
            break

    if not truncated and len(display_words) > len(words):
        for word in display_words[len(words) :]:
            words.append(WordPhoneSpan(text=word, start=len(phones), end=len(phones)))

    vowel_prefix = [0]
    for phone in phones:
        vowel_prefix.append(vowel_prefix[-1] + int(is_vowel_phone(phone)))

    return TextProgressMetadata(words=words, phones=phones, vowel_prefix=vowel_prefix)


def text_to_phone_tokens(
    text: str,
    config: SpeechGeneratorConfig,
    phone_to_token: Dict,
    phonemizer,
    normalize: bool = True,
    return_phonemes: bool = False,
    language: str = "en-us",
    full_streaming: bool = False,
) -> List[List[int]]:
    phone_tokens: List[int] = []
    punct_ins_indices: List[int] = []
    punct_del_indices: List[int] = []
    punct_tokens: List[int] = []

    if normalize:
        text = english_normalizer(text)

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

        if phones[-1].endswith(tuple(config.punct_map.keys())):
            phone = phones.pop()
            symb = phone[-1]
            if phone[:-1] != "":
                phones.append(phone[:-1])
            punct_tokens.append(phone_to_token[symb])
            punct_ins_indices.append(len(phones))

    for ph in phones:
        ph_list = []
        if ph not in phone_to_token:
            # TODO: Search for similar phones instead of skipping
            ph_list = ["unk"]
        else:
            ph_list.append(ph)

        for _ph in ph_list:
            phone_tokens.append(phone_to_token[_ph])

    punct_del_indices = list(punct_ins_indices + np.arange(len(punct_ins_indices)))
    output = [phone_tokens, punct_ins_indices, punct_del_indices, punct_tokens]

    if return_phonemes:
        output.append(phones)

    return output


def prepare_non_streaming_text(
    text: str,
    prompt_phone_tokens: torch.Tensor,
    prompt_punct_del_indices: torch.Tensor,
    prompt_len: int,
    ctx: GenerationContext,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Prepare phone embeddings for non-streaming mode."""
    phone_tokens_to_gen, punct_ins_indices, punct_del_indices, punct_tokens = (
        text_to_phone_tokens(
            text=text,
            config=ctx.config,
            phone_to_token=ctx.phone_to_token,
            phonemizer=ctx.phonemizer,
        )
    )

    max_len = ctx.config.max_phone_tokens - prompt_phone_tokens.shape[1]
    if len(phone_tokens_to_gen) > max_len:
        ctx.logger.warning(
            f"Input text exceeds max_phone_tokens ({ctx.config.max_phone_tokens}); trimming."
        )
        phone_tokens_to_gen = phone_tokens_to_gen[:max_len]
        punct_ins_indices = [idx for idx in punct_ins_indices if idx < max_len]
        punct_del_indices = punct_del_indices[: len(punct_ins_indices)]
        punct_tokens = punct_tokens[: len(punct_ins_indices)]

    phone_tokens_to_gen = np.insert(
        phone_tokens_to_gen, punct_ins_indices, punct_tokens
    ).tolist()

    phone_tokens_to_gen.extend([ctx.config.sil_token, ctx.config.eos_token])

    phone_tokens = torch.tensor(
        [phone_tokens_to_gen], device=ctx.device, dtype=torch.int64
    )
    punct_del_indices = torch.tensor(
        [punct_del_indices], device=ctx.device, dtype=torch.int64
    )

    if ctx.config.cfg_gamma is not None:
        phone_tokens = torch.cat(
            [
                phone_tokens,
                torch.tensor(
                    [[ctx.config.unk_token] * len(phone_tokens_to_gen)],
                    device=ctx.device,
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        )
        punct_del_indices = punct_del_indices.repeat(ctx.batch_size, 1)

    phone_tokens = torch.cat([prompt_phone_tokens, phone_tokens], dim=1)

    phone_emb = ctx.extract_phone_embeddings(phone_tokens, prompt_len=prompt_len)
    phone_seq_len = phone_emb.shape[1] - 2

    punct_del_indices_combined = prompt_punct_del_indices
    if len(punct_del_indices) > 0:
        punct_del_indices_combined = torch.cat(
            [
                punct_del_indices_combined,
                punct_del_indices + prompt_phone_tokens.shape[1],
            ],
            dim=1,
        )

    phone_emb = remove_punctuation(phone_emb, punct_del_indices_combined)
    phone_seq_len -= punct_del_indices_combined.shape[1]

    return phone_tokens, phone_emb, phone_seq_len


def prepare_streaming_text(
    text_gen: Iterator[str | None],
    phone_tokens: torch.Tensor,
    punct_del_indices: torch.Tensor,
    empty_text_stream: bool,
    prompt_len: int,
    ctx: GenerationContext,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, bool] | None:
    """Fetch next text chunk from generator and update phone tokens/embeddings."""
    try:
        text_chunk = next(text_gen)
        if text_chunk is None:
            return None
    except StopIteration:
        empty_text_stream = True

    punct_del_indices_chunk: List[int] = []
    if empty_text_stream:
        phone_tokens_to_gen = [ctx.config.sil_token, ctx.config.eos_token]
        punct_ins_indices, punct_del_indices_chunk, punct_tokens = [], [], []
    else:
        (
            phone_tokens_to_gen,
            punct_ins_indices,
            punct_del_indices_chunk,
            punct_tokens,
        ) = text_to_phone_tokens(
            text_chunk,
            config=ctx.config,
            phone_to_token=ctx.phone_to_token,
            phonemizer=ctx.phonemizer,
            full_streaming=True,
        )

    max_len = ctx.config.max_phone_tokens - phone_tokens.shape[1]
    if len(phone_tokens_to_gen) > max_len and not empty_text_stream:
        ctx.logger.warning(
            f"Input text stream is too long; trimming to fit max_phone_tokens ({ctx.config.max_phone_tokens})."
        )
        phone_tokens_to_gen = phone_tokens_to_gen[:max_len]
        punct_ins_indices = [idx for idx in punct_ins_indices if idx < max_len]
        punct_del_indices_chunk = punct_del_indices_chunk[: len(punct_ins_indices)]
        punct_tokens = punct_tokens[: len(punct_ins_indices)]
        phone_tokens_to_gen.extend([ctx.config.sil_token, ctx.config.eos_token])
        empty_text_stream = True

    phone_tokens_to_gen = (
        torch.from_numpy(
            np.insert(phone_tokens_to_gen, punct_ins_indices, punct_tokens)
        )
        .unsqueeze(0)
        .to(dtype=torch.int64, device=ctx.device)
    )

    phone_seq_len = phone_tokens.shape[1]
    punct_del_indices_chunk = torch.tensor(
        [punct_del_indices_chunk], device=ctx.device, dtype=torch.int64
    )

    if ctx.config.cfg_gamma is not None:
        phone_tokens_to_gen = torch.cat(
            [
                phone_tokens_to_gen,
                torch.tensor(
                    [[ctx.config.unk_token] * phone_tokens_to_gen.shape[1]],
                    device=ctx.device,
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        )
        punct_del_indices_chunk = punct_del_indices_chunk.repeat(ctx.batch_size, 1)

    phone_tokens = torch.cat([phone_tokens, phone_tokens_to_gen], dim=1)
    phone_emb = ctx.extract_phone_embeddings(phone_tokens, prompt_len=prompt_len)

    punct_del_indices = torch.cat(
        [punct_del_indices, punct_del_indices_chunk + phone_seq_len], dim=1
    )

    phone_seq_len = phone_emb.shape[1]
    if punct_del_indices.shape[1] > 0:
        phone_emb = remove_punctuation(phone_emb, punct_del_indices)
        phone_seq_len -= punct_del_indices.shape[1]

    if empty_text_stream:
        phone_seq_len -= 2

    return (
        phone_tokens,
        phone_emb,
        phone_seq_len,
        punct_del_indices,
        empty_text_stream,
    )
