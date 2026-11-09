from dataclasses import dataclass
from typing import Generator, List

import numpy as np
import torch

from voxtream.config import SpeechGeneratorConfig
from voxtream.utils.dataset.phoneme_vocab import (
    EOW,
    EOW_PAD,
    EOW_SET,
    NUM_ROOT_SPECIAL,
    NUM_STREAMS,
    PUNCT,
    PUNCT_PAD,
    PUNCT_TOKENS,
    ROOT_EOS,
    ROOT_UNK,
    PhonemeVocab,
    punct_token,
)
from voxtream.utils.generator.context import GenerationContext
from voxtream.utils.text.normalizer import english_normalizer

PUNCT_STREAM, EOW_STREAM = PUNCT, EOW

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
    punct_symbols = tuple(PUNCT_TOKENS)

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
    vocab: PhonemeVocab,
    phonemizer,
    normalize: bool = True,
    return_phonemes: bool = False,
    language: str = "en-us",
    full_streaming: bool = False,
):
    """
    Encode text as `[n_phonemes, 5]` root/prefix/suffix/punctuation/end-of-word tokens.

    Punctuation is not a token of its own: it rides on the last phoneme of the word it
    follows, together with the end-of-word mark, so the sequence stays one row per
    phoneme and the frame pointer indexes it directly.

    Returns the `[n_phonemes, 5]` tokens and the punctuation that found no phoneme to
    ride on, which happens when a stream chunk is punctuation alone (", " between two
    word chunks): it belongs to the last phoneme of the chunk before it.
    """
    if normalize:
        # a stream chunk keeps the pauses at its edges ("Wait" / "-" / "what")
        text = english_normalizer(text, strip_edges=not full_streaming)

    if full_streaming:
        text += config.text_context

    phonemes = phonemizer.phonemize(text, language=language)

    if full_streaming:
        phonemes = phonemes[: -config.text_context_length]

    rows: List[List[int]] = []
    phones: List[str] = []
    carry = ""
    for word in phonemes.split():
        word_phones = [ph for ph in word.split("|") if ph]
        if not word_phones:
            continue

        # espeak hangs the punctuation off the last phoneme of the word ("n." , "n?!")
        symbols, last = "", word_phones[-1]
        while last and last[-1] in PUNCT_TOKENS:
            symbols = last[-1] + symbols
            last = last[:-1]
        if symbols:
            word_phones.pop()
            if last:
                word_phones.append(last)
        if not word_phones:
            # nothing but punctuation; it belongs to the phoneme before it
            carry += symbols
            continue

        for ph in word_phones:
            root, prefix, suffix = vocab.normalize(ph)
            rows.append([root, prefix, suffix, PUNCT_PAD, EOW_PAD])
            phones.append(ph)
        rows[-1][PUNCT_STREAM] = punct_token(symbols)
        rows[-1][EOW_STREAM] = EOW_SET

    if carry and rows:
        rows[-1][PUNCT_STREAM] = punct_token(carry)
        rows[-1][EOW_STREAM] = EOW_SET
        carry = ""

    tokens = np.asarray(rows, dtype=np.int64).reshape(-1, NUM_STREAMS)
    return (tokens, carry, phones) if return_phonemes else (tokens, carry)


def special_rows(roots: List[int]) -> np.ndarray:
    """`[n, 5]` rows for BOS/EOS/PROMPT, PAD on every attribute stream."""
    rows = np.zeros((len(roots), NUM_STREAMS), dtype=np.int64)
    rows[:, 0] = roots
    return rows


def _as_batch(tokens: np.ndarray, ctx: GenerationContext) -> torch.Tensor:
    """`[1, L, 5]`, with an all-UNK copy alongside it when CFG is on."""

    def to_tensor(rows):
        return (
            torch.from_numpy(rows).unsqueeze(0).to(dtype=torch.int64, device=ctx.device)
        )

    batch = to_tensor(tokens)
    if ctx.config.cfg_gamma is not None:
        batch = torch.cat([batch, to_tensor(special_rows([ROOT_UNK] * len(tokens)))])
    return batch


def prepare_non_streaming_text(
    text: str,
    prompt_phone_tokens: torch.Tensor,
    prompt_len: int,
    ctx: GenerationContext,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Prepare phone embeddings for non-streaming mode."""
    tokens, _ = text_to_phone_tokens(
        text=text,
        config=ctx.config,
        vocab=ctx.vocab,
        phonemizer=ctx.phonemizer,
    )

    # one row is reserved for the [EOS] appended below
    max_len = ctx.config.max_phone_tokens - prompt_phone_tokens.shape[1] - 1
    if len(tokens) > max_len:
        ctx.logger.warning(
            f"Input text exceeds max_phone_tokens ({ctx.config.max_phone_tokens}); trimming."
        )
        tokens = tokens[:max_len]

    # the pointer holds the last phoneme through the closing release and then steps onto
    # [EOS], which is where training leaves it and where generation ends
    tokens = np.concatenate([tokens, special_rows([ROOT_EOS])])
    phone_tokens = torch.cat([prompt_phone_tokens, _as_batch(tokens, ctx)], dim=1)

    # the pointer starts at the end of the prompt and the whole text is already there
    phone_emb = ctx.extract_phone_embeddings(
        phone_tokens, prompt_len=prompt_len, look_ahead_phones=len(tokens)
    )
    # stop once the pointer would step off the last phoneme onto [EOS]
    phone_seq_len = phone_emb.shape[1] - 1

    return phone_tokens, phone_emb, phone_seq_len


def prepare_streaming_text(
    text_gen: Generator[str, None, None],
    phone_tokens: torch.Tensor,
    empty_text_stream: bool,
    prompt_len: int,
    pointer: int,
    ctx: GenerationContext,
) -> tuple[torch.Tensor, torch.Tensor, int, bool] | None:
    """
    Fetch next text chunk from generator and update phone tokens/embeddings.

    `pointer` is the phoneme row the frame loop has reached. What lies between it and the
    end of the sequence is the look-ahead the phoneme transformer can actually use, and
    the mask is chosen to match: at the head of a stream that is only a few phonemes, so
    the first frame does not wait for text that would only be encoded against a sequence
    about to grow. That is the full-sequence encoder (`phone_kv_cache: false`); the
    KV-cached one encodes the new rows under the largest mask and re-encodes the rows
    near the end as more text arrives.
    """
    try:
        text_chunk = next(text_gen)
        if text_chunk is None:
            return None
    except StopIteration:
        empty_text_stream = True

    carry = ""
    if empty_text_stream:
        tokens = special_rows([ROOT_EOS])
    else:
        tokens, carry = text_to_phone_tokens(
            text_chunk,
            config=ctx.config,
            vocab=ctx.vocab,
            phonemizer=ctx.phonemizer,
            full_streaming=True,
        )
    # the chunk was punctuation alone: hang it on the phoneme already sent, unless all
    # that has been sent is the prompt, whose masked rows carry nothing
    if carry and phone_tokens[0, -1, 0].item() >= NUM_ROOT_SPECIAL:
        phone_tokens = phone_tokens.clone()
        phone_tokens[:, -1, PUNCT_STREAM] = punct_token(carry)
        phone_tokens[:, -1, EOW_STREAM] = EOW_SET

    max_len = ctx.config.max_phone_tokens - phone_tokens.shape[1]
    if len(tokens) > max_len and not empty_text_stream:
        ctx.logger.warning(
            f"Input text stream is too long; trimming to fit max_phone_tokens ({ctx.config.max_phone_tokens})."
        )
        tokens = np.concatenate(
            [tokens[: max(max_len - 1, 0)], special_rows([ROOT_EOS])]
        )
        empty_text_stream = True

    phone_tokens = torch.cat([phone_tokens, _as_batch(tokens, ctx)], dim=1)

    phone_seq_len = phone_tokens.shape[1]
    if empty_text_stream:
        phone_seq_len -= 1
    phone_emb = ctx.extract_phone_embeddings(
        phone_tokens,
        prompt_len=prompt_len,
        look_ahead_phones=max(0, phone_seq_len - int(pointer)),
    )

    return phone_tokens, phone_emb, phone_seq_len, empty_text_stream
