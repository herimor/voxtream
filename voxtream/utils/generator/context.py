from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Protocol

import torch

from voxtream.config import SpeechGeneratorConfig
from voxtream.utils.dataset.phoneme_vocab import PhonemeVocab

if TYPE_CHECKING:
    from voxtream.utils.generator.flat_text import FlatPhonemeVocab


class PhonemeEmbeddingExtractor(Protocol):
    def __call__(
        self,
        phone_tokens: torch.Tensor,
        input_pos: torch.Tensor | None = ...,
        phoneme_embedding_indices: torch.Tensor | None = ...,
        prompt_len: int | None = ...,
        look_ahead_phones: int | None = ...,
        punct_del_indices: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...


@dataclass
class GenerationContext:
    config: SpeechGeneratorConfig
    logger: Any
    device: str
    dtype: torch.dtype
    batch_size: int
    extract_phone_embeddings: PhonemeEmbeddingExtractor
    phonemizer: Any
    vocab: "PhonemeVocab | FlatPhonemeVocab"
    phoneme_index_map: Dict
    mimi_prompt: Any
    spk_enc: Any
    sidon_se: Any
    vad: Any

    def encode_kwargs(
        self, enhance_prompt: bool, apply_vad: bool, normalize_loudness: bool
    ) -> Dict[str, Any]:
        """The arguments of `encode_prompt_audio`."""
        kwargs = self.prompt_kwargs(enhance_prompt, apply_vad, normalize_loudness)
        del kwargs["batch_size"], kwargs["vocab"]
        return kwargs

    def prompt_kwargs(
        self, enhance_prompt: bool, apply_vad: bool, normalize_loudness: bool
    ) -> Dict[str, Any]:
        """The arguments of `prepare_prompt`."""
        return dict(
            config=self.config,
            logger=self.logger,
            device=self.device,
            dtype=self.dtype,
            batch_size=self.batch_size,
            vocab=self.vocab,
            mimi_prompt=self.mimi_prompt,
            spk_enc=self.spk_enc,
            sidon_se=self.sidon_se,
            vad=self.vad,
            enhance_prompt=enhance_prompt,
            apply_vad=apply_vad,
            normalize_loudness=normalize_loudness,
        )


@dataclass
class FrameState:
    phone_emb_indices: torch.Tensor
    phone_emb_max_idx: int
    eos_idx: int
    state_counter: Dict
    # consecutive forced pointer steps, and the pointer before the first of them; see
    # `stall_kicks`
    kicks: int = 0
    stall_anchor: int = -1
