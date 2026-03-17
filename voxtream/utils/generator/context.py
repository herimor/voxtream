from dataclasses import dataclass
from typing import Any, Dict, Protocol

import torch

from voxtream.config import SpeechGeneratorConfig


class PhonemeEmbeddingExtractor(Protocol):
    def __call__(
        self,
        phone_tokens: torch.Tensor,
        input_pos: torch.Tensor | None = ...,
        phoneme_embedding_indices: torch.Tensor | None = ...,
        prompt_len: int | None = ...,
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
    phone_to_token: Dict
    phoneme_index_map: Dict
    mimi_prompt: Any
    spk_enc: Any
    sidon_se: Any
    vad: Any

    def prompt_kwargs(self, enhance_prompt: bool, apply_vad: bool) -> Dict[str, Any]:
        return dict(
            config=self.config,
            logger=self.logger,
            device=self.device,
            dtype=self.dtype,
            batch_size=self.batch_size,
            mimi_prompt=self.mimi_prompt,
            spk_enc=self.spk_enc,
            sidon_se=self.sidon_se,
            vad=self.vad,
            enhance_prompt=enhance_prompt,
            apply_vad=apply_vad,
        )


@dataclass
class FrameState:
    phone_emb_indices: torch.Tensor
    phone_emb_max_idx: int
    eos_idx: int
    state_counter: Dict
