from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class SinkAttentionConfig:
    audio_window_size: int
    min_recent_window: int = 1
    tail_budget_padding: int = 1
    tail_window_divisor: int = 2


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
            if local_pos >= self.config.audio_window_size:
                self._rebuild_temp_former_cache(
                    global_pos,
                    temp_former,
                    temp_former_causal_mask,
                    embed_audio_tokens,
                )
            local_input_pos = input_pos - self._sink_pos_offset

        curr_temp_former_mask = self._get_temp_former_sink_mask(
            local_input_pos, temp_former_causal_mask
        )

        return local_input_pos, curr_temp_former_mask

    def append_step(self, phone_emb: torch.Tensor, audio_tokens: torch.Tensor) -> None:
        if self._sink_prompt_len is None:
            return

        self._sink_phone_emb_buf.append(phone_emb.detach())
        self._sink_audio_tokens_buf.append(audio_tokens.detach())
        self._trim_sink_buffers()

    def _get_temp_former_sink_mask(
        self, input_pos: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
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

        return causal & (prompt_mask | recent_mask)

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
        mask = self._get_temp_former_sink_mask(input_pos, temp_former_causal_mask)
        temp_former(h, input_pos=input_pos, mask=mask).to(dtype=audio_emb.dtype)

        self._sink_pos_offset = global_pos - seq_len
