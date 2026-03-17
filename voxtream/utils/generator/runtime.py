from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
from moshi.models import MimiModel

from voxtream.config import SpeechGeneratorConfig
from voxtream.utils.generator.context import FrameState, GenerationContext


def decode_audio_frame(
    mimi: MimiModel,
    frame: torch.Tensor,
    sem_code: torch.Tensor,
    mimi_vocab_size: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Decode predicted frame into audio."""
    audio_frame = torch.cat([sem_code, frame[:, 1:]], dim=1)
    audio_frame = torch.clamp(audio_frame, 0, int(mimi_vocab_size - 1)).to(torch.int64)
    sem_code = frame[:, :1]

    audio_frame = mimi.decode(audio_frame.unsqueeze(-1)).squeeze()
    audio_frame = audio_frame.to(dtype=torch.float32).cpu().numpy()
    return audio_frame, sem_code


def update_indices_and_tokens(
    pred_shift: torch.Tensor,
    frame: torch.Tensor,
    idx: int,
    phone_seq_len: int,
    frame_state: FrameState,
    ctx: GenerationContext,
) -> Tuple[torch.Tensor, FrameState]:
    """Update phone embedding indices, audio tokens, and EOS logic."""
    pred_shift_int = int(pred_shift.item())
    shift, num_tokens = ctx.phoneme_index_map[str(pred_shift_int)]

    start = frame_state.phone_emb_max_idx + shift
    state = (start, start + num_tokens)
    if state in frame_state.state_counter:
        if start >= phone_seq_len - 2 and frame_state.state_counter[state] == 3:
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


def init_spk_rate_state(
    config: SpeechGeneratorConfig,
    target_spk_rate_cnt: List[int] | None,
    device: str,
) -> Tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    int | None,
    Deque[int] | None,
]:
    if target_spk_rate_cnt is None:
        return None, None, None, None

    target_spk_rate_cnt = torch.tensor(
        target_spk_rate_cnt,
        dtype=torch.int64,
        device=device,
    )
    cur_spk_rate_cnt = torch.ones_like(target_spk_rate_cnt)
    frames = (
        max(
            1,
            int(round(config.spk_rate_window_sec * 1000 / config.mimi_frame_ms)),
        )
        if config.spk_rate_window_sec is not None and config.spk_rate_window_sec > 0
        else None
    )
    return (
        target_spk_rate_cnt,
        cur_spk_rate_cnt,
        frames,
        deque() if frames else None,
    )
