from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterator, List, Tuple

import numpy as np
import torch
from moshi.models import MimiModel

from voxtream.config import SpeechGeneratorConfig
from voxtream.utils.generator.context import FrameState, GenerationContext
from voxtream.utils.generator.helpers import interpolate_speaking_rate_params


@dataclass
class SpeakingRateRuntimeState:
    cfg_gamma: float | None
    spk_rate_weight: float | None = None
    target_spk_rate_cnt: torch.Tensor | None = None
    cur_spk_rate_cnt: torch.Tensor | None = None
    spk_rate_window_frames: int | None = None
    spk_rate_history: Deque[int] | None = None
    last_speaking_rate: float | None = None


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


def init_current_duration_state(
    config: SpeechGeneratorConfig,
    device: str,
) -> Tuple[torch.Tensor, int | None, Deque[int] | None]:
    duration_bins = max(int(key) for key in config.phoneme_index_map) + 1
    cur_spk_rate_cnt = torch.ones(
        duration_bins,
        dtype=torch.int64,
        device=device,
    )
    frames = (
        max(
            1,
            int(round(config.spk_rate_window_sec * 1000 / config.mimi_frame_ms)),
        )
        if config.spk_rate_window_sec is not None and config.spk_rate_window_sec > 0
        else None
    )
    return cur_spk_rate_cnt, frames, deque() if frames else None


def update_speaking_rate_params(
    speaking_rate: Iterator[float] | None,
    speaking_rate_config: Dict[str, Dict[str, list | float]],
    state: SpeakingRateRuntimeState,
    config: SpeechGeneratorConfig,
    device: str,
    logger=None,
) -> SpeakingRateRuntimeState:
    if speaking_rate is None or speaking_rate_config is None:
        return state

    try:
        current_speaking_rate = float(next(speaking_rate))
    except StopIteration as exc:
        raise ValueError(
            "speaking_rate generator must yield indefinitely. "
            "For a fixed speaking rate, use an iterator that repeats one value."
        ) from exc

    if current_speaking_rate == state.last_speaking_rate:
        return state

    duration_state, state.spk_rate_weight, state.cfg_gamma = (
        interpolate_speaking_rate_params(
            speaking_rate_config,
            current_speaking_rate,
            logger=logger,
        )
    )

    if state.target_spk_rate_cnt is None:
        (
            state.target_spk_rate_cnt,
            state.cur_spk_rate_cnt,
            state.spk_rate_window_frames,
            state.spk_rate_history,
        ) = init_spk_rate_state(
            config=config,
            target_spk_rate_cnt=duration_state,
            device=device,
        )
    else:
        updated_target_spk_rate_cnt = torch.tensor(
            duration_state,
            dtype=torch.int64,
            device=device,
        )
        if updated_target_spk_rate_cnt.shape == state.target_spk_rate_cnt.shape:
            state.target_spk_rate_cnt = updated_target_spk_rate_cnt
        else:
            (
                state.target_spk_rate_cnt,
                state.cur_spk_rate_cnt,
                state.spk_rate_window_frames,
                state.spk_rate_history,
            ) = init_spk_rate_state(
                config=config,
                target_spk_rate_cnt=duration_state,
                device=device,
            )

    state.last_speaking_rate = current_speaking_rate
    return state


def update_speaking_rate_history(
    state: SpeakingRateRuntimeState,
    pred_shift: torch.Tensor,
) -> SpeakingRateRuntimeState:
    if state.spk_rate_history is None or state.cur_spk_rate_cnt is None:
        return state

    state.spk_rate_history.append(int(pred_shift.item()))
    if len(state.spk_rate_history) > state.spk_rate_window_frames:
        dropped = state.spk_rate_history.popleft()
        state.cur_spk_rate_cnt[dropped] -= 1
    return state


def progress_metadata(
    generated_audio_frames: int,
    audio_frame_sec: float,
    frame_state: FrameState,
    prompt_phone_end_idx: int,
    speaking_rate_enabled: bool,
    speaking_rate_state: SpeakingRateRuntimeState,
) -> Dict:
    def counter_to_list(counter):
        if counter is None:
            return None
        if isinstance(counter, torch.Tensor):
            return counter.detach().float().cpu().reshape(-1).tolist()
        return list(counter)

    return {
        "time_sec": generated_audio_frames * audio_frame_sec,
        "phone_position": max(
            0, int(frame_state.phone_emb_max_idx - prompt_phone_end_idx)
        ),
        "speaking_rate": (
            speaking_rate_state.last_speaking_rate if speaking_rate_enabled else None
        ),
        "target_duration_state": counter_to_list(
            speaking_rate_state.target_spk_rate_cnt
        ),
        "current_duration_state": counter_to_list(speaking_rate_state.cur_spk_rate_cnt),
    }
