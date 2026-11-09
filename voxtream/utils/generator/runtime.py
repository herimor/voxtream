from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterator, List, Tuple

import numpy as np
import torch
from moshi.models import MimiModel
from moshi.utils.compile import CUDAGraphed

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


class MimiFrameDecoder:
    """
    Streaming Mimi decoder for one frame per call.
    On CUDA the whole decode (RVQ dequantize, upsample, decoder transformer, SEANet)
    runs as a single CUDA graph. Moshi graphs only the transformer and SEANet,
    leaving the 16-codebook dequantize dominated by kernel launch overhead.
    Must be created after entering `mimi.streaming(...)`: the graph captures the
    streaming state buffers, which `reset_streaming()` updates in place.
    """

    def __init__(self, mimi: MimiModel):
        self.mimi = mimi
        disable = next(mimi.parameters()).device.type != "cuda"
        self._graphed_decode = CUDAGraphed(self._decode, disable=disable)

    def _decode(self, codes: torch.Tensor) -> torch.Tensor:
        emb = self.mimi.decode_latent(codes)
        emb = self.mimi._to_encoder_framerate(emb)
        if self.mimi.decoder_transformer is not None:
            (emb,) = self.mimi.decoder_transformer(emb)
        return self.mimi.decoder(emb).to(dtype=torch.float32)

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode `codes` (batch, num_codebooks, 1). On CUDA the output is a static
        graph buffer, overwritten by the next call."""
        return self._graphed_decode(codes)


# A decoded frame counts as sound above this level; below it the frame is silence and
# the opening of a draw is still held back (`start_retry_frames`). Mimi renders silence
# near -70 dBFS and speech above -40, so the threshold sits well clear of both.
SPEECH_DBFS = -50.0


def frame_has_speech(audio_frame: np.ndarray) -> bool:
    """Whether a decoded frame carries sound rather than silence."""
    if audio_frame.size == 0:
        return False
    power = float(np.mean(np.square(audio_frame, dtype=np.float64)))
    return 10.0 * np.log10(power + 1e-12) > SPEECH_DBFS


def decode_audio_frame(
    mimi_decoder: MimiFrameDecoder,
    frame: torch.Tensor,
    sem_code: torch.Tensor,
    mimi_vocab_size: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Decode predicted frame into audio."""
    audio_frame = torch.cat([sem_code, frame[:, 1:]], dim=1)
    audio_frame = torch.clamp(audio_frame, 0, int(mimi_vocab_size - 1)).to(torch.int64)
    sem_code = frame[:, :1]

    # .cpu() copies out of the static graph buffer.
    audio_frame = mimi_decoder(audio_frame.unsqueeze(-1)).squeeze().cpu().numpy()
    return audio_frame, sem_code


def update_indices_and_tokens(
    pred_shift: torch.Tensor,
    frame: torch.Tensor,
    idx: int,
    phone_seq_len: int,
    max_phone_idx: int,
    frame_state: FrameState,
    ctx: GenerationContext,
) -> Tuple[torch.Tensor, FrameState]:
    """
    Update phone embedding indices, audio tokens, and EOS logic.

    `phone_seq_len` is where the pointer has to stop; `max_phone_idx` is the last row the
    phoneme sequence actually has. They are not the same: while a text stream is still
    arriving there is no stop at all, and once it ends the pointer holds the last phoneme
    through the closing release and then steps onto `[EOS]`, which ends generation. Every
    index is clamped to `max_phone_idx` so neither case can reach past the embeddings.
    Their difference is also how this function knows the text has ended: only then do the
    two coincide.
    """
    pred_shift_int = int(pred_shift.item())
    shift, num_tokens = ctx.phoneme_index_map[str(pred_shift_int)]

    start = frame_state.phone_emb_max_idx + shift
    state = (start, start + num_tokens)
    kicked = False
    if state in frame_state.state_counter:
        repeats = frame_state.state_counter[state]
        # Generation ends when the model steps off the last phoneme onto `[EOS]`, which
        # a crop closing on a sentence end supervises. This is the net for when it does
        # not: once the text has run out and the pointer has held the final row long
        # enough to have rendered the release, take the step on the model's behalf. Both
        # guards matter: mid-stream the last row is simply the newest text, and forcing
        # the pointer off it would end generation in the middle of the utterance.
        text_ended = phone_seq_len <= max_phone_idx
        if (
            text_ended
            and start >= phone_seq_len - 1
            and repeats >= ctx.config.final_phone_hold_frames
        ):
            start += 1
            state = (start, start + num_tokens)
        # Push the model to move forward if it's stuck on the same state for too long
        elif repeats > ctx.config.frame_repeat_counter:
            start += 1
            state = (start, start + num_tokens)
            kicked = True

    # a step of the model's own ends a stall; a hold leaves the count as it is
    if kicked:
        kicks = frame_state.kicks + 1
        stall_anchor = (
            frame_state.stall_anchor if kicks > 1 else frame_state.phone_emb_max_idx
        )
    elif shift > 0:
        kicks, stall_anchor = 0, -1
    else:
        kicks, stall_anchor = frame_state.kicks, frame_state.stall_anchor

    eos_idx = frame_state.eos_idx

    if start >= phone_seq_len:
        # The loop runs one more frame after `eos_idx` is set, so this index is still
        # fed to the model and decoded into audio. Once the text has ended the row it
        # lands on is `[EOS]`, where training leaves the pointer through the silence
        # that closes a crop, so the frame renders that silence. While the text is still
        # arriving there is no `[EOS]` yet and the clamp lands on the newest row.
        val = [max(0, min(int(start), max_phone_idx))] * ctx.config.num_phones_per_frame
        eos_idx = idx
    else:
        val = [
            min(i, max_phone_idx) for i in range(int(start), int(start + num_tokens))
        ]
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
            kicks=kicks,
            stall_anchor=stall_anchor,
        ),
    )


def restart_from_stall(frame_state: FrameState, ctx: GenerationContext) -> FrameState:
    """The frame state rewound to the pointer the stall began on, its counters cleared
    so the rows ahead may be held again. The audio context is reset by the caller."""
    anchor = max(0, int(frame_state.stall_anchor))
    rows = [anchor] * ctx.config.num_phones_per_frame
    return FrameState(
        phone_emb_indices=torch.tensor(
            [[rows]] * ctx.batch_size,
            device=frame_state.phone_emb_indices.device,
            dtype=torch.int64,
        ),
        phone_emb_max_idx=anchor,
        eos_idx=frame_state.eos_idx,
        state_counter={
            s: c for s, c in frame_state.state_counter.items() if s[0] < anchor
        },
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
