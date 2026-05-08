import logging
import time
from pathlib import Path
from typing import Dict, Generator, Iterator, Optional

import numpy as np
import torch
import torch._inductor.config
from moshi.utils.compile import CUDAGraphed
from silero_vad import load_silero_vad

torch.set_float32_matmul_precision("medium")
torch._inductor.config.fx_graph_cache = True

from voxtream.config import SpeechGeneratorConfig
from voxtream.utils.generator import (
    DTYPE_MAP,
    configure_cpu_threads,
)
from voxtream.utils.generator.context import FrameState, GenerationContext
from voxtream.utils.generator.helpers import autocast_ctx
from voxtream.utils.generator.prompt import prepare_prompt
from voxtream.utils.generator.runtime import (
    SpeakingRateRuntimeState,
    decode_audio_frame,
    init_current_duration_state,
    progress_metadata,
    update_indices_and_tokens,
    update_speaking_rate_history,
    update_speaking_rate_params,
)
from voxtream.utils.generator.setup import (
    load_generator_model,
    load_mimi_model,
    load_speaker_encoder,
)
from voxtream.utils.generator.text import (
    prepare_non_streaming_text,
    prepare_streaming_text,
)
from voxtream.utils.model import patch_kv_cache_for_cuda_graph
from voxtream.utils.sidon_se import LazySidonSE
from voxtream.utils.text.phonemizer import ESpeak


class SpeechGenerator:
    def __init__(
        self,
        config: SpeechGeneratorConfig,
        spk_rate_config: Dict[str, Dict[str, list | float]] = None,
        compile: bool = False,
    ):
        self.config = config
        self.logger = logging.getLogger("voxtream")
        self.spk_rate_config = spk_rate_config

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = DTYPE_MAP[device]
        batch_size = 1 if config.cfg_gamma is None else 2

        model, phone_to_token = load_generator_model(
            config=config,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )

        self.model = model
        self.mimi = load_mimi_model(
            config=config,
            device=device,
            dtype=dtype,
            num_codebooks=config.num_codebooks,
        )
        mimi_prompt = load_mimi_model(
            config=config,
            device=device,
            dtype=dtype,
            num_codebooks=config.num_codebooks,
        )
        spk_enc = load_speaker_encoder(
            config=config,
            device=device,
            dtype=dtype,
        )
        sidon_se = LazySidonSE(
            device=device,
            reload_model=config.sidon_se_reload_model,
            logger=self.logger,
        )
        vad = load_silero_vad()

        self.ctx = GenerationContext(
            config=config,
            logger=self.logger,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            extract_phone_embeddings=self.model.extract_phoneme_embeddings,
            phonemizer=ESpeak(),
            phone_to_token=phone_to_token,
            phoneme_index_map=config.phoneme_index_map,
            mimi_prompt=mimi_prompt,
            spk_enc=spk_enc,
            sidon_se=sidon_se,
            vad=vad,
        )
        self._autocast_ctx = autocast_ctx(device=device, dtype=dtype)

        if self.ctx.device == "cuda":
            if compile:
                self.model._temp_former = torch.compile(
                    model=self.model._temp_former, dynamic=False, mode="max-autotune"
                )
                self.model._dep_former_init = torch.compile(
                    model=self.model._dep_former, dynamic=False, mode="max-autotune"
                )
                self.model._dep_former = torch.compile(
                    model=self.model._dep_former, dynamic=False, mode="max-autotune"
                )
            else:
                patch_kv_cache_for_cuda_graph()
                self.model._temp_former = CUDAGraphed(self.model._temp_former)
                self.model._dep_former_init = CUDAGraphed(self.model._dep_former)
                self.model._dep_former = CUDAGraphed(self.model._dep_former)
        else:
            configure_cpu_threads()

        self._mimi_stream_ctx = None
        self._mimi_streaming_started = False

    def _ensure_mimi_streaming(self, batch_size: int = 1) -> None:
        """
        Initialize Mimi streaming once and reuse the compiled state across streams.
        Resets streaming buffers for each new generation without reinitializing graphs.
        """
        if not self._mimi_streaming_started:
            self._mimi_stream_ctx = self.mimi.streaming(batch_size=batch_size)
            self._mimi_stream_ctx.__enter__()
            self._mimi_streaming_started = True
        else:
            self.mimi.reset_streaming()

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_audio_path: Path,
        text: str | Iterator[str | None],
        speaking_rate: Optional[Iterator[float]] = None,
        enhance_prompt: bool | None = None,
        apply_vad: bool | None = None,
        return_progress: bool = False,
        min_streaming_rtf: float | None = None,
    ) -> Generator[
        tuple[np.ndarray, float] | tuple[np.ndarray, float, Dict], None, None
    ]:
        # Override parameters
        enhance_prompt = (
            self.config.enhance_prompt if enhance_prompt is None else enhance_prompt
        )
        apply_vad = self.config.apply_vad if apply_vad is None else apply_vad

        (
            mimi_codes,
            spk_embedding,
            prompt_phone_tokens,
            prompt_phone_tokens_to_embed,
            punct_del_indices,
        ) = prepare_prompt(
            prompt_audio_path=prompt_audio_path,
            **self.ctx.prompt_kwargs(
                enhance_prompt=enhance_prompt, apply_vad=apply_vad
            ),
        )

        phone_emb_indices = prompt_phone_tokens_to_embed

        self.model.reset_caches()
        spk_embedding = self.model.spk_emb_proj(spk_embedding)

        if isinstance(text, str):
            phone_tokens, phone_emb, phone_seq_len = prepare_non_streaming_text(
                text,
                prompt_phone_tokens,
                punct_del_indices,
                prompt_phone_tokens.shape[1],
                self.ctx,
            )
            empty_text_stream = True
        else:
            phone_emb, phone_seq_len = None, 0
            phone_tokens = prompt_phone_tokens
            empty_text_stream = False

        max_seq_len = int(self.config.max_audio_length_ms / self.config.mimi_frame_ms)
        curr_pos = (
            torch.arange(0, mimi_codes.size(2))
            .unsqueeze(0)
            .to(self.ctx.device, dtype=torch.int64)
        )

        eos_idx = max_seq_len
        sem_code: torch.Tensor = None
        phone_emb_max_idx = int(phone_emb_indices[0, -1, -1].item())
        prompt_phone_end_idx = phone_emb_max_idx

        frame_state = FrameState(
            phone_emb_indices=phone_emb_indices,
            phone_emb_max_idx=phone_emb_max_idx,
            eos_idx=eos_idx,
            state_counter={},
        )
        spk_rate_state = SpeakingRateRuntimeState(cfg_gamma=self.config.cfg_gamma)
        if speaking_rate is None:
            (
                spk_rate_state.cur_spk_rate_cnt,
                spk_rate_state.spk_rate_window_frames,
                spk_rate_state.spk_rate_history,
            ) = init_current_duration_state(
                config=self.config,
                device=self.ctx.device,
            )

        self._ensure_mimi_streaming()

        start_time = time.perf_counter()
        generated_audio_frames = 0
        audio_frame_sec = self.config.mimi_frame_ms / 1000.0

        for idx in range(max_seq_len):
            if not empty_text_stream:
                is_enough_text = False
                while not is_enough_text and not empty_text_stream:
                    result = prepare_streaming_text(
                        text_gen=text,
                        phone_tokens=phone_tokens,
                        punct_del_indices=punct_del_indices,
                        empty_text_stream=empty_text_stream,
                        prompt_len=prompt_phone_tokens.shape[1],
                        ctx=self.ctx,
                    )
                    if result is not None:
                        (
                            phone_tokens,
                            phone_emb,
                            phone_seq_len,
                            punct_del_indices,
                            empty_text_stream,
                        ) = result

                    if (
                        not empty_text_stream
                        and phone_seq_len - frame_state.phone_emb_max_idx
                        < self.config.min_look_ahead_phones
                    ):
                        time.sleep(0.01)
                    else:
                        is_enough_text = True

            assert (
                frame_state.phone_emb_max_idx < phone_emb.shape[1]
            ), "Phone embedding index out of range!"
            phone_emb_chunk = self.model.reorder_phone_emb(phone_emb, phone_emb_indices)
            spk_rate_state = update_speaking_rate_params(
                speaking_rate=speaking_rate,
                speaking_rate_config=self.spk_rate_config,
                state=spk_rate_state,
                config=self.config,
                device=self.ctx.device,
                logger=self.logger,
            )

            with self._autocast_ctx:
                (
                    frame,
                    pred_shift,
                    spk_rate_state.cur_spk_rate_cnt,
                ) = self.model.generate_frame(
                    config=self.config,
                    phone_emb=phone_emb_chunk,
                    audio_tokens=mimi_codes,
                    input_pos=curr_pos,
                    spk_embeddings=spk_embedding,
                    cfg_gamma=spk_rate_state.cfg_gamma,
                    spk_rate_weight=spk_rate_state.spk_rate_weight,
                    cur_spk_rate_cnt=spk_rate_state.cur_spk_rate_cnt,
                    target_spk_rate_cnt=spk_rate_state.target_spk_rate_cnt,
                )

            spk_rate_state = update_speaking_rate_history(
                spk_rate_state,
                pred_shift,
            )

            if idx >= self.config.audio_delay_frames:
                audio_frame, sem_code = decode_audio_frame(
                    mimi=self.mimi,
                    frame=frame,
                    sem_code=sem_code,
                    mimi_vocab_size=self.config.mimi_vocab_size,
                )
                gen_time = time.perf_counter() - start_time
                generated_audio_frames += 1

                if min_streaming_rtf is not None:
                    min_frame_time = audio_frame_sec * min_streaming_rtf
                    if gen_time < min_frame_time:
                        sleep_time = min_frame_time - gen_time
                        sleep_start = time.perf_counter()
                        time.sleep(sleep_time)
                        actual_sleep_time = time.perf_counter() - sleep_start
                        gen_time += actual_sleep_time

                start_time = time.perf_counter()
            else:
                sem_code = frame[:, :1]

            eos_reached = frame_state.eos_idx <= idx
            if not eos_reached:
                mimi_codes, frame_state = update_indices_and_tokens(
                    pred_shift=pred_shift,
                    frame=frame,
                    idx=idx,
                    phone_seq_len=phone_seq_len,
                    frame_state=frame_state,
                    ctx=self.ctx,
                )
                phone_emb_indices = frame_state.phone_emb_indices

            if idx >= self.config.audio_delay_frames:
                if return_progress:
                    yield audio_frame, gen_time, progress_metadata(
                        generated_audio_frames=generated_audio_frames,
                        audio_frame_sec=audio_frame_sec,
                        frame_state=frame_state,
                        prompt_phone_end_idx=prompt_phone_end_idx,
                        speaking_rate_enabled=speaking_rate is not None,
                        speaking_rate_state=spk_rate_state,
                    )
                else:
                    yield audio_frame, gen_time

            if eos_reached:
                break

            curr_pos = curr_pos[:, -1:] + 1

        if self.config.reset_streaming_state:
            self._mimi_stream_ctx.__exit__(None, None, None)
            self._mimi_streaming_started = False
