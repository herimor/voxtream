import logging
import time
from pathlib import Path
from typing import Generator, List

import numpy as np
import torch
import torch._inductor.config
from moshi.utils.compile import CUDAGraphed
from silero_vad import load_silero_vad

torch.set_float32_matmul_precision("medium")
torch._inductor.config.fx_graph_cache = True

from voxtream.config import SpeechGeneratorConfig
from voxtream.utils.generator import DTYPE_MAP, configure_cpu_threads
from voxtream.utils.generator.context import FrameState, GenerationContext
from voxtream.utils.generator.helpers import autocast_ctx
from voxtream.utils.generator.prompt import prepare_prompt
from voxtream.utils.generator.runtime import (
    decode_audio_frame,
    init_spk_rate_state,
    update_indices_and_tokens,
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
from voxtream.utils.sidon_se import SidonSE
from voxtream.utils.text.phonemizer import ESpeak


class SpeechGenerator:
    def __init__(self, config: SpeechGeneratorConfig, compile: bool = False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = DTYPE_MAP[device]
        batch_size = 1 if config.cfg_gamma is None else 2

        model, phone_to_token = load_generator_model(
            config=config,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )

        self.config = config
        self.logger = logging.getLogger("voxtream")

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
        sidon_se = SidonSE(device=device, reload_model=config.sidon_se_reload_model)
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
        text: str | Generator[str, None, None],
        target_spk_rate_cnt: List[int] = None,
        spk_rate_weight: float = None,
        cfg_gamma: float = None,
        enhance_prompt: bool = None,
        apply_vad: bool = None,
    ) -> Generator[np.ndarray, None, None]:
        # Override parameters
        cfg_gamma = self.config.cfg_gamma if cfg_gamma is None else cfg_gamma
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

        frame_state = FrameState(
            phone_emb_indices=phone_emb_indices,
            phone_emb_max_idx=phone_emb_max_idx,
            eos_idx=eos_idx,
            state_counter={},
        )

        (
            target_spk_rate_cnt,
            cur_spk_rate_cnt,
            spk_rate_window_frames,
            spk_rate_history,
        ) = init_spk_rate_state(
            config=self.config,
            target_spk_rate_cnt=target_spk_rate_cnt,
            device=self.ctx.device,
        )

        self._ensure_mimi_streaming()

        start_time = time.perf_counter()
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

            with self._autocast_ctx:
                frame, pred_shift, cur_spk_rate_cnt = self.model.generate_frame(
                    config=self.config,
                    phone_emb=phone_emb_chunk,
                    audio_tokens=mimi_codes,
                    input_pos=curr_pos,
                    spk_embeddings=spk_embedding,
                    cfg_gamma=cfg_gamma,
                    spk_rate_weight=spk_rate_weight,
                    cur_spk_rate_cnt=cur_spk_rate_cnt,
                    target_spk_rate_cnt=target_spk_rate_cnt,
                )

            if spk_rate_history is not None and cur_spk_rate_cnt is not None:
                spk_rate_history.append(int(pred_shift.item()))
                if len(spk_rate_history) > spk_rate_window_frames:
                    dropped = spk_rate_history.popleft()
                    cur_spk_rate_cnt[dropped] -= 1

            if idx >= self.config.audio_delay_frames:
                audio_frame, sem_code = decode_audio_frame(
                    mimi=self.mimi,
                    frame=frame,
                    sem_code=sem_code,
                    mimi_vocab_size=self.config.mimi_vocab_size,
                )
                gen_time = time.perf_counter() - start_time
                start_time = time.perf_counter()
                yield audio_frame, gen_time
            else:
                sem_code = frame[:, :1]

            if frame_state.eos_idx <= idx:
                break

            mimi_codes, frame_state = update_indices_and_tokens(
                pred_shift=pred_shift,
                frame=frame,
                idx=idx,
                phone_seq_len=phone_seq_len,
                frame_state=frame_state,
                ctx=self.ctx,
            )
            phone_emb_indices = frame_state.phone_emb_indices

            curr_pos = curr_pos[:, -1:] + 1

        if self.config.reset_streaming_state:
            self._mimi_stream_ctx.__exit__(None, None, None)
            self._mimi_streaming_started = False
