import importlib.util
import logging
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Generator, Iterator

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
    flat_text,
)
from voxtream.utils.generator.context import FrameState, GenerationContext
from voxtream.utils.generator.helpers import autocast_ctx
from voxtream.utils.generator.prompt import (
    STAGE_MIMI,
    STAGE_SPEAKER,
    encode_prompt_audio,
    prepare_prompt,
)
from voxtream.utils.generator.runtime import (
    MimiFrameDecoder,
    SpeakingRateRuntimeState,
    decode_audio_frame,
    frame_has_speech,
    init_current_duration_state,
    progress_metadata,
    restart_from_stall,
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

# the steps of a stream before its first frame, after the prompt's (see prompt.py)
STAGE_PHONEMES = "phonemes"
STAGE_GENERATION = "generation"
STAGES = (STAGE_MIMI, STAGE_SPEAKER, STAGE_PHONEMES, STAGE_GENERATION)


@dataclass(frozen=True)
class TextFrontend:
    """The text side of a generation, which depends on the phoneme vocabulary."""

    prepare_non_streaming_text: Callable
    prepare_streaming_text: Callable
    update_indices_and_tokens: Callable


FACTORED_FRONTEND = TextFrontend(
    prepare_non_streaming_text=prepare_non_streaming_text,
    prepare_streaming_text=prepare_streaming_text,
    update_indices_and_tokens=update_indices_and_tokens,
)
# VoXtream2 and VoXtream2.1
FLAT_FRONTEND = TextFrontend(
    prepare_non_streaming_text=flat_text.prepare_non_streaming_text,
    prepare_streaming_text=flat_text.prepare_streaming_text,
    update_indices_and_tokens=flat_text.update_indices_and_tokens,
)


def _once_per_stage(on_stage: Callable[[str], None]) -> Callable[[str], None]:
    """`on_stage` told of each stage once: a redrawn opening runs the stages again."""
    if on_stage is None:
        return None
    seen = set()

    def report(stage: str) -> None:
        if stage not in seen:
            seen.add(stage)
            on_stage(stage)

    return report


class SpeechGenerator:
    def __init__(
        self,
        config: SpeechGeneratorConfig,
        spk_rate_config: Dict[str, Dict[str, list | float]] = None,
        compile: bool = False,
    ):
        """
        Without a GPU, the transformer stacks run through llama.cpp (q8_0) when
        `config.gguf` is set, from `config.gguf_dir` or the model repo, on
        `config.cpu_threads` threads; they fall back to PyTorch when llama-cpp-python is
        not installed.
        """
        self.config = config
        self.logger = logging.getLogger("voxtream")
        self.spk_rate_config = spk_rate_config

        gguf = config.gguf and not torch.cuda.is_available()
        if gguf and importlib.util.find_spec("llama_cpp") is None:
            self.logger.warning(
                "llama-cpp-python is not installed; running the transformer stacks in "
                "PyTorch"
            )
            gguf = False
        gguf_args = None
        if gguf:
            gguf_args = dict(dir=config.gguf_dir, threads=config.cpu_threads)
            # llama.cpp executes on the host, so the torch side stays on the CPU
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = DTYPE_MAP[device]
        batch_size = 1 if config.cfg_gamma is None else 2

        model, vocab = load_generator_model(
            config=config,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            gguf=gguf_args,
        )
        self.gguf = gguf_args

        self.model = model
        self.frontend = (
            FLAT_FRONTEND if model.config.flat_phone_vocab else FACTORED_FRONTEND
        )
        self.mimi = load_mimi_model(
            config=config,
            device=device,
            dtype=dtype,
            num_codebooks=config.num_codebooks,
            gguf=gguf_args,
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

        if config.phonemizer not in ("espeak", "espeak-lib"):
            raise ValueError(
                f"phonemizer must be 'espeak' or 'espeak-lib', got {config.phonemizer!r}"
            )

        self.ctx = GenerationContext(
            config=config,
            logger=self.logger,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            extract_phone_embeddings=(
                self.model.encode_phonemes
                if config.phone_kv_cache
                else self.model.extract_phoneme_embeddings
            ),
            phonemizer=ESpeak(library=config.phonemizer),
            vocab=vocab,
            phoneme_index_map=config.phoneme_index_map,
            mimi_prompt=mimi_prompt,
            spk_enc=spk_enc,
            sidon_se=sidon_se,
            vad=vad,
        )
        self._autocast_ctx = autocast_ctx(device=device, dtype=dtype)

        if self.ctx.device == "cuda":
            mog = self.model.config.dep_former_head == "mog"
            if compile:
                self.model._temp_former = torch.compile(
                    model=self.model._temp_former, dynamic=False, mode="max-autotune"
                )
                if not mog:
                    self.model._dep_former_init = torch.compile(
                        model=self.model._dep_former, dynamic=False, mode="max-autotune"
                    )
                self.model._dep_former = torch.compile(
                    model=self.model._dep_former, dynamic=False, mode="max-autotune"
                )
            else:
                patch_kv_cache_for_cuda_graph()
                self.model._temp_former = CUDAGraphed(self.model._temp_former)
                if not mog:
                    self.model._dep_former_init = CUDAGraphed(self.model._dep_former)
                    self.model._dep_former = CUDAGraphed(self.model._dep_former)
            if mog and not compile:
                # The MoG decoding of a frame is a single CUDA graph, its depth
                # transformer steps included
                self.model._generate_acoustic_mog = CUDAGraphed(
                    partial(
                        self.model.generate_acoustic_mog,
                        dtype=dtype,
                        **self.model.mog_decoding_kwargs(config, batch_size == 2),
                    )
                )
            if config.phone_kv_cache:
                # a word of text is one forward of a fixed number of rows, one graph
                phone_encoder = self.model.phone_sink_attention
                phone_encoder.enable_graph(CUDAGraphed(phone_encoder._step))
        else:
            configure_cpu_threads(config.cpu_threads)

        self._mimi_stream_ctx = None
        self._mimi_streaming_started = False
        self._mimi_decoder: MimiFrameDecoder | None = None

    def _ensure_mimi_streaming(self, batch_size: int = 1) -> None:
        """
        Initialize Mimi streaming once and reuse the compiled state across streams.
        Resets streaming buffers for each new generation without reinitializing graphs.
        """
        if not self._mimi_streaming_started:
            self._mimi_stream_ctx = self.mimi.streaming(batch_size=batch_size)
            self._mimi_stream_ctx.__enter__()
            self._mimi_streaming_started = True
            # New streaming state buffers, so the decode graph must be captured again.
            self._mimi_decoder = MimiFrameDecoder(self.mimi)
        else:
            self.mimi.reset_streaming()

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt_audio_path: Path,
        enhance_prompt: bool = None,
        apply_vad: bool = None,
        normalize_loudness: bool = None,
        on_stage: Callable[[str], None] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The Mimi codes and the speaker embedding of a prompt audio, what a cache of
        prompts holds; `generate_stream` takes the pair as `prompt_encoding`. The
        options default to the config. `on_stage` is told when the Mimi encoder
        (`STAGE_MIMI`) and the speaker encoder (`STAGE_SPEAKER`) start.
        """
        return encode_prompt_audio(
            prompt_audio_path=Path(prompt_audio_path),
            on_stage=on_stage,
            **self.ctx.encode_kwargs(
                enhance_prompt=(
                    self.config.enhance_prompt
                    if enhance_prompt is None
                    else enhance_prompt
                ),
                apply_vad=self.config.apply_vad if apply_vad is None else apply_vad,
                normalize_loudness=(
                    self.config.normalize_prompt_loudness
                    if normalize_loudness is None
                    else normalize_loudness
                ),
            ),
        )

    def generate_stream(
        self,
        prompt_audio_path: Path,
        text: str | Generator[str, None, None],
        speaking_rate: Iterator[float] = None,
        enhance_prompt: bool = None,
        apply_vad: bool = None,
        normalize_loudness: bool = None,
        return_progress: bool = False,
        max_stream_lead_sec: float = None,
        prompt_encoding: tuple[torch.Tensor, torch.Tensor] = None,
        on_stage: Callable[[str], None] = None,
    ) -> Generator[
        tuple[np.ndarray, float] | tuple[np.ndarray, float, Dict], None, None
    ]:
        """
        Speak `text` in the voice of `prompt_audio_path`, one Mimi frame at a time.

        The prompt is encoded once (`encode_prompt`), before the first draw, unless
        `prompt_encoding` brings its codes and speaker embedding, say from a cache.
        `on_stage` is told once when each step before the first frame starts: the
        Mimi encoder, the speaker encoder, the phonemes of a text given whole
        (`STAGE_PHONEMES`) and the generation itself (`STAGE_GENERATION`).

        A draw that opens with `start_retry_frames` of silence is abandoned and taken
        again: the model sometimes parks the pointer on the first phoneme and renders
        silence, and another draw of the same utterance comes out clean. Holding the
        opening frames back costs the listener nothing, since silence carries nothing,
        and it is what makes the retry invisible: nothing audible has been handed over
        yet. The frames held back are also where the opening silence is trimmed, down to
        `start_lead_in_frames`. Once a frame carries sound the stream runs straight
        through to the end.

        A text stream is consumed as it is read, so there is no second draw to take and
        the first one is streamed as is.

        Pacing: `max_stream_lead_sec` caps how far the audio may run ahead of the wall
        clock since the first frame, which bounds the latency of the speaking-rate
        control while a lead lost to a slow frame is won back at full speed.
        """
        on_stage = _once_per_stage(on_stage)
        if prompt_encoding is None:
            prompt_encoding = self.encode_prompt(
                prompt_audio_path,
                enhance_prompt=enhance_prompt,
                apply_vad=apply_vad,
                normalize_loudness=normalize_loudness,
                on_stage=on_stage,
            )
        once = partial(
            self._stream_once,
            prompt_audio_path=prompt_audio_path,
            speaking_rate=speaking_rate,
            enhance_prompt=enhance_prompt,
            apply_vad=apply_vad,
            normalize_loudness=normalize_loudness,
            prompt_encoding=prompt_encoding,
            on_stage=on_stage,
            return_progress=return_progress,
            max_stream_lead_sec=max_stream_lead_sec,
        )
        budget = self.config.start_retry_frames
        attempts = self.config.start_retry_attempts if isinstance(text, str) else 1
        if budget <= 0 or attempts <= 1:
            yield from once(text=text)
            return

        # derived from the caller's seed, so a run stays reproducible across retries
        base_seed = torch.initial_seed()
        keep = max(0, self.config.start_lead_in_frames)
        for attempt in range(attempts):
            if attempt:
                torch.manual_seed((base_seed + 1_000_003 * attempt) % (2**63 - 1))
            opening, started, stalled = [], False, False
            stream = once(text=text)
            for item in stream:
                if started:
                    yield item
                elif frame_has_speech(item[0]):
                    started = True
                    for held in opening[len(opening) - keep :] if keep else []:
                        yield held
                    opening.clear()
                    yield item
                elif len(opening) < budget:
                    opening.append(item)
                else:
                    stalled = True
                    stream.close()
                    break
            if not stalled:
                return
            self.logger.debug(
                "silent opening, redrawing (%d/%d)", attempt + 1, attempts
            )
        # every draw stalled: repeat the first one and keep it, whatever it does. It is
        # the draw the generator would have produced with no retry at all, so the worst
        # case is today's behaviour rather than a stream cut off at the budget.
        torch.manual_seed(base_seed)
        yield from once(text=text)

    @torch.inference_mode()
    def _stream_once(
        self,
        prompt_audio_path: Path,
        text: str | Generator[str, None, None],
        speaking_rate: Iterator[float] = None,
        enhance_prompt: bool = None,
        apply_vad: bool = None,
        normalize_loudness: bool = None,
        return_progress: bool = False,
        max_stream_lead_sec: float = None,
        prompt_encoding: tuple[torch.Tensor, torch.Tensor] = None,
        on_stage: Callable[[str], None] = None,
    ) -> Generator[
        tuple[np.ndarray, float] | tuple[np.ndarray, float, Dict], None, None
    ]:
        """One draw, streamed frame by frame. `generate_stream` is the entry point."""
        # Override parameters
        enhance_prompt = (
            self.config.enhance_prompt if enhance_prompt is None else enhance_prompt
        )
        apply_vad = self.config.apply_vad if apply_vad is None else apply_vad
        normalize_loudness = (
            self.config.normalize_prompt_loudness
            if normalize_loudness is None
            else normalize_loudness
        )

        (
            mimi_codes,
            spk_embedding,
            prompt_phone_tokens,
            prompt_phone_tokens_to_embed,
        ) = prepare_prompt(
            prompt_audio_path=prompt_audio_path,
            prompt_encoding=prompt_encoding,
            on_stage=on_stage,
            **self.ctx.prompt_kwargs(
                enhance_prompt=enhance_prompt,
                apply_vad=apply_vad,
                normalize_loudness=normalize_loudness,
            ),
        )

        phone_emb_indices = prompt_phone_tokens_to_embed

        self.model.reset_caches()
        if self.config.phone_kv_cache:
            # the prompt rows are known before any text: encode them now, so the first
            # chunk of text only costs its own rows
            self.ctx.extract_phone_embeddings(prompt_phone_tokens)
        spk_embedding = self.model.spk_emb_proj(spk_embedding)

        if isinstance(text, str):
            if on_stage:
                on_stage(STAGE_PHONEMES)
            phone_tokens, phone_emb, phone_seq_len = (
                self.frontend.prepare_non_streaming_text(
                    text,
                    prompt_phone_tokens,
                    prompt_phone_tokens.shape[1],
                    self.ctx,
                )
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
        if on_stage:
            on_stage(STAGE_GENERATION)

        start_time = time.perf_counter()
        generated_audio_frames = 0
        stream_start = None
        audio_frame_sec = self.config.mimi_frame_ms / 1000.0
        restarted = set()  # stall positions already restarted from, see `stall_kicks`

        for idx in range(max_seq_len):
            if not empty_text_stream:
                is_enough_text = False
                while not is_enough_text and not empty_text_stream:
                    result = self.frontend.prepare_streaming_text(
                        text_gen=text,
                        phone_tokens=phone_tokens,
                        empty_text_stream=empty_text_stream,
                        prompt_len=prompt_phone_tokens.shape[1],
                        pointer=frame_state.phone_emb_max_idx,
                        ctx=self.ctx,
                    )
                    if result is not None:
                        (
                            phone_tokens,
                            phone_emb,
                            phone_seq_len,
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
                    mimi_decoder=self._mimi_decoder,
                    frame=frame,
                    sem_code=sem_code,
                    mimi_vocab_size=self.config.mimi_vocab_size,
                )
                gen_time = time.perf_counter() - start_time
                generated_audio_frames += 1

                if max_stream_lead_sec is not None:
                    # Audio produced against the wall clock since the first frame.
                    # Only a lead above the cap is slept off, so a lead lost to a
                    # slow frame is won back at full speed.
                    now = time.perf_counter()
                    if stream_start is None:
                        stream_start = now
                    lead = generated_audio_frames * audio_frame_sec - (
                        now - stream_start
                    )
                    if lead > max_stream_lead_sec:
                        time.sleep(lead - max_stream_lead_sec)
                        gen_time += time.perf_counter() - now

                start_time = time.perf_counter()
            else:
                sem_code = frame[:, :1]

            eos_reached = frame_state.eos_idx <= idx
            if not eos_reached:
                mimi_codes, frame_state = self.frontend.update_indices_and_tokens(
                    pred_shift=pred_shift,
                    frame=frame,
                    idx=idx,
                    phone_seq_len=phone_seq_len,
                    max_phone_idx=phone_emb.shape[1] - 1,
                    frame_state=frame_state,
                    ctx=self.ctx,
                )
                phone_emb_indices = frame_state.phone_emb_indices
                if (
                    self.config.stall_kicks
                    and frame_state.kicks >= self.config.stall_kicks
                    and frame_state.stall_anchor not in restarted
                    and frame_state.eos_idx > idx
                ):
                    restarted.add(frame_state.stall_anchor)
                    self.logger.info(
                        "stalled at %.1f s, restarting from phoneme row %d",
                        generated_audio_frames * audio_frame_sec,
                        frame_state.stall_anchor,
                    )
                    frame_state = restart_from_stall(frame_state, self.ctx)
                    phone_emb_indices = frame_state.phone_emb_indices
                    self.model.sink_attention.restart()

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
