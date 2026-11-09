import argparse
import json
import time
import uuid
import warnings
from itertools import chain
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import gradio as gr
import numpy as np
import soundfile as sf

from voxtream.config import SpeechGeneratorConfig
from voxtream.generator import (
    STAGE_GENERATION,
    STAGE_MIMI,
    STAGE_PHONEMES,
    STAGE_SPEAKER,
    SpeechGenerator,
)
from voxtream.utils.app import (
    CUSTOM_CSS,
    AppConfig,
    GenerationControl,
    PromptCache,
    SharedGenerationState,
    SpeakingRateState,
    StageReporter,
    VisualizationState,
    build_low_latency_audio_head,
    clear_outputs,
    empty_rate_plot,
    float32_to_int16,
    load_app_config,
    render_audio_stream,
    render_text_progress,
)
from voxtream.utils.generator import (
    existing_file,
    text_generator,
)
from voxtream.utils.generator.text import build_text_progress_metadata

# What the stream box says at each step before the first frame of a generation
STAGE_STATUS = {
    STAGE_MIMI: "Extracting Mimi codes of the prompt…",
    STAGE_SPEAKER: "Extracting the speaker embedding…",
    STAGE_PHONEMES: "Extracting phonemes…",
    STAGE_GENERATION: "Generating speech…",
}
CACHED_PROMPT_STATUS = "Prompt loaded from the cache…"
# the prompt options of the page as it opens, the settings the cache is built with
PROMPT_ENHANCEMENT_DEFAULT = False
VOICE_ACTIVITY_DETECTION_DEFAULT = False


def generation_button_updates(running: bool, paused: bool = False):
    if not running:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    if paused:
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
            gr.update(visible=True, interactive=True),
        )
    return (
        gr.update(visible=True, interactive=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def demo_app(
    config: SpeechGeneratorConfig,
    app_config: AppConfig,
    demo_examples,
    synthesize_fn,
    speaking_rate_state: SpeakingRateState,
    generation_control: GenerationControl,
    shared_generation_state: SharedGenerationState,
) -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        head=build_low_latency_audio_head(app_config),
        title="VoXtream2",
    ) as demo:
        gr.Markdown("# VoXtream2 TTS demo")

        with gr.Row(equal_height=True, elem_id="cols"):
            with gr.Column(scale=1, elem_id="left-col"):
                prompt_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label=(
                        "Prompt audio (3-10 sec of target voice. "
                        f"Max {config.max_prompt_sec} sec)"
                    ),
                )
                with gr.Accordion("Advanced options", open=False):
                    enable_speaking_rate = gr.Checkbox(
                        label="Use speaking rate control", value=True
                    )
                    prompt_enhancement = gr.Checkbox(
                        label="Prompt enhancement", value=PROMPT_ENHANCEMENT_DEFAULT
                    )
                    prompt_enhancement_msg = gr.Markdown(
                        "⚠️ First 3-5 runs may have higher latency due to model "
                        "loading and warmup.",
                        visible=False,
                    )
                    voice_activity_detection = gr.Checkbox(
                        label="Voice activity detection",
                        value=VOICE_ACTIVITY_DETECTION_DEFAULT,
                    )
                    streaming_input = gr.Checkbox(label="Streaming input", value=False)

            with gr.Column(scale=1, elem_id="right-col"):
                target_text = gr.Textbox(
                    lines=4,
                    max_length=config.max_phone_tokens,
                    label=(
                        "Target text (Required, "
                        f"max {config.max_phone_tokens} chars)"
                    ),
                    placeholder="What you want the model to say",
                )
                output_audio = gr.Audio(
                    label="Synthesized audio",
                    interactive=False,
                    streaming=False,
                    autoplay=False,
                    show_download_button=True,
                    show_share_button=False,
                    visible=False,
                )
                stream_audio = gr.HTML(
                    render_audio_stream(app_config), elem_id="audio-stream-container"
                )

        with gr.Row():
            clear_btn = gr.Button("Clear", elem_id="clear", variant="secondary")
            submit_btn = gr.Button(
                "Submit", elem_id="submit", variant="primary", interactive=False
            )
            pause_btn = gr.Button(
                "Pause", elem_id="pause", variant="secondary", visible=False
            )
            resume_btn = gr.Button(
                "Resume", elem_id="resume", variant="primary", visible=False
            )
            stop_btn = gr.Button("Stop", elem_id="stop", variant="stop", visible=False)

        validation_msg = gr.Markdown("", visible=False)
        speaking_rate_control = gr.Slider(
            minimum=app_config.speaking_rate_min,
            maximum=app_config.speaking_rate_max,
            step=app_config.speaking_rate_step,
            value=app_config.speaking_rate_default,
            label="Speaking rate (SPS). Change the speed of speech synthesis in real-time. ",
        )
        rate_plot = gr.HTML(empty_rate_plot(app_config), elem_id="rate-plot-container")
        text_progress = gr.HTML(
            render_text_progress(app_config, None), elem_id="text-progress-container"
        )
        # The generation a page controls. Pause, resume, stop and the speaking rate
        # go through `shared_generation_state` under this id, so they reach the
        # worker running the generation whichever worker handles the click.
        generation_session = gr.State("")

        def validate_inputs(audio, ttext):
            if not audio:
                return gr.update(
                    visible=True, value="⚠️ Please provide a prompt audio."
                ), gr.update(interactive=False)
            if not ttext or not ttext.strip():
                return gr.update(
                    visible=True, value="⚠️ Please provide target text."
                ), gr.update(interactive=False)
            return gr.update(visible=False, value=""), gr.update(interactive=True)

        enable_speaking_rate.change(
            fn=lambda enabled: gr.update(interactive=enabled),
            inputs=enable_speaking_rate,
            outputs=speaking_rate_control,
        )
        prompt_enhancement.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=prompt_enhancement,
            outputs=prompt_enhancement_msg,
        )

        def update_speaking_rate(value, session_id):
            speaking_rate_state.update(value)
            shared_generation_state.update_speaking_rate(session_id, value)

        speaking_rate_control.input(
            fn=update_speaking_rate,
            inputs=[speaking_rate_control, generation_session],
            queue=False,
            show_progress="hidden",
        )
        speaking_rate_control.release(
            fn=update_speaking_rate,
            inputs=[speaking_rate_control, generation_session],
            queue=False,
            show_progress="hidden",
        )

        for inp in [prompt_audio, target_text]:
            inp.change(
                fn=validate_inputs,
                inputs=[prompt_audio, target_text],
                outputs=[validation_msg, submit_btn],
            )

        def prepare_generation(speaking_rate, enable_rate):
            session_id = shared_generation_state.create(speaking_rate)
            generation_control.start()
            speaking_rate_state.start(speaking_rate)
            return (
                gr.update(value=None, visible=False),
                gr.update(interactive=False),
                empty_rate_plot(app_config, show_target=enable_rate),
                render_text_progress(app_config, None),
                render_audio_stream(app_config, session_id=session_id),
                *generation_button_updates(running=True),
                session_id,
            )

        submit_btn.click(
            fn=prepare_generation,
            inputs=[speaking_rate_control, enable_speaking_rate],
            outputs=[
                output_audio,
                enable_speaking_rate,
                rate_plot,
                text_progress,
                stream_audio,
                pause_btn,
                resume_btn,
                stop_btn,
                generation_session,
            ],
            show_progress="hidden",
        ).then(
            fn=synthesize_fn,
            inputs=[
                prompt_audio,
                target_text,
                prompt_enhancement,
                voice_activity_detection,
                streaming_input,
                speaking_rate_control,
                enable_speaking_rate,
                generation_session,
            ],
            outputs=[
                output_audio,
                enable_speaking_rate,
                rate_plot,
                text_progress,
                stream_audio,
                pause_btn,
                resume_btn,
                stop_btn,
                generation_session,
            ],
        )

        def pause_generation(session_id):
            generation_control.pause()
            shared_generation_state.pause(session_id)
            return generation_button_updates(running=True, paused=True)

        def resume_generation(session_id):
            generation_control.resume()
            shared_generation_state.resume(session_id)
            return generation_button_updates(running=True)

        def stop_generation(session_id):
            generation_control.stop()
            speaking_rate_state.stop()
            shared_generation_state.stop(session_id)
            return generation_button_updates(running=False)

        pause_btn.click(
            fn=pause_generation,
            inputs=generation_session,
            outputs=[pause_btn, resume_btn, stop_btn],
            js=(
                "() => { if (window.voxtreamLowLatencyAudio) { "
                "window.voxtreamLowLatencyAudio.pause(); } return []; }"
            ),
            queue=False,
        )
        resume_btn.click(
            fn=resume_generation,
            inputs=generation_session,
            outputs=[pause_btn, resume_btn, stop_btn],
            js=(
                "() => { if (window.voxtreamLowLatencyAudio) { "
                "window.voxtreamLowLatencyAudio.resume(); } return []; }"
            ),
            queue=False,
        )
        stop_btn.click(
            fn=stop_generation,
            inputs=generation_session,
            outputs=[pause_btn, resume_btn, stop_btn],
            js=(
                "() => { if (window.voxtreamLowLatencyAudio) { "
                "window.voxtreamLowLatencyAudio.stop(); } return []; }"
            ),
            queue=False,
        )

        def clear_generation(session_id):
            generation_control.stop()
            speaking_rate_state.stop()
            shared_generation_state.stop(session_id)
            return (
                gr.update(value=None),
                gr.update(value=""),
                gr.update(value=None, visible=False),
                gr.update(visible=False, value=""),
                gr.update(interactive=False),
                gr.update(interactive=True),
                empty_rate_plot(app_config),
                render_text_progress(app_config, None),
                render_audio_stream(app_config, session_id=uuid.uuid4().hex),
                *generation_button_updates(running=False),
                "",
            )

        clear_btn.click(
            fn=clear_generation,
            inputs=generation_session,
            outputs=[
                prompt_audio,
                target_text,
                output_audio,
                validation_msg,
                submit_btn,
                enable_speaking_rate,
                rate_plot,
                text_progress,
                stream_audio,
                pause_btn,
                resume_btn,
                stop_btn,
                generation_session,
            ],
        )

        gr.Markdown("### Examples")
        ex = gr.Examples(
            examples=demo_examples,
            inputs=[
                prompt_audio,
                target_text,
                prompt_enhancement,
                voice_activity_detection,
                streaming_input,
                speaking_rate_control,
                enable_speaking_rate,
            ],
            outputs=[
                output_audio,
                enable_speaking_rate,
                rate_plot,
                text_progress,
                stream_audio,
                pause_btn,
                resume_btn,
                stop_btn,
                generation_session,
            ],
            fn=synthesize_fn,
            cache_examples=False,
        )

        ex.dataset.click(
            fn=lambda: (*clear_outputs(app_config), ""),
            inputs=[],
            outputs=[
                output_audio,
                rate_plot,
                text_progress,
                stream_audio,
                generation_session,
            ],
            queue=False,
        ).then(
            fn=validate_inputs,
            inputs=[prompt_audio, target_text],
            outputs=[validation_msg, submit_btn],
            queue=False,
        )

    return demo


def build_synthesize_fn(
    config: SpeechGeneratorConfig,
    app_config: AppConfig,
    speech_generator: SpeechGenerator,
    speaking_rate_state: SpeakingRateState,
    generation_control: GenerationControl,
    shared_generation_state: SharedGenerationState,
    prompt_cache: PromptCache | None = None,
):
    chunk_size = int(config.mimi_sr * app_config.min_chunk_sec)

    def synthesize_fn(
        prompt_audio_path,
        target_text,
        prompt_enhancement,
        voice_activity_detection,
        streaming_input,
        speaking_rate_control,
        enable_speaking_rate=True,
        generation_session_id="",
    ):
        control_session_id = generation_session_id or shared_generation_state.create(
            speaking_rate_control
        )
        stream_session_id = control_session_id or uuid.uuid4().hex
        stream_seq = 0

        if not prompt_audio_path or not target_text:
            speaking_rate_state.stop()
            generation_control.finish()
            shared_generation_state.finish(control_session_id)
            yield (
                gr.update(value=None, visible=False),
                gr.update(interactive=True),
                empty_rate_plot(app_config, show_target=enable_speaking_rate),
                render_text_progress(app_config, None),
                render_audio_stream(app_config, session_id=stream_session_id),
                *generation_button_updates(running=False),
                control_session_id,
            )
            return

        if shared_generation_state.is_stopped(control_session_id):
            speaking_rate_state.stop()
            generation_control.finish()
            shared_generation_state.finish(control_session_id)
            yield (
                gr.update(value=None, visible=False),
                gr.update(interactive=True),
                empty_rate_plot(app_config, show_target=enable_speaking_rate),
                render_text_progress(app_config, None),
                render_audio_stream(
                    app_config,
                    session_id=stream_session_id,
                    active=False,
                    final=True,
                ),
                *generation_button_updates(running=False),
                control_session_id,
            )
            return

        speaking_rate_state.ensure_started(speaking_rate_control)
        speaking_rate_gen = (
            shared_generation_state.speaking_rate_values(
                control_session_id, speaking_rate_control
            )
            if enable_speaking_rate
            else None
        )
        text_metadata = build_text_progress_metadata(
            target_text,
            config=config,
            phonemizer=speech_generator.ctx.phonemizer,
            max_phone_tokens=config.max_phone_tokens,
        )
        rate_window_sec = (
            config.spk_rate_window_sec
            if config.spk_rate_window_sec and config.spk_rate_window_sec > 0
            else app_config.plot_window_sec
        )
        frame_sec = config.mimi_frame_ms / 1000.0
        text_progress_delay_sec = (
            app_config.audio_stream_start_delay_sec
            + config.audio_delay_frames * frame_sec
        )
        visualization = VisualizationState(
            text_metadata=text_metadata,
            app_config=app_config,
            rate_window_sec=rate_window_sec,
            frame_sec=frame_sec,
            text_progress_delay_sec=text_progress_delay_sec,
            show_target=enable_speaking_rate,
        )

        def stage_update(status: str):
            """The page while a step before the first frame runs."""
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                render_audio_stream(
                    app_config, session_id=stream_session_id, status=status
                ),
                *generation_button_updates(
                    running=True,
                    paused=shared_generation_state.is_paused(control_session_id),
                ),
                control_session_id,
            )

        def with_stage_updates(stages):
            """Yields a page update per stage of `stages`, returns its result."""
            while True:
                try:
                    stage = next(stages)
                except StopIteration as done:
                    return done.value
                yield stage_update(STAGE_STATUS[stage])

        # The steps before the first frame, each shown as it starts: the prompt's
        # Mimi codes and speaker embedding (or the cache entry holding them), the
        # phonemes of the text, then the generation. On a CPU the prompt alone takes
        # longer than the first frame, which a listener would otherwise take for a
        # slow start.
        encode_options = dict(
            enhance_prompt=prompt_enhancement,
            apply_vad=voice_activity_detection,
            normalize_loudness=config.normalize_prompt_loudness,
        )
        prompt_encoding = (
            prompt_cache.load(prompt_audio_path, **encode_options)
            if prompt_cache
            else None
        )
        if prompt_encoding is None:
            reporter = StageReporter()
            prompt_encoding = yield from with_stage_updates(
                reporter.run(
                    lambda: speech_generator.encode_prompt(
                        Path(prompt_audio_path), on_stage=reporter, **encode_options
                    )
                )
            )
            if prompt_cache:
                try:
                    prompt_cache.save(
                        prompt_audio_path, *prompt_encoding, **encode_options
                    )
                except OSError as exc:
                    speech_generator.logger.warning(
                        f"could not write the prompt cache entry: {exc}"
                    )
        else:
            yield stage_update(CACHED_PROMPT_STATUS)

        reporter = StageReporter()
        stream = speech_generator.generate_stream(
            prompt_audio_path=Path(prompt_audio_path),
            text=text_generator(target_text) if streaming_input else target_text,
            speaking_rate=speaking_rate_gen,
            enhance_prompt=prompt_enhancement,
            apply_vad=voice_activity_detection,
            return_progress=True,
            max_stream_lead_sec=app_config.max_stream_lead_sec,
            prompt_encoding=prompt_encoding,
            on_stage=reporter,
        )
        stream_iter = iter(stream)
        first = yield from with_stage_updates(
            reporter.run(lambda: next(stream_iter, None))
        )
        stream_items = chain([first], stream_iter) if first is not None else iter(())

        buffer = []
        buffer_len = 0
        total_buffer = []
        stopped = False
        sent_text = None
        # The page keeps only the latest update and drops the ones it did not get
        # to render, so it is updated no more often than `min_stream_update_sec`:
        # frames that come sooner are merged into the next chunk, which hands the
        # player the same audio in fewer updates
        min_update_sec = app_config.min_stream_update_sec
        last_update = None

        def wait_for_update_slot():
            if last_update is not None:
                wait = last_update + min_update_sec - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            return time.perf_counter()

        while True:
            if not shared_generation_state.wait_if_paused(control_session_id):
                stopped = True
                break
            try:
                frame, _, progress = next(stream_items)
            except StopIteration:
                break
            if shared_generation_state.is_stopped(control_session_id):
                stopped = True
                break

            buffer.append(frame)
            total_buffer.append(frame)
            buffer_len += frame.shape[0]
            plot_update, _ = visualization.update(progress)

            if buffer_len >= chunk_size:
                if shared_generation_state.is_stopped(control_session_id):
                    stopped = True
                    break
                now = time.perf_counter()
                if last_update is not None and now - last_update < min_update_sec:
                    continue  # held for the next frame
                last_update = now
                audio = np.concatenate(buffer)
                stream_seq += 1
                # The text is re-rendered on whichever frame crosses a
                # `visual_update_sec` mark and a chunk spans several frames, so the
                # latest render is sent whenever it changed rather than the update
                # of the chunk's last frame (which would be a no-op most of the time)
                text_update = gr.update()
                if visualization.latest_text != sent_text:
                    sent_text = text_update = visualization.latest_text
                yield (
                    gr.update(),
                    gr.update(),
                    plot_update,
                    text_update,
                    render_audio_stream(
                        app_config,
                        session_id=stream_session_id,
                        seq=stream_seq,
                        sample_rate=config.mimi_sr,
                        audio=float32_to_int16(audio),
                        active=True,
                    ),
                    *generation_button_updates(
                        running=True,
                        paused=shared_generation_state.is_paused(control_session_id),
                    ),
                    control_session_id,
                )

                buffer = []
                buffer_len = 0

        stopped = stopped or shared_generation_state.is_stopped(control_session_id)
        if stopped and hasattr(stream, "close"):
            stream.close()
        final_text = visualization.final_text()

        if buffer_len > 0 and not stopped:
            final = np.concatenate(buffer)
            nfade = min(int(config.mimi_sr * app_config.fade_out_sec), final.shape[0])
            if nfade > 0:
                fade = np.linspace(1.0, 0.0, nfade, dtype=np.float32)
                final[-nfade:] *= fade
            stream_seq += 1
            last_update = wait_for_update_slot()
            yield (
                gr.update(),
                gr.update(),
                visualization.latest_plot,
                visualization.latest_text,
                render_audio_stream(
                    app_config,
                    session_id=stream_session_id,
                    seq=stream_seq,
                    sample_rate=config.mimi_sr,
                    audio=float32_to_int16(final),
                    active=True,
                ),
                *generation_button_updates(
                    running=True,
                    paused=shared_generation_state.is_paused(control_session_id),
                ),
                control_session_id,
            )

        if len(total_buffer) > 0:
            full_audio = np.concatenate(total_buffer)
            nfade = min(
                int(config.mimi_sr * app_config.fade_out_sec), full_audio.shape[0]
            )
            if nfade > 0:
                fade = np.linspace(1.0, 0.0, nfade, dtype=np.float32)
                full_audio[-nfade:] *= fade

            file_path = f"/tmp/voxtream_{uuid.uuid4().hex}.wav"
            sf.write(file_path, float32_to_int16(full_audio), config.mimi_sr)

            speaking_rate_state.stop()
            generation_control.finish()
            shared_generation_state.finish(control_session_id)
            wait_for_update_slot()
            yield (
                gr.update(value=file_path, visible=True),
                gr.update(interactive=True),
                visualization.latest_plot,
                final_text,
                render_audio_stream(
                    app_config,
                    session_id=stream_session_id,
                    seq=stream_seq + 1,
                    sample_rate=config.mimi_sr,
                    active=False,
                    final=True,
                ),
                *generation_button_updates(running=False),
                control_session_id,
            )
        else:
            speaking_rate_state.stop()
            generation_control.finish()
            shared_generation_state.finish(control_session_id)
            wait_for_update_slot()
            yield (
                gr.update(value=None, visible=False),
                gr.update(interactive=True),
                visualization.latest_plot,
                final_text,
                render_audio_stream(
                    app_config,
                    session_id=stream_session_id,
                    seq=stream_seq + 1,
                    sample_rate=config.mimi_sr,
                    active=False,
                    final=True,
                ),
                *generation_button_updates(running=False),
                control_session_id,
            )

    return synthesize_fn


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=existing_file,
        help="Path to the config file",
        default="configs/generator.json",
    )
    parser.add_argument(
        "--app-config",
        type=existing_file,
        help="Path to the app config file",
        default="configs/app.json",
    )
    parser.add_argument(
        "--spk-rate-config",
        type=existing_file,
        help="Path to the speaking rate config file",
        default="configs/speaking_rate.json",
    )
    parser.add_argument(
        "--examples-config",
        type=existing_file,
        help="Path to the examples config file",
        default="assets/examples.json",
    )
    parser.add_argument(
        "--prompt-cache-dir",
        type=str,
        default="assets/prompt_cache",
        help="Directory of encoded prompts (Mimi codes and speaker embedding, see "
        "--build-prompt-cache); a prompt found there starts generating at once, one "
        "that is not is encoded and written there. Empty disables the cache",
    )
    parser.add_argument(
        "--build-prompt-cache",
        action="store_true",
        help="Encode the prompts of the examples into --prompt-cache-dir with the "
        "options the page opens with, then exit",
    )
    return parser.parse_args(argv)


def load_generator(args) -> SpeechGenerator:
    with open(args.config) as f:
        config = SpeechGeneratorConfig(**json.load(f))
    with open(args.spk_rate_config) as f:
        spk_rate_config = json.load(f)
    # the backend (`gguf`, `gguf_dir`) and the CPU threads come from the config
    return SpeechGenerator(config, spk_rate_config)


def load_examples(args) -> list:
    with open(args.examples_config) as f:
        return json.load(f).get("examples", [])


def build_prompt_cache(args) -> None:
    """Encode the prompt of every example into the cache."""
    speech_generator = load_generator(args)
    cache = PromptCache(args.prompt_cache_dir, speech_generator.config)
    options = dict(
        enhance_prompt=PROMPT_ENHANCEMENT_DEFAULT,
        apply_vad=VOICE_ACTIVITY_DETECTION_DEFAULT,
        normalize_loudness=speech_generator.config.normalize_prompt_loudness,
    )
    for audio_path in dict.fromkeys(example[0] for example in load_examples(args)):
        if sf.info(audio_path).subtype != "PCM_16":
            print(
                f"{audio_path}: not 16-bit PCM, so the copy gradio makes of it will "
                "not match this entry"
            )
        path = cache.save(
            audio_path,
            *speech_generator.encode_prompt(Path(audio_path), **options),
            **options,
        )
        print(f"{audio_path} -> {path}")
    print(f"{len(cache.entries())} entries in {cache.directory}")


def build_demo(args) -> gr.Blocks:
    app_config = load_app_config(args.app_config)
    demo_examples = load_examples(args)
    speech_generator = load_generator(args)
    config = speech_generator.config
    backend = (
        f"llama.cpp q8_0 from {speech_generator.gguf['dir']}"
        if speech_generator.gguf
        else f"PyTorch on {speech_generator.ctx.device}"
    )
    print(f"Backend: {backend}")
    prompt_cache = (
        PromptCache(args.prompt_cache_dir, config) if args.prompt_cache_dir else None
    )
    if prompt_cache:
        print(
            f"Prompt cache: {len(prompt_cache.entries())} entries in {prompt_cache.directory}"
        )

    speaking_rate_state = SpeakingRateState(app_config.speaking_rate_default)
    generation_control = GenerationControl()
    shared_generation_state = SharedGenerationState()

    synthesize_fn = build_synthesize_fn(
        config,
        app_config,
        speech_generator,
        speaking_rate_state,
        generation_control,
        shared_generation_state,
        prompt_cache=prompt_cache,
    )
    return demo_app(
        config,
        app_config,
        demo_examples,
        synthesize_fn,
        speaking_rate_state,
        generation_control,
        shared_generation_state,
    )


def main():
    args = parse_args()
    if args.build_prompt_cache:
        build_prompt_cache(args)
        return
    demo = build_demo(args)
    demo.queue(default_concurrency_limit=1).launch()


if __name__ == "__main__":
    main()
