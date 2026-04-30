import argparse
import json
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from voxtream.config import SpeechGeneratorConfig
from voxtream.generator import SpeechGenerator
from voxtream.utils.app import (
    CUSTOM_CSS,
    AppConfig,
    GenerationControl,
    SpeakingRateState,
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
):
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
                        label="Prompt enhancement", value=False
                    )
                    prompt_enhancement_msg = gr.Markdown(
                        "⚠️ First 3-5 runs may have higher latency due to model "
                        "loading and warmup.",
                        visible=False,
                    )
                    voice_activity_detection = gr.Checkbox(
                        label="Voice activity detection", value=False
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

        def validate_inputs(audio, ttext):
            if not audio:
                return gr.update(
                    visible=True, value="⚠️ Please provide a prompt audio."
                ), gr.update(interactive=False)
            if not ttext.strip():
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
        speaking_rate_control.release(
            fn=lambda value: speaking_rate_state.update(value),
            inputs=speaking_rate_control,
            queue=False,
        )

        for inp in [prompt_audio, target_text]:
            inp.change(
                fn=validate_inputs,
                inputs=[prompt_audio, target_text],
                outputs=[validation_msg, submit_btn],
            )

        def prepare_generation(speaking_rate, enable_rate):
            generation_control.start()
            speaking_rate_state.start(speaking_rate)
            return (
                gr.update(value=None, visible=False),
                gr.update(interactive=False),
                empty_rate_plot(app_config, show_target=enable_rate),
                render_text_progress(app_config, None),
                render_audio_stream(app_config, session_id=uuid.uuid4().hex),
                *generation_button_updates(running=True),
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
            ],
        )

        def pause_generation():
            generation_control.pause()
            return generation_button_updates(running=True, paused=True)

        def resume_generation():
            generation_control.resume()
            return generation_button_updates(running=True)

        def stop_generation():
            generation_control.stop()
            speaking_rate_state.stop()
            return generation_button_updates(running=False)

        pause_btn.click(
            fn=pause_generation,
            inputs=[],
            outputs=[pause_btn, resume_btn, stop_btn],
            js=(
                "() => { if (window.voxtreamLowLatencyAudio) { "
                "window.voxtreamLowLatencyAudio.pause(); } return []; }"
            ),
            queue=False,
        )
        resume_btn.click(
            fn=resume_generation,
            inputs=[],
            outputs=[pause_btn, resume_btn, stop_btn],
            js=(
                "() => { if (window.voxtreamLowLatencyAudio) { "
                "window.voxtreamLowLatencyAudio.resume(); } return []; }"
            ),
            queue=False,
        )
        stop_btn.click(
            fn=stop_generation,
            inputs=[],
            outputs=[pause_btn, resume_btn, stop_btn],
            js=(
                "() => { if (window.voxtreamLowLatencyAudio) { "
                "window.voxtreamLowLatencyAudio.stop(); } return []; }"
            ),
            queue=False,
        )

        clear_btn.click(
            fn=lambda: (
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
            ),
            inputs=[],
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
            ],
            fn=synthesize_fn,
            cache_examples=False,
        )

        ex.dataset.click(
            fn=lambda: clear_outputs(app_config),
            inputs=[],
            outputs=[
                output_audio,
                rate_plot,
                text_progress,
                stream_audio,
            ],
            queue=False,
        ).then(
            fn=validate_inputs,
            inputs=[prompt_audio, target_text],
            outputs=[validation_msg, submit_btn],
            queue=False,
        )

    demo.launch()


def main():
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
    args = parser.parse_args()

    with open(args.config) as f:
        config = SpeechGeneratorConfig(**json.load(f))

    with open(args.spk_rate_config) as f:
        spk_rate_config = json.load(f)

    app_config = load_app_config(args.app_config)

    with open(args.examples_config) as f:
        examples_config = json.load(f)
    demo_examples = examples_config.get("examples", [])

    speech_generator = SpeechGenerator(config, spk_rate_config)
    speaking_rate_state = SpeakingRateState(app_config.speaking_rate_default)
    generation_control = GenerationControl()
    chunk_size = int(config.mimi_sr * app_config.min_chunk_sec)

    def synthesize_fn(
        prompt_audio_path,
        target_text,
        prompt_enhancement,
        voice_activity_detection,
        streaming_input,
        speaking_rate_control,
        enable_speaking_rate=True,
    ):
        stream_session_id = uuid.uuid4().hex
        stream_seq = 0

        if not prompt_audio_path or not target_text:
            speaking_rate_state.stop()
            generation_control.finish()
            yield (
                gr.update(value=None, visible=False),
                gr.update(interactive=True),
                empty_rate_plot(app_config, show_target=enable_speaking_rate),
                render_text_progress(app_config, None),
                render_audio_stream(app_config, session_id=stream_session_id),
                *generation_button_updates(running=False),
            )
            return

        speaking_rate_state.ensure_started(speaking_rate_control)
        speaking_rate_gen = (
            speaking_rate_state.values() if enable_speaking_rate else None
        )
        text_metadata = build_text_progress_metadata(
            target_text,
            config=config,
            phone_to_token=speech_generator.ctx.phone_to_token,
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

        stream = speech_generator.generate_stream(
            prompt_audio_path=Path(prompt_audio_path),
            text=text_generator(target_text) if streaming_input else target_text,
            speaking_rate=speaking_rate_gen,
            enhance_prompt=prompt_enhancement,
            apply_vad=voice_activity_detection,
            return_progress=True,
            min_streaming_rtf=app_config.min_streaming_rtf,
        )

        buffer = []
        buffer_len = 0
        total_buffer = []
        stopped = False

        stream_iter = iter(stream)
        while True:
            if not generation_control.wait_if_paused():
                stopped = True
                break
            try:
                frame, _, progress = next(stream_iter)
            except StopIteration:
                break
            if generation_control.is_stopped():
                stopped = True
                break

            buffer.append(frame)
            total_buffer.append(frame)
            buffer_len += frame.shape[0]
            plot_update, text_update = visualization.update(progress)

            if buffer_len >= chunk_size:
                if generation_control.is_stopped():
                    stopped = True
                    break
                audio = np.concatenate(buffer)
                stream_seq += 1
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
                        running=True, paused=generation_control.is_paused()
                    ),
                )

                buffer = []
                buffer_len = 0

        stopped = stopped or generation_control.is_stopped()
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
                    running=True, paused=generation_control.is_paused()
                ),
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
            )
        else:
            speaking_rate_state.stop()
            generation_control.finish()
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
            )

    demo_app(
        config,
        app_config,
        demo_examples,
        synthesize_fn,
        speaking_rate_state,
        generation_control,
    )


if __name__ == "__main__":
    main()
