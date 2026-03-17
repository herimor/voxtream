import argparse
import json
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from voxtream.config import SpeechGeneratorConfig
from voxtream.generator import SpeechGenerator
from voxtream.utils.generator import (
    existing_file,
    interpolate_speaking_rate_params,
    text_generator,
)

MIN_CHUNK_SEC = 0.01
FADE_OUT_SEC = 0.10
CUSTOM_CSS = """
/* overall width */
.gradio-container {max-width: 1100px !important}
/* stack labels tighter and even heights */
#cols .wrap > .form {gap: 10px}
#left-col, #right-col {gap: 14px}
/* make submit centered + bigger */
#submit {width: 260px; margin: 10px auto 0 auto;}
/* make clear align left and look secondary */
#clear {width: 120px;}
/* give audio a little breathing room */
audio {outline: none;}
"""


def float32_to_int16(audio_float32: np.ndarray) -> np.ndarray:
    """
    Convert float32 audio samples (-1.0 to 1.0) to int16 PCM samples.

    Parameters:
        audio_float32 (np.ndarray): Input float32 audio samples.

    Returns:
        np.ndarray: Output int16 audio samples.
    """
    if audio_float32.dtype != np.float32:
        raise ValueError("Input must be a float32 numpy array")

    # Clip to avoid overflow after scaling
    audio_clipped = np.clip(audio_float32, -1.0, 1.0)

    # Scale and convert
    audio_int16 = (audio_clipped * 32767).astype(np.int16)

    return audio_int16


def _clear_outputs():
    # clears the player + hides file (download btn mirrors file via .change)
    return gr.update(value=None), gr.update(value=None, visible=False)


def demo_app(config: SpeechGeneratorConfig, demo_examples, synthesize_fn):
    with gr.Blocks(css=CUSTOM_CSS, title="VoXtream2") as demo:
        gr.Markdown("# VoXtream2 TTS demo")
        gr.Markdown(
            "⚠️ First 3-5 runs may have higher latency due to model loading and warmup."
        )

        with gr.Row(equal_height=True, elem_id="cols"):
            with gr.Column(scale=1, elem_id="left-col"):
                prompt_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label=f"Prompt audio (3-10 sec of target voice. Max {config.max_prompt_sec} sec)",
                )
                with gr.Accordion("Advanced options", open=False):
                    prompt_enhancement = gr.Checkbox(
                        label="Prompt enhancement", value=True
                    )
                    voice_activity_detection = gr.Checkbox(
                        label="Voice activity detection", value=True
                    )
                    streaming_input = gr.Checkbox(label="Streaming input", value=False)

            with gr.Column(scale=1, elem_id="right-col"):
                target_text = gr.Textbox(
                    lines=3,
                    max_length=config.max_phone_tokens,
                    label=f"Target text (Required, max {config.max_phone_tokens} chars)",
                    placeholder="What you want the model to say",
                )
                speaking_rate_control = gr.Slider(
                    minimum=1,
                    maximum=7,
                    step=0.1,
                    value=4,
                    label="Speaking rate (syllables per second)",
                )
                enable_speaking_rate = gr.Checkbox(
                    label="Use speaking rate control", value=True
                )
                enable_speaking_rate.change(
                    fn=lambda enabled: gr.update(interactive=enabled),
                    inputs=enable_speaking_rate,
                    outputs=speaking_rate_control,
                )
                output_audio = gr.Audio(
                    label="Synthesized audio",
                    interactive=False,
                    streaming=True,
                    autoplay=True,
                    show_download_button=False,
                    show_share_button=False,
                    visible=False,
                )

                # appears only when file is ready
                download_btn = gr.DownloadButton(
                    "Download audio",
                    visible=False,
                )

        with gr.Row():
            clear_btn = gr.Button("Clear", elem_id="clear", variant="secondary")
            submit_btn = gr.Button(
                "Submit", elem_id="submit", variant="primary", interactive=False
            )

        # Message box for validation errors
        validation_msg = gr.Markdown("", visible=False)

        # --- Validation logic ---
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

        # Live validation whenever inputs change
        for inp in [prompt_audio, target_text]:
            inp.change(
                fn=validate_inputs,
                inputs=[prompt_audio, target_text],
                outputs=[validation_msg, submit_btn],
            )

        # clear outputs before streaming
        submit_btn.click(
            fn=lambda a, t: (
                gr.update(value=None, visible=True),
                gr.update(value=None, visible=False),
            ),
            inputs=[prompt_audio, target_text],
            outputs=[output_audio, download_btn],
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
            outputs=[output_audio, download_btn],
        )

        clear_btn.click(
            fn=lambda: (
                gr.update(value=None),
                gr.update(value=""),
                gr.update(value=None, visible=False),  # output_audio
                gr.update(value=None, visible=False),  # download_btn
                gr.update(visible=False, value=""),  # validation_msg
                gr.update(interactive=False),  # submit_btn
            ),
            inputs=[],
            outputs=[
                prompt_audio,
                target_text,
                output_audio,
                download_btn,
                validation_msg,
                submit_btn,
            ],
        )

        # --- Add Examples ---
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
            outputs=[output_audio, download_btn],
            fn=synthesize_fn,
            cache_examples=False,
        )

        ex.dataset.click(
            fn=_clear_outputs,
            inputs=[],
            outputs=[output_audio, download_btn],
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
        speaking_rate_config = json.load(f)

    with open(args.examples_config) as f:
        examples_config = json.load(f)
    demo_examples = examples_config.get("examples", [])

    speech_generator = SpeechGenerator(config)
    CHUNK_SIZE = int(config.mimi_sr * MIN_CHUNK_SEC)

    def synthesize_fn(
        prompt_audio_path,
        target_text,
        prompt_enhancement,
        voice_activity_detection,
        streaming_input,
        speaking_rate_control,
        enable_speaking_rate=True,
    ):
        if not prompt_audio_path or not target_text:
            return None, gr.update(value=None, visible=False)

        if enable_speaking_rate:
            duration_state, weight, cfg_gamma = interpolate_speaking_rate_params(
                speaking_rate_config, speaking_rate_control
            )
        else:
            duration_state, weight, cfg_gamma = None, None, None

        stream = speech_generator.generate_stream(
            prompt_audio_path=Path(prompt_audio_path),
            text=text_generator(target_text) if streaming_input else target_text,
            target_spk_rate_cnt=duration_state,
            spk_rate_weight=weight,
            cfg_gamma=cfg_gamma,
            enhance_prompt=prompt_enhancement,
            apply_vad=voice_activity_detection,
        )

        buffer = []
        buffer_len = 0
        total_buffer = []

        for frame, _ in stream:
            buffer.append(frame)
            total_buffer.append(frame)
            buffer_len += frame.shape[0]

            if buffer_len >= CHUNK_SIZE:
                audio = np.concatenate(buffer)
                yield (config.mimi_sr, float32_to_int16(audio)), None

                # Reset buffer and length
                buffer = []
                buffer_len = 0

        # Handle any remaining audio in the buffer
        if buffer_len > 0:
            final = np.concatenate(buffer)
            nfade = min(int(config.mimi_sr * FADE_OUT_SEC), final.shape[0])
            if nfade > 0:
                fade = np.linspace(1.0, 0.0, nfade, dtype=np.float32)
                final[-nfade:] *= fade
            yield (config.mimi_sr, float32_to_int16(final)), None

        # Save the full audio to a file for download
        if len(total_buffer) > 0:
            full_audio = np.concatenate(total_buffer)
            nfade = min(int(config.mimi_sr * FADE_OUT_SEC), full_audio.shape[0])
            if nfade > 0:
                fade = np.linspace(1.0, 0.0, nfade, dtype=np.float32)
                full_audio[-nfade:] *= fade

            file_path = f"/tmp/voxtream_{uuid.uuid4().hex}.wav"
            sf.write(file_path, float32_to_int16(full_audio), config.mimi_sr)

            yield None, gr.update(value=file_path, visible=True)
        else:
            yield None, gr.update(value=None, visible=False)

    demo_app(config, demo_examples, synthesize_fn)


if __name__ == "__main__":
    main()
