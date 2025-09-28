import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np

from voxtream.generator import SpeechGenerator, SpeechGeneratorConfig
from voxtream.utils.generator import existing_file

MIN_CHUNK_SEC = 0.2
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=existing_file,
        help="Path to the config file",
        default="configs/generator.json",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = SpeechGeneratorConfig(**json.load(f))
    speech_generator = SpeechGenerator(config)
    CHUNK_SIZE = int(config.mimi_sr * MIN_CHUNK_SEC)

    def synthesize_fn(prompt_audio_path, prompt_text, target_text):
        if not prompt_audio_path or not target_text:
            return None
        stream = speech_generator.generate_stream(
            prompt_text=prompt_text,
            prompt_audio_path=Path(prompt_audio_path),
            text=target_text,
        )

        buffer = []
        buffer_len = 0

        for frame, _ in stream:
            buffer.append(frame)
            buffer_len += frame.shape[0]

            if buffer_len >= CHUNK_SIZE:
                audio = np.concatenate(buffer)
                yield (config.mimi_sr, float32_to_int16(audio))

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
            yield (config.mimi_sr, float32_to_int16(final))

    with gr.Blocks(css=CUSTOM_CSS, title="VoXtream") as demo:
        gr.Markdown("# VoXtream TTS demo")

        with gr.Row(equal_height=True, elem_id="cols"):
            with gr.Column(scale=1, elem_id="left-col"):
                prompt_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Prompt audio (3-5 sec of target voice. Max 10 sec)",
                )
                prompt_text = gr.Textbox(
                    lines=3,
                    max_length=config.max_prompt_chars,
                    label=f"Prompt transcript (Required, max {config.max_prompt_chars} chars)",
                    placeholder="Text that matches the prompt audio",
                )

            with gr.Column(scale=1, elem_id="right-col"):
                target_text = gr.Textbox(
                    lines=3,
                    max_length=config.max_phone_tokens,
                    label=f"Target text (Required, max {config.max_phone_tokens} chars)",
                    placeholder="What you want the model to say",
                )
                output_audio = gr.Audio(
                    label="Synthesized audio",
                    interactive=False,
                    streaming=True,
                    autoplay=True,
                )

        with gr.Row():
            clear_btn = gr.Button("Clear", elem_id="clear", variant="secondary")
            submit_btn = gr.Button("Submit", elem_id="submit", variant="primary")
        
        # Message box for validation errors
        validation_msg = gr.Markdown("", visible=False)

        # --- Validation logic ---
        def validate_inputs(audio, ptext, ttext):
            if not audio:
                return gr.update(visible=True, value="⚠️ Please provide a prompt audio."), gr.update(interactive=False)
            if not ptext.strip():
                return gr.update(visible=True, value="⚠️ Please provide a prompt transcript."), gr.update(interactive=False)
            if not ttext.strip():
                return gr.update(visible=True, value="⚠️ Please provide target text."), gr.update(interactive=False)
            return gr.update(visible=False, value=""), gr.update(interactive=True)

        # Live validation whenever inputs change
        for inp in [prompt_audio, prompt_text, target_text]:
            inp.change(
                fn=validate_inputs,
                inputs=[prompt_audio, prompt_text, target_text],
                outputs=[validation_msg, submit_btn],
            )

        # --- Wire up actions ---
        submit_btn.click(
            fn=lambda a, p, t: None,  # clears the audio value
            inputs=[prompt_audio, prompt_text, target_text],
            outputs=output_audio,
            show_progress="hidden",
        ).then(
            fn=synthesize_fn,
            inputs=[prompt_audio, prompt_text, target_text],
            outputs=output_audio,
        )

        clear_btn.click(
            fn=lambda: (None, "", "", None, gr.update(visible=False, value=""), gr.update(interactive=False)),
            inputs=[],
            outputs=[prompt_audio, prompt_text, target_text, output_audio, validation_msg, submit_btn],
        )

        # --- Add Examples ---
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                [
                    "assets/app/male.wav",
                    "You could take the easy route or a situation that makes sense which a lot of you do",
                    "Hey, how are you doing? I just uhm want to make sure everything is okay."
                ],
                [
                    "assets/app/female.wav",
                    "I would certainly anticipate some pushback whereas most people know if you followed my work.",
                    "Hello, hello. Let's have a quick chat, uh, in an hour. I need to share something with you."
                ],
            ],
            inputs=[prompt_audio, prompt_text, target_text],
            outputs=output_audio,
            fn=synthesize_fn,
            cache_examples=True,
        )

    demo.launch()


if __name__ == "__main__":
    main()
