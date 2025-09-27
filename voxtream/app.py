import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np

from voxtream.generator import SpeechGenerator, SpeechGeneratorConfig
from voxtream.utils.generator import existing_file

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

    def synthesize_fn(prompt_audio_path, prompt_text, target_text):
        if not prompt_audio_path or not target_text:
            return None
        stream = speech_generator.generate_stream(
            prompt_text=prompt_text,
            prompt_audio_path=Path(prompt_audio_path),
            text=target_text,
        )
        frames = [frame for frame, _ in stream]
        if not frames:
            return None
        waveform = np.concatenate(frames).astype(np.float32)

        # Fade out
        fade_len_sec = 0.1
        fade_out = np.linspace(1.0, 0.0, int(config.mimi_sr * fade_len_sec))
        waveform[-int(config.mimi_sr * fade_len_sec) :] *= fade_out

        return (config.mimi_sr, waveform)

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
                    label=f"Prompt transcript. Max characters: {config.max_prompt_chars} (Required)",
                    placeholder="Text that matches the prompt audio",
                )

            with gr.Column(scale=1, elem_id="right-col"):
                target_text = gr.Textbox(
                    lines=3,
                    max_length=config.max_phone_tokens,
                    label=f"Target text. Max characters: {config.max_phone_tokens}",
                    placeholder="What you want the model to say",
                )
                output_audio = gr.Audio(
                    type="numpy",
                    label="Synthesized audio",
                    interactive=False,
                )

        with gr.Row():
            clear_btn = gr.Button("Clear", elem_id="clear", variant="secondary")
            submit_btn = gr.Button("Submit", elem_id="submit", variant="primary")

        # wire up actions
        submit_btn.click(
            fn=synthesize_fn,
            inputs=[prompt_audio, prompt_text, target_text],
            outputs=output_audio,
        )

        # reset everything
        clear_btn.click(
            fn=lambda: (None, "", "", None),
            inputs=[],
            outputs=[prompt_audio, prompt_text, target_text, output_audio],
        )

    demo.launch()


if __name__ == "__main__":
    main()
