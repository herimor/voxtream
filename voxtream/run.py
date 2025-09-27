import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from voxtream.generator import SpeechGenerator, SpeechGeneratorConfig
from voxtream.utils.generator import existing_file, set_seed, text_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pa",
        "--prompt-audio",
        type=existing_file,
        help="Path to the prompt audio file (3-5 sec of target voice. Max 10 sec).",
        default="assets/audio/male.wav",
    )
    parser.add_argument(
        "-pt",
        "--prompt-text",
        type=str,
        help="Text transcription to the prompt audio (Max 250 characters).",
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        help="Text to be synthesized (Max 1000 characters).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output audio file",
        default="output.wav",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=existing_file,
        help="Path to the config file",
        default="configs/generator.json",
    )
    parser.add_argument(
        "-fs", "--full-stream", action="store_true", help="Enables full-streaming mode"
    )
    args = parser.parse_args()

    set_seed()
    with open(args.config) as f:
        config = SpeechGeneratorConfig(**json.load(f))

    speech_generator = SpeechGenerator(config)
    speech_stream = speech_generator.generate_stream(
        prompt_text=args.prompt_text,
        prompt_audio_path=Path(args.prompt_audio),
        text=text_generator(args.text) if args.full_stream else args.text,
    )

    audio_frames = [audio_frame for audio_frame, _ in speech_stream]
    sf.write(args.output, np.concatenate(audio_frames), config.mimi_sr)
    speech_generator.logger.info(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
