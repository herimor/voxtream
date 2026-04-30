import argparse
import json
from itertools import repeat
from pathlib import Path

import numpy as np
import soundfile as sf

from voxtream.config import SpeechGeneratorConfig
from voxtream.generator import SpeechGenerator
from voxtream.utils.generator import (
    existing_file,
    set_seed,
    text_generator,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pa",
        "--prompt-audio",
        type=existing_file,
        help="Path to the prompt audio file (5-10 sec of target voice. Max 20 sec).",
        default="assets/audio/english_male.wav",
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
        "--spk-rate-config",
        type=existing_file,
        help="Path to the speaking rate config file",
        default="configs/speaking_rate.json",
    )
    parser.add_argument(
        "--spk-rate",
        type=float,
        help="Speaking rate (syllables per second)",
        default=None,
    )
    parser.add_argument(
        "-fs", "--full-stream", action="store_true", help="Enables full-streaming mode"
    )
    parser.add_argument(
        "-pe",
        "--prompt-enhancement",
        action="store_true",
        help="Enables prompt enhancement",
    )
    args = parser.parse_args()

    set_seed()
    with open(args.config) as f:
        config = SpeechGeneratorConfig(**json.load(f))

    with open(args.spk_rate_config) as f:
        spk_rate_config = json.load(f)

    speech_generator = SpeechGenerator(config, spk_rate_config)

    if args.text is None:
        speech_generator.logger.error("No text provided.")
        exit(0)

    speaking_rate = repeat(args.spk_rate) if args.spk_rate is not None else None

    speech_stream = speech_generator.generate_stream(
        prompt_audio_path=Path(args.prompt_audio),
        text=text_generator(args.text) if args.full_stream else args.text,
        speaking_rate=speaking_rate,
        enhance_prompt=args.prompt_enhancement,
    )

    audio_frames = [audio_frame for audio_frame, _ in speech_stream]
    sf.write(args.output, np.concatenate(audio_frames), config.mimi_sr)
    speech_generator.logger.info(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
