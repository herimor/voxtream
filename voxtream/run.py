import argparse
from itertools import repeat
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf

from voxtream.config import (
    load_generator_config,
    load_speaking_rate_config,
    resolve_data_path,
)
from voxtream.generator import SpeechGenerator
from voxtream.utils.generator import (
    set_seed,
    text_generator,
)


def _audio_frame(result: tuple[Any, ...]) -> np.ndarray[Any, Any]:
    return cast(np.ndarray[Any, Any], result[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pa",
        "--prompt-audio",
        type=Path,
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
        type=Path,
        help="Path to the config file",
        default="configs/generator.json",
    )
    parser.add_argument(
        "--spk-rate-config",
        type=Path,
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
    config = load_generator_config(args.config)
    spk_rate_config = load_speaking_rate_config(args.spk_rate_config)

    speech_generator = SpeechGenerator(config, spk_rate_config)

    if args.text is None:
        speech_generator.logger.error("No text provided.")
        raise SystemExit(2)

    speaking_rate = repeat(args.spk_rate) if args.spk_rate is not None else None

    speech_stream = speech_generator.generate_stream(
        prompt_audio_path=resolve_data_path(
            args.prompt_audio, "assets/audio/english_male.wav"
        ),
        text=text_generator(args.text) if args.full_stream else args.text,
        speaking_rate=speaking_rate,
        enhance_prompt=args.prompt_enhancement,
    )

    with sf.SoundFile(args.output, "w", samplerate=config.mimi_sr, channels=1) as f:
        for result in speech_stream:
            f.write(_audio_frame(result))
    speech_generator.logger.info(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
