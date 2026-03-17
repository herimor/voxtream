import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from voxtream.config import SpeechGeneratorConfig
from voxtream.generator import SpeechGenerator
from voxtream.utils.generator import (
    existing_file,
    interpolate_speaking_rate_params,
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
    args = parser.parse_args()

    set_seed()
    with open(args.config) as f:
        config = SpeechGeneratorConfig(**json.load(f))

    with open(args.spk_rate_config) as f:
        speaking_rate_config = json.load(f)

    speech_generator = SpeechGenerator(config)

    if args.text is None:
        speech_generator.logger.error("No text provided.")
        exit(0)

    if args.spk_rate is not None:
        duration_state, weight, cfg_gamma = interpolate_speaking_rate_params(
            speaking_rate_config, args.spk_rate, logger=speech_generator.logger
        )
    else:
        duration_state, weight, cfg_gamma = None, None, None

    speech_stream = speech_generator.generate_stream(
        prompt_audio_path=Path(args.prompt_audio),
        text=text_generator(args.text) if args.full_stream else args.text,
        target_spk_rate_cnt=duration_state,
        spk_rate_weight=weight,
        cfg_gamma=cfg_gamma,
    )

    audio_frames = [audio_frame for audio_frame, _ in speech_stream]
    sf.write(args.output, np.concatenate(audio_frames), config.mimi_sr)
    speech_generator.logger.info(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
