import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch._inductor.config
from tqdm.auto import tqdm

from voxtream.generator import SpeechGenerator, SpeechGeneratorConfig
from voxtream.utils.generator import existing_file, set_seed, text_generator

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.fx_graph_cache = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--compile", action="store_true", help="Compile graph")
    parser.add_argument(
        "-cfg",
        "--config",
        type=existing_file,
        help="Path to the config file",
        default="configs/generator.json",
    )
    parser.add_argument(
        "-m",
        "--meta",
        type=existing_file,
        help="Path to the metadata file",
        default="assets/benchmark/meta.csv",
    )
    args = parser.parse_args()

    set_seed()
    with open(args.config) as f:
        config = SpeechGeneratorConfig(**json.load(f))
    speech_generator = SpeechGenerator(config, compile=args.compile)

    meta = pd.read_csv(args.meta)

    audio_frames, first_packet_latency, gen_times = [], [], []
    for idx, row in tqdm(meta.iterrows(), total=len(meta)):
        speech_stream = speech_generator.generate_stream(
            prompt_text=row.prompt_text,
            prompt_audio_path=Path(row.prompt_audio),
            text=text_generator(row.text),
        )

        if idx == 0:
            # warmup
            for _, _ in speech_stream:
                pass
            continue

        for i, (audio_frame, gen_time) in enumerate(speech_stream):
            audio_frames.append(audio_frame)
            if i == 0:
                first_packet_latency.append(gen_time)
            else:
                gen_times.append(gen_time)

    rtf = (np.mean(gen_times) * 1000) / config.mimi_frame_ms
    print(f"First packet latency: {round(np.mean(first_packet_latency) * 1000)} ms")
    print(f"RTF: {round(rtf, 2)}")


if __name__ == "__main__":
    main()
