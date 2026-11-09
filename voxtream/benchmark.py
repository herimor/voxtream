import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from voxtream.config import SpeechGeneratorConfig
from voxtream.generator import SpeechGenerator
from voxtream.utils.generator import existing_file, set_seed, text_generator


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
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default=None,
        help="Also write the generated audio here, one wav per row of the metadata. "
        "Timings are reported either way, but writing the files costs a little of the "
        "measured time, so leave it off for a pure RTF run",
    )
    parser.add_argument(
        "-fs", "--full-stream", action="store_true", help="Enables full-streaming mode"
    )
    args = parser.parse_args()

    set_seed()
    with open(args.config) as f:
        config = SpeechGeneratorConfig(**json.load(f))

    # the backend (`gguf`, `gguf_dir`) and the CPU threads come from the config
    speech_generator = SpeechGenerator(config, compile=args.compile)

    meta = pd.read_csv(args.meta)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    first_packet_latency, gen_times = [], []
    for idx, row in tqdm(meta.iterrows(), total=len(meta)):
        audio_frames = []
        speech_stream = speech_generator.generate_stream(
            prompt_audio_path=Path(row.prompt_audio),
            text=text_generator(row.text) if args.full_stream else row.text,
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

        if out_dir is not None and audio_frames:
            audio = np.concatenate([np.atleast_1d(f) for f in audio_frames])
            sf.write(
                out_dir / f"{idx:04d}_{Path(row.prompt_audio).stem}.wav",
                audio,
                config.mimi_sr,
            )

    rtf = (np.mean(gen_times) * 1000) / config.mimi_frame_ms
    fpl = np.mean(first_packet_latency) * 1000
    device = speech_generator.ctx.device
    backend = "llama.cpp q8_0" if speech_generator.gguf else f"PyTorch {device}"
    threads = config.cpu_threads if device == "cpu" else None
    print(f"Backend: {backend}" + (f", {threads} threads" if threads else ""))
    print(f"First packet latency: {round(fpl)} ms")
    print(f"RTF: {round(rtf, 3)}")
    # one machine-readable line for sweeps over backends and thread counts
    print(
        f"RESULT backend={backend!r} threads={threads} fpl_ms={fpl:.1f} rtf={rtf:.3f} "
        f"frames={len(gen_times) + len(first_packet_latency)}"
    )


if __name__ == "__main__":
    main()
