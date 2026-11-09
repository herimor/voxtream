"""
Client of `voxtream-server`: streams a text word by word and plays the audio as it
arrives, or writes it to `--output`.

    python voxtream/client.py --prompt-audio assets/audio/english_male.wav
"""

import argparse
import asyncio
import json

import numpy as np
import websockets


async def main(args: argparse.Namespace) -> None:
    async with websockets.connect(args.uri, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "event": "init",
                    # a path on the server, or use prompt_audio_b64
                    "prompt_audio_path": args.prompt_audio,
                    "full_stream": True,
                }
            )
        )

        # Stream the text word by word
        for word in args.text.split():
            await ws.send(json.dumps({"event": "text", "chunk": word + " "}))
        await ws.send(json.dumps({"event": "eot"}))

        # Receive config
        cfg = json.loads(await ws.recv())
        if cfg["type"] == "error":
            print("Error from server:", cfg["message"])
            return

        assert cfg["type"] == "config"
        sr = int(cfg["sample_rate"])

        frames = []
        stream = None
        if args.output is None:
            import sounddevice as sd

            stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
            stream.start()
        try:
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    audio = np.frombuffer(msg, dtype=np.float32)
                    if stream is not None:  # play audio as it arrives
                        stream.write(audio)
                    else:
                        frames.append(audio)
                else:
                    data = json.loads(msg)
                    if data.get("type") == "error":
                        print("Error from server:", data["message"])
                    if data.get("type") == "eos":
                        break
        finally:
            if stream is not None:
                stream.stop()
                stream.close()

        if args.output is not None:
            import soundfile as sf

            audio = np.concatenate(frames) if frames else np.zeros(0, np.float32)
            sf.write(args.output, audio, sr)
            print(f"{args.output}: {len(audio) / sr:.2f} s")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a text to voxtream-server, play or save the audio"
    )
    parser.add_argument("--uri", default="ws://localhost:7860/voxtream")
    parser.add_argument(
        "--prompt-audio",
        default="assets/audio/english_male.wav",
        help="Prompt audio, a path on the server",
    )
    parser.add_argument("--text", default="Streaming synthesis starts right away")
    parser.add_argument(
        "--output", default=None, help="Write the audio to this file instead of playing"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
