import asyncio
import json
import numpy as np
import sounddevice as sd
import websockets


async def main():
    uri = "ws://localhost:7860/voxtream"
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({
            "event": "init",
            "prompt_text": "The liquor was first created as 'Brandy Milk', produced with milk, brandy and vanilla.",
            "prompt_audio_path": "assets/audio/male.wav",  # or use prompt_audio_b64
            "full_stream": True
        }))

        # Example: stream a sentence word-by-word:
        for word in "Streaming synthesis starts right away".split():
            await ws.send(json.dumps({"event":"text","chunk": word + " "}))
        await ws.send(json.dumps({"event":"eot"}))

        # Receive config
        cfg = json.loads(await ws.recv())
        if cfg["type"] == "error":
            print("Error from server:", cfg["message"])
            return
        
        assert cfg["type"] == "config"
        sr = int(cfg["sample_rate"])

        # Play audio as it arrives
        sd.default.samplerate = sr
        sd.default.channels = 1
        with sd.OutputStream(dtype="float32") as stream:
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    audio = np.frombuffer(msg, dtype=np.float32)
                    stream.write(audio)
                else:
                    data = json.loads(msg)
                    if data.get("type") == "eos":
                        break

asyncio.run(main())
