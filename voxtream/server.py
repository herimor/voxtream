#!/usr/bin/env python3
"""
A tiny WebSocket TTS server for VoXtream.

Protocol (very simple):

Client connects to ws://HOST:PORT/voxtream and first sends a JSON text frame:

{
  "event": "init",
  "prompt_text": "3-5s prompt transcript",
  // Either provide a path on the server or a base64-encoded wav/ogg data string:
  "prompt_audio_path": "/path/to/prompt.wav",        // optional if prompt_audio_b64 given
  "prompt_audio_b64": "data:audio/wav;base64,...",   // optional if prompt_audio_path given
  // Provide text *either* as a single string:
  "text": "Hello world, streaming back now!",
  // Or stream text afterwards using {"event":"text","chunk":"..."} messages.
  // Optional knobs:
  "sample_rate": null,      // if null, use config.mimi_sr
  "full_stream": true       // when true, we'll treat following "text" events as streaming input
}

Then, if full_stream=true and no "text" field was provided in init:
- send any number of JSON text frames: {"event":"text","chunk":"next words..."}
- when done, send {"event":"eot"} (end of text)

Server responses:
- First a JSON text frame with synthesis config:
  {"type":"config","sample_rate":24000,"dtype":"float32","channels":1}

- Then many binary frames, each = one audio frame (float32 PCM, mono, little-endian).
  You can play them as they arrive.

- Final JSON text frame:
  {"type":"eos"}
"""

import sys
import asyncio
import base64
import json
import os
import re
import tempfile
from contextlib import asynccontextmanager

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

sys.path.append(str(Path(__file__).parent.parent))
from voxtream.generator import SpeechGenerator, SpeechGeneratorConfig  # type: ignore
from voxtream.utils.generator import set_seed  # type: ignore


# ---------- Helpers ----------

DATA_URL_RE = re.compile(r"^data:.*?;base64,(.*)$", re.IGNORECASE)

def _b64_to_bytes(s: str) -> bytes:
    m = DATA_URL_RE.match(s)
    payload = m.group(1) if m else s
    return base64.b64decode(payload)

def _ensure_prompt_audio_file(prompt_audio_path: Optional[str], prompt_audio_b64: Optional[str]) -> Path:
    """
    Returns a filesystem Path to the prompt audio.
    If base64 is provided, writes a temp file with the decoded bytes (wav/ogg input supported by voxtream).
    """
    if prompt_audio_path:
        return Path(prompt_audio_path)
    if not prompt_audio_b64:
        raise ValueError("Either 'prompt_audio_path' or 'prompt_audio_b64' must be provided.")
    raw = _b64_to_bytes(prompt_audio_b64)
    # Write as-is; VoXtream reads by extension, so try to infer (default to .wav)
    suffix = ".wav"
    if prompt_audio_b64.lower().startswith("data:audio/ogg"):
        suffix = ".ogg"
    fd, tmp = tempfile.mkstemp(prefix="voxtream_prompt_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return Path(tmp)

def get_generator_from_state(app: FastAPI) -> tuple[SpeechGenerator, SpeechGeneratorConfig]:
    try:
        return app.state.speech_generator, app.state.config
    except AttributeError:
        raise HTTPException(status_code=503, detail="Model is initializing, try again")

@asynccontextmanager
async def lifespan(app: FastAPI):
    set_seed()
    config_path = "configs/generator.json"
    with open(config_path) as f:
        config = SpeechGeneratorConfig(**json.load(f))
    
    app.state.config = config
    app.state.speech_generator = SpeechGenerator(config)

    try:
        yield
    finally:
        # optional: add teardown if SpeechGenerator exposes one
        pass


class _QueueIterator(Iterator[str]):
    """
    Iterator backed by an asyncio.Queue, to be consumed from a non-async thread.
    We capture the main asyncio loop and use run_coroutine_threadsafe(q.get()).
    Putting None into the queue signals StopIteration.
    """
    def __init__(self, q: "asyncio.Queue[Optional[str]]", loop: asyncio.AbstractEventLoop):
        self._q = q
        self._loop = loop

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = asyncio.run_coroutine_threadsafe(self._q.get(), self._loop).result()
        if item is None:
            raise StopIteration
        return item


# ---------- App setup ----------

app = FastAPI(title="VoXtream WebSocket TTS", lifespan=lifespan)


@app.get("/")
def index():
    return HTMLResponse(
        """
        <html>
          <body>
            <h2>VoXtream WebSocket is running.</h2>
            <p>Connect a client to <code>/voxtream</code> and follow the protocol in server.py docstring.</p>
          </body>
        </html>
        """
    )


@app.websocket("/voxtream")
async def synthesis(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()  # <— capture once here

    try:
        speech_generator, config = get_generator_from_state(ws.app)
    except AttributeError:
        # App is not fully started (rare) or shutting down
        await ws.close(code=1013, reason="Service unavailable, initializing")
        return

    try:
        # --- 1) Receive INIT ---
        init_msg = await ws.receive_text()
        try:
            init = json.loads(init_msg)
        except Exception:
            await ws.send_text(json.dumps({"type": "error", "message": "First message must be JSON init."}))
            await ws.close()
            return

        if init.get("event") != "init":
            await ws.send_text(json.dumps({"type": "error", "message": "First message must be {'event':'init',...}."}))
            await ws.close()
            return

        prompt_text: str = init.get("prompt_text", "")
        prompt_audio_path: Optional[str] = init.get("prompt_audio_path")
        prompt_audio_b64: Optional[str] = init.get("prompt_audio_b64")
        text_initial: Optional[str] = init.get("text")
        full_stream: bool = bool(init.get("full_stream", False))

        # Optional: override sample rate (if you down/up-sample on client); we always generate at config.mimi_sr.
        sample_rate = config.mimi_sr

        try:
            prompt_path = _ensure_prompt_audio_file(prompt_audio_path, prompt_audio_b64)
        except Exception as e:
            await ws.send_text(json.dumps({"type": "error", "message": f"Invalid prompt audio: {e}"}))
            await ws.close()
            return

        # Tell client what to expect (mono, float32, PCM)
        await ws.send_text(json.dumps({
            "type": "config",
            "sample_rate": sample_rate,
            "dtype": "float32",
            "channels": 1
        }))

        # --- 2) Prepare text source (string OR generator) ---
        # If full_stream, we build an iterator fed by subsequent websocket messages.
        text_source: str | Iterator[str]
        queue: Optional["asyncio.Queue[Optional[str]]"] = None

        # --- Prepare text source ---
        if full_stream:
            queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
            feeder_done = asyncio.Event()

            async def recv_text_chunks():
                try:
                    if text_initial:
                        await queue.put(text_initial)
                    while True:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        if "text" in msg:
                            try:
                                payload = json.loads(msg["text"])
                            except json.JSONDecodeError:
                                # treat raw text as a chunk
                                await queue.put(msg["text"])
                                continue

                            ev = payload.get("event")
                            if ev == "text":
                                chunk = payload.get("chunk", "")
                                if chunk:
                                    await queue.put(chunk)
                            elif ev == "eot":
                                break
                        # ignore binary
                finally:
                    await queue.put(None)  # signal end-of-text
                    feeder_done.set()

            asyncio.create_task(recv_text_chunks())
            text_source = _QueueIterator(queue, loop)   # <— pass loop in
        else:
            # ... (unchanged one-shot fallback)
            text_source = text_initial or ""

        # --- Streaming out audio (reworked worker error path) ---
        audio_q: "asyncio.Queue[tuple[Optional[np.ndarray], Optional[str]]]" = asyncio.Queue(maxsize=8)
        done_evt = asyncio.Event()

        def _run_generator():
            return speech_generator.generate_stream(
                prompt_text=prompt_text,
                prompt_audio_path=prompt_path,
                text=text_source,
            )

        def _worker():
            err: Optional[str] = None
            try:
                for audio_frame, _meta in _run_generator():
                    asyncio.run_coroutine_threadsafe(audio_q.put((audio_frame, None)), loop).result()
            except Exception as e:
                err = str(e)
            finally:
                # Always send poison pill; attach error message once
                asyncio.run_coroutine_threadsafe(audio_q.put((None, err)), loop).result()
                # done_evt.set() is not a coroutine; schedule it thread-safely
                loop.call_soon_threadsafe(done_evt.set)

        import threading
        threading.Thread(target=_worker, daemon=True).start()

        # Tell client config before streaming
        await ws.send_text(json.dumps({
            "type": "config",
            "sample_rate": config.mimi_sr,
            "dtype": "float32",
            "channels": 1
        }))

        # Pump frames out; if we receive an error, send it (if socket still open) then end.
        while True:
            frame, err = await audio_q.get()
            if frame is None:
                if err:
                    # best-effort error message; ignore if already closed remotely
                    try:
                        await ws.send_text(json.dumps({"type": "error", "message": err}))
                    except Exception:
                        pass
                break
            if frame.dtype != np.float32:
                frame = frame.astype(np.float32, copy=False)
            frame = np.ascontiguousarray(frame)
            await ws.send_bytes(frame.tobytes())

        # graceful end
        try:
            await ws.send_text(json.dumps({"type": "eos"}))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass

    except WebSocketDisconnect:
        return
    except Exception as e:
        # Last-ditch: only if still open
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
