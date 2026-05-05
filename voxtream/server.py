#!/usr/bin/env python3
"""
A tiny WebSocket TTS server for VoXtream.

Protocol (very simple):

Client connects to ws://HOST:PORT/voxtream and first sends a JSON text frame:

{
  "event": "init",
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

import asyncio
import base64
import binascii
import json
import logging
import os
import re
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, cast

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from voxtream.config import (  # type: ignore
    SpeechGeneratorConfig,
    load_generator_config,
    load_speaking_rate_config,
)
from voxtream.generator import SpeechGenerator  # type: ignore
from voxtream.utils.generator import set_seed  # type: ignore

# ---------- Helpers ----------

DATA_URL_RE = re.compile(r"^data:.*?;base64,(.*)$", re.IGNORECASE)
ALLOWED_PROMPT_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
MAX_PROMPT_AUDIO_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHUNK_CHARS = 4_000
MAX_INITIAL_TEXT_CHARS = 20_000
MAX_GENERATION_WORKERS = 1
LOGGER = logging.getLogger("voxtream.server")


def _b64_to_bytes(s: str, max_bytes: int = MAX_PROMPT_AUDIO_BYTES) -> bytes:
    m = DATA_URL_RE.match(s)
    payload = m.group(1) if m else s
    compact_payload = "".join(payload.split())
    if len(compact_payload) > ((max_bytes + 2) // 3) * 4:
        raise ValueError(f"Prompt audio upload exceeds {max_bytes} bytes")
    try:
        raw = base64.b64decode(compact_payload, validate=True)
    except binascii.Error as err:
        raise ValueError("prompt_audio_b64 is not valid base64") from err
    if len(raw) > max_bytes:
        raise ValueError(f"Prompt audio upload exceeds {max_bytes} bytes")
    return raw


def _validate_prompt_audio_path(prompt_audio_path: str, prompt_root: Path) -> Path:
    path = Path(prompt_audio_path).expanduser().resolve()
    prompt_root = prompt_root.expanduser().resolve()
    if not path.is_relative_to(prompt_root):
        raise ValueError(f"prompt_audio_path must be inside {prompt_root}")
    if not path.is_file():
        raise ValueError("prompt_audio_path must point to an existing file")
    if path.suffix.lower() not in ALLOWED_PROMPT_SUFFIXES:
        raise ValueError(
            f"prompt_audio_path must use one of {sorted(ALLOWED_PROMPT_SUFFIXES)}"
        )
    if path.stat().st_size > MAX_PROMPT_AUDIO_BYTES:
        raise ValueError(f"Prompt audio file exceeds {MAX_PROMPT_AUDIO_BYTES} bytes")
    return path


def _ensure_prompt_audio_file(
    prompt_audio_path: Optional[str], prompt_audio_b64: Optional[str], prompt_root: Path
) -> tuple[Path, bool]:
    """
    Returns a filesystem Path to the prompt audio.
    If base64 is provided, writes a temp file with the decoded bytes (wav/ogg input supported by voxtream).
    """
    if prompt_audio_path:
        return _validate_prompt_audio_path(prompt_audio_path, prompt_root), False
    if not prompt_audio_b64:
        raise ValueError(
            "Either 'prompt_audio_path' or 'prompt_audio_b64' must be provided."
        )
    raw = _b64_to_bytes(prompt_audio_b64)
    # Write as-is; VoXtream reads by extension, so try to infer (default to .wav)
    suffix = ".wav"
    if prompt_audio_b64.lower().startswith("data:audio/ogg"):
        suffix = ".ogg"
    fd, tmp = tempfile.mkstemp(prefix="voxtream_prompt_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return Path(tmp), True


def get_generator_from_state(
    app: FastAPI,
) -> tuple[SpeechGenerator, SpeechGeneratorConfig]:
    try:
        return app.state.speech_generator, app.state.config
    except AttributeError as err:
        raise HTTPException(
            status_code=503, detail="Model is initializing, try again"
        ) from err


@asynccontextmanager
async def lifespan(app: FastAPI):
    set_seed()
    config = load_generator_config()
    spk_rate_config = load_speaking_rate_config()

    app.state.config = config
    app.state.speech_generator = SpeechGenerator(config, spk_rate_config)
    app.state.generation_semaphore = threading.BoundedSemaphore(MAX_GENERATION_WORKERS)
    app.state.prompt_root = Path(os.environ.get("VOXTREAM_PROMPT_ROOT", ".")).resolve()

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

    def __init__(
        self, q: "asyncio.Queue[Optional[str]]", loop: asyncio.AbstractEventLoop
    ):
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
            await ws.send_text(
                json.dumps(
                    {"type": "error", "message": "First message must be JSON init."}
                )
            )
            await ws.close()
            return

        if init.get("event") != "init":
            await ws.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": "First message must be {'event':'init',...}.",
                    }
                )
            )
            await ws.close()
            return

        prompt_audio_path_value = init.get("prompt_audio_path")
        prompt_audio_b64_value = init.get("prompt_audio_b64")
        text_initial_value = init.get("text")
        if prompt_audio_path_value is not None and not isinstance(prompt_audio_path_value, str):
            await ws.send_text(
                json.dumps({"type": "error", "message": "prompt_audio_path must be a string"})
            )
            await ws.close()
            return
        if prompt_audio_b64_value is not None and not isinstance(prompt_audio_b64_value, str):
            await ws.send_text(
                json.dumps({"type": "error", "message": "prompt_audio_b64 must be a string"})
            )
            await ws.close()
            return
        if text_initial_value is not None and not isinstance(text_initial_value, str):
            await ws.send_text(
                json.dumps({"type": "error", "message": "text must be a string"})
            )
            await ws.close()
            return

        prompt_audio_path: Optional[str] = prompt_audio_path_value
        prompt_audio_b64: Optional[str] = prompt_audio_b64_value
        text_initial: Optional[str] = text_initial_value
        if text_initial is not None and len(text_initial) > MAX_INITIAL_TEXT_CHARS:
            await ws.send_text(
                json.dumps({"type": "error", "message": "Initial text is too long"})
            )
            await ws.close()
            return
        full_stream: bool = bool(init.get("full_stream", False))

        # Optional: override sample rate (if you down/up-sample on client); we always generate at config.mimi_sr.
        sample_rate = config.mimi_sr

        temp_prompt_path: Optional[Path] = None
        try:
            prompt_path, is_temp_prompt = _ensure_prompt_audio_file(
                prompt_audio_path,
                prompt_audio_b64,
                prompt_root=ws.app.state.prompt_root,
            )
            if is_temp_prompt:
                temp_prompt_path = prompt_path
        except Exception as e:
            await ws.send_text(
                json.dumps({"type": "error", "message": f"Invalid prompt audio: {e}"})
            )
            await ws.close()
            return

        # Tell client what to expect (mono, float32, PCM)
        await ws.send_text(
            json.dumps(
                {
                    "type": "config",
                    "sample_rate": sample_rate,
                    "dtype": "float32",
                    "channels": 1,
                }
            )
        )

        # --- 2) Prepare text source (string OR generator) ---
        # If full_stream, we build an iterator fed by subsequent websocket messages.
        text_source: str | Iterator[str | None]

        # --- Prepare text source ---
        if full_stream:
            text_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
            feeder_done = asyncio.Event()

            async def recv_text_chunks():
                try:
                    if text_initial:
                        await text_queue.put(text_initial)
                    while True:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        if "text" in msg:
                            try:
                                payload = json.loads(msg["text"])
                            except json.JSONDecodeError:
                                # treat raw text as a chunk
                                await text_queue.put(msg["text"])
                                continue

                            ev = payload.get("event")
                            if ev == "text":
                                chunk = payload.get("chunk", "")
                                if not isinstance(chunk, str):
                                    continue
                                if len(chunk) > MAX_TEXT_CHUNK_CHARS:
                                    await text_queue.put(None)
                                    break
                                if chunk:
                                    await text_queue.put(chunk)
                            elif ev == "eot":
                                break
                        # ignore binary
                finally:
                    await text_queue.put(None)  # signal end-of-text
                    feeder_done.set()

            asyncio.create_task(recv_text_chunks())
            text_source = _QueueIterator(text_queue, loop)  # <— pass loop in
        else:
            # ... (unchanged one-shot fallback)
            text_source = text_initial or ""

        # --- Streaming out audio (reworked worker error path) ---
        audio_q: "asyncio.Queue[tuple[np.ndarray[Any, Any] | None, str | None]]" = (
            asyncio.Queue(maxsize=8)
        )
        done_evt = asyncio.Event()

        def _run_generator():
            return speech_generator.generate_stream(
                prompt_audio_path=prompt_path,
                text=iter(text_source) if not isinstance(text_source, str) else text_source,
            )

        def _worker():
            err: Optional[str] = None
            acquired_generation_slot = False
            try:
                semaphore = ws.app.state.generation_semaphore
                acquired = semaphore.acquire(blocking=False)
                if not acquired:
                    err = "Server is busy; try again later."
                    return
                acquired_generation_slot = True
                for result in _run_generator():
                    audio_frame = result[0]
                    asyncio.run_coroutine_threadsafe(
                        audio_q.put((cast(np.ndarray[Any, Any], audio_frame), None)), loop
                    ).result()
            except Exception as e:
                err = str(e)
            finally:
                if acquired_generation_slot:
                    ws.app.state.generation_semaphore.release()
                if temp_prompt_path is not None:
                    try:
                        temp_prompt_path.unlink(missing_ok=True)
                    except OSError:
                        LOGGER.warning("Failed to delete temp prompt %s", temp_prompt_path)
                # Always send poison pill; attach error message once
                asyncio.run_coroutine_threadsafe(
                    audio_q.put((None, err)), loop
                ).result()
                # done_evt.set() is not a coroutine; schedule it thread-safely
                loop.call_soon_threadsafe(done_evt.set)

        threading.Thread(target=_worker, daemon=True).start()

        # Pump frames out; if we receive an error, send it (if socket still open) then end.
        while True:
            frame, err = await audio_q.get()
            if frame is None:
                if err:
                    # best-effort error message; ignore if already closed remotely
                    try:
                        await ws.send_text(
                            json.dumps({"type": "error", "message": err})
                        )
                    except WebSocketDisconnect:
                        return
                    except Exception as send_err:
                        LOGGER.debug("Failed to send websocket error", exc_info=send_err)
                break
            if frame.dtype != np.float32:
                frame = frame.astype(np.float32, copy=False)
            frame = np.ascontiguousarray(frame)
            await ws.send_bytes(frame.tobytes())

        # graceful end
        try:
            await ws.send_text(json.dumps({"type": "eos"}))
        except WebSocketDisconnect:
            return
        except Exception as send_err:
            LOGGER.debug("Failed to send websocket eos", exc_info=send_err)
        try:
            await ws.close()
        except Exception as close_err:
            LOGGER.debug("Failed to close websocket", exc_info=close_err)

    except WebSocketDisconnect:
        return
    except Exception as e:
        # Last-ditch: only if still open
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            await ws.close()
        except Exception as send_err:
            LOGGER.debug("Failed to send last-ditch websocket error", exc_info=send_err)


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
