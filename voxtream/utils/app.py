import base64
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Mapping

import gradio as gr
import numpy as np

from voxtream.utils.generator.text import TextProgressMetadata

CUSTOM_CSS = """
/* overall width */
.gradio-container {max-width: 1100px !important}
/* stack labels tighter and even heights */
#cols .wrap > .form {gap: 10px}
#left-col, #right-col {gap: 14px}
/* make submit centered + bigger */
#submit {width: 260px; margin: 10px auto 0 auto;}
/* make clear align left and look secondary */
#clear {width: 120px;}
#pause, #resume, #stop {width: 120px; margin: 10px 0 0 0;}
/* give audio a little breathing room */
audio {outline: none;}
#rate-plot svg {width: 100%; height: auto; display: block;}
#rate-plot .rate-plot-target {stroke: #ff9f43;}
#voxtream-audio-stream {
  border: 1px solid #d9dce3;
  border-radius: 8px;
  padding: 10px 12px;
  margin: 8px 0;
  background: #fff;
  color: #252934;
  font-size: 14px;
}
.stream-indicator {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  margin-right: 8px;
  background: #6b7280;
}
.stream-indicator.active {
  background: #16a34a;
}
.text-progress-title {
  margin: 0 0 6px 0;
  color: #252934;
  font-size: 14px;
  font-weight: 600;
}
#text-progress {
  border: 1px solid #d9dce3;
  border-radius: 8px;
  padding: 14px 16px;
  min-height: 64px;
  font-size: 18px;
  line-height: 1.65;
  background: #fff;
}
#text-progress .text-progress-current {
  color: #ff9f43;
}
"""

LOW_LATENCY_AUDIO_HEAD_TEMPLATE = """
<script>
(() => {
  const START_DELAY_SEC = __START_DELAY_SEC__;
  const DEFAULT_SAMPLE_RATE = __DEFAULT_SAMPLE_RATE__;
  const state = {
    ctx: null,
    nextTime: 0,
    paused: false,
    session: null,
    seq: -1,
    sources: []
  };

  function ensureContext(sampleRate, shouldResume = true) {
    if (!state.ctx || state.ctx.state === "closed") {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) return null;
      try {
        state.ctx = new AudioContextClass({ sampleRate });
      } catch (_) {
        state.ctx = new AudioContextClass();
      }
    }
    if (state.paused) {
      if (state.ctx.state === "running") {
        state.ctx.suspend().catch(() => {});
      }
      return state.ctx;
    }
    if (shouldResume && state.ctx.state === "suspended") {
      state.ctx.resume().catch(() => {});
    }
    return state.ctx;
  }

  function stopScheduled() {
    for (const source of state.sources) {
      try {
        source.stop();
      } catch (_) {}
    }
    state.sources = [];
    state.nextTime = 0;
  }

  function pausePlayback() {
    state.paused = true;
    if (state.ctx && state.ctx.state === "running") {
      state.ctx.suspend().catch(() => {});
    }
  }

  function resumePlayback() {
    state.paused = false;
    if (state.ctx && state.ctx.state === "suspended") {
      state.ctx.resume().catch(() => {});
    } else {
      ensureContext(DEFAULT_SAMPLE_RATE);
    }
  }

  function stopPlayback() {
    state.paused = false;
    stopScheduled();
    if (state.ctx && state.ctx.state === "suspended") {
      state.ctx.resume().catch(() => {});
    }
  }

  function reset(session) {
    if (state.session !== session) {
      state.paused = false;
      stopScheduled();
      state.session = session;
      state.seq = -1;
    }
  }

  function decodeBase64(base64) {
    const binary = window.atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  function playPcm(base64, sampleRate) {
    const ctx = ensureContext(sampleRate, !state.paused);
    if (!ctx || !base64) return;

    const bytes = decodeBase64(base64);
    const samples = new Int16Array(
      bytes.buffer,
      bytes.byteOffset,
      Math.floor(bytes.byteLength / 2)
    );
    if (!samples.length) return;

    const buffer = ctx.createBuffer(1, samples.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i += 1) {
      channel[i] = Math.max(-1, Math.min(1, samples[i] / 32768));
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => {
      state.sources = state.sources.filter((item) => item !== source);
    };

    const minStart = ctx.currentTime + START_DELAY_SEC;
    if (!state.nextTime || state.nextTime < minStart) {
      state.nextTime = minStart;
    }
    source.start(state.nextTime);
    state.nextTime += buffer.duration;
    state.sources.push(source);
  }

  function processNode(node) {
    if (!node || !node.dataset) return;
    const session = node.dataset.session || "";
    const seq = Number(node.dataset.seq || "0");
    const sampleRate = Number(node.dataset.sr || DEFAULT_SAMPLE_RATE);

    reset(session);
    if (seq <= state.seq) return;
    state.seq = seq;
    playPcm(node.dataset.pcm || "", sampleRate);
  }

  function scan() {
    processNode(document.getElementById("voxtream-audio-stream"));
  }

  window.voxtreamLowLatencyAudio = {
    begin: () => ensureContext(DEFAULT_SAMPLE_RATE, !state.paused),
    pause: pausePlayback,
    reset: stopPlayback,
    resume: resumePlayback,
    stop: stopPlayback,
    scan
  };

  document.addEventListener(
    "pointerdown",
    () => ensureContext(DEFAULT_SAMPLE_RATE, !state.paused),
    { capture: true }
  );
  document.addEventListener(
    "keydown",
    () => ensureContext(DEFAULT_SAMPLE_RATE, !state.paused),
    { capture: true }
  );

  const observer = new MutationObserver(scan);
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true
  });
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scan);
  } else {
    scan();
  }
})();
</script>
"""


@dataclass(frozen=True)
class AppConfig:
    min_chunk_sec: float
    fade_out_sec: float
    plot_window_sec: float
    visual_update_sec: float
    future_phone_limit: int
    plot_width: int
    plot_height: int
    plot_left: int
    plot_right: int
    plot_top: int
    plot_bottom: int
    plot_y_max: int
    plot_y_tick: int
    plot_x_tick_sec: int
    audio_stream_start_delay_sec: float
    audio_stream_sample_rate: int
    speaking_rate_min: float
    speaking_rate_max: float
    speaking_rate_step: float
    speaking_rate_default: float
    min_streaming_rtf: float


def load_app_config(path: str | Path) -> AppConfig:
    with open(path) as f:
        return AppConfig(**json.load(f))


def build_low_latency_audio_head(app_config: AppConfig) -> str:
    return LOW_LATENCY_AUDIO_HEAD_TEMPLATE.replace(
        "__START_DELAY_SEC__", f"{app_config.audio_stream_start_delay_sec:.3f}"
    ).replace("__DEFAULT_SAMPLE_RATE__", str(int(app_config.audio_stream_sample_rate)))


class SpeakingRateState:
    def __init__(self, initial_value: float):
        self._value = float(initial_value)
        self._active = False
        self._lock = threading.Lock()

    def start(self, value: float) -> None:
        with self._lock:
            self._value = float(value)
            self._active = True

    def ensure_started(self, value: float) -> None:
        with self._lock:
            if not self._active:
                self._value = float(value)
            self._active = True

    def stop(self) -> None:
        with self._lock:
            self._active = False

    def update(self, value: float) -> None:
        with self._lock:
            if self._active:
                self._value = float(value)

    def values(self):
        while True:
            with self._lock:
                value = self._value
            yield value


class SharedGenerationState:
    """File-backed generation controls for callbacks that may run out of process."""

    def __init__(self, directory: str | Path = "/tmp/voxtream_generation_state"):
        self._directory = Path(directory)
        self._lock = threading.Lock()

    def _path(self, session_id: str) -> Path | None:
        if not session_id:
            return None
        return self._directory / f"{session_id}.json"

    def _read_unlocked(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id)
        if path is None or not path.exists():
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_unlocked(self, session_id: str, state: Mapping[str, Any]) -> None:
        path = self._path(session_id)
        if path is None:
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        with open(tmp_path, "w") as f:
            json.dump(dict(state), f)
        tmp_path.replace(path)

    def create(self, speaking_rate: float) -> str:
        session_id = uuid.uuid4().hex
        self.start(session_id, speaking_rate)
        return session_id

    def start(self, session_id: str, speaking_rate: float) -> None:
        with self._lock:
            self._write_unlocked(
                session_id,
                {
                    "active": True,
                    "paused": False,
                    "stopped": False,
                    "speaking_rate": float(speaking_rate),
                },
            )

    def update(self, session_id: str, **updates: Any) -> None:
        if not session_id:
            return
        with self._lock:
            state = self._read_unlocked(session_id)
            if not state:
                return
            state.update(updates)
            self._write_unlocked(session_id, state)

    def update_speaking_rate(self, session_id: str, speaking_rate: float) -> None:
        self.update(session_id, speaking_rate=float(speaking_rate))

    def pause(self, session_id: str) -> None:
        self.update(session_id, paused=True)

    def resume(self, session_id: str) -> None:
        self.update(session_id, paused=False)

    def stop(self, session_id: str) -> None:
        self.update(session_id, paused=False, stopped=True)

    def finish(self, session_id: str) -> None:
        self.update(session_id, active=False, paused=False, stopped=False)

    def read(self, session_id: str) -> dict[str, Any]:
        return self._read_unlocked(session_id)

    def is_paused(self, session_id: str) -> bool:
        return bool(self.read(session_id).get("paused", False))

    def is_stopped(self, session_id: str) -> bool:
        return bool(self.read(session_id).get("stopped", False))

    def wait_if_paused(self, session_id: str) -> bool:
        while True:
            state = self.read(session_id)
            if state.get("stopped", False):
                return False
            if not state.get("paused", False):
                return True
            time.sleep(0.05)

    def speaking_rate_values(self, session_id: str, initial_value: float):
        value = float(initial_value)
        while True:
            state = self.read(session_id)
            if "speaking_rate" in state:
                value = float(state["speaking_rate"])
            yield value


def float32_to_int16(audio_float32: np.ndarray) -> np.ndarray:
    """
    Convert float32 audio samples (-1.0 to 1.0) to int16 PCM samples.

    Parameters:
        audio_float32 (np.ndarray): Input float32 audio samples.

    Returns:
        np.ndarray: Output int16 audio samples.
    """
    if audio_float32.dtype != np.float32:
        raise ValueError("Input must be a float32 numpy array")

    audio_clipped = np.clip(audio_float32, -1.0, 1.0)
    return (audio_clipped * 32767).astype(np.int16)


def _scale_plot_x(
    app_config: AppConfig,
    time_sec: float,
    x_min: float,
    x_max: float,
    plot_x0: float | None = None,
    plot_x1: float | None = None,
) -> float:
    if plot_x0 is None:
        plot_x0 = app_config.plot_left
    if plot_x1 is None:
        plot_x1 = app_config.plot_width - app_config.plot_right
    return plot_x0 + (time_sec - x_min) / max(x_max - x_min, 1e-6) * (plot_x1 - plot_x0)


def _scale_plot_y(
    app_config: AppConfig,
    rate: float,
    y_max: float,
    plot_y0: float | None = None,
    plot_y1: float | None = None,
) -> float:
    if plot_y0 is None:
        plot_y0 = app_config.plot_height - app_config.plot_bottom
    if plot_y1 is None:
        plot_y1 = app_config.plot_top
    rate = min(max(rate, 0.0), y_max)
    return plot_y0 - rate / max(y_max, 1e-6) * (plot_y0 - plot_y1)


def _polyline(
    app_config: AppConfig,
    points,
    x_min: float,
    x_max: float,
    y_max: float,
    plot_x0: float | None = None,
    plot_x1: float | None = None,
    plot_y0: float | None = None,
    plot_y1: float | None = None,
):
    coords = []
    for time_sec, value in points:
        if value is None or not np.isfinite(value):
            continue
        x = _scale_plot_x(app_config, time_sec, x_min, x_max, plot_x0, plot_x1)
        y = _scale_plot_y(app_config, max(0.0, value), y_max, plot_y0, plot_y1)
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def _step_path(
    app_config: AppConfig,
    points,
    x_min: float,
    x_max: float,
    y_max: float,
    plot_x0: float | None = None,
    plot_x1: float | None = None,
    plot_y0: float | None = None,
    plot_y1: float | None = None,
) -> str:
    clean_points = [
        (time_sec, value)
        for time_sec, value in points
        if value is not None and np.isfinite(value)
    ]
    if not clean_points:
        return ""

    first_x = _scale_plot_x(
        app_config, clean_points[0][0], x_min, x_max, plot_x0, plot_x1
    )
    first_y = _scale_plot_y(
        app_config, max(0.0, clean_points[0][1]), y_max, plot_y0, plot_y1
    )
    parts = [f"M {first_x:.1f} {first_y:.1f}"]
    prev_value = clean_points[0][1]
    for time_sec, value in clean_points[1:]:
        x = _scale_plot_x(app_config, time_sec, x_min, x_max, plot_x0, plot_x1)
        prev_y = _scale_plot_y(
            app_config, max(0.0, prev_value), y_max, plot_y0, plot_y1
        )
        y = _scale_plot_y(app_config, max(0.0, value), y_max, plot_y0, plot_y1)
        parts.append(f"L {x:.1f} {prev_y:.1f} L {x:.1f} {y:.1f}")
        prev_value = value
    return " ".join(parts)


def _duration_distribution(duration_state, bins: int = 6) -> list[float] | None:
    if duration_state is None:
        return None

    values = np.asarray(duration_state, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None
    if values.size < bins:
        values = np.pad(values, (0, bins - values.size))
    elif values.size > bins:
        values = values[:bins]

    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0:
        return [0.0] * bins
    return (values / total).tolist()


def _render_duration_histogram(
    target_duration_state,
    current_duration_state,
    panel_x0: float,
    panel_x1: float,
    y0: float,
    y1: float,
    show_target: bool,
) -> str:
    target_dist = _duration_distribution(target_duration_state) if show_target else None
    current_dist = _duration_distribution(current_duration_state)
    chart_x0 = panel_x0 + 34
    chart_x1 = panel_x1 - 6
    bins = 6
    group_width = (chart_x1 - chart_x0) / bins
    target_bar_width = group_width * 0.70
    current_bar_width = group_width * 0.50

    def scale_prob(value: float) -> float:
        return y0 - min(max(value, 0.0), 1.0) * (y0 - y1)

    grid = []
    labels = []
    for value, label in [(0.0, "0"), (0.5, ".5"), (1.0, "1")]:
        y = scale_prob(value)
        grid.append(
            f'<line x1="{chart_x0:.1f}" y1="{y:.1f}" '
            f'x2="{chart_x1:.1f}" y2="{y:.1f}" '
            'stroke="#e6e8ef" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{chart_x0 - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            'font-size="11" fill="#4a4f5c">'
            f"{label}</text>"
        )

    target_bars = []
    if target_dist is not None:
        for idx, value in enumerate(target_dist):
            center = chart_x0 + group_width * (idx + 0.5)
            bar_height_y = scale_prob(value)
            target_bars.append(
                f'<rect x="{center - target_bar_width / 2:.1f}" '
                f'y="{bar_height_y:.1f}" width="{target_bar_width:.1f}" '
                f'height="{y0 - bar_height_y:.1f}" fill="#ff9f43" '
                'fill-opacity="0.9"/>'
            )

    current_bars = []
    if current_dist is not None:
        for idx, value in enumerate(current_dist):
            center = chart_x0 + group_width * (idx + 0.5)
            bar_height_y = scale_prob(value)
            current_bars.append(
                f'<rect x="{center - current_bar_width / 2:.1f}" '
                f'y="{bar_height_y:.1f}" width="{current_bar_width:.1f}" '
                f'height="{y0 - bar_height_y:.1f}" fill="#1f77b4" '
                'fill-opacity="0.7"/>'
            )

    x_labels = []
    for idx in range(bins):
        center = chart_x0 + group_width * (idx + 0.5)
        x_labels.append(
            f'<text x="{center:.1f}" y="{y0 + 16:.1f}" text-anchor="middle" '
            'font-size="11" fill="#4a4f5c">'
            f"{idx + 1}</text>"
        )

    overlap = None
    if target_dist is not None and current_dist is not None:
        overlap = sum(
            min(target, current) for target, current in zip(target_dist, current_dist)
        )
    overlap_text = (
        f'<text x="{panel_x1:.1f}" y="47" text-anchor="end" font-size="12" '
        f'fill="#252934">Overlap {overlap * 100:.2f}%</text>'
        if overlap is not None
        else ""
    )

    return f"""
    <text x="{panel_x0:.1f}" y="27" font-size="18" font-weight="600"
          fill="#252934">Duration state</text>
    {overlap_text}
    {''.join(grid)}
    <line x1="{chart_x0:.1f}" y1="{y0:.1f}" x2="{chart_x1:.1f}" y2="{y0:.1f}"
          stroke="#252934" stroke-width="1.2"/>
    <line x1="{chart_x0:.1f}" y1="{y0:.1f}" x2="{chart_x0:.1f}" y2="{y1:.1f}"
          stroke="#252934" stroke-width="1.2"/>
    {''.join(labels)}
    {''.join(target_bars)}
    {''.join(current_bars)}
    {''.join(x_labels)}
    <text x="{(chart_x0 + chart_x1) / 2:.1f}" y="{y0 + 32:.1f}"
          text-anchor="middle" font-size="12" fill="#252934">Duration bin</text>
"""


def render_rate_plot(
    app_config: AppConfig,
    times,
    generated_rates,
    target_rates,
    show_target: bool,
    target_duration_state=None,
    current_duration_state=None,
) -> str:
    latest_time = float(times[-1]) if times else 0.0
    x_min = max(0.0, latest_time - app_config.plot_window_sec)
    x_max = max(app_config.plot_window_sec, latest_time)
    visible = [
        (time_sec, gen_rate, target_rate)
        for time_sec, gen_rate, target_rate in zip(times, generated_rates, target_rates)
        if time_sec >= x_min
    ]

    y_max = float(app_config.plot_y_max)

    x0 = app_config.plot_left
    y0 = app_config.plot_height - app_config.plot_bottom
    available_x1 = app_config.plot_width - app_config.plot_right
    duration_panel_width = min(
        max(220, int(app_config.plot_width * 0.24)),
        max(180, available_x1 - x0 - 360),
    )
    duration_x0 = available_x1 - duration_panel_width
    duration_x1 = available_x1
    plot_gap = 34
    x1 = duration_x0 - plot_gap
    y1 = app_config.plot_top
    grid = []
    labels = []
    for y_value in range(0, app_config.plot_y_max + 1, app_config.plot_y_tick):
        y = _scale_plot_y(app_config, y_value, y_max, y0, y1)
        grid.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
            'stroke="#e6e8ef" stroke-width="1"/>'
        )
        if y_value != 0:
            labels.append(
                f'<text x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end" '
                'font-size="13" fill="#4a4f5c">'
                f"{y_value:.0f}</text>"
            )

    first_x_tick = int(np.ceil(x_min))
    last_x_tick = int(np.floor(x_max))
    for x_value in range(first_x_tick, last_x_tick + 1, app_config.plot_x_tick_sec):
        x = _scale_plot_x(app_config, x_value, x_min, x_max, x0, x1)
        grid.append(
            f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" '
            'stroke="#f1f2f6" stroke-width="1"/>'
        )
        if x_value != 0:
            labels.append(
                f'<text x="{x:.1f}" y="{y0 + 18:.1f}" text-anchor="middle" '
                'font-size="13" fill="#4a4f5c">'
                f"{x_value:.0f}</text>"
            )

    generated_points = [(time_sec, gen_rate) for time_sec, gen_rate, _ in visible]
    target_points = [(time_sec, target_rate) for time_sec, _, target_rate in visible]
    generated_polyline = _polyline(
        app_config, generated_points, x_min, x_max, y_max, x0, x1, y0, y1
    )
    target_path = (
        _step_path(app_config, target_points, x_min, x_max, y_max, x0, x1, y0, y1)
        if show_target
        else ""
    )

    generated_line = (
        f'<polyline points="{generated_polyline}" fill="none" '
        'stroke="#1f77b4" stroke-width="2.5" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
        if generated_polyline
        else ""
    )
    target_line = (
        f'<path d="{target_path}" class="rate-plot-target" fill="none" '
        'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
        if target_path
        else ""
    )
    target_legend = (
        f'<line class="rate-plot-target" x1="{x1 - 354:.1f}" y1="25" '
        f'x2="{x1 - 324:.1f}" y2="25" stroke-width="2.2"/>'
        f'<text x="{x1 - 316:.1f}" y="30" font-size="13" '
        'fill="#252934">Target</text>'
        if show_target
        else ""
    )
    generated_legend_x = x1 - 164
    duration_histogram = _render_duration_histogram(
        target_duration_state,
        current_duration_state,
        duration_x0,
        duration_x1,
        y0,
        y1,
        show_target=show_target,
    )

    return f"""
<div id="rate-plot">
  <svg viewBox="0 0 {app_config.plot_width} {app_config.plot_height}" role="img"
       aria-label="Dynamic speaking rate control plot">
    <rect x="1" y="1" width="{app_config.plot_width - 2}" height="{app_config.plot_height - 2}"
          rx="8" fill="#ffffff" stroke="#d9dce3" stroke-width="1.2"/>
    <text x="{x0}" y="27" font-size="18" font-weight="600"
          fill="#252934">Speaking-rate control</text>
    {''.join(grid)}
    <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#252934" stroke-width="1.4"/>
    <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#252934" stroke-width="1.4"/>
    {''.join(labels)}
    {target_line}
    {generated_line}
    {target_legend}
    <line x1="{generated_legend_x:.1f}" y1="25" x2="{generated_legend_x + 30:.1f}"
          y2="25" stroke="#1f77b4" stroke-width="2.5"/>
    <text x="{generated_legend_x + 38:.1f}" y="30" font-size="13"
          fill="#252934">Generated</text>
    <text x="{(x0 + x1) / 2:.1f}" y="{y0 + 35:.1f}" text-anchor="middle"
          font-size="15" fill="#252934">Time (seconds)</text>
    <text x="36" y="{(y0 + y1) / 2:.1f}" text-anchor="middle"
          transform="rotate(-90 36 {(y0 + y1) / 2:.1f})"
          font-size="15" fill="#252934">Speaking rate (SPS)</text>
    {duration_histogram}
  </svg>
</div>
"""


def empty_rate_plot(app_config: AppConfig, show_target: bool = True) -> str:
    return render_rate_plot(app_config, [], [], [], show_target=show_target)


def render_text_progress(
    app_config: AppConfig,
    metadata: TextProgressMetadata | None,
    phone_position: int = 0,
) -> str:
    if metadata is None or not metadata.words:
        return (
            '<div class="text-progress-title">Input text stream</div>'
            '<div id="text-progress"></div>'
        )

    phone_position = max(0, int(phone_position))
    current_idx = None
    for idx, word in enumerate(metadata.words):
        if word.end > phone_position:
            current_idx = idx
            break

    future_limit = phone_position + app_config.future_phone_limit
    rendered_words = []
    for idx, word in enumerate(metadata.words):
        if current_idx is None:
            color = "#111827"
        elif idx < current_idx:
            color = "#111827"
        elif idx == current_idx:
            rendered_words.append(
                f'<span class="text-progress-current">{escape(word.text)}</span>'
            )
            continue
        else:
            if word.end > future_limit:
                break
            color = "#9ca3af"

        rendered_words.append(
            f'<span style="color: {color};">{escape(word.text)}</span>'
        )

    return (
        '<div class="text-progress-title">Input text stream</div>'
        f'<div id="text-progress">{" ".join(rendered_words)}</div>'
    )


def render_audio_stream(
    app_config: AppConfig,
    session_id: str = "",
    seq: int = 0,
    sample_rate: int | None = None,
    audio: np.ndarray | None = None,
    active: bool = False,
    final: bool = False,
) -> str:
    if sample_rate is None:
        sample_rate = app_config.audio_stream_sample_rate

    if audio is None:
        pcm_base64 = ""
    else:
        pcm_base64 = base64.b64encode(np.ascontiguousarray(audio).tobytes()).decode(
            "ascii"
        )

    status = "Complete" if final else ("Playing" if active else "Ready")
    indicator_class = (
        "stream-indicator active" if active and not final else "stream-indicator"
    )
    return (
        f'<div id="voxtream-audio-stream" data-session="{escape(session_id)}" '
        f'data-seq="{int(seq)}" data-sr="{int(sample_rate)}" '
        f'data-final="{int(final)}" data-pcm="{pcm_base64}">'
        f'<span class="{indicator_class}"></span>{status}</div>'
    )


def clear_outputs(app_config: AppConfig):
    return (
        gr.update(value=None, visible=False),
        empty_rate_plot(app_config),
        render_text_progress(app_config, None),
        render_audio_stream(app_config, session_id=uuid.uuid4().hex),
    )


def vowels_at_position(metadata: TextProgressMetadata, phone_position: int) -> int:
    phone_position = max(0, min(int(phone_position), len(metadata.vowel_prefix) - 1))
    return metadata.vowel_prefix[phone_position]


def phone_position_at_time(phone_history, target_time: float) -> int:
    """Return the newest phone position generated no later than target_time."""
    position = 0
    for time_sec, phone_position in phone_history:
        if time_sec > target_time:
            break
        position = phone_position
    return position


@dataclass
class VisualizationState:
    text_metadata: TextProgressMetadata
    app_config: AppConfig
    rate_window_sec: float
    frame_sec: float
    text_progress_delay_sec: float
    show_target: bool
    rate_history: deque = field(default_factory=deque)
    phone_history: deque = field(default_factory=lambda: deque([(0.0, 0)]))
    plot_times: list[float] = field(default_factory=list)
    generated_rates: list[float] = field(default_factory=list)
    target_rates: list[float | None] = field(default_factory=list)
    target_duration_state: list[float] | None = field(init=False, default=None)
    current_duration_state: list[float] | None = field(init=False, default=None)
    latest_plot: str = field(init=False)
    latest_text: str = field(init=False)
    last_phone_position: int = field(init=False, default=0)
    last_visual_update_time: float = field(init=False)

    def __post_init__(self) -> None:
        self.latest_plot = empty_rate_plot(
            self.app_config, show_target=self.show_target
        )
        self.latest_text = render_text_progress(self.app_config, self.text_metadata, 0)
        self.last_visual_update_time = -self.app_config.visual_update_sec

    def update(self, progress: Mapping[str, Any], force: bool = False):
        time_sec = float(progress.get("time_sec", 0.0))
        phone_position = max(
            self.last_phone_position, int(progress.get("phone_position", 0))
        )
        phone_position = min(phone_position, len(self.text_metadata.vowel_prefix) - 1)
        self.last_phone_position = phone_position
        self.phone_history.append((time_sec, phone_position))
        delayed_phone_position = phone_position_at_time(
            self.phone_history, max(0.0, time_sec - self.text_progress_delay_sec)
        )

        vowel_count = vowels_at_position(self.text_metadata, phone_position)
        self.rate_history.append((time_sec, vowel_count))
        cutoff = max(0.0, time_sec - self.rate_window_sec)
        while len(self.rate_history) > 1 and self.rate_history[1][0] <= cutoff:
            self.rate_history.popleft()

        base_time, base_vowels = self.rate_history[0]
        elapsed = max(time_sec - base_time, self.frame_sec)
        generated_rate = max(0.0, (vowel_count - base_vowels) / elapsed)
        target_rate = (
            float(progress["speaking_rate"])
            if self.show_target and progress.get("speaking_rate") is not None
            else None
        )
        if self.show_target and progress.get("target_duration_state") is not None:
            self.target_duration_state = list(progress["target_duration_state"])
        if progress.get("current_duration_state") is not None:
            self.current_duration_state = list(progress["current_duration_state"])

        self.plot_times.append(time_sec)
        self.generated_rates.append(generated_rate)
        self.target_rates.append(target_rate)
        while (
            self.plot_times
            and self.plot_times[0] < time_sec - self.app_config.plot_window_sec
        ):
            self.plot_times.pop(0)
            self.generated_rates.pop(0)
            self.target_rates.pop(0)

        should_update = (
            force
            or time_sec - self.last_visual_update_time
            >= self.app_config.visual_update_sec
        )
        self.latest_plot = render_rate_plot(
            self.app_config,
            self.plot_times,
            self.generated_rates,
            self.target_rates,
            show_target=self.show_target,
            target_duration_state=self.target_duration_state,
            current_duration_state=self.current_duration_state,
        )
        if not should_update:
            return self.latest_plot, gr.update()

        self.latest_text = render_text_progress(
            self.app_config, self.text_metadata, delayed_phone_position
        )
        self.last_visual_update_time = time_sec
        return self.latest_plot, self.latest_text

    def final_text(self) -> str:
        return render_text_progress(
            self.app_config, self.text_metadata, self.last_phone_position
        )


class GenerationControl:
    def __init__(self):
        self._condition = threading.Condition()
        self._paused = False
        self._stopped = False

    def start(self) -> None:
        with self._condition:
            self._paused = False
            self._stopped = False
            self._condition.notify_all()

    def pause(self) -> None:
        with self._condition:
            if not self._stopped:
                self._paused = True

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._paused = False
            self._condition.notify_all()

    def finish(self) -> None:
        with self._condition:
            self._paused = False
            self._stopped = False
            self._condition.notify_all()

    def is_paused(self) -> bool:
        with self._condition:
            return self._paused

    def is_stopped(self) -> bool:
        with self._condition:
            return self._stopped

    def wait_if_paused(self) -> bool:
        with self._condition:
            while self._paused and not self._stopped:
                self._condition.wait(timeout=0.05)
            return not self._stopped
