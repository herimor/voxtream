import json
import re
import sys
from itertools import count
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm

FRAMES_PER_SEC = 12.5  # 80 ms frames
MIN_RATE, MAX_RATE = 2.0, 6.0
SMOOTH_SIGMA = 3  # frames
ALIGNMENT_EXT = ".alignment.json"


def count_syllables(word: str) -> int:
    """Vowel-group heuristic with a silent trailing "e"."""
    word = re.sub(r"[^a-z]", "", word.lower())
    num = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and num > 1:
        num -= 1
    return max(num, 1)


def load_alignment(audio_path: Path) -> dict:
    with open(audio_path.with_suffix(ALIGNMENT_EXT)) as f:
        return json.load(f)


def target_rate_frames(
    words: list[dict], min_rate: float = MIN_RATE, max_rate: float = MAX_RATE
) -> np.ndarray:
    """Per-frame control signal: each word (with its preceding pause) gets its
    clipped rate and lasts syllables / rate seconds."""
    syllables, rates, prev_end = [], [], 0.0
    for word in words:
        duration = max(word["end"] - word["start"], 1e-3) + max(
            word["start"] - prev_end, 0.0
        )
        syllables.append(count_syllables(word["word"]))
        rates.append(np.clip(syllables[-1] / duration, min_rate, max_rate))
        prev_end = word["end"]

    ends = np.cumsum(np.array(syllables) / np.array(rates)) * FRAMES_PER_SEC
    frames = np.full(int(np.ceil(ends[-1] - 1e-6)), rates[-1])
    for start, end, rate in zip(np.r_[0, ends[:-1]], ends, rates):
        frames[int(round(start)) : int(round(end))] = rate
    return frames


def rate_iterator(frames: np.ndarray, delay_frames: int = 0) -> Iterator[float]:
    """Endless per-step speaking rate for `SpeechGenerator.generate_stream`."""
    for step in count():
        yield float(frames[min(max(step - delay_frames, 0), len(frames) - 1)])


def rate_curve(alignment: dict) -> np.ndarray | None:
    """Per-frame realized rate on the alignment's own timeline."""
    curve = np.full(int(alignment["duration"] * FRAMES_PER_SEC), np.nan)
    prev_end = 0.0
    for word in alignment["words"]:
        duration = round(word["end"] - prev_end, 3)
        if duration <= 0:
            return None
        rate = round(count_syllables(word["word"]) / duration, 2)
        start, end = round(prev_end, 3), round(prev_end + duration, 3)
        curve[int(start * FRAMES_PER_SEC) : int(end * FRAMES_PER_SEC)] = rate
        prev_end = word["end"]

    if len(curve) < 2 or np.isnan(curve).all():
        return None
    return pd.Series(curve).ffill().bfill().to_numpy()


def curve_metrics(target: np.ndarray, generated: np.ndarray) -> tuple[float, float]:
    """Pearson r and MAE between the target curve and the generated curve
    resampled to the target length, both Gaussian-smoothed."""
    generated = np.interp(
        np.linspace(0, 1, len(target)), np.linspace(0, 1, len(generated)), generated
    )
    generated = gaussian_filter1d(generated, SMOOTH_SIGMA)
    target = gaussian_filter1d(target, SMOOTH_SIGMA)
    corr = np.nan
    if target.std() > 0 and generated.std() > 0:
        corr = np.corrcoef(target, generated)[0, 1]
    return corr, np.abs(target - generated).mean()


def align(
    meta: pd.DataFrame,
    dataset_dir: Path,
    aligner_root: Path,
    file_ext: str,
    device: str = "cuda",
):
    """Seamless word alignment of every `row.path`, saved next to the audio."""
    seamless_dir = aligner_root / "aligners/seamless"
    deps_dir = seamless_dir / "seamless_unity_aligner/deps"
    (deps_dir / "fairseq2/assets/cards").mkdir(exist_ok=True)  # not tracked by git
    sys.path[:0] = [str(seamless_dir), str(deps_dir)]
    from seamless_unity_aligner import SeamlessUnityAligner

    aligner = None
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Aligning"):
        audio_path = (dataset_dir / row.path).with_suffix(f".{file_ext}")
        if audio_path.with_suffix(ALIGNMENT_EXT).exists():
            continue
        if aligner is None:
            aligner = SeamlessUnityAligner(device=torch.device(device))

        words, _ = aligner(audio=str(audio_path), transcript=row.text)
        alignment = {
            "duration": round(sf.info(audio_path).duration, 3),
            "words": [
                {
                    "word": w.word,
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                }
                for w in words.itertuples()
            ],
        }
        with open(audio_path.with_suffix(ALIGNMENT_EXT), "w") as f:
            json.dump(alignment, f)


def main(
    meta: pd.DataFrame,
    dataset_dir: Path,
    reference_dir: Path,
    aligner_root: Path,
    file_ext: str,
):
    """Correlation and MAE between reference and generated speaking-rate curves."""
    align(meta, dataset_dir, aligner_root, file_ext)

    scores = []
    for _, row in meta.iterrows():
        target = rate_curve(load_alignment(reference_dir / row.path))
        generated = rate_curve(load_alignment(dataset_dir / row.path))
        if target is None or generated is None:
            continue
        scores.append((row.path, *curve_metrics(target, generated)))

    scores = pd.DataFrame(scores, columns=["path", "corr", "mae"])
    scores.to_csv(dataset_dir / "speaking_rate.csv", index=False)
    mean = scores.drop(columns="path").mean().round(3)
    print(f"Speaking rate : R {mean['corr']}, MAE {mean['mae']}")
