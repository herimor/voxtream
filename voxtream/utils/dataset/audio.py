"""
Audio helpers shared by the dataset stages and the inference frontend.

Loudness is the one thing the two sides have to agree on. The model learns the level the
store was encoded at, so a prompt that arrives louder or quieter than that is a mismatch
the speaker embedding cannot absorb, and a cluster assembled out of utterances recorded at
different levels teaches a step change in gain at every join. Both go through
`normalize_loudness`, at the same target.

The BS.1770 gating is done here rather than in `pyloudnorm.Meter.integrated_loudness`,
which is the same algorithm written a block at a time: it copies the signal, re-derives
the biquad coefficients on every filter call, runs one `lfilter` per stage, and then sums
each 400 ms block in a Python loop - about 800 `np.sum` calls for an 80 s entry. Folding
the stages into one `sosfilt` and taking the block energies off a cumulative sum measures
2.3x faster on real Emilia segments and agrees with `pyloudnorm` to 1e-13 LU, which
matters because metering is half of what the Mimi stage's audio workers do.
"""

from typing import Dict, Optional

import numpy as np
import pyloudnorm as pyln
import scipy.signal

# ITU-R BS.1770-4
BLOCK_SEC = 0.400  # gating block
OVERLAP = 0.75  # of a block
ABSOLUTE_GATE = -70.0  # LUFS
RELATIVE_GATE = -10.0  # LU below the ungated mean

_METERS: Dict[int, pyln.Meter] = {}
_WEIGHTING: Dict[int, np.ndarray] = {}


def loudness_meter(sample_rate: int) -> pyln.Meter:
    """One ITU-R BS.1770 meter per sample rate; building one is not free."""
    if sample_rate not in _METERS:
        _METERS[sample_rate] = pyln.Meter(
            sample_rate, block_size=BLOCK_SEC, overlap=OVERLAP
        )
    return _METERS[sample_rate]


def k_weighting(sample_rate: int) -> np.ndarray:
    """The meter's K-weighting stages as one second-order-section matrix."""
    if sample_rate not in _WEIGHTING:
        stages = list(loudness_meter(sample_rate)._filters.values())
        sos = np.zeros((len(stages), 6))
        for i, stage in enumerate(stages):
            # `b` and `a` are properties that re-derive the coefficients on every read,
            # so they are taken once here; the passband gain is a scalar and folds into
            # the numerator instead of costing a pass over the signal
            sos[i, :3] = np.asarray(stage.b, dtype=np.float64) * stage.passband_gain
            sos[i, 3:] = np.asarray(stage.a, dtype=np.float64)
        _WEIGHTING[sample_rate] = sos
    return _WEIGHTING[sample_rate]


def _block_power(weighted: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """Mean square of every gating block, off one cumulative sum."""
    step = 1.0 - OVERLAP  # of a block, as the reference writes it
    count = int(
        np.round((len(weighted) / sample_rate - BLOCK_SEC) / (BLOCK_SEC * step)) + 1
    )
    if count < 1:
        return None

    energy = np.empty(len(weighted) + 1, dtype=np.float64)
    energy[0] = 0.0
    np.cumsum(np.square(weighted), out=energy[1:])

    # the bounds are grouped exactly as the reference groups them, so the truncation to
    # whole samples lands on the same sample and the two agree bit for bit
    block = np.arange(count)
    lo = (BLOCK_SEC * (block * step) * sample_rate).astype(int)
    # rounding `count` up can put the last block's end past the signal; the reference
    # lets the slice truncate and still divides by a whole block, so the bound is
    # clamped rather than the block dropped
    hi = np.minimum(
        (BLOCK_SEC * (block * step + 1) * sample_rate).astype(int), len(weighted)
    )
    return (energy[hi] - energy[lo]) / (BLOCK_SEC * sample_rate)


def measure_loudness(signal: np.ndarray, sample_rate: int) -> Optional[float]:
    """
    Integrated loudness in LUFS, or `None` when it cannot be measured.

    `None` means exactly that: the signal is shorter than the meter's 0.4 s block, or it
    is quiet enough that the gate rejects every block and the loudness comes back `-inf`.
    Scaling by an unmeasurable gain is never the right answer, so the caller decides -
    a short clip in the middle of a longer recording is usually better served by the
    gain its neighbours got than by being left at its own level.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < int(BLOCK_SEC * sample_rate):
        return None

    power = _block_power(
        scipy.signal.sosfilt(k_weighting(sample_rate), signal), sample_rate
    )
    if power is None:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        loud = -0.691 + 10.0 * np.log10(power)
        # the absolute gate first, then a relative one off what it left (eq. 6, 7)
        kept = loud >= ABSOLUTE_GATE
        if not kept.any():
            return None
        threshold = -0.691 + 10.0 * np.log10(power[kept].mean()) + RELATIVE_GATE
        kept = (loud > threshold) & (loud > ABSOLUTE_GATE)
        if not kept.any():
            return None
        loudness = float(-0.691 + 10.0 * np.log10(power[kept].mean()))

    return loudness if np.isfinite(loudness) else None


def loudness_gain(loudness: float, target_lufs: float) -> float:
    """The linear gain that moves `loudness` onto `target_lufs`."""
    return float(10.0 ** ((target_lufs - loudness) / 20.0))


def normalize_loudness(
    signal: np.ndarray, sample_rate: int, target_lufs: float
) -> Optional[np.ndarray]:
    """
    Scale `signal` to `target_lufs` integrated loudness, or `None` if it cannot be measured.

    `None` means exactly that: the signal is shorter than the meter's 0.4 s block, or it is
    silent enough that the loudness comes back `-inf`. The caller decides what to do with
    it, because scaling by an unmeasurable gain is never the right answer.

    There is no clipping guard here on purpose. Something that concatenates several
    normalized signals has to check the peak of the finished assembly, not of each part:
    rescaling the parts one at a time undoes the match it just made.
    """
    loudness = measure_loudness(signal, sample_rate)
    if loudness is None:
        return None
    return np.asarray(signal, dtype=np.float32) * loudness_gain(loudness, target_lufs)
