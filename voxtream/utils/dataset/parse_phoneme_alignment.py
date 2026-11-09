"""
Phoneme duration statistics of the Clap-IPA alignments (`phoneme_align.py`).

Reads the phoneme alignment `{data_dir}/{parent}/{name}.json` of every protocol row (see
`paths.py`), counts the duration of
every phoneme and fits, per phoneme, the duration model that `emilia_alignment_filter.py`
scores the utterances with.

Durations are counted in aligner frames: Clap-IPA works on 20 ms frames (10 ms mel hop,
stride-2 conv) and its forced alignment gives every phoneme at least one frame, so every
duration is a whole number of frames. The workers therefore return histograms of
(phoneme, context, frames) instead of lists of floats, which keeps the RAM flat, and the
statistics are computed exactly from the merged histograms.

Context. There is no silence token between the words of a chunk, so silence is absorbed
by the phonemes next to it. Measured on 40k files (median duration):
  - inside continuous speech, the boundary phonemes behave like the rest
    (word-final 60 ms, medial 60 ms, word-initial 80 ms),
  - next to a pause (a gap between consecutive words in the output) both neighbours
    stretch: the last phoneme before it 140 ms (p90 400 ms), the first after it 120 ms,
  - the first and the last phoneme of the utterance are squashed by the [SIL] tokens
    of the aligner (median 20 ms, i.e. one frame).
The phonemes of the last two groups are counted as `edge` and kept out of the statistics;
`phoneme_contexts` is the single definition of that split, use it in the filter as well.

Example:
    python -m voxtream.utils.dataset.parse_phoneme_alignment \
        -m emilia/metadata.parquet --mask emilia/sentence_filter.npy \
        -d /data/alignment/emilia -o emilia/phoneme_stats.json -nj 32

Output (`phoneme_stats.json`), sorted by count; durations in seconds:
    {"t": {"count": 9053819, "median": 0.05647, "p0.5": 0.01037, "p99.5": 0.27308,
           "min_frame_frac": 0.23247, "mu": -2.6735, "sigma": 0.41}, ...}
`min_frame_frac` is the share of one-frame (squashed) durations. `mu` and `sigma` are the
median and scaled MAD of the log-durations of 2+ frames: one-frame durations are frequent
(10-25% of most phonemes) and would inflate the MAD of a fit over all of them.
The histograms go next to it as `phoneme_stats.npz` (`phonemes` [P], `contexts` [C],
`counts` [P, C, MAX_FRAMES + 1]) to recompute the statistics without reading the tree.
"""

import argparse
import json
import multiprocessing as mp
from array import array
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from voxtream.utils.dataset.paths import alignment_path, check_protocol

try:
    import orjson

    loads = orjson.loads
except ImportError:  # ~3x slower to parse, same result
    loads = json.loads

FRAME = 0.02  # Clap-IPA frame, seconds
MAX_FRAMES = 255  # longer durations land in the last bin
N_BINS = MAX_FRAMES + 1
CONTEXTS = ("medial", "initial", "final", "single", "edge")
MEDIAL, INITIAL, FINAL, SINGLE, EDGE = range(len(CONTEXTS))
N_CTX = len(CONTEXTS)
# a pause shows up as a gap between two words, a chunk boundary as a gap or an overlap;
# timestamps are rounded to 1 ms, so contiguous words may differ by that much
GAP_TOL = 0.005


# ---------------------------
# Counting
# ---------------------------


def phoneme_contexts(words: List[dict]) -> Iterator[Tuple[str, int, int]]:
    """Yield (phoneme, context, duration in frames) for every phoneme of an alignment."""
    last = len(words) - 1
    for i, word in enumerate(words):
        phonemes = word["phonemes"]
        n = len(phonemes) - 1
        # the boundary phonemes of a word next to a pause or the utterance edge are not
        # their own: they absorb the silence or lose frames to the [SIL] tokens
        edge_l = i == 0 or abs(word["start"] - words[i - 1]["end"]) > GAP_TOL
        edge_r = i == last or abs(words[i + 1]["start"] - word["end"]) > GAP_TOL
        for j, (phoneme, start, end) in enumerate(phonemes):
            if (j == 0 and edge_l) or (j == n and edge_r):
                ctx = EDGE
            elif n == 0:
                ctx = SINGLE
            elif j == 0:
                ctx = INITIAL
            elif j == n:
                ctx = FINAL
            else:
                ctx = MEDIAL
            yield phoneme, ctx, min(round((end - start) / FRAME), MAX_FRAMES)


# protocol paths as one Arrow array, set before the workers are forked: they slice their
# shard out of it, so no path list is pickled and no Python strings are shared
PATHS: Optional[pa.ChunkedArray] = None


def count_shard(
    job: Tuple[int, int, Path],
) -> Tuple[List[str], np.ndarray, np.ndarray, int, int]:
    """
    Histogram the phoneme durations of the files `PATHS[start:stop]`.

    Returns
    -------
    Tuple[List[str], np.ndarray, np.ndarray, int, int]
        (shard vocabulary, flat (phoneme, context, frames) keys over that vocabulary,
        their counts, number of files, number of files without an alignment)
    """
    start, stop, data_dir = job
    paths = PATHS.slice(start, stop - start).to_pylist()
    vocab: Dict[str, int] = {}
    keys = array("q")
    missing = 0
    for path in paths:
        try:
            with open(alignment_path(path, data_dir), "rb") as f:
                words = loads(f.read())["words"]
        except FileNotFoundError:
            missing += 1
            continue
        for phoneme, ctx, frames in phoneme_contexts(words):
            pid = vocab.setdefault(phoneme, len(vocab))
            keys.append((pid * N_CTX + ctx) * N_BINS + frames)
    uniq, counts = np.unique(np.frombuffer(keys, dtype=np.int64), return_counts=True)
    return list(vocab), uniq, counts, len(paths), missing


# ---------------------------
# Statistics
# ---------------------------

# bin f holds durations in [f - 0.5, f + 0.5) frames; the counts are spread uniformly
# over the bin, so quantiles and MAD do not snap to the 20 ms grid
EDGES = np.concatenate([[0.0], np.arange(N_BINS) + 0.5]) * FRAME


def _quantile(cdf: np.ndarray, q: float) -> float:
    b = min(int(np.searchsorted(cdf[1:], q)), N_BINS - 1)
    frac = (q - cdf[b]) / max(cdf[b + 1] - cdf[b], 1e-12)
    return float(EDGES[b] + frac * (EDGES[b + 1] - EDGES[b]))


def log_median_mad(counts: np.ndarray) -> Optional[Tuple[float, float]]:
    """Median (seconds) and scaled MAD of the log-durations of a frame histogram."""
    total = counts.sum()
    if total == 0:
        return None
    cdf = np.concatenate([[0.0], np.cumsum(counts)]) / total
    med = _quantile(cdf, 0.5)
    # MAD of log-durations: the half-width r of the log-interval around the median that
    # holds half of the mass, i.e. F(med * e^r) - F(med / e^r) = 0.5
    lo_r, hi_r = 0.0, 10.0
    for _ in range(50):
        r = 0.5 * (lo_r + hi_r)
        mass = np.interp(med * np.exp(r), EDGES, cdf) - np.interp(
            med * np.exp(-r), EDGES, cdf
        )
        lo_r, hi_r = (r, hi_r) if mass < 0.5 else (lo_r, r)
    return med, 1.4826 * hi_r  # scaled MAD ~ std for normal data


def hist_stats(counts: np.ndarray) -> Optional[dict]:
    """Statistics of a frame histogram, see the module docstring."""
    total = int(counts.sum())
    if total == 0:
        return None
    cdf = np.concatenate([[0.0], np.cumsum(counts)]) / total
    # the log-normal is fitted on durations of 2+ frames: one-frame durations form a
    # spike of their own (squashed phonemes) that would blow the MAD up
    body = counts.copy()
    body[:2] = 0
    fit = log_median_mad(body)
    return {
        "count": total,
        "median": round(_quantile(cdf, 0.5), 5),  # typical duration, seconds
        "p0.5": round(_quantile(cdf, 0.005), 5),
        "p99.5": round(_quantile(cdf, 0.995), 5),
        "min_frame_frac": round(float(counts[:2].sum()) / total, 5),  # <= 1 frame
        "mu": round(float(np.log(fit[0])), 5) if fit else None,  # log-seconds
        "sigma": round(fit[1], 5) if fit else None,
    }


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-m",
        "--meta-path",
        required=True,
        help="protocol parquet with the relative `path` of every utterance",
    )
    p.add_argument(
        "--mask",
        default="",
        help="boolean .npy row mask over the protocol (e.g. the sentence filter); "
        "every row is read without one",
    )
    p.add_argument(
        "-d",
        "--data-dir",
        required=True,
        help="root of the `{parent}/{name}.json` alignment tree",
    )
    p.add_argument(
        "-o",
        "--out-path",
        required=True,
        help="statistics JSON; the histograms go next to it as .npz",
    )
    p.add_argument("-nj", "--num-jobs", type=int, default=32)
    p.add_argument(
        "--shard-size", type=int, default=10000, help="files per worker task"
    )
    p.add_argument(
        "-n", "--limit", type=int, default=None, help="only N files (test mode)"
    )
    args = p.parse_args()

    global PATHS
    # pandas would hold the 18M paths as Python strings: 22 s and 9.6 GB vs 4 s and 3 GB
    PATHS = pq.read_table(args.meta_path, columns=["path"]).column("path")
    check_protocol(PATHS[0].as_py())
    if args.mask:
        PATHS = PATHS.filter(pa.array(np.load(args.mask)))
    n_files = min(len(PATHS), args.limit or len(PATHS))
    data_dir = Path(args.data_dir)

    # contiguous shards keep every worker inside a few directories at a time
    jobs = [
        (i, min(i + args.shard_size, n_files), data_dir)
        for i in range(0, n_files, args.shard_size)
    ]
    vocab: Dict[str, int] = {}
    hist = np.zeros((0, N_CTX, N_BINS), dtype=np.int64)
    missing = 0
    with (
        mp.get_context("fork").Pool(args.num_jobs) as pool,
        tqdm(total=n_files, unit="file", unit_scale=True) as bar,
    ):
        for shard_vocab, keys, counts, n_shard, n_missing in pool.imap_unordered(
            count_shard, jobs
        ):
            # shard vocabulary -> global vocabulary
            remap = np.array(
                [vocab.setdefault(ph, len(vocab)) for ph in shard_vocab],
                dtype=np.int64,
            )
            if len(vocab) > len(hist):
                hist = np.concatenate(
                    [hist, np.zeros((len(vocab) - len(hist), N_CTX, N_BINS), np.int64)]
                )
            pid, rest = np.divmod(keys, N_CTX * N_BINS)
            ctx, frames = np.divmod(rest, N_BINS)
            np.add.at(hist, (remap[pid], ctx, frames), counts)
            missing += n_missing
            bar.update(n_shard)

    phonemes = list(vocab)
    print(
        f"{n_files - missing:,} files, {missing:,} without alignment | "
        f"{len(phonemes):,} phonemes, {int(hist.sum()):,} occurrences, "
        f"{hist[:, EDGE].sum() / max(hist.sum(), 1):.1%} at an edge",
        flush=True,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path.with_suffix(".npz"),
        phonemes=np.array(phonemes),
        contexts=np.array(CONTEXTS),
        counts=hist,
    )

    clean = hist[:, :EDGE].sum(axis=1)  # every context but the edge
    stats = {}
    for i in np.argsort(-clean.sum(axis=1), kind="stable"):
        s = hist_stats(clean[i])
        if s is not None:  # phonemes seen only at an edge have no statistics
            stats[phonemes[i]] = s
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
