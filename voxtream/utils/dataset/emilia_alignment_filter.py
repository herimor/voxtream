"""
Utterance-level filter of the Clap-IPA phoneme alignments (`phoneme_align.py`).

Every utterance gets four alignment-quality features, computed from its phoneme alignment
`{data_dir}/{parent}/{name}.json`, its Seamless word alignment `{name}.seamless.json` (see
`paths.py`) and the per-phoneme duration model of `parse_phoneme_alignment.py`:
  - stretch: share of phonemes far longer than expected at the speaking rate of the
    utterance, i.e. log-duration z-score minus its utterance median above `Z_MAX`,
  - squash: share of phonemes squashed to a single aligner frame,
  - rate: phonemes per second of audio, low when the audio holds speech the transcript
    does not or long pauses,
  - seamless_off: share of words whose centre is more than `OFF_MAX` away from the
    Seamless alignment of the same word, an independent aligner.
The first two skip the phonemes at an edge (`phoneme_contexts`), whose durations hold
pauses or lose frames to the [SIL] tokens.

The features are combined Fisher-style: each one becomes the share of the utterances that
are at least as bad on it (a tail probability), and the score is the sum of their negative
logs, so an utterance is dropped when it is extreme on one feature or bad on several. The
`--drop` share of the worst-scoring utterances is filtered out.

Checked with whisper-large-v3 against the protocol transcript on 400 dropped and 400 kept
utterances of a 300k-file run at `--drop 0.02`: WER mean 0.47 / median 0.26 / 58% above
0.2 for the dropped, 0.08 / 0.05 / 10% for the kept. Every feature pulls its weight: the
utterances dropped mostly for one of them have a mean WER of 0.82 (rate), 0.50
(seamless_off), 0.38 (stretch) and 0.24 (squash).

Example:
    python -m voxtream.utils.dataset.emilia_alignment_filter \
        -m emilia/metadata.parquet --mask emilia/sentence_filter.npy \
        -s emilia/phoneme_stats.npz -d /data/alignment/emilia -o emilia -nj 32

Output, row-aligned with the protocol:
  - `alignment_scores.parquet`: the features and the score (NaN outside `--mask` and
    for the files without an alignment),
  - `alignment_filter.npy`: boolean keep mask. False outside `--mask`, for the files
    without an alignment and for the dropped ones, so it replaces `--mask`.
`--from-scores` rebuilds the mask from an existing `alignment_scores.parquet` with another
`--drop`, without reading the tree.
"""

import argparse
import math
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from voxtream.utils.dataset.parse_phoneme_alignment import (
    EDGE,
    FRAME,
    hist_stats,
    loads,
    phoneme_contexts,
)
from voxtream.utils.dataset.paths import (
    SEAMLESS_SUFFIX,
    alignment_path,
    check_protocol,
)
from voxtream.utils.dataset.phoneme_align import strip_punctuation

Z_MAX = 3.0  # stretch threshold on the tempo-normalised log-duration z-score
OFF_MAX = 0.2  # seconds between the Clap-IPA and the Seamless centre of a word
MIN_PHONEMES = 10  # below that, stretch and squash are too noisy and stay NaN
MIN_COUNT = (
    1000  # rarer phonemes back off to their unstressed form, then to all phonemes
)
STRESS = str.maketrans("", "", "ˈˌ")

# feature -> True if higher is worse
FEATURES = {"stretch": True, "squash": True, "rate": False, "seamless_off": True}


# ---------------------------
# Duration model
# ---------------------------


def duration_model(stats_path: Path) -> Tuple[Dict[str, Tuple[float, float]], tuple]:
    """
    Per-phoneme (mu, sigma) of the log-durations, from the histograms of
    `parse_phoneme_alignment.py`.

    Returns
    -------
    Tuple[Dict[str, Tuple[float, float]], tuple]
        (model of every phoneme of the histograms, model of all phonemes together for
        the phonemes never seen)
    """
    z = np.load(stats_path)
    clean = z["counts"][:, :EDGE].sum(axis=1)  # [P, bins]

    def fit(counts: np.ndarray) -> Optional[Tuple[float, float]]:
        s = hist_stats(counts) if counts.sum() >= MIN_COUNT else None
        return (s["mu"], s["sigma"]) if s and s["mu"] is not None else None

    fallback = fit(clean.sum(axis=0))
    phonemes = [str(p) for p in z["phonemes"]]
    unstressed: Dict[str, np.ndarray] = {}
    for p, counts in zip(phonemes, clean):
        key = p.translate(STRESS)
        unstressed[key] = unstressed.get(key, 0) + counts
    model = {}
    for p, counts in zip(phonemes, clean):
        model[p] = fit(counts) or fit(unstressed[p.translate(STRESS)]) or fallback
    return model, fallback


# ---------------------------
# Features
# ---------------------------


def seamless_offset(words: List[dict], seamless: List[dict]) -> float:
    """Share of the words whose centre is more than `OFF_MAX` from the Seamless one."""
    # the phoneme alignment holds the Seamless words in order, minus the unaligned ones
    j, n_off, n = 0, 0, 0
    for word in words:
        while (
            j < len(seamless) and strip_punctuation(seamless[j]["word"]) != word["text"]
        ):
            j += 1
        if j == len(seamless):
            break
        ref = seamless[j]
        # centres rather than starts: Clap-IPA puts the onsets ~1 frame early
        off = 0.5 * abs(word["start"] + word["end"] - ref["start"] - ref["end"])
        n_off += off > OFF_MAX
        n += 1
        j += 1
    return n_off / n if n else math.nan


def utterance_features(
    words: List[dict],
    seamless: List[dict],
    duration: float,
    model: Dict[str, Tuple[float, float]],
    fallback: Tuple[float, float],
) -> Tuple[int, float, float, float, float]:
    """(phonemes off the edges, stretch, squash, rate, seamless_off) of one utterance."""
    zs, n_all, n_one = [], 0, 0
    for phoneme, ctx, frames in phoneme_contexts(words):
        n_all += 1
        if ctx == EDGE:
            continue
        mu, sigma = model.get(phoneme, fallback)
        # zero frames only come from clamping at the end of the audio
        zs.append((math.log(max(frames, 0.5) * FRAME) - mu) / sigma)
        n_one += frames <= 1

    n = len(zs)
    stretch = squash = math.nan
    if n >= MIN_PHONEMES:
        z = np.asarray(zs)
        # the utterance median is its speaking rate: fast and slow speakers are not
        # outliers, phonemes far off their own speaker's tempo are
        stretch = float((z - np.median(z) > Z_MAX).mean())
        squash = n_one / n
    rate = n_all / duration if duration > 0 else math.nan
    return n, stretch, squash, rate, seamless_offset(words, seamless)


# ---------------------------
# Workers
# ---------------------------

# set before the workers are forked, see `parse_phoneme_alignment.PATHS`
PATHS: Optional[pa.ChunkedArray] = None
DURATIONS: Optional[np.ndarray] = None
ROWS: Optional[np.ndarray] = None  # protocol row of every entry of PATHS
MODEL: Optional[Tuple[Dict[str, Tuple[float, float]], tuple]] = None


def score_shard(job: Tuple[int, int, Path]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Features of the files `PATHS[start:stop]`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        (protocol rows of the aligned files, their [n, 5] features, number of files)
    """
    start, stop, data_dir = job
    rows, feats = [], []
    for k, path in enumerate(PATHS.slice(start, stop - start).to_pylist()):
        try:
            with open(alignment_path(path, data_dir), "rb") as f:
                words = loads(f.read())["words"]
            with open(alignment_path(path, data_dir, SEAMLESS_SUFFIX), "rb") as f:
                seamless = loads(f.read())["words"]
        except FileNotFoundError:
            continue
        if not words:
            continue
        rows.append(ROWS[start + k])
        feats.append(utterance_features(words, seamless, DURATIONS[start + k], *MODEL))
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(feats, dtype=np.float64),
        stop - start,
    )


# ---------------------------
# Score and mask
# ---------------------------


def tail_probability(x: np.ndarray, higher_is_worse: bool) -> np.ndarray:
    """Share of the finite values at least as bad as each value; 1 for NaN."""
    ok = np.isfinite(x)
    ref = np.sort(x[ok] if higher_is_worse else -x[ok])
    v = x if higher_is_worse else -x
    p = np.ones_like(x)
    p[ok] = 1.0 - np.searchsorted(ref, v[ok], side="left") / max(len(ref), 1)
    return p


def score(df: pd.DataFrame, features: Dict[str, bool] = FEATURES) -> np.ndarray:
    """Fisher combination of the tail probabilities of the features (higher is worse)."""
    out = np.zeros(len(df))
    for name, higher_is_worse in features.items():
        out -= np.log(tail_probability(df[name].to_numpy(), higher_is_worse))
    return out


def build_mask(df: pd.DataFrame, drop: float) -> np.ndarray:
    """Keep the scored utterances except the `drop` share with the highest score."""
    s = df["score"].to_numpy()
    scored = np.isfinite(s)
    if not scored.any():
        return scored
    return scored & (s <= np.quantile(s[scored], 1.0 - drop))


def summarize(
    df: pd.DataFrame, keep: np.ndarray, features: Dict[str, bool] = FEATURES
) -> None:
    scored = np.isfinite(df["score"].to_numpy())
    dropped = scored & ~keep
    print(
        f"{scored.sum():,} scored, {dropped.sum():,} dropped "
        f"({dropped.sum() / max(scored.sum(), 1):.2%}), {keep.sum():,} kept"
    )
    q = [0.5, 0.99]
    for name in features:
        kept, drop = df[name][keep], df[name][dropped]
        print(
            f"  {name:13s} kept p50/p99 {kept.quantile(q[0]):.3f}/{kept.quantile(q[1]):.3f}"
            f" | dropped p50 {drop.median():.3f}"
        )


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
        default=None,
        help="protocol parquet with the relative `path` and `duration` of every "
        "utterance (not needed with --from-scores)",
    )
    p.add_argument(
        "--mask",
        default="",
        help="boolean .npy row mask over the protocol (e.g. the sentence filter); "
        "every row is scored without one",
    )
    p.add_argument(
        "-s",
        "--stats-path",
        default=None,
        help="duration histograms of `parse_phoneme_alignment.py` "
        "(not needed with --from-scores)",
    )
    p.add_argument(
        "-d",
        "--data-dir",
        default=None,
        help="root of the `{parent}/{name}.json` alignment tree "
        "(not needed with --from-scores)",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        required=True,
        help="where `alignment_scores.parquet` and `alignment_filter.npy` go",
    )
    p.add_argument(
        "--drop", type=float, default=0.02, help="share of the scored files to drop"
    )
    p.add_argument(
        "--from-scores",
        action="store_true",
        help="rebuild the mask from the existing alignment_scores.parquet",
    )
    p.add_argument("-nj", "--num-jobs", type=int, default=32)
    p.add_argument(
        "--shard-size", type=int, default=10000, help="files per worker task"
    )
    p.add_argument(
        "-n", "--limit", type=int, default=None, help="only N files (test mode)"
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    scores_path = out_dir / "alignment_scores.parquet"
    if args.from_scores:
        df = pd.read_parquet(scores_path)
        keep = build_mask(df, args.drop)
        summarize(df, keep)
        np.save(out_dir / "alignment_filter.npy", keep)
        return

    for name in ("meta_path", "stats_path", "data_dir"):
        if getattr(args, name) is None:
            p.error(f"--{name.replace('_', '-')} is required unless --from-scores")

    global PATHS, DURATIONS, ROWS, MODEL
    table = pq.read_table(args.meta_path, columns=["path", "duration"])
    check_protocol(table.column("path")[0].as_py())
    n_rows = len(table)
    ROWS = np.arange(n_rows)
    if args.mask:
        mask = np.load(args.mask)
        table = table.filter(pa.array(mask))
        ROWS = ROWS[mask]
    PATHS = table.column("path")
    DURATIONS = table.column("duration").to_numpy()
    MODEL = duration_model(Path(args.stats_path))
    n_files = min(len(PATHS), args.limit or len(PATHS))
    data_dir = Path(args.data_dir)

    jobs = [
        (i, min(i + args.shard_size, n_files), data_dir)
        for i in range(0, n_files, args.shard_size)
    ]
    feats = np.full((n_rows, 5), np.nan)
    with (
        mp.get_context("fork").Pool(args.num_jobs) as pool,
        tqdm(total=n_files, unit="file", unit_scale=True) as bar,
    ):
        for rows, shard, n_shard in pool.imap_unordered(score_shard, jobs):
            if len(rows):
                feats[rows] = shard
            bar.update(n_shard)

    df = pd.DataFrame(
        {
            "n_phonemes": feats[:, 0],
            **{name: feats[:, i + 1] for i, name in enumerate(FEATURES)},
        }
    ).astype(np.float32)
    scored = np.isfinite(df["n_phonemes"].to_numpy())
    print(f"{scored.sum():,} of {n_files:,} files have an alignment", flush=True)
    df["score"] = np.nan
    df.loc[scored, "score"] = score(df[scored])
    keep = build_mask(df, args.drop)
    summarize(df, keep)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(scores_path)
    np.save(out_dir / "alignment_filter.npy", keep)


if __name__ == "__main__":
    main()
