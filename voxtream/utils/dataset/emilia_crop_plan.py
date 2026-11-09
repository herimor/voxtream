"""
Crop plan of the speaker clusters: how every training entry is cut out of the audio.

A cluster (`emilia_speaker_clusters.py`) is one training entry: its utterances, in the
order the cluster parquet lists them, are concatenated into a single recording of
exactly `--target-sec` seconds. This script decides, without touching any audio, where
each utterance is cut and where the concatenated recording ends, and writes that decision
down so that the Mimi codes, the phoneme sequence and the word alignment of an entry are
built against the same timeline.

Three things are decided per cluster:

  - What goes in. A sentence goes in whole: the pauses between its words are left as
    the speaker made them, because that rhythm is what the model has to learn, and the
    breath before a phrase and the release after it come with it. Only the silence
    around a sentence is bounded, to `--max-edge-pause` on each side, so a join carries
    a believable pause instead of a second of dead air - and the silence that is kept is
    the speakers' own room tone, never anything inserted. That budget applies wherever a
    sentence ends, so a wide gap between two sentences of the same file splits it in two
    and the middle is left out. A pause longer than `--max-inner-pause` *inside* a
    sentence is a hole in the recording rather than prosody, and drops the file.

  - The end of the recording. A cluster holds 52-87 s of audio, more than the target, and
    an entry must not end in the middle of a phrase. The last word boundary that fits
    before `--target-sec` and is followed by a pause of at least `--min-end-pause` ends
    the recording, keeping `--max-end-pause` of that pause. Utterances past that point
    are dropped and the one that straddles it is cut short.

  - The padding. Everything from the end of the speech to `--target-sec` is digital
    silence. It is encoded with the rest, so its Mimi codes are whatever the codec makes
    of silence, and the dataloader masks it out of the loss.

Output in `--out-dir`:
  - `crop_plan.parquet`, one row per used utterance: `cluster_id`, `order`, `row` (the
    protocol row), `src_start`, `src_end` (the sample range to take out of that file) and
    `dst_start` (where it lands in the entry). All sample counts at `--sample-rate`, so
    the Mimi stage never has to redo a floating point calculation.
  - `crop_clusters.parquet`, one row per entry: `cluster_id`, `plan_start`, `plan_count`
    (its slice of `crop_plan.parquet`), `speech_samples` and `num_speech_frames`.
  - `crop_plan_meta.json`, the settings the plan was built with, and the file names (not
    the paths) of the protocol and the cluster directory it was built from.

The plan is the complete record of which audio went into which training entry and in
what order: together with the protocol, the alignments, the speaker embeddings and the
audio it rebuilds a store (`emilia_train_arrays.py`, `emilia_mimi_codes.py`) without
repeating any filtering or clustering.

Reading the plan back (`--plan`) re-prints the statistics and checks it against the
cluster parquet; `load_plan()` is what the Mimi stage and the array builder import, so
every later stage repeats this exact crop:

    for cluster_id, segments, num_speech_frames in iter_clusters(plan, clusters, paths):
        audio = np.zeros(total_samples, dtype=np.float32)
        for path, src_start, src_end, dst_start in segments:
            chunk = decode(audio_path(path, audio_dir))[src_start:src_end]
            audio[dst_start : dst_start + len(chunk)] = chunk

Example:
    python -m voxtream.utils.dataset.emilia_crop_plan \
        -m emilia/metadata.parquet -c emilia/clusters \
        -d /data/alignment/emilia -o emilia/crop_plan -nj 32
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from voxtream.utils.dataset.parse_phoneme_alignment import loads
from voxtream.utils.dataset.paths import alignment_path, check_protocol
from voxtream.utils.dataset.phoneme_vocab import TERMINAL_PUNCT

# set before the workers are forked, see `parse_phoneme_alignment.PATHS`
PATHS: Optional[pa.ChunkedArray] = None
DURATIONS: Optional[np.ndarray] = None
CLUSTER_ROWS: Optional[np.ndarray] = None  # protocol row of every clustered utterance
CLUSTER_OFFSETS: Optional[np.ndarray] = None  # CSR offsets into CLUSTER_ROWS
CLUSTER_IDS: Optional[np.ndarray] = None  # cluster id of every CSR slot
SETTINGS: Optional["Settings"] = None


class Settings:
    """Everything the worker needs, in samples where it can be."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.data_dir = Path(args.data_dir)
        self.sr = args.sample_rate
        self.frame_samples = round(args.sample_rate / args.mimi_tps)
        self.total = round(args.target_sec * args.sample_rate)
        self.max_edge_pause = args.max_edge_pause
        self.max_inner_pause = args.max_inner_pause
        self.min_end_pause = round(args.min_end_pause * args.sample_rate)
        self.max_end_pause = round(args.max_end_pause * args.sample_rate)
        self.min_speech = round(args.min_speech_sec * args.sample_rate)


# ---------------------------
# One cluster
# ---------------------------


def sentence_bounds(
    words: List[dict], punctuation: Optional[List]
) -> List[Tuple[int, int]]:
    """`(first, last)` word index of every sentence of a file."""
    breaks = sorted(
        {int(i) - 1 for symbol, i in punctuation or [] if symbol in TERMINAL_PUNCT}
    )
    bounds: List[Tuple[int, int]] = []
    first = 0
    for b in breaks:
        if first <= b < len(words):
            bounds.append((first, b))
            first = b + 1
    if first < len(words):
        bounds.append((first, len(words) - 1))
    return bounds


def utterance_pieces(
    words: List[dict], punctuation: Optional[List], duration: float, cfg: Settings
) -> List[Tuple[int, int, int, int]]:
    """
    Stretches of a file to take, as `(src_start, src_end, first word, last word)`.

    A sentence keeps its own rhythm: the pauses between its words are left exactly as the
    speaker made them. What is bounded is the silence *around* a sentence - at most
    `--max-edge-pause` before its first word and the same after its last - so neither the
    edges of a file nor the gap between two of its sentences can drop a stretch of dead
    air into the entry. A gap wide enough to exceed that budget from both sides splits the
    file, and the audio in the middle is left out rather than encoded.

    An empty list means the file is unusable. A pause longer than `--max-inner-pause`
    inside a sentence is a hole in the recording rather than prosody, and nothing in the
    phoneme stream tells the model why the voice stopped, so the whole file is dropped.
    """
    bounds = sentence_bounds(words, punctuation)
    if not bounds:
        return []
    for first, last in bounds:
        for i in range(first, last):
            if words[i + 1]["start"] - words[i]["end"] > cfg.max_inner_pause:
                return []

    cuts = [0]  # sentence index that starts each piece
    for k in range(len(bounds) - 1):
        gap = words[bounds[k + 1][0]]["start"] - words[bounds[k][1]]["end"]
        if gap > 2 * cfg.max_edge_pause:
            cuts.append(k + 1)
    cuts.append(len(bounds))

    pieces: List[Tuple[int, int, int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        first, last = bounds[a][0], bounds[b - 1][1]
        start = max(0.0, words[first]["start"] - cfg.max_edge_pause)
        end = min(duration, words[last]["end"] + cfg.max_edge_pause)
        if end > start:
            pieces.append((round(start * cfg.sr), round(end * cfg.sr), first, last))
    return pieces


def plan_cluster(
    rows: np.ndarray, cfg: Settings
) -> Optional[Tuple[List[Tuple[int, int, int, int]], int]]:
    """
    Lay a cluster out on one timeline and decide where it ends.

    Returns
    -------
    Optional[Tuple[List[Tuple[int, int, int, int]], int]]
        (`(row, src_start, src_end, dst_start)` per used utterance, speech samples),
        or None when the cluster has no usable audio.
    """
    spans: List[Tuple[int, int, int, int]] = []  # (row, src_start, src_end, dst_start)
    word_ends: List[int] = []  # end of every word, in cluster samples
    word_pauses: List[int] = []  # silence that follows it, in cluster samples
    dst = 0
    prev_end: Optional[int] = None  # end of the previous word, in cluster samples

    full = False
    for row in rows:
        try:
            with open(alignment_path(PATHS[row].as_py(), cfg.data_dir), "rb") as f:
                data = loads(f.read())
        except (FileNotFoundError, KeyError, ValueError):
            continue
        words = data.get("words") or []
        if not words:
            continue

        for src_start, src_end, first, last in utterance_pieces(
            words, data.get("punctuation"), float(DURATIONS[row]), cfg
        ):
            offset = dst - src_start
            for word in words[first : last + 1]:
                start = round(word["start"] * cfg.sr) + offset
                end = round(word["end"] * cfg.sr) + offset
                if prev_end is not None:
                    word_pauses.append(max(0, start - prev_end))
                word_ends.append(end)
                prev_end = end

            spans.append((int(row), src_start, src_end, dst))
            dst += src_end - src_start
            # the kept tail of this piece is the start of the pause after its last word
            if dst >= cfg.total + cfg.max_end_pause:
                full = True
                break
        if full:
            break

    if prev_end is None:
        return None
    # the last word's pause is the tail of the last file
    word_pauses.append(max(0, dst - prev_end))

    speech_end = choose_end(np.asarray(word_ends), np.asarray(word_pauses), dst, cfg)
    if speech_end < cfg.min_speech:
        return None

    segments: List[Tuple[int, int, int, int]] = []
    for row, src_start, src_end, dst_start in spans:
        if dst_start >= speech_end:
            break
        segments.append(
            (
                row,
                src_start,
                min(src_end, src_start + (speech_end - dst_start)),
                dst_start,
            )
        )

    return segments, speech_end


def choose_end(
    ends: np.ndarray, pauses: np.ndarray, available: int, cfg: Settings
) -> int:
    """
    Last word boundary that fits in `total` and is followed by a long enough pause.

    The pause itself is kept, up to `max_end_pause`, so an entry closes on the release of
    the last word rather than on its final frame.
    """
    limit = min(cfg.total, available)
    kept = np.minimum(pauses, cfg.max_end_pause)
    fits = (ends + kept <= limit) & (ends > 0)

    good = fits & (pauses >= cfg.min_end_pause)
    if good.any():
        last = int(np.flatnonzero(good)[-1])
        return int(ends[last] + kept[last])
    if fits.any():  # no pause of that length: settle for any word boundary
        last = int(np.flatnonzero(fits)[-1])
        return int(ends[last] + kept[last])
    return limit


# ---------------------------
# Workers
# ---------------------------


def plan_shard(job: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Plan the clusters `[start, stop)` of the CSR index.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        (`[n_segments, 6]` of (cluster_id, order, row, src_start, src_end, dst_start)
        packed as int64, `[n_clusters, 3]` of (cluster_id, speech_samples, n_segments),
        clusters seen)
    """
    start, stop = job
    seg_rows: List[Tuple[int, int, int, int, int, int]] = []
    cluster_rows: List[Tuple[int, int, int]] = []

    for i in range(start, stop):
        rows = CLUSTER_ROWS[CLUSTER_OFFSETS[i] : CLUSTER_OFFSETS[i + 1]]
        planned = plan_cluster(rows, SETTINGS)
        if planned is None:
            continue
        segments, speech_end = planned
        if not segments:
            continue
        cluster_id = int(CLUSTER_IDS[i])
        for order, (row, src_start, src_end, dst_start) in enumerate(segments):
            seg_rows.append((cluster_id, order, row, src_start, src_end, dst_start))
        cluster_rows.append((cluster_id, speech_end, len(segments)))

    return (
        np.asarray(seg_rows, dtype=np.int64).reshape(-1, 6),
        np.asarray(cluster_rows, dtype=np.int64).reshape(-1, 3),
        stop - start,
    )


# ---------------------------
# Plan IO
# ---------------------------


def load_plan(out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read back `(crop_plan, crop_clusters, meta)`."""
    out_dir = Path(out_dir)
    plan = pd.read_parquet(out_dir / "crop_plan.parquet")
    clusters = pd.read_parquet(out_dir / "crop_clusters.parquet")
    with open(out_dir / "crop_plan_meta.json") as f:
        meta = json.load(f)
    return plan, clusters, meta


def iter_clusters(
    plan: pd.DataFrame, clusters: pd.DataFrame, paths: Optional[pa.ChunkedArray] = None
) -> Iterator[Tuple[int, List[Tuple[int, int, int, int]], int]]:
    """
    Yield `(cluster_id, segments, num_speech_frames)` in the order the plan stores them.

    A segment is `(row, src_start, src_end, dst_start)`, or `(path, ...)` with the
    relative protocol path when the protocol paths are given.
    """
    cols = plan[["row", "src_start", "src_end", "dst_start"]].to_numpy()
    for cluster_id, plan_start, plan_count, num_speech_frames in zip(
        clusters["cluster_id"].to_numpy(),
        clusters["plan_start"].to_numpy(),
        clusters["plan_count"].to_numpy(),
        clusters["num_speech_frames"].to_numpy(),
        strict=True,
    ):
        block = cols[plan_start : plan_start + plan_count]
        segments = [
            (
                paths[int(row)].as_py() if paths is not None else int(row),
                int(src_start),
                int(src_end),
                int(dst_start),
            )
            for row, src_start, src_end, dst_start in block
        ]
        yield int(cluster_id), segments, int(num_speech_frames)


def summarize(plan: pd.DataFrame, clusters: pd.DataFrame, meta: dict) -> None:
    sr, total = meta["sample_rate"], meta["total_samples"]
    speech = clusters["speech_samples"].to_numpy() / sr
    used = (plan["src_end"].to_numpy() - plan["src_start"].to_numpy()) / sr
    print(
        f"{len(clusters):,} entries, {len(plan):,} utterances "
        f"({len(plan) / max(len(clusters), 1):.1f} per entry)"
    )
    print(
        f"speech  mean {speech.mean():.1f}s  p01/p50/p99 "
        f"{np.quantile(speech, 0.01):.1f}/{np.median(speech):.1f}/"
        f"{np.quantile(speech, 0.99):.1f}s  of {total / sr:.0f}s"
    )
    print(
        f"padding mean {total / sr - speech.mean():.1f}s "
        f"({1 - speech.mean() * sr / total:.1%} of the frames are masked)"
    )
    print(f"utterance mean {used.mean():.1f}s, total {used.sum() / 3600:,.0f} h")


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
        "utterance",
    )
    p.add_argument(
        "-c",
        "--clusters-dir",
        default=None,
        help="directory holding `clusters.parquet` of `emilia_speaker_clusters.py` "
        "or `hifitts2_speaker_groups.py`",
    )
    p.add_argument(
        "-d",
        "--data-dir",
        default=None,
        help="root of the `{parent}/{name}.json` alignment tree",
    )
    p.add_argument("-o", "--out-dir", default=None, help="where the plan goes")
    p.add_argument(
        "--plan",
        default=None,
        help="read an existing plan directory back and re-print its statistics",
    )
    p.add_argument("--target-sec", type=float, default=80.0, help="length of an entry")
    p.add_argument(
        "--min-speech-sec",
        type=float,
        default=30.0,
        help="drop a cluster whose speech is shorter than this",
    )
    p.add_argument(
        "--max-edge-pause",
        type=float,
        default=1.0,
        help="silence kept before the first word of a sentence and after its last, seconds",
    )
    p.add_argument(
        "--max-inner-pause",
        type=float,
        default=2.0,
        help="a pause this long inside a sentence drops the whole file, seconds",
    )
    p.add_argument(
        "--min-end-pause",
        type=float,
        default=0.20,
        help="pause a word must be followed by to end an entry",
    )
    p.add_argument(
        "--max-end-pause",
        type=float,
        default=0.30,
        help="how much of that pause is kept",
    )
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--mimi-tps", type=float, default=12.5)
    p.add_argument("-nj", "--num-jobs", type=int, default=32)
    p.add_argument(
        "--shard-size", type=int, default=2000, help="clusters per worker task"
    )
    p.add_argument(
        "-n", "--limit", type=int, default=None, help="only N clusters (test mode)"
    )
    args = p.parse_args()

    if args.plan is not None:
        plan, clusters, meta = load_plan(Path(args.plan))
        summarize(plan, clusters, meta)
        return

    for name in ("meta_path", "clusters_dir", "data_dir", "out_dir"):
        if getattr(args, name) is None:
            p.error(f"--{name.replace('_', '-')} is required unless --plan is given")

    global PATHS, DURATIONS, CLUSTER_ROWS, CLUSTER_OFFSETS, CLUSTER_IDS, SETTINGS
    table = pq.read_table(args.meta_path, columns=["path", "duration"])
    check_protocol(table.column("path")[0].as_py())
    # one chunk, so a worker indexes a scattered cluster row in constant time
    PATHS = table.column("path").combine_chunks()
    DURATIONS = table.column("duration").to_numpy()
    SETTINGS = Settings(args)

    # the cluster parquet is already grouped by cluster_id and ordered within a cluster
    members = pq.read_table(
        Path(args.clusters_dir) / "clusters.parquet", columns=["cluster_id", "row"]
    )
    cluster_id = members.column("cluster_id").to_numpy()
    CLUSTER_ROWS = members.column("row").to_numpy()
    starts = np.flatnonzero(np.r_[True, cluster_id[1:] != cluster_id[:-1]])
    CLUSTER_OFFSETS = np.r_[starts, len(cluster_id)].astype(np.int64)
    CLUSTER_IDS = cluster_id[starts]
    n_clusters = min(len(CLUSTER_IDS), args.limit or len(CLUSTER_IDS))
    print(f"{n_clusters:,} clusters, {len(CLUSTER_ROWS):,} utterances", flush=True)

    jobs = [
        (i, min(i + args.shard_size, n_clusters))
        for i in range(0, n_clusters, args.shard_size)
    ]
    seg_blocks, cluster_blocks = [], []
    with (
        mp.get_context("fork").Pool(args.num_jobs) as pool,
        tqdm(total=n_clusters, unit="cluster", unit_scale=True) as bar,
    ):
        for segs, cls, n_done in pool.imap(plan_shard, jobs):
            if len(segs):
                seg_blocks.append(segs)
                cluster_blocks.append(cls)
            bar.update(n_done)

    segs = np.concatenate(seg_blocks) if seg_blocks else np.zeros((0, 6), np.int64)
    cls = (
        np.concatenate(cluster_blocks) if cluster_blocks else np.zeros((0, 3), np.int64)
    )

    plan = pd.DataFrame(
        {
            "cluster_id": segs[:, 0].astype(np.int32),
            "order": segs[:, 1].astype(np.int16),
            "row": segs[:, 2].astype(np.int64),
            "src_start": segs[:, 3].astype(np.int32),
            "src_end": segs[:, 4].astype(np.int32),
            "dst_start": segs[:, 5].astype(np.int32),
        }
    )
    if not len(cls):
        print(
            "no cluster could be planned; check the cluster parquet and the alignments"
        )
        return

    # the segments were emitted cluster by cluster in this order, so the counts the
    # workers reported turn straight into the slice of `crop_plan.parquet` of an entry
    counts = cls[:, 2]
    clusters = pd.DataFrame(
        {
            "cluster_id": cls[:, 0].astype(np.int32),
            "plan_start": np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64),
            "plan_count": counts.astype(np.int16),
            "speech_samples": cls[:, 1].astype(np.int32),
            "num_speech_frames": (cls[:, 1] // SETTINGS.frame_samples).astype(np.int16),
        }
    )

    meta = {
        "sample_rate": args.sample_rate,
        "mimi_tps": args.mimi_tps,
        "frame_samples": SETTINGS.frame_samples,
        "total_samples": SETTINGS.total,
        "num_frames": SETTINGS.total // SETTINGS.frame_samples,
        "target_sec": args.target_sec,
        "max_edge_pause": args.max_edge_pause,
        "max_inner_pause": args.max_inner_pause,
        "min_end_pause": args.min_end_pause,
        "max_end_pause": args.max_end_pause,
        "min_speech_sec": args.min_speech_sec,
        # file names only: the plan is shared without the machine it was built on
        "meta_path": Path(args.meta_path).name,
        "clusters_dir": Path(args.clusters_dir).name,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan.to_parquet(out_dir / "crop_plan.parquet", index=False)
    clusters.to_parquet(out_dir / "crop_clusters.parquet", index=False)
    with open(out_dir / "crop_plan_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    summarize(plan, clusters, meta)


if __name__ == "__main__":
    main()
