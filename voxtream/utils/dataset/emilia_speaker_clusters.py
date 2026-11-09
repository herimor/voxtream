"""
Speaker clusters of ~70 s for Emilia, built from ReDimNet2-B3 speaker embeddings.

The speaker labels of Emilia (`EN_B00000_S00000` in `EN_B00000_S00000_W000000`) come from
the diarization of a single source video and are weak: a label can mix several speakers.
This script regroups the utterances into clusters of one speaker each, of a total
duration of about `--target-duration` seconds, with the embeddings of
`emilia_speaker_encoder.py` (one per file, from its first 3 s).

Stage 1 (within speaker labels). The utterances of every label are clustered with
agglomerative clustering (cosine similarity, `--linkage` linkage, merged while the
linkage similarity is >= `--threshold`). Every group of it is cut, in file name order,
into contiguous pieces of ~`--target-duration` s, each within
[`--min-duration`, `--max-duration`] (see `split_durations`). Labels shorter than
`--min-duration` in total, groups too short to make a piece and the pieces that miss
the bounds are outliers.

Stage 2 (outliers, regardless of the labels). The outliers are pooled, cut into buckets
of at most `--max-bucket-size` utterances with recursive spherical k-means (exact
agglomerative clustering of millions of points is out of reach, and the utterances of
one speaker land in one bucket), and every bucket is clustered and cut into pieces as
in stage 1, with `--outlier-threshold` (stricter by default: two utterances of two
labels are far less likely to share a speaker than two of one label). This is repeated
`--outlier-rounds` times with new buckets, on the outliers of the previous round, so
that a speaker split by the bucket boundaries gets another chance. What remains, the
files with no embedding (unreadable files have an all-zero one) and the files outside of
the `--mask` row masks (e.g. the sentence and alignment filters) stay unclustered.

The utterances of a cluster are ordered by file name. Outputs, in `--output-dir`:

    clusters.parquet    one row per clustered utterance, ordered by cluster and name:
                        cluster_id, row (protocol row), name, speaker, duration, stage
    cluster_ids.npy     int32 [len(protocol)], the cluster of every protocol row, -1 if none
    summary.json        arguments (input paths cut to their file names) and statistics

Example (first 10k speaker labels, to tune the thresholds):
    python -m voxtream.utils.dataset.emilia_speaker_clusters \
        -m emilia/metadata.parquet -e emilia/spk_emb.npy \
        -o /tmp/clusters_test --limit 10000

Example (whole protocol, sentence and alignment filters):
    python -m voxtream.utils.dataset.emilia_speaker_clusters \
        -m emilia/metadata.parquet -e emilia/spk_emb.npy \
        --mask emilia/sentence_filter.npy emilia/alignment_filter.npy \
        --target-duration 80 -o emilia/clusters -j 40
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from voxtream.utils.dataset.paths import check_protocol, file_names

# a set of clusters: the rows of all clusters concatenated, and the offsets of the
# clusters in it (cluster k is rows[offsets[k]:offsets[k + 1]])
Clusters = Tuple[np.ndarray, np.ndarray]

# set in the parent before the workers are forked, shared copy-on-write
EMBEDDINGS: Optional[np.ndarray] = None  # memmap [len(protocol), 192] float16
DURATIONS: Optional[np.ndarray] = None  # float32 [len(protocol)]
NAME_RANK: Optional[np.ndarray] = None  # int64 [len(protocol)], rank of the file name
ARGS: Optional[argparse.Namespace] = None


# ---------------------------
# Clustering
# ---------------------------


def load_embeddings(rows: np.ndarray) -> np.ndarray:
    """Embeddings of `rows` as float32, read from the memmap in increasing row order."""
    order = np.argsort(rows, kind="stable")
    embeddings = np.empty((len(rows), EMBEDDINGS.shape[1]), dtype=np.float32)
    embeddings[order] = EMBEDDINGS[rows[order]]
    return embeddings


def agglomerate(embeddings: np.ndarray, threshold: float, method: str) -> np.ndarray:
    """
    Agglomerative clustering on cosine similarity, merging clusters while their linkage
    similarity is >= `threshold`. `embeddings` are L2-normalized.

    Returns
    -------
    np.ndarray
        The cluster label of every embedding, from 0.
    """
    if len(embeddings) == 1:
        return np.zeros(1, dtype=np.int64)
    distances = 1.0 - embeddings @ embeddings.T
    np.clip(distances, 0.0, 2.0, out=distances)
    np.fill_diagonal(distances, 0.0)
    tree = linkage(squareform(distances, checks=False), method=method)
    return fcluster(tree, t=1.0 - threshold, criterion="distance") - 1


def split_durations(
    durations: np.ndarray, target: float, min_duration: float, max_duration: float
) -> List[Tuple[int, int]]:
    """
    Cut a sequence of utterances into contiguous pieces of ~`target` seconds.

    The number of pieces `n` is the one closest to `total / target` that gives pieces
    within [min_duration, max_duration], and the cuts are the utterance boundaries
    closest to multiples of `total / n`. If there is no such `n` (`total` falls between
    `n * max_duration` and `(n + 1) * min_duration`), there are `floor(total /
    min_duration)` pieces of ~`target` seconds and the tail is left out. Pieces outside
    of the bounds, because the utterances are too long to cut closer, are dropped.

    Returns
    -------
    List[Tuple[int, int]]
        The [start, end) utterance indices of every piece.
    """
    total = float(durations.sum())
    if total < min_duration:
        return []

    n_min = math.ceil(total / max_duration)
    n_max = math.floor(total / min_duration)
    if n_min <= n_max:
        n = min(max(round(total / target), n_min), n_max)
        cuts = total / n * np.arange(1, n)
        end = total
    else:
        n = n_max
        cuts = target * np.arange(1, n)
        end = target * n

    # boundary k sits after utterance k - 1, at time ends[k]
    ends = np.concatenate([[0.0], np.cumsum(durations)])
    bounds = np.abs(ends[None, :] - np.append(cuts, end)[:, None]).argmin(axis=1)
    bounds = np.concatenate([[0], bounds])

    pieces = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop > start and min_duration <= ends[stop] - ends[start] <= max_duration:
            pieces.append((int(start), int(stop)))
    return pieces


def cluster_rows(
    rows: np.ndarray, threshold: float
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Cluster `rows` and cut every group, in file name order, into pieces.

    Returns
    -------
    Tuple[List[np.ndarray], np.ndarray]
        The rows of every piece, ordered by file name, and the rows left out.
    """
    args = ARGS
    durations = DURATIONS[rows]
    if durations.sum() < args.min_duration:
        return [], rows

    embeddings = load_embeddings(rows)
    labels = agglomerate(embeddings, threshold, args.linkage)

    pieces, outliers = [], []
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    for group in groups:
        group = group[np.argsort(NAME_RANK[rows[group]])]
        kept = np.zeros(len(group), dtype=bool)
        for start, stop in split_durations(
            durations[group], args.target_duration, args.min_duration, args.max_duration
        ):
            pieces.append(rows[group[start:stop]])
            kept[start:stop] = True
        outliers.append(rows[group[~kept]])
    return pieces, np.concatenate(outliers)


def pack(pieces: List[np.ndarray]) -> Clusters:
    offsets = np.cumsum([0] + [len(piece) for piece in pieces])
    rows = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    return rows, offsets


def run_speakers(task: Tuple[np.ndarray, np.ndarray]) -> Tuple[Clusters, np.ndarray]:
    """Stage 1 on a range of speaker labels: the rows (name-ordered) and label starts."""
    rows, starts = task
    pieces, outliers = [], []
    for start, stop in zip(starts[:-1], starts[1:]):
        speaker_pieces, speaker_outliers = cluster_rows(
            rows[start:stop], ARGS.threshold
        )
        pieces.extend(speaker_pieces)
        outliers.append(speaker_outliers)
    return pack(pieces), np.concatenate(outliers)


def run_bucket(rows: np.ndarray) -> Tuple[Clusters, np.ndarray]:
    """Stage 2 on a bucket of outliers."""
    pieces, outliers = cluster_rows(rows, ARGS.outlier_threshold)
    return pack(pieces), outliers


def init_worker() -> None:
    # the parallelism is across processes, one BLAS thread each
    threadpool_limits(1)


# ---------------------------
# Buckets
# ---------------------------


def assign(
    embeddings: np.ndarray, centroids: np.ndarray, chunk: int = 65536
) -> np.ndarray:
    """Index of the most similar centroid of every embedding."""
    labels = np.empty(len(embeddings), dtype=np.int64)
    for start in range(0, len(embeddings), chunk):
        labels[start : start + chunk] = (
            embeddings[start : start + chunk] @ centroids.T
        ).argmax(axis=1)
    return labels


def spherical_kmeans(
    embeddings: np.ndarray,
    k: int,
    rng: np.random.Generator,
    iterations: int = 20,
    sample_per_centroid: int = 256,
) -> np.ndarray:
    """
    Spherical k-means, fit on a sample of `sample_per_centroid * k` embeddings.

    Returns
    -------
    np.ndarray
        The centroid of every embedding.
    """
    sample_size = min(len(embeddings), sample_per_centroid * k)
    sample = embeddings[rng.choice(len(embeddings), sample_size, replace=False)]
    centroids = sample[rng.choice(sample_size, k, replace=False)]
    for _ in range(iterations):
        labels = assign(sample, centroids)
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, sample)
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        empty = norms[:, 0] == 0
        # an empty centroid is re-seeded on a random sample
        sums[empty] = sample[rng.choice(sample_size, int(empty.sum()), replace=False)]
        norms[empty] = 1.0
        centroids = sums / norms
    return assign(embeddings, centroids)


def make_buckets(
    rows: np.ndarray, bucket_size: int, max_bucket_size: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """
    Cut `rows` into buckets of similar embeddings, of ~`bucket_size` rows and at most
    `max_bucket_size`: k-means, and k-means again inside of the buckets that are too
    large (k-means buckets are uneven).
    """
    embeddings = load_embeddings(rows)
    buckets, pending = [], [np.arange(len(rows))]
    progress = tqdm(desc="buckets", unit="row", unit_scale=True, total=len(rows))
    while pending:
        indices = pending.pop()
        if len(indices) <= max_bucket_size:
            buckets.append(rows[indices])
            progress.update(len(indices))
            continue
        k = max(2, round(len(indices) / bucket_size))
        labels = spherical_kmeans(embeddings[indices], k, rng)
        order = np.argsort(labels, kind="stable")
        splits = np.split(indices[order], np.flatnonzero(np.diff(labels[order])) + 1)
        if len(splits) == 1:
            # identical embeddings, k-means cannot split them: arbitrary chunks
            splits = np.array_split(indices, math.ceil(len(indices) / bucket_size))
        pending.extend(splits)
    progress.close()
    # the largest buckets first, so that the pool does not end on one of them
    return sorted(buckets, key=len, reverse=True)


# ---------------------------
# Main
# ---------------------------


def speaker_tasks(starts: np.ndarray, rows_per_task: int) -> Iterator[Tuple[int, int]]:
    """Ranges [first, last) of speaker labels with ~`rows_per_task` rows each."""
    first = 0
    while first < len(starts) - 1:
        last = int(np.searchsorted(starts, starts[first] + rows_per_task, side="right"))
        last = min(max(last - 1, first + 1), len(starts) - 1)
        yield first, last
        first = last


def run_pool(
    func, tasks: list, num_workers: int, desc: str
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Run `func` over `tasks`. Returns the rows of every cluster and the outliers."""
    pieces, outliers = [], []
    ctx = mp.get_context("fork")
    with ctx.Pool(num_workers, initializer=init_worker) as pool:
        for (rows, offsets), task_outliers in tqdm(
            pool.imap(func, tasks), total=len(tasks), desc=desc
        ):
            pieces.extend(np.split(rows, offsets[1:-1]) if len(offsets) > 1 else [])
            outliers.append(task_outliers)
    outliers = np.concatenate(outliers) if outliers else np.empty(0, dtype=np.int64)
    return pieces, outliers


def describe(pieces: List[np.ndarray]) -> dict:
    if not pieces:
        return {"clusters": 0}
    durations = np.array([DURATIONS[piece].sum() for piece in pieces])
    sizes = np.array([len(piece) for piece in pieces])
    return {
        "clusters": len(pieces),
        "utterances": int(sizes.sum()),
        "hours": float(durations.sum() / 3600),
        "duration_mean": float(durations.mean()),
        "duration_p01_p50_p99": np.percentile(durations, [1, 50, 99]).round(1).tolist(),
        "utterances_mean": float(sizes.mean()),
    }


def main():
    global EMBEDDINGS, DURATIONS, NAME_RANK, ARGS

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--meta-path",
        type=str,
        required=True,
        help="Protocol parquet with the relative `path` and `duration` of every file",
    )
    parser.add_argument(
        "-e",
        "--embeddings-path",
        type=str,
        required=True,
        help="Speaker embeddings [len(protocol), dim], L2-normalized, row-aligned with the protocol",
    )
    parser.add_argument(
        "--mask",
        type=str,
        nargs="*",
        default=[],
        help="Boolean .npy row masks over the protocol (combined with AND), "
        "the rows outside of them are never clustered",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument("-j", "--num-workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--target-duration",
        type=float,
        default=70.0,
        help="Target cluster duration (s)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=52.5, help="Minimum cluster duration (s)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=87.5, help="Maximum cluster duration (s)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Cosine similarity threshold of the clustering within speaker labels",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold of the clustering of the outliers",
    )
    parser.add_argument(
        "--linkage",
        choices=["average", "complete", "single", "weighted"],
        default="average",
        help="Agglomerative clustering linkage",
    )
    parser.add_argument(
        "--outlier-rounds", type=int, default=2, help="Rounds of outlier clustering"
    )
    parser.add_argument(
        "--bucket-size", type=int, default=4000, help="Target outlier bucket size"
    )
    parser.add_argument(
        "--max-bucket-size", type=int, default=8000, help="Maximum outlier bucket size"
    )
    parser.add_argument(
        "--rows-per-task", type=int, default=20000, help="Rows per stage 1 task"
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Cluster only the first N speaker labels (in name order), to tune the thresholds",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of the k-means")
    args = parser.parse_args()
    assert args.min_duration <= args.target_duration <= args.max_duration
    ARGS = args

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    meta = pd.read_parquet(args.meta_path, columns=["path", "duration"])
    check_protocol(meta["path"].iloc[0])
    names = meta["path"].str.rsplit("/", n=1).str[-1]
    speakers = names.str.rsplit("_", n=1).str[0].to_numpy()
    names = names.to_numpy()
    DURATIONS = meta["duration"].to_numpy(dtype=np.float32)
    num_protocol = len(meta)
    del meta

    # the speaker label is a prefix of the name: in name order the labels are contiguous
    by_name = np.argsort(names, kind="stable")
    NAME_RANK = np.empty(num_protocol, dtype=np.int64)
    NAME_RANK[by_name] = np.arange(num_protocol)
    sorted_speakers = speakers[by_name]
    starts = np.flatnonzero(sorted_speakers[1:] != sorted_speakers[:-1]) + 1
    starts = np.concatenate([[0], starts, [num_protocol]])
    if args.limit:
        starts = starts[: args.limit + 1]
    print(
        f"{num_protocol:,} files, {len(starts) - 1:,} speaker labels | "
        f"loaded in {time.perf_counter() - t0:.0f} s",
        flush=True,
    )

    EMBEDDINGS = np.load(args.embeddings_path, mmap_mode="r")
    assert (
        len(EMBEDDINGS) == num_protocol
    ), "Embeddings are not aligned with the protocol"
    # unreadable files have an all-zero embedding, they are never clustered
    selected = by_name[: starts[-1]]
    norms = np.empty(len(selected), dtype=np.float32)
    for start in range(0, len(selected), 1_000_000):
        chunk = load_embeddings(selected[start : start + 1_000_000])
        norms[start : start + 1_000_000] = np.linalg.norm(chunk, axis=1)
    valid = norms > 0.5
    num_no_embedding = int((~valid).sum())
    mask = np.ones(num_protocol, dtype=bool)
    for path in args.mask:
        path_mask = np.load(path)
        assert (
            path_mask.shape == (num_protocol,) and path_mask.dtype == bool
        ), f"{path} is not a boolean mask over the protocol"
        mask &= path_mask
    num_masked = int((~mask[selected]).sum())
    valid &= mask[selected]
    num_invalid = int((~valid).sum())
    if num_invalid:
        keep = np.concatenate([[0], np.cumsum(valid)])
        selected, starts = selected[valid], np.unique(keep[starts])
    print(
        f"{num_no_embedding:,} files without embedding, {num_masked:,} masked out",
        flush=True,
    )

    # Stage 1: within speaker labels
    tasks = [
        (
            selected[starts[first] : starts[last]],
            starts[first : last + 1] - starts[first],
        )
        for first, last in speaker_tasks(starts, args.rows_per_task)
    ]
    t = time.perf_counter()
    stage1, outliers = run_pool(run_speakers, tasks, args.num_workers, "stage 1")
    del tasks
    stats = {"stage1": describe(stage1)}
    print(
        f"stage 1: {stats['stage1']} | {len(outliers):,} outliers "
        f"({DURATIONS[outliers].sum() / 3600:,.0f} h) | {time.perf_counter() - t:.0f} s",
        flush=True,
    )

    # Stage 2: outliers, regardless of the speaker labels
    rng = np.random.default_rng(args.seed)
    stage2 = []
    for outlier_round in range(args.outlier_rounds):
        if DURATIONS[outliers].sum() < args.min_duration:
            break
        t = time.perf_counter()
        buckets = make_buckets(outliers, args.bucket_size, args.max_bucket_size, rng)
        pieces, outliers = run_pool(
            run_bucket, buckets, args.num_workers, f"stage 2, round {outlier_round + 1}"
        )
        stage2.extend(pieces)
        stats[f"stage2_round{outlier_round + 1}"] = describe(pieces)
        print(
            f"stage 2, round {outlier_round + 1}: {len(buckets):,} buckets "
            f"(max {len(buckets[0]):,}) | {describe(pieces)} | {len(outliers):,} outliers "
            f"({DURATIONS[outliers].sum() / 3600:,.0f} h) | {time.perf_counter() - t:.0f} s",
            flush=True,
        )

    # Outputs
    clusters = stage1 + stage2
    rows, offsets = pack(clusters)
    cluster_ids = np.full(num_protocol, -1, dtype=np.int32)
    cluster_ids[rows] = np.repeat(
        np.arange(len(clusters), dtype=np.int32), np.diff(offsets)
    )
    np.save(output_dir / "cluster_ids.npy", cluster_ids)

    table = pd.DataFrame(
        {
            "cluster_id": cluster_ids[rows],
            "row": rows,
            "name": names[rows],
            "speaker": speakers[rows],
            "duration": DURATIONS[rows],
            "stage": np.where(cluster_ids[rows] < len(stage1), 1, 2).astype(np.int8),
        }
    )
    table.to_parquet(output_dir / "clusters.parquet", index=False)

    num_selected = int(starts[-1] - starts[0]) + num_invalid
    hours_selected = DURATIONS[by_name[:num_selected]].sum() / 3600
    labels_per_cluster = table.groupby("cluster_id")["speaker"].nunique()
    stats["all"] = describe(clusters)
    stats["all"].update(
        {
            "files": num_selected,
            "files_without_embedding": num_no_embedding,
            "files_masked": num_masked,
            "hours_total": float(hours_selected),
            "hours_clustered_fraction": float(
                stats["all"].get("hours", 0) / hours_selected
            ),
            "speaker_labels_per_cluster_mean": float(labels_per_cluster.mean()),
            "unclustered_files": len(outliers) + num_invalid,
        }
    )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "args": file_names(
                    vars(args), ("meta_path", "embeddings_path", "mask", "output_dir")
                ),
                "stats": stats,
            },
            f,
            indent=2,
        )
    print(json.dumps(stats["all"], indent=2))
    print(f"Saved {output_dir} | {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
