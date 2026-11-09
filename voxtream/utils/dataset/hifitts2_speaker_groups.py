"""
Training entries of HiFiTTS-2, from its true speaker and session labels.

HiFiTTS-2 is LibriVox read by named speakers, so unlike Emilia it needs no clustering:
the labels are in the protocol path, `{speaker}/{session}/{speaker}_{session}_{book}_...`,
and they are right. `emilia_speaker_clusters.py` exists because Emilia's labels come from
the diarization of a single video and a label can mix speakers; embedding-clustering this
dataset would only add a way to be wrong about something it already states.

What carries over from that script is the cutting. An entry is a contiguous run of
utterances in file name order totalling about `--target-duration` seconds, and
`split_durations` picks how many pieces a run makes and where the cuts fall. This applies
it twice.

File name order here is not a string sort. The index of an utterance inside its chapter is
the unpadded tail of the name, so sorting the names as text reads 0, 1, 10, 11, ... 19, 2
and fills an entry with utterances from all over the chapter; `name_order` compares that
tail as a number, which is what the padded counter in an Emilia name does for free. It
takes the longest run of consecutive utterances in a session from 20 to 65.

Stage 1, one session. The utterances of every `(speaker, session)` pair are cut into
pieces. A session is one sitting with one book, so an entry built inside it holds one
voice, one room and one microphone, which is what a long-form entry should be. The key is
the pair and not the session alone: the session directories are book ids, the same book is
read by several speakers, and grouping by it would mix them.

Stage 2, several sessions of one speaker. Sessions shorter than `--min-duration` and the
tails stage 1 could not place are pooled per speaker, left in file name order so the
sessions stay contiguous, and cut the same way. Such an entry crosses a session boundary,
which changes the room mid-entry but not the voice. Neither stage ever crosses a speaker.

Measured on the 15.6k h protocol behind the sentence filter, at 80 s: stage 1 places
14,828 h in 668k entries and leaves 783 h, stage 2 places 657 h more, and 126 h in
speakers with too little audio to fill one entry are left out. That is 99.2% of the hours
in 698k entries of 68-92 s, against 90.7% for Emilia's clustering, and 1.2% of the entries
span more than one session.

The outputs are what `emilia_crop_plan.py` reads, so everything downstream - the crop
plan, the training arrays and the Mimi codes - is the Emilia pipeline unchanged:

    clusters.parquet   one row per used utterance, ordered by entry and file name:
                       cluster_id, row (protocol row), name, speaker, session, duration,
                       stage (1 single session, 2 several)
    cluster_ids.npy    int32 [len(protocol)], the entry of every protocol row, -1 if none
    summary.json       arguments (input paths cut to their file names) and statistics

Example:
    python -m voxtream.utils.dataset.hifitts2_speaker_groups \
        -m hifitts2/metadata.parquet \
        --mask hifitts2/sentence_filter.npy hifitts2/alignment_filter.npy \
        -e hifitts2/spk_emb.npy --target-duration 80 -o hifitts2/clusters
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from voxtream.utils.dataset.emilia_speaker_clusters import split_durations
from voxtream.utils.dataset.paths import check_protocol, file_names


def path_labels(paths: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(speaker, session, name)` of every file, read off `{speaker}/{session}/{name}`."""
    parts = paths.str.split("/", expand=True)
    if parts.shape[1] != 3 or parts.isna().any().any():
        raise SystemExit(
            "HiFiTTS-2 protocol paths must be `{speaker}/{session}/{name}`"
        )
    return parts[0].to_numpy(), parts[1].to_numpy(), parts[2].to_numpy()


def name_order(names: np.ndarray) -> np.ndarray:
    """
    The order the utterances were recorded in, as their names give it.

    The index of an utterance inside its chapter is the tail of the name and is *not*
    zero-padded (it runs 0..405), so a plain string sort reads 0, 1, 10, 11, ... 19, 2,
    20 and builds entries out of utterances scattered through the chapter. Emilia's names
    carry a padded counter (`W000000`) and sort into recording order by accident; here
    the tail has to be compared as the number it is.

    Everything before the tail is compared as text, which keeps the files of a speaker
    and of a session contiguous: a name begins with `{speaker}_{session}_`, and those are
    string prefixes, so all the names sharing one sort together.
    """
    series = pd.Series(names)
    head = series.str.rsplit("_", n=1).str[0]
    tail = pd.to_numeric(series.str.rsplit("_", n=1).str[1], errors="coerce")
    # a name that does not end in a number is compared whole, and sorts before the ones
    # of its chapter that do
    head = head.where(tail.notna(), series)
    return np.lexsort((tail.fillna(-1).to_numpy(), pd.factorize(head, sort=True)[0]))


def group_bounds(keys: np.ndarray) -> np.ndarray:
    """Start of every run of equal keys, plus the end; `keys` is already grouped."""
    if not len(keys):
        return np.zeros(1, dtype=np.int64)
    changes = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    return np.concatenate([[0], changes, [len(keys)]])


def cut_runs(
    rows: np.ndarray, keys: np.ndarray, durations: np.ndarray, args: argparse.Namespace
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Cut every run of equal `keys` into entries.

    `rows` are protocol rows already ordered by key and then by file name, so a run is a
    contiguous slice and the utterances inside it are in the order they were recorded.

    Returns
    -------
    Tuple[List[np.ndarray], np.ndarray]
        The rows of every entry, and the rows no entry took.
    """
    pieces: List[np.ndarray] = []
    leftover: List[np.ndarray] = []
    bounds = group_bounds(keys)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        run = rows[start:stop]
        run_durations = durations[run]
        taken = np.zeros(len(run), dtype=bool)
        for first, last in split_durations(
            run_durations, args.target_duration, args.min_duration, args.max_duration
        ):
            pieces.append(run[first:last])
            taken[first:last] = True
        leftover.append(run[~taken])
    rest = np.concatenate(leftover) if leftover else np.empty(0, dtype=np.int64)
    return pieces, rest


def describe(pieces: List[np.ndarray], durations: np.ndarray) -> dict:
    if not pieces:
        return {"clusters": 0}
    totals = np.array([durations[piece].sum() for piece in pieces])
    sizes = np.array([len(piece) for piece in pieces])
    return {
        "clusters": len(pieces),
        "utterances": int(sizes.sum()),
        "hours": float(totals.sum() / 3600),
        "duration_mean": float(totals.mean()),
        "duration_p01_p50_p99": np.percentile(totals, [1, 50, 99]).round(1).tolist(),
        "utterances_mean": float(sizes.mean()),
    }


def load_mask(paths: List[str], num_rows: int) -> np.ndarray:
    mask = np.ones(num_rows, dtype=bool)
    for path in paths:
        row_mask = np.load(path)
        assert (
            row_mask.shape == (num_rows,) and row_mask.dtype == bool
        ), f"{path} is not a boolean mask over the protocol"
        mask &= row_mask
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--meta-path",
        required=True,
        help="protocol parquet with the relative `path` and `duration` of every file",
    )
    parser.add_argument(
        "--mask",
        nargs="*",
        default=[],
        help="boolean .npy row masks over the protocol (combined with AND); the rows "
        "outside of them never enter an entry",
    )
    parser.add_argument(
        "-e",
        "--embeddings-path",
        default=None,
        help="optional speaker embeddings row-aligned with the protocol; when given, the "
        "files whose embedding is all zero - the ones the speaker encoder could not "
        "read - are left out, as the array builder would drop them anyway",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, help="where the entries go"
    )
    parser.add_argument(
        "--target-duration", type=float, default=80.0, help="target entry duration (s)"
    )
    parser.add_argument(
        "--min-duration", type=float, default=68.0, help="minimum entry duration (s)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=92.0, help="maximum entry duration (s)"
    )
    parser.add_argument(
        "--single-session-only",
        action="store_true",
        help="skip stage 2, so every entry comes from one session",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="only the first N speakers (in name order), to try the settings out",
    )
    args = parser.parse_args()
    assert args.min_duration <= args.target_duration <= args.max_duration

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    table = pq.read_table(args.meta_path, columns=["path", "duration"])
    check_protocol(table.column("path")[0].as_py())
    speakers, sessions, names = path_labels(table.column("path").to_pandas())
    durations = table.column("duration").to_numpy().astype(np.float32)
    num_protocol = len(durations)
    del table

    mask = load_mask(args.mask, num_protocol)
    num_masked = int((~mask).sum())
    num_no_embedding = 0
    if args.embeddings_path:
        embeddings = np.load(args.embeddings_path, mmap_mode="r")
        assert (
            len(embeddings) == num_protocol
        ), "embeddings are not aligned with the protocol"
        usable = np.zeros(num_protocol, dtype=bool)
        for start in range(0, num_protocol, 1_000_000):
            chunk = np.asarray(embeddings[start : start + 1_000_000], dtype=np.float32)
            usable[start : start + 1_000_000] = np.linalg.norm(chunk, axis=1) > 0.5
        num_no_embedding = int((~usable & mask).sum())
        mask &= usable

    # name order makes the sessions of a speaker, and the utterances of a session,
    # contiguous and chronological: the name is `{speaker}_{session}_{chapter}_{index}`
    rows = np.flatnonzero(mask)
    rows = rows[name_order(names[rows])]

    # the labels are only ever compared for equality, and comparing four million Python
    # strings costs minutes where integers cost seconds, so they are numbered once here.
    # `factorize` numbers in order of first appearance, which is name order, so speaker
    # `k` is the k-th speaker by name and `--limit` is a comparison
    speaker_code = pd.factorize(speakers[rows])[0]
    if args.limit:
        keep = speaker_code < args.limit
        rows, speaker_code = rows[keep], speaker_code[keep]
    if not len(rows):
        raise SystemExit("every row is masked out; check --mask against this protocol")
    session_code = pd.factorize(sessions[rows])[0]
    # the session directories are book ids and the same book is read by several speakers,
    # so the group is the pair; mixing the two codes keeps that one integer
    pair_code = (
        speaker_code.astype(np.int64) * (int(session_code.max()) + 1) + session_code
    )
    speaker_of_row = np.full(num_protocol, -1, dtype=np.int64)
    speaker_of_row[rows] = speaker_code
    pair_of_row = np.full(num_protocol, -1, dtype=np.int64)
    pair_of_row[rows] = pair_code

    hours_available = float(durations[rows].sum() / 3600)
    print(
        f"{num_protocol:,} files, {num_masked:,} masked out, "
        f"{num_no_embedding:,} without an embedding | {len(rows):,} usable, "
        f"{hours_available:,.0f} h | {int(speaker_code.max()) + 1:,} speakers, "
        f"{len(np.unique(pair_code)):,} sessions "
        f"| loaded in {time.perf_counter() - t0:.0f} s",
        flush=True,
    )

    # Stage 1: inside one session of one speaker
    t = time.perf_counter()
    stage1, leftover = cut_runs(rows, pair_code, durations, args)
    stats = {"stage1": describe(stage1, durations)}
    print(
        f"stage 1 (one session): {stats['stage1']} | {len(leftover):,} files left over "
        f"({durations[leftover].sum() / 3600:,.0f} h) | {time.perf_counter() - t:.0f} s",
        flush=True,
    )

    # Stage 2: the leftovers of a speaker, across its sessions
    stage2: List[np.ndarray] = []
    unused = leftover
    if not args.single_session_only:
        t = time.perf_counter()
        leftover = leftover[name_order(names[leftover])]
        stage2, unused = cut_runs(leftover, speaker_of_row[leftover], durations, args)
        stats["stage2"] = describe(stage2, durations)
        print(
            f"stage 2 (several sessions, one speaker): {stats['stage2']} | "
            f"{len(unused):,} files unused ({durations[unused].sum() / 3600:,.0f} h) | "
            f"{time.perf_counter() - t:.0f} s",
            flush=True,
        )

    # Outputs, in the layout `emilia_crop_plan.py` reads
    entries = stage1 + stage2
    if not entries:
        raise SystemExit("no entry could be formed; is the mask or the protocol empty?")
    offsets = np.cumsum([0] + [len(piece) for piece in entries])
    entry_rows = np.concatenate(entries)
    cluster_ids = np.full(num_protocol, -1, dtype=np.int32)
    cluster_ids[entry_rows] = np.repeat(
        np.arange(len(entries), dtype=np.int32), np.diff(offsets)
    )
    np.save(output_dir / "cluster_ids.npy", cluster_ids)

    table = pd.DataFrame(
        {
            "cluster_id": cluster_ids[entry_rows],
            "row": entry_rows,
            "name": names[entry_rows],
            "speaker": speakers[entry_rows],
            "session": sessions[entry_rows],
            "duration": durations[entry_rows],
            "stage": np.where(cluster_ids[entry_rows] < len(stage1), 1, 2).astype(
                np.int8
            ),
        }
    )
    table.to_parquet(output_dir / "clusters.parquet", index=False)

    # an entry of two speakers would be a silent corruption of the training data, so it
    # is checked here rather than trusted to the grouping above
    starts, counts = offsets[:-1], np.diff(offsets)
    entry_speaker = speaker_of_row[entry_rows]
    assert np.array_equal(
        entry_speaker, np.repeat(entry_speaker[starts], counts)
    ), "an entry mixes speakers"
    # a session is contiguous inside an entry, the rows being in name order, so counting
    # the runs of the pair code counts the sessions
    entry_pair = pair_of_row[entry_rows]
    new_session = np.ones(len(entry_rows), dtype=bool)
    new_session[1:] = entry_pair[1:] != entry_pair[:-1]
    # two consecutive entries usually come out of the same session, so the first row of
    # an entry always starts a run however it compares to the last row of the one before
    new_session[starts] = True
    sessions_per_entry = np.add.reduceat(new_session, starts)

    stats["all"] = describe(entries, durations)
    stats["all"].update(
        {
            "files": int(len(rows)),
            "files_masked": num_masked,
            "files_without_embedding": num_no_embedding,
            "hours_total": hours_available,
            "hours_clustered_fraction": float(
                stats["all"].get("hours", 0) / hours_available
            ),
            "speakers": int(len(np.unique(entry_speaker))),
            "sessions_per_cluster_mean": float(sessions_per_entry.mean()),
            "clusters_several_sessions": int((sessions_per_entry > 1).sum()),
            "unclustered_files": int(len(unused)),
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
