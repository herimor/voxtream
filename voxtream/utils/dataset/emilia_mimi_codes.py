"""
Mimi codes for a store, one row per entry.

The last stage of the pipeline. `emilia_crop_plan.py` decided where every utterance is cut
and where it lands on the cluster's timeline; `emilia_train_arrays.py` turned the
alignments into the phoneme and word arrays and wrote `cluster_id.npy`. This repeats the
plan's crop on the audio and encodes each assembled entry in one pass.

Assembly, per entry:

  - read every utterance of the cluster whole, as the plan takes it,
  - normalize each one to `--lufs` integrated loudness *before* it is placed, so a cluster
    stitched out of recordings at different levels does not teach a step change in gain at
    every join,
  - place it at `dst_start`, leaving the gaps the plan left as digital silence,
  - encode the whole timeline at once, so the joins stay continuous for the codec.

The peak is checked on the finished entry, not on its parts: rescaling each utterance
would undo the loudness match that was the point of normalizing them.

Row alignment is the thing to get right. `cluster_id.npy` holds the clusters that survived
the array builder, which is a subset of the plan's - entries die on a missing alignment, a
pointer walk the frame states cannot represent, or an unusable speaker embedding. This
drives its loop from `cluster_id.npy` and looks the plan up by cluster id, so row `i` of
`mimi_codes.npy` is row `i` of every other array in the store. Iterating the plan instead
would silently pair every row after the first drop with the wrong audio.

Output: `uint16 [num_entries, codebooks, num_frames]` in shards of `--shard-codebooks`,
each file named after the index of its first codebook - `mimi_codes.npy` for 0..15,
`mimi_codes_16.npy` for 16..31. A run that trains on the lower 16 then opens one file and
never pages in the rest, and `DatasetStore.read_codes` concatenates only when asked for
more than the first shard holds. The files are written in place through memory maps, so
the work spreads over several GPUs and a restart fills what is already allocated.

Example:
    python -m voxtream.utils.dataset.emilia_mimi_codes \\
        -m emilia/metadata.parquet -p emilia/crop_plan -a /data/emilia \\
        -o /data/stores/emilia -cb 32 -g 0,1,2,3 -bs 8 -w 4
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.multiprocessing as mp
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from voxtream.utils.dataset.audio import loudness_gain, measure_loudness
from voxtream.utils.dataset.emilia_crop_plan import load_plan
from voxtream.utils.dataset.paths import audio_path, check_protocol_file

MIMI_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_WEIGHT = "tokenizer-e351c8d8-checkpoint125.safetensors"
PEAK_CEILING = 0.99


def shard_layout(num_codebooks: int, shard_size: int) -> List[Tuple[str, int, int]]:
    """
    `(filename, first codebook, count)` per shard.

    The codes go out in shards of `shard_size` codebooks, each file named after the index
    of its first one, so a run that trains on the lower 16 opens only the first file and
    never pages in the rest. `DatasetStore` walks the same names to find them.
    """
    shards, offset = [], 0
    while offset < num_codebooks:
        count = min(shard_size, num_codebooks - offset)
        name = "mimi_codes.npy" if offset == 0 else f"mimi_codes_{offset}.npy"
        shards.append((name, offset, count))
        offset += count
    return shards


# ---------------------------
# Audio
# ---------------------------


# one MP3 frame. The plan takes every file whole, so `start` is 0 and this never bites,
# but a non-zero start would: the first frame decoded after a seek has no bit reservoir
# to draw on, so mpg123 hands back a garbled frame, says `part2_3_length too large for
# available bit count` on fd 2, and resyncs on the next frame. Seeking a frame early and
# dropping the lead-in costs nothing measurable and keeps that frame out of the audio.
_LEAD_IN_SAMPLES = 1152


@contextmanager
def _quiet_decoder():
    """
    Drop what libsndfile's bundled mpg123 writes to fd 2 while a slice is decoded.

    The frame it gives up on is reported as `part2_3_length (N) too large for available
    bit count`. It is written from C straight to the descriptor, so neither `logging` nor
    `warnings` can filter it, and it is pure noise: measured over a built plan, every
    read that emitted it returned audio identical to a full decode of the file, because
    the lead-in above keeps the frame mpg123 gave up on out of what is returned. Scoped
    to the decode, so tracebacks and the progress bar still reach the terminal.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def read_slice(path: str, start: int, stop: int, sample_rate: int) -> np.ndarray:
    """`[start:stop]` of one utterance as mono float32 at `sample_rate`."""
    start, stop = int(start), int(stop)
    with _quiet_decoder(), sf.SoundFile(path) as f:
        if f.samplerate == sample_rate:
            # the common case: read only as far as the plan asks for
            lead = min(_LEAD_IN_SAMPLES, start)
            f.seek(start - lead)
            audio = f.read(stop - start + lead, dtype="float32", always_2d=False)
            audio = audio[lead:]
        else:
            audio = f.read(dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = (
                torchaudio.functional.resample(
                    torch.from_numpy(audio), f.samplerate, sample_rate
                )
                .numpy()
                .astype(np.float32)
            )
            # the plan's sample indices are at `sample_rate`, so slice after resampling
            audio = audio[start:stop]
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


class ClusterAudio(Dataset):
    """One assembled, loudness-normalized entry per item, in store row order."""

    def __init__(
        self,
        rows: np.ndarray,
        spans: np.ndarray,
        meta_path: str,
        audio_dir: str,
        plan_dir: Path,
        total_samples: int,
        sample_rate: int,
        target_lufs: float,
    ) -> None:
        self.rows = rows
        self.spans = spans
        self.meta_path = meta_path
        self.audio_dir = audio_dir
        self.plan_dir = plan_dir
        self.total = total_samples
        self.sr = sample_rate
        self.lufs = target_lufs
        self._paths = (
            pq.read_table(meta_path, columns=["path"]).column("path").combine_chunks()
        )
        self._cols = (
            load_plan(plan_dir)[0][["row", "src_start", "src_end", "dst_start"]]
            .to_numpy()
            .astype(np.int64)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = int(self.rows[i])
        start, count = self.spans[row]
        unread = unmeasured = 0

        # read and measure first, place second. About 2% of the plan's segments are
        # shorter than the meter's 0.4 s block, and those are better served by the gain
        # their neighbours got - same speaker, same session - than by being left at
        # whatever level they came in at, which is a step change at that join.
        chunks: List[Tuple[int, np.ndarray, float]] = []
        for src_row, src_start, src_end, dst_start in self._cols[start : start + count]:
            try:
                chunk = read_slice(
                    audio_path(self._paths[int(src_row)].as_py(), self.audio_dir),
                    src_start,
                    src_end,
                    self.sr,
                )
            except Exception:
                # a file that cannot be decoded leaves its stretch of the timeline silent
                # rather than taking the whole entry, and is reported at the end
                unread += 1
                continue
            loudness = measure_loudness(chunk, self.sr)
            if loudness is None:
                unmeasured += 1
                chunks.append((int(dst_start), chunk, np.nan))
            else:
                chunks.append(
                    (int(dst_start), chunk, loudness_gain(loudness, self.lufs))
                )

        gains = np.array([g for _, _, g in chunks], dtype=np.float64)
        fallback = np.nanmedian(gains) if np.isfinite(gains).any() else 1.0

        audio = np.zeros(self.total, dtype=np.float32)
        for at, chunk, gain in chunks:
            n = max(0, min(len(chunk), self.total - at))
            if n:
                audio[at : at + n] = chunk[:n] * (
                    gain if np.isfinite(gain) else fallback
                )

        # the peak of the assembly, not of its parts
        peak = float(np.abs(audio).max())
        clipped = peak > PEAK_CEILING
        if clipped:
            audio *= PEAK_CEILING / peak
        return row, torch.from_numpy(audio), unread, unmeasured, int(clipped)


def init_worker(_: int) -> None:
    """
    One thread per audio worker.

    Torch sizes its intra-op pool from the physical core count, and a forked DataLoader
    worker inherits that number, so `num_workers x GPUs` workers ask for twenty threads
    each the moment one of them touches a torch op - the resampling path a fifth of the
    files take. What a worker does is a serial MP3 decode and an IIR filter; neither gains
    anything from a thread pool, and the oversubscription is what makes the workers thrash
    and the GPUs start minutes apart.
    """
    torch.set_num_threads(1)


def collate(batch):
    rows, audio, unread, unmeasured, clipped = zip(*batch)
    return (
        np.asarray(rows, dtype=np.int64),
        torch.stack(audio),
        sum(unread),
        sum(unmeasured),
        sum(clipped),
    )


# ---------------------------
# Plan -> store rows
# ---------------------------


def plan_spans(store_dir: Path, plan_dir: Path) -> Tuple[np.ndarray, dict]:
    """`(plan_start, plan_count)` per store row, in store row order."""
    _, clusters, meta = load_plan(plan_dir)
    cluster_file = store_dir / "cluster_id.npy"
    if not cluster_file.exists():
        raise SystemExit(
            f"{cluster_file} is missing. Build the store with emilia_train_arrays.py "
            "first: this stage writes one row per surviving entry, and only that file "
            "says which entries survived."
        )
    kept = np.load(cluster_file)

    index = {
        int(cid): (int(ps), int(pc))
        for cid, ps, pc in zip(
            clusters["cluster_id"].to_numpy(),
            clusters["plan_start"].to_numpy(),
            clusters["plan_count"].to_numpy(),
            strict=True,
        )
    }
    missing = [int(c) for c in kept if int(c) not in index]
    if missing:
        raise SystemExit(
            f"{len(missing)} store entries name a cluster the plan does not have "
            f"(first: {missing[:5]}); the store and the plan are from different runs"
        )
    spans = np.asarray([index[int(c)] for c in kept], dtype=np.int64)
    return spans, meta


# ---------------------------
# One GPU
# ---------------------------


def run_gpu(slot: int, gpus: List[int], rows: np.ndarray, args, spans, meta) -> None:
    gpu = gpus[slot]
    device = f"cuda:{gpu}"
    # the model is placed explicitly, but the DataLoader's pinning thread and anything
    # else that asks for "the current device" would otherwise land on cuda:0 in every
    # worker, leaving a stray context per GPU on one card and crowding it out of memory
    torch.cuda.set_device(gpu)
    store_dir = Path(args.store_dir)
    mine = rows[slot :: len(gpus)]

    model = (
        loaders.get_mimi(
            filename=hf_hub_download(repo_id=MIMI_REPO, filename=MIMI_WEIGHT),
            device=device,
            num_codebooks=args.num_codebooks,
        )
        .eval()
        .half()
    )

    shards = [
        (np.lib.format.open_memmap(store_dir / name, mode="r+"), first, count)
        for name, first, count in shard_layout(args.num_codebooks, args.shard_codebooks)
    ]
    num_frames = shards[0][0].shape[2]

    loader = DataLoader(
        ClusterAudio(
            rows=mine,
            spans=spans,
            meta_path=args.meta_path,
            audio_dir=args.audio_dir,
            plan_dir=Path(args.plan_dir),
            total_samples=int(meta["total_samples"]),
            sample_rate=int(meta["sample_rate"]),
            target_lufs=args.lufs,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        # an entry is 80 s of float32, 7.7 MiB, and every queued one is pinned host
        # memory; four deep per worker was gigabytes of churn for a queue the encoder
        # drains faster than the workers fill it
        prefetch_factor=2 if args.num_workers else None,
        worker_init_fn=init_worker if args.num_workers else None,
        # fork, so the plan and the path column stay shared with this process
        multiprocessing_context="fork" if args.num_workers else None,
    )

    unread = unmeasured = clipped = 0
    bar = tqdm(
        loader,
        position=slot,
        desc=f"gpu{gpu}",
        unit="entry",
        unit_scale=args.batch_size,
    )
    for batch_rows, audio, n_unread, n_unmeasured, n_clipped in bar:
        with torch.no_grad():
            encoded = model.encode(
                audio.unsqueeze(1).to(device, non_blocking=True).half()
            )
        block = encoded[:, : args.num_codebooks, :num_frames].cpu().numpy()
        if block.shape[2] < num_frames:  # a short tail would leave the row half written
            block = np.pad(block, ((0, 0), (0, 0), (0, num_frames - block.shape[2])))
        block = block.astype(np.uint16)
        for codes, first, count in shards:
            codes[batch_rows] = block[:, first : first + count]
        unread += n_unread
        unmeasured += n_unmeasured
        clipped += n_clipped
    bar.close()
    for codes, _, _ in shards:
        codes.flush()

    with open(store_dir / f".mimi_stats_{slot}.json", "w") as f:
        json.dump(
            {
                "gpu": gpu,
                "entries": int(len(mine)),
                "unreadable_utterances": int(unread),
                "unmeasurable_utterances": int(unmeasured),
                "entries_rescaled_for_peak": int(clipped),
            },
            f,
        )


# ---------------------------
# Entry point
# ---------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument(
        "-m",
        "--meta-path",
        required=True,
        help="the protocol parquet, with paths relative to --audio-dir",
    )
    p.add_argument("-p", "--plan-dir", required=True, help="the crop plan directory")
    p.add_argument(
        "-a",
        "--audio-dir",
        required=True,
        help="absolute root of the audio: utterance `path` is `{audio_dir}/{path}.mp3`",
    )
    p.add_argument(
        "-o", "--store-dir", required=True, help="the store, where mimi_codes.npy goes"
    )
    p.add_argument("-cb", "--num-codebooks", type=int, default=32)
    p.add_argument(
        "--shard-codebooks",
        type=int,
        default=16,
        help="codebooks per file; 32 in shards of 16 lets a 16-codebook run open one file",
    )
    p.add_argument(
        "-g", "--gpus", default="0", help="comma separated GPU ids, e.g. 0,1,2,3"
    )
    p.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=8,
        help="entries per encoder pass. Throughput is flat in this - 4, 8, 16 and 24 all "
        "measure 34.5 entries/s on one card - so it only decides how much memory the "
        "run holds: about 1.2 GiB of activations per 80 s entry, 9.5 GiB at 8, 18.7 GiB "
        "at 16, OOM at 32. There is nothing to win by raising it",
    )
    p.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="audio workers per GPU, each pinned to one thread. Decoding and metering an "
        "entry costs about 125 ms of one core against a 34.5 entries/s encoder, so four "
        "workers very nearly keep a card fed and a fifth would finish the job. Past that "
        "they only compete: `num_workers x GPUs` is the whole CPU budget of the run",
    )
    p.add_argument(
        "--lufs",
        type=float,
        default=-23.0,
        help="integrated loudness every utterance is normalized to before assembly",
    )
    p.add_argument(
        "-n", "--limit", type=int, default=None, help="only N entries (test mode)"
    )
    args = p.parse_args()

    args.audio_dir = os.path.abspath(args.audio_dir)
    check_protocol_file(args.meta_path)
    store_dir = Path(args.store_dir)
    spans, meta = plan_spans(store_dir, Path(args.plan_dir))
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]

    with open(store_dir / "meta.json") as f:
        store_meta = json.load(f)
    num_entries, num_frames = len(spans), int(store_meta["num_frames"])
    rows = np.arange(
        num_entries if args.limit is None else min(args.limit, num_entries)
    )

    layout = shard_layout(args.num_codebooks, args.shard_codebooks)
    for name, first, count in layout:
        out = store_dir / name
        shape = (num_entries, count, num_frames)
        if out.exists():
            existing = np.lib.format.open_memmap(out, mode="r")
            bad = existing.shape != shape or existing.dtype != np.uint16
            del existing
            if bad:
                raise SystemExit(
                    f"{out} does not match uint16 {shape}; remove it or point -o at a "
                    "fresh store"
                )
            print(f"{out.name} already exists, filling it in place")
        else:
            gib = float(np.prod(shape)) * 2 / 1024**3
            print(
                f"allocating {out.name} - codebooks {first}..{first + count - 1}, "
                f"uint16 {shape}, {gib:.1f} GiB"
            )
            np.lib.format.open_memmap(
                out, mode="w+", dtype=np.uint16, shape=shape
            ).flush()

    print(
        f"{len(rows):,} entries over {len(gpus)} GPU(s) {gpus}, "
        f"{args.num_codebooks} codebooks, {num_frames} frames, "
        f"loudness target {args.lufs} LUFS"
    )
    mp.spawn(run_gpu, args=(gpus, rows, args, spans, meta), nprocs=len(gpus), join=True)

    stats, total = [], {}
    for slot in range(len(gpus)):
        path = store_dir / f".mimi_stats_{slot}.json"
        if path.exists():
            with open(path) as f:
                stats.append(json.load(f))
            os.remove(path)
    for key in (
        "entries",
        "unreadable_utterances",
        "unmeasurable_utterances",
        "entries_rescaled_for_peak",
    ):
        total[key] = sum(s[key] for s in stats)
    print(
        f"done: {total['entries']:,} entries; "
        f"{total['unreadable_utterances']:,} utterances could not be decoded, "
        f"{total['unmeasurable_utterances']:,} too short or quiet to measure, "
        f"{total['entries_rescaled_for_peak']:,} entries rescaled to stay under full scale"
    )
    with open(store_dir / "mimi_codes_meta.json", "w") as f:
        json.dump(
            {
                "num_codebooks": args.num_codebooks,
                "shard_codebooks": args.shard_codebooks,
                "shards": [name for name, _, _ in layout],
                "num_frames": num_frames,
                "lufs": args.lufs,
                "mimi_repo": MIMI_REPO,
                "mimi_weight": MIMI_WEIGHT,
                # the name only: a store is shared without the machine it was built on
                "plan_dir": Path(args.plan_dir).resolve().name,
                **total,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
