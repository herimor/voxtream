"""
Speaker embeddings for Emilia and HiFiTTS-2 with ReDimNet2-B3 (vb2+vox2+cnc2).

Like `speaker_encoder.py`, every file of the protocol is embedded from its first 3 s,
zero-padded if shorter (Emilia never is, its shortest file is 3.0 s; 78k files of
HiFiTTS-2 are, down to 2.0 s). The model is ReDimNet2-B3 fine-tuned with large margin on
VoxBlink2 + VoxCeleb2 + CN-Celeb2 (`PalabraAI/redimnet2`, release v1.0.0,
`b3-vb2+vox2+cnc2_v0-lm.pt`).

Only the first 3 s of a file are decoded (libsndfile reads the head of the mp3 or flac
from NFS and stops), which is ~3x cheaper than decoding the whole file as
`speaker_encoder.py` does. Files libsndfile cannot open fall back to a full ffmpeg decode.
Any sample rate is resampled to 16 kHz (Emilia: 24/32 kHz mp3, HiFiTTS-2: 24 kHz mp3 and
22.05 kHz flac).

The protocol holds paths relative to `--audio-dir` (`{audio_dir}/{path}.mp3`, see
`paths.py`). It is cut into contiguous shards of `--shard-size` rows, dealt round-robin to
the GPU processes in an order that alternates between the filesystems the audio lives on
(`interleave_storage`), which matters when the audio directory gathers several network
mounts. Each process reads its shards with one dataloader and writes each as

    {out_dir}/shards/spk_emb_{shard:05d}.npy    float16 [rows, 192]

and shards already on disk are skipped, so the job is resumable. Once every shard is
there they are merged into

    {out_dir}/spk_emb.npy                       float16 [len(protocol), 192]

row `i` being the embedding of row `i` of the protocol. Embeddings are L2-normalized;
the rows of unreadable files are all-zero, so they are the rows of zero norm.

Throughput (RTX 3090, 16 workers, torch.compile max-autotune, 100k random files):
~630 files/s end to end, i.e. ~8 h for the 18.1M files of Emilia on one GPU. The model
alone runs at ~915 files/s in fp16 (590 uncompiled); the rest is NFS reads, which vary a
lot with the load of the shared servers. fp16 matches fp32 (cosine >= 0.999) and the
partial libsndfile decode matches a full ffmpeg decode (mean cosine 0.9998, min 0.997).

Example (benchmark, 100k random files on GPU 1, written to {output_dir}/limit_100000):
    python -m voxtream.utils.dataset.emilia_speaker_encoder \
        -m emilia/metadata.parquet -a /data/emilia -o /data/spk_emb/emilia \
        --gpus 1 -n 100000

Example (whole protocol on 4 GPUs):
    python -m voxtream.utils.dataset.emilia_speaker_encoder \
        -m emilia/metadata.parquet -a /data/emilia -o /data/spk_emb/emilia \
        --gpus 1,2,3,4

HiFiTTS-2 (4.69M files) runs the same way with its own protocol and audio:
    python -m voxtream.utils.dataset.emilia_speaker_encoder \
        -m hifitts2/metadata.parquet -a /data/hifitts2 -o /data/spk_emb/hifitts2 --gpus 1
"""

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.multiprocessing as mp
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from tqdm.auto import tqdm

from voxtream.utils.dataset.paths import audio_path, check_protocol

warnings.filterwarnings("ignore")

EMBEDDING_DIM = 192


# ---------------------------
# Data
# ---------------------------


def load_head(path: str, seconds: float) -> Tuple[torch.Tensor, int]:
    """
    Decode the first `seconds` of a file, mixed down to mono.

    Returns
    -------
    Tuple[torch.Tensor, int]
        The waveform [T] at the native rate, and the rate.
    """
    try:
        with sf.SoundFile(path) as f:
            sr = f.samplerate
            waveform = f.read(int(seconds * sr), dtype="float32", always_2d=True)
        return torch.from_numpy(waveform).mean(dim=1), sr
    except Exception:
        waveform, sr = torchaudio.load(path)
        return waveform.mean(dim=0)[: int(seconds * sr)], sr


class HeadDataset(Dataset):
    """Returns the first `target_length_sec` seconds of a file at `sample_rate`."""

    def __init__(
        self,
        paths: List[str],
        audio_dir: str,
        sample_rate: int = 16000,
        target_length_sec: float = 3,
    ):
        self.paths = paths
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_length_sec = target_length_sec
        self.target_length = int(target_length_sec * sample_rate)

        self._resamplers: Dict[int, Resample] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Optional[Tuple[int, torch.Tensor]]:
        try:
            waveform, sr = load_head(
                audio_path(self.paths[idx], self.audio_dir), self.target_length_sec
            )
        except Exception:
            return None

        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = Resample(sr, self.sample_rate)
            waveform = self._resamplers[sr](waveform)

        if waveform.shape[0] < self.target_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.target_length - waveform.shape[0])
            )

        return idx, waveform[: self.target_length]


def collate_heads(
    items: List[Optional[Tuple[int, torch.Tensor]]],
) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
    """
    Stack the readable files of a batch into [B, T].

    Returns the number of files in the batch, readable or not, the dataset indices of
    the readable ones and their waveforms (None if there are none).
    """
    num_files = len(items)
    items = [item for item in items if item is not None]
    if not items:
        return num_files, torch.empty(0, dtype=torch.long), None

    indices = torch.tensor([idx for idx, _ in items])
    waveforms = torch.stack([waveform for _, waveform in items])
    return num_files, indices, waveforms


# ---------------------------
# Encoding
# ---------------------------


def load_model(device: str, compile_mode: str = "none"):
    """ReDimNet2-B3, large-margin fine-tuned on VoxBlink2 + VoxCeleb2 + CN-Celeb2."""
    model = (
        torch.hub.load(
            "PalabraAI/redimnet2:main",
            "redimnet2",
            model_name="b3",
            train_type="lm",
            dataset="vb2+vox2+cnc2_v0",
            trust_repo=True,
            verbose=False,
        )
        .to(device)
        .half()
    )
    model.spec.float()
    model.bn.float()
    model.eval()

    # the model is bound by memory-bound elementwise and norm kernels, which inductor
    # fuses: 590 -> 750 (default) / 915 (max-autotune) files/s on an RTX 3090, at the
    # cost of ~1 / ~5 min of compilation per process
    if compile_mode == "default":
        model = torch.compile(model, dynamic=False)
    elif compile_mode == "max-autotune":
        model = torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=False)
    return model


@torch.no_grad()
def encode_batch(
    model, waveforms: torch.Tensor, device: str, batch_size: Optional[int] = None
) -> np.ndarray:
    """
    Embed a batch, zero-padded to `batch_size` rows so that a compiled model always
    sees the same shape.

    Returns
    -------
    np.ndarray
        Embeddings [B, 192], L2-normalized, float16.
    """
    num_rows = waveforms.shape[0]
    if batch_size is not None and num_rows < batch_size:
        waveforms = torch.nn.functional.pad(waveforms, (0, 0, 0, batch_size - num_rows))

    waveforms = waveforms.to(device, non_blocking=True).half()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        embeddings = model(waveforms)
    embeddings = embeddings[:num_rows].float()
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return embeddings.cpu().numpy().astype(np.float16)


# ---------------------------
# Worker
# ---------------------------


def shard_path(shards_dir: Path, shard: int) -> Path:
    return shards_dir / f"spk_emb_{shard:05d}.npy"


def run_worker(
    rank: int,
    args,
    paths: List[str],
    shards: List[int],
    done: "mp.Value",
    failed: "mp.Value",
) -> None:
    """
    Embed `shards` (protocol shard ids) on one GPU.

    The shards are read by one dataloader as a single sequence of files, so its workers
    and prefetch queue survive the shard boundaries, and every shard is written as
    soon as its last file comes out (the dataloader keeps the order).
    """
    gpu = args.gpu_ids[rank]
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    model = load_model(device, args.compile)
    pad_to = args.batch_size if args.compile != "none" else None

    bounds = [
        (s * args.shard_size, min((s + 1) * args.shard_size, len(paths)))
        for s in shards
    ]
    # position of every shard in the sequence read by this process
    starts = np.cumsum([0] + [hi - lo for lo, hi in bounds])
    rank_paths = [path for lo, hi in bounds for path in paths[lo:hi]]

    dataset = HeadDataset(
        rank_paths,
        args.audio_dir,
        sample_rate=args.sample_rate,
        target_length_sec=args.target_length_sec,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_heads,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        # this process is spawned, so its workers would be spawned as well and
        # re-import torch one by one; they never touch CUDA, forking them is safe
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )

    # unreadable files are dropped by the dataloader and keep their zero row
    buffers: Dict[int, np.ndarray] = {}
    position, next_shard = 0, 0
    t_data = t_encode = 0.0
    t = time.perf_counter()
    for num_files, indices, waveforms in dataloader:
        t_data += time.perf_counter() - t
        t = time.perf_counter()
        if waveforms is not None:
            embeddings = encode_batch(model, waveforms, device, pad_to)
            owners = np.searchsorted(starts, indices.numpy(), side="right") - 1
            for k in np.unique(owners):
                if k not in buffers:
                    lo, hi = bounds[k]
                    buffers[k] = np.zeros((hi - lo, EMBEDDING_DIM), dtype=np.float16)
                mask = owners == k
                buffers[k][indices.numpy()[mask] - starts[k]] = embeddings[mask]
        t_encode += time.perf_counter() - t

        position += num_files
        while next_shard < len(shards) and starts[next_shard + 1] <= position:
            lo, hi = bounds[next_shard]
            embeddings = buffers.pop(
                next_shard, np.zeros((hi - lo, EMBEDDING_DIM), dtype=np.float16)
            )
            out_path = shard_path(args.shards_dir, shards[next_shard])
            tmp_path = out_path.with_suffix(".tmp.npy")
            np.save(tmp_path, embeddings)
            os.replace(tmp_path, out_path)
            next_shard += 1

        with done.get_lock():
            done.value += num_files
        with failed.get_lock():
            failed.value += num_files - len(indices)
        t = time.perf_counter()

    print(
        f"rank {rank} (gpu {gpu}): {len(shards)} shards, {position:,} files | "
        f"waiting for data {t_data:.0f} s, encoding {t_encode:.0f} s",
        flush=True,
    )


# ---------------------------
# Main
# ---------------------------


def interleave_storage(
    shards: List[int], paths: List[str], shard_size: int, audio_dir: str
) -> List[int]:
    """
    Order the shards so that the filesystems they live on are read at the same time.

    The audio directory can gather several network mounts (e.g. the batches of Emilia
    linked in from two NFS servers), and the protocol is made of contiguous blocks on
    each. The shards are grouped by the filesystem of their first file and every group
    is spread evenly over the order, which keeps all the servers busy until the end of
    the job; with one filesystem this is the plain shard order.
    """
    groups: Dict[Optional[int], List[int]] = {}
    for shard in shards:
        try:
            device: Optional[int] = os.stat(
                audio_path(paths[shard * shard_size], audio_dir)
            ).st_dev
        except OSError:
            device = None
        groups.setdefault(device, []).append(shard)

    return [
        shard
        for _, shard in sorted(
            ((i + 0.5) / len(group), shard)
            for group in groups.values()
            for i, shard in enumerate(group)
        )
    ]


def merge_shards(shards_dir: Path, num_shards: int, num_rows: int, out_path: Path):
    """Concatenate the shards into one [num_rows, 192] array, row-aligned with the protocol."""
    merged = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16, shape=(num_rows, EMBEDDING_DIM)
    )
    row = 0
    for shard in tqdm(range(num_shards), desc="merge"):
        embeddings = np.load(shard_path(shards_dir, shard))
        merged[row : row + len(embeddings)] = embeddings
        row += len(embeddings)
    assert row == num_rows, f"merged {row} rows, protocol has {num_rows}"
    merged.flush()

    norms = np.linalg.norm(merged[: min(num_rows, 100_000)].astype(np.float32), axis=1)
    assert np.all(
        (np.abs(norms - 1) < 1e-2) | (norms == 0)
    ), "Embeddings are not normalized"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--meta-path",
        type=str,
        required=True,
        help="Protocol parquet with the relative `path` of every file",
    )
    parser.add_argument(
        "-a",
        "--audio-dir",
        type=str,
        required=True,
        help="Absolute root of the audio: file `path` is `{audio_dir}/{path}.mp3`",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="1",
        help="Comma-separated GPU ids, one process each; repeat an id to run several",
    )
    parser.add_argument(
        "-nw",
        "--num-workers",
        type=int,
        default=16,
        help="Dataloader workers PER PROCESS",
    )
    parser.add_argument("-bs", "--batch-size", type=int, help="Batch size", default=256)
    parser.add_argument(
        "-tl",
        "--target-length-sec",
        type=float,
        help="Target length in seconds",
        default=3,
    )
    parser.add_argument(
        "-sr", "--sample-rate", type=int, help="Model sample rate", default=16000
    )
    parser.add_argument(
        "--shard-size", type=int, default=100_000, help="Protocol rows per shard"
    )
    parser.add_argument(
        "--compile",
        choices=["none", "default", "max-autotune"],
        default="max-autotune",
        help="torch.compile the model (see `load_model` for the trade-off)",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Embed only N random files (benchmark mode), written to {output_dir}/limit_{N}",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=0, help="Seed of the --limit sample"
    )
    args = parser.parse_args()

    args.gpu_ids = [int(x) for x in args.gpus.split(",")]
    args.audio_dir = os.path.abspath(args.audio_dir)
    output_dir = Path(args.output_dir)

    meta = pd.read_parquet(args.meta_path, columns=["path"])
    check_protocol(meta["path"].iloc[0])
    num_protocol = len(meta)
    if args.limit:
        # a random sample, the protocol is ordered by batch and the first files of it
        # can sit on one disk or in the page cache
        meta = meta.sample(n=args.limit, random_state=args.sample_seed)
        output_dir = output_dir / f"limit_{args.limit}"
    args.shards_dir = output_dir / "shards"
    args.shards_dir.mkdir(parents=True, exist_ok=True)
    if args.limit:
        np.save(output_dir / "rows.npy", meta.index.values)

    paths = meta["path"].tolist()
    num_shards = -(-len(paths) // args.shard_size)
    pending = [
        s for s in range(num_shards) if not shard_path(args.shards_dir, s).exists()
    ]
    num_pending = sum(
        min(args.shard_size, len(paths) - s * args.shard_size) for s in pending
    )
    print(
        f"{len(paths):,} files (protocol: {num_protocol:,}) | {num_shards} shards, "
        f"{len(pending)} pending ({num_pending:,} files) | "
        f"{len(args.gpu_ids)} processes x {args.num_workers} workers",
        flush=True,
    )

    if pending:
        order = interleave_storage(pending, paths, args.shard_size, args.audio_dir)
        world = len(args.gpu_ids)

        ctx = mp.get_context("spawn")
        done = ctx.Value("q", 0)
        failed = ctx.Value("q", 0)

        procs = [
            ctx.Process(
                target=run_worker,
                args=(rank, args, paths, order[rank::world], done, failed),
            )
            for rank in range(world)
        ]
        for proc in procs:
            proc.start()

        t0 = None
        progress = tqdm(total=num_pending, unit="file", unit_scale=True, smoothing=0.05)
        while any(proc.is_alive() for proc in procs):
            time.sleep(2)
            if t0 is None and done.value:
                # throughput is measured from the first batch, model loading and the
                # warm-up of the dataloaders are a fixed cost
                t0, n0 = time.perf_counter(), done.value
            progress.n = done.value
            progress.refresh()
        progress.n = done.value
        progress.close()
        for proc in procs:
            proc.join()
        assert all(proc.exitcode == 0 for proc in procs), "A worker failed"

        if t0 is not None and done.value > n0:
            rate = (done.value - n0) / (time.perf_counter() - t0)
            print(
                f"{done.value:,} files, {failed.value:,} unreadable | "
                f"steady state: {rate:,.0f} files/s -> "
                f"{num_protocol / rate / 3600:.2f} h for the whole protocol "
                f"({num_protocol:,} files) with this setup",
                flush=True,
            )

    merge_shards(
        args.shards_dir,
        num_shards,
        len(paths),
        output_dir / "spk_emb.npy",
    )
    print(f"Saved {output_dir / 'spk_emb.npy'}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
