"""
Which Mimi tokens mean silence, for one training store.

A crop that closes on a sentence end holds the release of the last phoneme and then
steps onto `[EOS]`, and the frames it stands on there have to be real silence: that
step is what ends generation, and a model shown speech under `[EOS]` learns nothing
about stopping. The alignment cannot say whether they are silent. An entry
concatenates source utterances that were VAD-trimmed before they were joined, so at a
join the gap the word timings report is the next recording starting, not a pause -
measured on Emilia, the sentence ends a crop would pick are silent 33% of the time
overall and 16% of the time at a join, against 80% mid-utterance.

Mimi's first codebook answers it directly. Silence is a small and stable set of tokens,
so decoding a sample of entries, labelling their frames against the loud level of the
entry and keeping the tokens that are silent at least `min_silent` of the time
identifies silent frames at ~99% precision on held-out entries. The dataloader then
reads codebook 0 of an entry - 2 kB - and needs no audio and no decoder.

Decoding a whole store to measure the energy exactly would cost about 43 GPU-hours for
Emilia at 40k hours; this is a GPU-minute.

    python -m voxtream.utils.dataset.silence_tokens -d /path/to/store -g 0

Writes `silence_tokens.json` beside the arrays. Re-run it whenever the Mimi codes are
rebuilt; nothing else in the store depends on it.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from tqdm.auto import tqdm

from voxtream.utils.dataset.store import DatasetStore

MIMI_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
# codebooks to decode: the energy envelope is settled well before the last of them, and
# fewer codebooks is less memory for the same answer
DECODE_CODEBOOKS = 8
VOCAB = 2048


def frame_energy(
    mimi, store: DatasetStore, ids: np.ndarray, batch_size: int, device: str
):
    """Mean square per frame of each entry, decoded from its Mimi codes."""
    num_frames = int(store.num_frames)
    for k in range(0, len(ids), batch_size):
        chunk = [int(i) for i in ids[k : k + batch_size]]
        codes = np.stack(
            [
                np.asarray(store.read_codes(i, DECODE_CODEBOOKS, 0, num_frames))
                for i in chunk
            ]
        )
        tokens = torch.as_tensor(np.clip(codes, 0, VOCAB - 1)).long().to(device)
        with torch.no_grad():
            wav = mimi.decode(tokens).float()
            energy = (wav.view(len(chunk), 1, num_frames, -1) ** 2).mean(-1)
        energy = energy.squeeze(1).cpu().numpy()
        del wav, tokens
        for j, i in enumerate(chunk):
            yield i, energy[j]


def calibrate(
    store: DatasetStore,
    entries: int,
    silence_db: float,
    batch_size: int,
    device: str,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`P(silent | token)` over a sample of decoded entries, and its support."""
    weight = hf_hub_download(MIMI_REPO, MIMI_NAME)
    mimi = loaders.get_mimi(
        filename=weight, device=device, num_codebooks=DECODE_CODEBOOKS
    ).eval()

    rng = np.random.default_rng(seed)
    n = len(store.num_speech_frames)
    ids = rng.choice(n, size=min(entries, n), replace=False)

    count = np.zeros(VOCAB, dtype=np.int64)
    silent = np.zeros(VOCAB, dtype=np.int64)
    for i, energy in tqdm(
        frame_energy(mimi, store, ids, batch_size, device),
        total=len(ids),
        desc="calibrating",
    ):
        num_speech = int(store.num_speech_frames[i])
        energy = energy[:num_speech]
        if num_speech < 2 or not np.isfinite(energy).all():
            continue
        # against the loud part of this entry, the way the generator measures its own
        # output, so a quiet speaker is not called silent throughout
        loud = np.percentile(energy, 95)
        if loud <= 0:
            continue
        is_silent = energy < loud * 10 ** (silence_db / 10.0)
        codes = np.asarray(store.read_codes(i, 1, 0, num_speech))[0].astype(np.int64)
        count += np.bincount(codes, minlength=VOCAB)
        silent += np.bincount(codes[is_silent], minlength=VOCAB)

    share = np.divide(silent, count, out=np.zeros(VOCAB), where=count > 0)
    return share, count, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--dataset-dir", type=str, required=True, help="store root"
    )
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument(
        "-n",
        "--entries",
        type=int,
        default=500,
        help="entries to decode for calibration",
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=4)
    parser.add_argument(
        "--silence-db",
        type=float,
        default=-30.0,
        help="a frame this far below the loud level of its entry counts as silent",
    )
    parser.add_argument(
        "--min-count", type=int, default=10, help="frames a token needs to be judged"
    )
    parser.add_argument(
        "--min-silent",
        type=float,
        default=0.90,
        help="share of a token's frames that must be silent for it to mean silence",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    store = DatasetStore(root)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    share, count, ids = calibrate(
        store, args.entries, args.silence_db, args.batch_size, device, args.seed
    )
    keep = (count >= args.min_count) & (share >= args.min_silent)
    tokens = np.flatnonzero(keep)
    if not len(tokens):
        raise SystemExit(
            "no token qualified as silence; the sample may be too small or the store "
            "may hold no silence at all"
        )

    seen = int(count.sum())
    covered = int(count[keep].sum())
    payload = {
        "tokens": tokens.tolist(),
        "silence_db": args.silence_db,
        "min_count": args.min_count,
        "min_silent": args.min_silent,
        "calibration_entries": int(len(ids)),
        "frames_seen": seen,
        "frames_called_silent": covered,
    }
    out = root / "silence_tokens.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"{len(tokens)} of {VOCAB} tokens mean silence; they cover "
        f"{covered / max(seen, 1) * 100:.1f}% of the {seen:,} frames decoded\n"
        f"written to {out}"
    )


if __name__ == "__main__":
    main()
