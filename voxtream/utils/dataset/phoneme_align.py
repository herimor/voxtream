"""
Phoneme alignment of Emilia and HiFiTTS-2, driven by the Seamless word alignments.

Every utterance of the protocol parquet has a Seamless word alignment next to where its
phoneme alignment goes (`emilia_seamless_align.py`; the layout is in `paths.py`):

    {data_dir}/{parent}/{name}.seamless.json
    {"duration": 6.264,
     "words": [{"word": "You", "start": 0.04, "end": 0.24, "score": 0.0139}, ...]}

It holds the transcript (one entry per whitespace token, punctuation attached), the word
timestamps and the audio duration; the audio itself is `{audio_dir}/{path}.mp3`, `path`
being the relative protocol path. For every file the script:
  1. normalizes and phonemizes the transcript (espeak-ng), mapping the phonemes onto the
     words and pulling the punctuation out of the sequence,
  2. groups consecutive words into chunks of up to 5 s, glued with at most 100 ms of the
     original audio (50 ms per side) and zero-padded to 5 s,
  3. force-aligns the chunks with Clap-IPA, in GPU batches that mix chunks of many files,
  4. maps the phoneme timestamps back to the original audio timeline and writes them
     next to the Seamless alignment as `{data_dir}/{parent}/{name}.json`, overwriting
     whatever is there.

The text side, audio decoding and chunk assembly run in the dataloader workers; the
audio stays at its native rate (24 kHz) and is resampled on the GPU inside the aligner,
which takes ~30% off the CPU cost of a file.

Example:
    python -m voxtream.utils.dataset.phoneme_align \
        -m emilia/metadata.parquet -a /data/emilia -d /data/alignment/emilia \
        --gpus 0,1,4,5 -nw 8

HiFiTTS-2 has the same layout (`{data_dir}/{session}/{name}.seamless.json`) and runs with
its own protocol:
    python -m voxtream.utils.dataset.phoneme_align \
        -m hifitts2/metadata.parquet -a /data/hifitts2 -d /data/alignment/hifitts2 \
        --gpus 2,3,4,5 -nw 4

Output (`{name}.json`), "Hello, world.":
    {
      "words": [
        {"text": "Hello", "start": 1.02, "end": 1.34,
         "phonemes": [["h", 1.02, 1.08], ["ə", 1.08, 1.14], ...]},
        {"text": "world", "start": 1.44, "end": 1.81, "phonemes": [...]}
      ],
      "punctuation": [[",", 1], [".", 2]]
    }
Punctuation is kept out of both the word text and the phonemes; its index is the insertion
position in the word list (`np.insert`).

Working from word timestamps instead of sentence spans keeps the long pauses out of the
aligner input, which is what Clap-IPA is worst at. Word timestamps are the ones of the
alignment, not of the Seamless input: Clap-IPA places an onset at the last frame it assigns
to the preceding token, a systematic -50 ms offset.

The phonemes come from the espeak-ng shared library (`ESpeak(library="espeak-lib")`), as at
inference; `--phonemizer espeak-cli` runs the command-line tool, which differs from it in a
stress mark on rare clauses and costs ~50 ms per file.
"""

import argparse
import io
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from voxtream.utils.aligner.clap_ipa import ClapIPA
from voxtream.utils.dataset.paths import (
    AUDIO_SUFFIX,
    SEAMLESS_SUFFIX,
    alignment_path,
    audio_path,
    check_protocol,
)
from voxtream.utils.text.normalizer import english_normalizer
from voxtream.utils.text.phonemizer import ESpeak

SR = 16000
PUNCTUATION = (".", ",", "?", "!")
# Whisper encoder positional embeddings limit the input to 30 s
MAX_ENCODER_SECONDS = 30.0


# ---------------------------
# Text side
# ---------------------------


@dataclass
class Word:
    """A transcript word with its timestamps and phonemes."""

    text: str
    start: float  # seconds, original audio timeline
    end: float
    phones: List[str]
    punct: List[str] = field(default_factory=list)  # punctuation following it


def strip_punctuation(text: str, punctuation: Tuple[str, ...] = PUNCTUATION) -> str:
    """Drop the trailing punctuation of a word ("Hello," -> "Hello")."""
    stripped = text.rstrip("".join(punctuation))
    return stripped or text


def split_punctuation(
    phones: List[str], punctuation: Tuple[str, ...] = PUNCTUATION
) -> Tuple[List[str], List[str]]:
    """Move the punctuation of a word out of its phoneme sequence."""
    symbols: List[str] = []
    while phones and phones[-1].endswith(punctuation):
        phone = phones.pop()
        symbols.append(phone[-1])
        if phone[:-1]:
            phones.append(phone[:-1])
            break
    symbols.reverse()
    return phones, symbols


def phonemize_words(
    text: str,
    words: List[str],
    phonemizer: ESpeak,
    sep: str = "|",
) -> List[List[str]]:
    """
    Phonemize a sentence and return one phoneme group per word.

    espeak sometimes glues function words to their neighbour ("the city" comes
    back as a single token), which breaks the word mapping for ~12% of the
    sentences; those fall back to phonemizing every word on its own.
    """

    def split_groups(seq: str) -> List[List[str]]:
        return [[ph for ph in group.split(sep) if ph] for group in seq.split()]

    norm = english_normalizer(text)
    if norm.strip():
        try:
            groups = split_groups(phonemizer.phonemize(norm))
        except Exception:
            groups = []
        if len(groups) == len(words):
            return groups

    groups = []
    for word in words:
        norm = english_normalizer(word)
        if not norm.strip():
            groups.append([])
            continue
        try:
            # a word may normalize to several tokens ("25" -> "twenty-five")
            groups.append(
                [ph for g in split_groups(phonemizer.phonemize(norm)) for ph in g]
            )
        except Exception:
            groups.append([])
    return groups


def parse_transcript(entries: List[dict], phonemizer: ESpeak) -> List[Word]:
    """
    Phonemize the transcript and return its words in order.

    Words without usable timestamps are dropped; the punctuation they carry is
    kept and moves to the preceding word.
    """
    words: List[Word] = []
    for entry in entries:
        entry_words = entry.get("words", [])
        if not entry_words:
            continue

        groups = phonemize_words(
            entry["transcript"], [w["word"] for w in entry_words], phonemizer
        )
        if len(groups) != len(entry_words):
            continue

        for meta, phones in zip(entry_words, groups):
            phones, symbols = split_punctuation(list(phones))

            start, end = meta.get("start"), meta.get("end")
            usable = start is not None and end is not None and end > start
            if phones and usable:
                words.append(
                    Word(
                        text=strip_punctuation(meta["word"]),
                        start=float(start),
                        end=float(end),
                        phones=phones,
                    )
                )
            if symbols:
                if words:
                    words[-1].punct.extend(symbols)
                # punctuation before the very first word is dropped

    return words


# ---------------------------
# Batch preparation
# ---------------------------


def build_chunks(
    words: List[Word],
    max_dur: float = 5.0,
    max_single: float = 29.0,
    pad: float = 0.1,
) -> Tuple[List[List[Word]], int]:
    """
    Group consecutive words into chunks of at most `max_dur` seconds.

    A pause between two words contributes at most `pad` seconds to the chunk.
    Words longer than `max_dur` become a chunk of their own; words longer than
    `max_single` are skipped (the aligner cannot see past 30 s).

    Returns
    -------
    Tuple[List[List[Word]], int]
        (chunks, number of skipped words)
    """
    chunks: List[List[Word]] = []
    skipped = 0

    current: List[Word] = []
    current_dur = 0.0

    for word in words:
        dur = word.end - word.start

        if dur > max_dur:
            if current:
                chunks.append(current)
                current, current_dur = [], 0.0
            if dur <= max_single:
                chunks.append([word])
            else:
                skipped += 1
            continue

        gap_pad = 0.0
        if current:
            gap_pad = min(max(0.0, word.start - current[-1].end), pad)
            if current_dur + gap_pad + dur > max_dur:
                chunks.append(current)
                current, current_dur, gap_pad = [], 0.0, 0.0

        current.append(word)
        current_dur += gap_pad + dur

    if current:
        chunks.append(current)

    return chunks, skipped


def assemble_chunk(
    audio: np.ndarray,
    chunk: List[Word],
    sample_rate: int,
    pad: float = 0.1,
    target_dur: float = 5.0,
    tail_dur: float = 0.2,
) -> Tuple[np.ndarray, List[float]]:
    """
    Concatenate the audio of the chunk words and pad the tail with zeros.

    Pauses shorter than `pad` are kept as they are, longer ones are cut down to
    `pad` seconds of the original audio, half of it taken after the previous word
    and half before the next one. The same half-pad of original audio is prepended
    and appended to the chunk.

    Returns
    -------
    Tuple[np.ndarray, List[float]]
        (chunk waveform, chunk-timeline offset of every word, in seconds)
    """

    def idx(t: float) -> int:
        return int(min(max(round(t * sample_rate), 0), len(audio)))

    half = int(round(0.5 * pad * sample_rate))
    parts: List[np.ndarray] = []
    offsets: List[float] = []
    cursor = 0  # samples

    head = audio[max(idx(chunk[0].start) - half, 0) : idx(chunk[0].start)]
    parts.append(head)
    cursor += len(head)

    for i, word in enumerate(chunk):
        if i > 0:
            prev_end, cur_start = idx(chunk[i - 1].end), idx(word.start)
            gap = cur_start - prev_end
            if gap <= 0:
                bridge = []
            elif gap <= 2 * half:
                bridge = [audio[prev_end:cur_start]]
            else:
                bridge = [
                    audio[prev_end : prev_end + half],
                    audio[cur_start - half : cur_start],
                ]
            for part in bridge:
                parts.append(part)
                cursor += len(part)

        offsets.append(cursor / sample_rate)
        speech = audio[idx(word.start) : idx(word.end)]
        parts.append(speech)
        cursor += len(speech)

    tail = audio[idx(chunk[-1].end) : idx(chunk[-1].end) + half]
    parts.append(tail)
    cursor += len(tail)

    waveform = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    # zero-pad the rest of the chunk, the trailing [SIL] token attaches to it
    total = max(
        int(round(target_dur * sample_rate)),
        len(waveform) + int(round(tail_dur * sample_rate)),
    )
    total = min(total, int(round(MAX_ENCODER_SECONDS * sample_rate)))
    if len(waveform) < total:
        waveform = np.concatenate(
            [waveform, np.zeros(total - len(waveform), dtype=waveform.dtype)]
        )
    else:
        waveform = waveform[:total]

    return waveform.astype(np.float32), offsets


def build_tokens(
    chunk: List[Word], pad_token: str = "[SIL]"
) -> Tuple[List[str], List[Optional[Tuple[int, int]]]]:
    """
    Build the phoneme sequence of a chunk.

    Words follow each other without a separator; a single `pad_token` closes the
    sequence and absorbs the zero padding, its predicted start time being the end
    of the last phoneme. Both are dropped after the alignment.

    Returns
    -------
    Tuple[List[str], List[Optional[Tuple[int, int]]]]
        (interior tokens, owner of every token as (word idx, phoneme idx))
    """
    tokens: List[str] = []
    owners: List[Optional[Tuple[int, int]]] = []

    for i, word in enumerate(chunk):
        for j, phone in enumerate(word.phones):
            tokens.append(phone)
            owners.append((i, j))

    tokens.append(pad_token)
    owners.append(None)

    return tokens, owners


@dataclass
class FileBatch:
    """Everything the aligner needs for one audio file."""

    rel_path: str
    out_path: Path
    duration: float
    chunks: List[List[Word]]
    audios: List[np.ndarray]
    tokens: List[List[str]]
    owners: List[List[Optional[Tuple[int, int]]]]
    offsets: List[List[float]]
    skipped: int


# ---------------------------
# Post processing
# ---------------------------


def postprocess_chunk(
    chunk: List[Word],
    owners: List[Optional[Tuple[int, int]]],
    offsets: List[float],
    times: Optional[List[float]],
    chunk_dur: float,
    file_dur: float,
) -> List[Optional[dict]]:
    """
    Map the chunk-level phoneme alignment back to the original audio timeline.

    `times[j]` is the predicted start of interior token `j`, so the end of a
    phoneme is the start of the next token; the trailing `[SIL]` is dropped here.

    Returns
    -------
    List[Optional[dict]]
        One output word per chunk word, None where the alignment is unusable.
    """
    if times is None or len(times) != len(owners):
        return [None] * len(chunk)

    spans: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for j, owner in enumerate(owners):
        if owner is None:
            continue
        word_idx, _ = owner
        start = times[j]
        end = times[j + 1] if j + 1 < len(times) else chunk_dur
        spans[word_idx].append((start, max(start, end)))

    results: List[Optional[dict]] = []
    for word_idx, word in enumerate(chunk):
        stamps = spans.get(word_idx, [])
        if len(stamps) != len(word.phones):
            results.append(None)
            continue

        # chunk timeline -> original timeline (the offset of a word is its start,
        # the pauses around it map linearly onto the original pauses)
        shift = word.start - offsets[word_idx]
        phonemes = []
        for phone, (start, end) in zip(word.phones, stamps):
            start = min(max(start + shift, 0.0), file_dur)
            end = min(max(end + shift, start), file_dur)
            phonemes.append([phone, round(start, 3), round(end, 3)])

        results.append(
            {
                "text": word.text,
                "start": phonemes[0][1],
                "end": phonemes[-1][2],
                "phonemes": phonemes,
            }
        )

    return results


def postprocess_file(
    batch: FileBatch,
    alignments: List[Optional[List[float]]],
    sample_rate: int,
) -> dict:
    """Flatten the aligned chunks into the word list of the whole file."""
    words: List[dict] = []
    punctuation: List[list] = []

    for chunk, owners, offsets, waveform, times in zip(
        batch.chunks, batch.owners, batch.offsets, batch.audios, alignments
    ):
        aligned = postprocess_chunk(
            chunk=chunk,
            owners=owners,
            offsets=offsets,
            times=times,
            chunk_dur=len(waveform) / sample_rate,
            file_dur=batch.duration,
        )
        for word, result in zip(chunk, aligned):
            if result is not None:
                words.append(result)
            # punctuation of a dropped word sticks to the last kept one
            for symbol in word.punct:
                punctuation.append([symbol, len(words)])

    return {"words": words, "punctuation": punctuation}


def build_phonemizer(kind: str) -> ESpeak:
    return ESpeak("en", library="espeak" if kind == "espeak-cli" else "espeak-lib")


# ---------------------------
# Data loading
# ---------------------------


class WordAlignmentDataset(Dataset):
    """Reads the Seamless alignment and the audio, and prepares the chunks of one file."""

    def __init__(
        self,
        paths: List[str],
        audio_dir: str,
        data_dir: Path,
        out_dir: Path,
        phonemizer: str = "espeak-lib",
        max_dur: float = 5.0,
        max_single: float = 29.0,
        pad: float = 0.1,
        pad_token: str = "[SIL]",
    ):
        self.paths = paths  # relative protocol paths
        self.audio_dir = audio_dir
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.kind = phonemizer
        self.max_dur = max_dur
        self.max_single = max_single
        self.pad = pad
        self.pad_token = pad_token
        self._ph = None

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def phonemizer(self):
        # espeak is not fork-safe to share, build it lazily per worker
        if self._ph is None:
            self._ph = build_phonemizer(self.kind)
        return self._ph

    def __getitem__(self, i: int) -> Optional[Tuple[FileBatch, int]]:
        path = self.paths[i]
        seamless_path = alignment_path(path, self.data_dir, SEAMLESS_SUFFIX)
        rel_path = (
            f"{seamless_path.parent.name}/{seamless_path.name[: -len(SEAMLESS_SUFFIX)]}"
        )
        try:
            with open(seamless_path) as f:
                meta = json.load(f)
            entries = meta.get("words") or []
            # one transcript entry for the whole utterance, in the seamless-interaction layout
            transcript = {
                "transcript": " ".join(w["word"] for w in entries),
                "words": entries,
            }
            words = parse_transcript([transcript], self.phonemizer)
            if not words:
                return None

            # the cache holds no audio path, it comes from the protocol parquet
            with open(audio_path(path, self.audio_dir), "rb") as f:
                raw = f.read()
            wav, sr = torchaudio.load(io.BytesIO(raw), format=AUDIO_SUFFIX.lstrip("."))
            # native rate, the aligner resamples on the GPU
            audio = wav.mean(0).numpy().astype(np.float32)
        except Exception:
            return None

        duration = len(audio) / sr
        words = [w for w in words if w.start < duration]
        for word in words:
            word.end = min(word.end, duration)
        words = [w for w in words if w.end > w.start]
        if not words:
            return None

        chunks, skipped = build_chunks(
            words, max_dur=self.max_dur, max_single=self.max_single, pad=self.pad
        )

        audios, tokens, owners, offsets = [], [], [], []
        for chunk in chunks:
            waveform, chunk_offsets = assemble_chunk(
                audio, chunk, sample_rate=sr, pad=self.pad, target_dur=self.max_dur
            )
            chunk_tokens, chunk_owners = build_tokens(chunk, self.pad_token)
            audios.append(waveform)
            offsets.append(chunk_offsets)
            tokens.append(chunk_tokens)
            owners.append(chunk_owners)

        batch = FileBatch(
            rel_path=rel_path,
            out_path=alignment_path(path, self.out_dir),
            duration=duration,
            chunks=chunks,
            audios=audios,
            tokens=tokens,
            owners=owners,
            offsets=offsets,
            skipped=skipped,
        )
        return batch, sr


def collate(
    items: List[Optional[Tuple[FileBatch, int]]],
) -> Tuple[List[Optional[Tuple[FileBatch, int]]], torch.Tensor]:
    """
    Move the chunk waveforms of all files of the item into one flat tensor.

    A tensor reaches the main process through shared memory, while numpy arrays are
    pickled through the worker pipe; at ~1.2 MB of chunks per file the pipe held a GPU
    to ~130 files/s. The arrays are put back as views by `uncollate`.
    """
    audios = [a for item in items if item is not None for a in item[0].audios]
    flat = np.concatenate(audios) if audios else np.zeros(0, dtype=np.float32)
    for item in items:
        if item is not None:
            # keep only the lengths, the samples travel in `flat`
            item[0].audios = [len(a) for a in item[0].audios]
    return items, torch.from_numpy(flat)


def uncollate(
    packed: Tuple[List[Optional[Tuple[FileBatch, int]]], torch.Tensor],
) -> List[Optional[Tuple[FileBatch, int]]]:
    items, flat = packed
    flat = flat.numpy()
    pos = 0
    for item in items:
        if item is None:
            continue
        lengths = item[0].audios
        item[0].audios = [
            flat[pos + s - n : pos + s] for n, s in zip(lengths, np.cumsum(lengths))
        ]
        pos += int(sum(lengths))
    return items


# ---------------------------
# Batched alignment
# ---------------------------


def align_chunks(
    aligner: ClapIPA, files: List[Tuple[FileBatch, int]], batch_size: int
) -> List[List[Optional[List[float]]]]:
    """Align the chunks of all `files` together, `batch_size` chunks per forward pass."""
    results: List[List[Optional[List[float]]]] = [
        [None] * len(batch.audios) for batch, _ in files
    ]

    # a forward pass resamples a single rate, so the chunks are grouped by it
    by_rate: Dict[int, List[Tuple[int, int]]] = {}
    for f, (batch, sr) in enumerate(files):
        for c in range(len(batch.audios)):
            by_rate.setdefault(sr, []).append((f, c))

    for sr, jobs in by_rate.items():
        # chunks are padded to 5 s, sorting only groups the rare longer ones
        jobs.sort(key=lambda fc: len(files[fc[0]][0].audios[fc[1]]))
        for i in range(0, len(jobs), batch_size):
            idx = jobs[i : i + batch_size]
            audios = [files[f][0].audios[c] for f, c in idx]
            phonemes = [
                [aligner.pad_token] + files[f][0].tokens[c] + [aligner.pad_token]
                for f, c in idx
            ]
            try:
                out = aligner(audios, phonemes, orig_sr=None if sr == SR else sr)
            except Exception:
                out = [None] * len(idx)
            for (f, c), times in zip(idx, out):
                results[f][c] = times
    return results


# ---------------------------
# Worker
# ---------------------------


def run_shard(rank: int, args, paths: List[str], total_files: int) -> None:
    torch.cuda.set_device(args.gpu_ids[rank])
    torch.set_num_threads(1)

    dataset = WordAlignmentDataset(
        paths,
        audio_dir=args.audio_dir,
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir or args.data_dir),
        phonemizer=args.phonemizer,
        max_dur=args.max_duration,
        max_single=args.max_single,
        pad=args.pad,
    )
    dl = DataLoader(
        dataset,
        batch_size=args.files_per_item,
        num_workers=args.num_workers,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=8 if args.num_workers > 0 else None,
        # this process is spawned, so its workers would be spawned as well and re-import
        # torch one by one (~10 s each); they never touch CUDA, forking them is safe
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )

    aligner = ClapIPA(sample_rate=SR)

    world = len(args.gpu_ids)
    pending: List[Tuple[FileBatch, int]] = []
    n_chunks = 0
    n_ok = n_empty = n_failed = 0
    secs = 0.0
    t_align = t_write = 0.0
    t0 = time.perf_counter()
    t_first: Optional[float] = None

    # rank 0 stands in for the whole job: the shards are round-robin slices of the same
    # protocol, so its counts scale by `world` and its ETA is the job's ETA
    bar = (
        tqdm(
            total=total_files,
            unit="file",
            unit_scale=True,
            smoothing=0.05,
            mininterval=5.0,
            dynamic_ncols=True,
            desc=f"align ({world} gpus, est.)",
        )
        if rank == 0
        else None
    )

    def flush() -> None:
        nonlocal n_ok, n_failed, n_chunks, secs, t_align, t_write
        if not pending:
            return
        t = time.perf_counter()
        alignments = align_chunks(aligner, pending, args.batch_size)
        t_align += time.perf_counter() - t
        t = time.perf_counter()
        for (batch, sr), times in zip(pending, alignments):
            secs += batch.duration
            result = postprocess_file(batch, times, sr)
            if not result["words"]:
                n_failed += 1
                continue
            batch.out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(batch.out_path, "w") as f:
                json.dump(result, f)
            n_ok += 1
        t_write += time.perf_counter() - t
        pending.clear()
        n_chunks = 0

    def tick(n: int) -> None:
        if bar is None:
            return
        el = time.perf_counter() - t0
        bar.set_postfix_str(
            f"{secs * world / 3600:,.0f} audio-h, {secs * world / el:,.0f} audio-h/h",
            refresh=False,
        )
        bar.update(n)

    for packed in dl:
        if t_first is None:
            t_first = time.perf_counter() - t0
        items = uncollate(packed)
        for item in items:
            if item is None:
                n_empty += 1
                continue
            pending.append(item)
            n_chunks += len(item[0].audios)
        tick(len(items) * world)
        if n_chunks >= args.chunks_per_pass:
            flush()
    flush()
    if bar is not None:
        bar.close()

    el = time.perf_counter() - t0
    print(
        f"rank {rank} (gpu {args.gpu_ids[rank]}): {n_ok:,} written, "
        f"{n_empty:,} unreadable, {n_failed:,} without alignment, "
        f"{secs / 3600:.2f} audio-h in {el / 60:.1f} min -> {secs / el:,.0f} audio-h/h | "
        f"startup {t_first or 0.0:.0f} s, align {t_align:.0f} s, write {t_write:.0f} s, "
        f"waiting for data {el - (t_first or 0.0) - t_align - t_write:.0f} s",
        flush=True,
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
        required=True,
        help="protocol parquet with the relative `path` of every utterance",
    )
    p.add_argument(
        "-a",
        "--audio-dir",
        required=True,
        help="absolute root of the audio: utterance `path` is `{audio_dir}/{path}.mp3`",
    )
    p.add_argument(
        "-d",
        "--data-dir",
        required=True,
        help="root of the `{parent}/{name}.seamless.json` alignment tree",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        default=None,
        help="write the JSON under this root instead of next to the Seamless alignment",
    )
    p.add_argument(
        "--gpus", default="0,1,4,5", help="comma-separated GPU ids, one process each"
    )
    p.add_argument(
        "-nw",
        "--num-workers",
        type=int,
        default=8,
        help="dataloader workers PER PROCESS",
    )
    p.add_argument(
        "--phonemizer", choices=["espeak-lib", "espeak-cli"], default="espeak-lib"
    )
    p.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=128,
        help="maximum number of chunks per forward pass",
    )
    p.add_argument(
        "--chunks-per-pass",
        type=int,
        default=512,
        help="collect files until they hold this many chunks, then align them",
    )
    p.add_argument(
        "--files-per-item", type=int, default=8, help="files per dataloader item"
    )
    p.add_argument(
        "-md",
        "--max-duration",
        type=float,
        default=5.0,
        help="maximum duration of a chunk, seconds",
    )
    p.add_argument(
        "-ms",
        "--max-single",
        type=float,
        default=29.0,
        help="maximum duration of a stand-alone word, seconds",
    )
    p.add_argument(
        "-p",
        "--pad",
        type=float,
        default=0.1,
        help="maximum pause kept between adjacent words, seconds",
    )
    p.add_argument(
        "-n", "--limit", type=int, default=None, help="only N files (test mode)"
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="with --limit, draw the N files at random with this seed (benchmarking)",
    )
    args = p.parse_args()

    args.gpu_ids = [int(x) for x in args.gpus.split(",")]
    world = len(args.gpu_ids)

    args.audio_dir = os.path.abspath(args.audio_dir)
    meta = pd.read_parquet(args.meta_path, columns=["path", "duration"])
    check_protocol(meta["path"].iloc[0])
    total_hours = meta["duration"].sum() / 3600
    if args.limit:
        if args.sample_seed is not None:
            meta = meta.sample(n=args.limit, random_state=args.sample_seed).sort_index()
        meta = meta.head(args.limit)
    print(
        f"{len(meta):,} files, {meta['duration'].sum() / 3600:,.1f} audio-h "
        f"(protocol: {total_hours:,.1f} h) | "
        f"{world} processes x {args.num_workers} workers | "
        f"phonemizer: {args.phonemizer}",
        flush=True,
    )

    paths = meta["path"].tolist()
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=run_shard, args=(r, args, paths[r::world], len(paths)))
        for r in range(world)
    ]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
