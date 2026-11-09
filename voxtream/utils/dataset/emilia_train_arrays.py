"""
Training artifacts of the speaker clusters: phonemes, word borders and frame alignment.

Reads the crop plan (`emilia_crop_plan.py`) and the Clap-IPA alignments
(`phoneme_align.py`) and writes one store (`store.py`) holding, per cluster,
everything the dataloader needs except the Mimi codes, which the Mimi stage writes into
the same directory against the same plan.

Per entry the builder lays the phoneme alignments of its utterances on the cluster
timeline and then does what `PhonemeAligner` did per training sample, once per entry:

  - Phonemes are made contiguous. A phoneme runs until the next one starts, so a pause
    between two words extends the last phoneme of the first word. The pause is not a
    token of its own; the end-of-word stream tells the model where the word ended and
    hence why that phoneme is long.

  - Every 80 ms frame takes the two phonemes it holds most of. A phoneme too short to win
    a slot in any frame is dropped from the sequence, except that one of them survives in
    each gap: the frame-to-phoneme pointer may advance by at most two phonemes per frame,
    and a shift of two is exactly "one phoneme was passed over". That bound is what makes
    the six-state shift label of the semantic head enough to drive the pointer.

  - The alignment is stored as one shift state per frame, `2 * shift + (n_phonemes - 1)`,
    the same encoding the generator steps with at inference. Absolute phoneme indices are
    a cumulative sum away, and keeping the relative form is what lets the dataloader crop
    anywhere without rewriting the alignment.

The phoneme sequence is five streams (`phoneme_vocab.py`): root, stress prefix, length
suffix, the punctuation that follows the word, and an end-of-word mark. Punctuation rides
on the phoneme that precedes it instead of being inserted into the sequence and deleted
from the transformer output, so phoneme indices and tokens stay one to one.

Word borders are stored so the dataloader can cut a training crop at a word rather than
mid-phoneme: first phoneme, first and last frame, and flags for the punctuation that
follows and for the utterance joins.

Output in `--out-dir` (see `store.py` for the full layout):
    meta.json  phone_tokens.*  words.*  word_*.values.npy
    frame_state.npy  num_speech_frames.npy  spk_emb.npy  cluster_id.npy
    phoneme_vocab.json

Example:
    python -m voxtream.utils.dataset.emilia_train_arrays \
        -m emilia/metadata.parquet -p emilia/crop_plan -e emilia/spk_emb.npy \
        -v valid_phoneme_vocab.json -d /data/alignment/emilia \
        -o /data/stores/emilia -nj 32
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from voxtream.utils.dataset.emilia_crop_plan import load_plan
from voxtream.utils.dataset.parse_phoneme_alignment import loads
from voxtream.utils.dataset.paths import alignment_path, check_protocol
from voxtream.utils.dataset.phoneme_vocab import (
    EOW_SET,
    NUM_STREAMS,
    PUNCT_PAD,
    TERMINAL_PUNCT_TOKENS,
    PhonemeVocab,
    punct_token,
)
from voxtream.utils.dataset.store import (
    PHONE_STREAMS,
    WORD_ANY_PUNCT,
    WORD_TERMINAL_PUNCT,
    WORD_UTTERANCE_END,
    build_offsets,
    indices_to_states,
    states_to_indices,
)

MAX_SHIFT = 2  # phonemes the pointer may advance in one frame
CUT_TOLERANCE_SEC = 0.02  # rounding slack when matching a word against the plan's cut
PHONES_PER_FRAME = 2

# set before the workers are forked, see `parse_phoneme_alignment.PATHS`
PATHS: Optional[pa.ChunkedArray] = None
SPK_EMB: Optional[np.ndarray] = None
VOCAB: Optional[PhonemeVocab] = None
PLAN: Optional[np.ndarray] = (
    None  # [n_segments, 4] of (row, src_start, src_end, dst_start)
)
CLUSTERS: Optional[np.ndarray] = None  # [n_clusters, 4] of (id, start, count, frames)
CFG: Optional["Settings"] = None


class Settings:
    def __init__(self, args: argparse.Namespace, meta: dict) -> None:
        self.data_dir = Path(args.data_dir)
        self.sr = meta["sample_rate"]
        frame_ms = 1000 * meta["frame_samples"] / meta["sample_rate"]
        if frame_ms != int(frame_ms):
            raise ValueError(
                f"a frame is {frame_ms} ms, which this builder cannot time in whole "
                "milliseconds; change the frame rate or move the timeline to samples"
            )
        self.frame_ms = int(frame_ms)
        self.num_frames = meta["num_frames"]
        self.min_phonemes = args.min_phonemes


# ---------------------------
# Timeline
# ---------------------------


class Word:
    """One word on the cluster timeline."""

    __slots__ = ("start_ms", "end_ms", "phonemes", "punct", "utterance_end")

    def __init__(self, start_ms: int, end_ms: int, phonemes: List[Tuple[str, int]]):
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.phonemes = phonemes  # (phoneme, start_ms)
        self.punct = PUNCT_PAD
        self.utterance_end = False


def collect_words(segments: np.ndarray, cfg: Settings) -> List[Word]:
    """Lay the alignments of a cluster's utterances on one timeline, in ms."""
    words: List[Word] = []
    for row, src_start, src_end, dst_start in segments:
        try:
            with open(alignment_path(PATHS[int(row)].as_py(), cfg.data_dir), "rb") as f:
                data = loads(f.read())
        except (FileNotFoundError, KeyError, ValueError):
            continue
        src_words = data.get("words") or []
        if not src_words:
            continue

        # the plan takes `[src_start, src_end)` of this file; keep the words inside it.
        # a file whose sentences sit too far apart is planned as several segments, so
        # this may be the second or a later stretch of the same alignment
        offset_ms = (int(dst_start) - int(src_start)) * 1000.0 / cfg.sr
        # the aligner rounds to the millisecond and the protocol duration is itself
        # approximate, so a last word may end a hair past the cut; the cut the plan
        # really makes is a word boundary plus its pause budget, far outside this
        start_sec = int(src_start) / cfg.sr - CUT_TOLERANCE_SEC
        end_sec = int(src_end) / cfg.sr + CUT_TOLERANCE_SEC
        kept: List[Word] = []
        first_word = 0
        for w, word in enumerate(src_words):
            if word["end"] <= start_sec:  # belongs to an earlier segment of this file
                first_word = w + 1
                continue
            if word["end"] > end_sec or not word["phonemes"]:
                break
            kept.append(
                Word(
                    round(word["start"] * 1000 + offset_ms),
                    round(word["end"] * 1000 + offset_ms),
                    [
                        (ph, round(s * 1000 + offset_ms))
                        for ph, s, _ in word["phonemes"]
                    ],
                )
            )
        if not kept:
            continue

        # punctuation is `[symbol, insertion index in the word list]`, so it follows the
        # word before that index; several symbols on one word collapse to the strongest
        per_word: Dict[int, str] = {}
        for symbol, index in data.get("punctuation") or []:
            w = int(index) - 1 - first_word  # `index` counts from the start of the file
            if 0 <= w < len(kept):
                per_word[w] = per_word.get(w, "") + symbol
        for w, symbols in per_word.items():
            kept[w].punct = punct_token(symbols)

        kept[-1].utterance_end = True
        words.extend(kept)
    return words


# ---------------------------
# Frame alignment
# ---------------------------


def ragged_arange(counts: np.ndarray) -> np.ndarray:
    """`0..c0-1, 0..c1-1, ...` for every count, as one flat array."""
    counts = counts[counts > 0]
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    out[0] = 0
    out[np.cumsum(counts)[:-1]] = 1 - counts[:-1]
    return np.cumsum(out)


def assign_frames(
    bounds: np.ndarray, num_frames: int, frame_ms: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Give every frame the phoneme it holds most of, plus that phoneme's louder neighbour.

    `bounds[p]` is the start of phoneme `p` in ms and `bounds[-1]` the end of the last
    one; phonemes are contiguous and cover `[0, num_frames * frame_ms)`.

    The pair is always two phonemes that follow each other, never the two longest of the
    frame with a short one between them: the generator walks the sequence with a single
    pointer and a span, so it can only ever hold neighbours, and a training frame it
    cannot represent would teach a pointer move that does not exist.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (first phoneme of every frame, last phoneme of every frame)
    """
    starts, ends = bounds[:-1], bounds[1:]
    # one (phoneme, frame) pair per frame a phoneme reaches into
    f_lo = np.maximum(starts // frame_ms, 0)
    f_hi = np.minimum((ends - 1) // frame_ms, num_frames - 1)
    n_touch = np.maximum(f_hi - f_lo + 1, 0)
    phone = np.repeat(np.arange(len(starts)), n_touch)
    frame = ragged_arange(n_touch) + np.repeat(f_lo, n_touch)
    overlap = np.minimum(ends[phone], (frame + 1) * frame_ms) - np.maximum(
        starts[phone], frame * frame_ms
    )

    if not len(frame):
        return np.full(num_frames, -1, dtype=np.int64), np.full(
            num_frames, -1, dtype=np.int64
        )

    # group by frame, phonemes of a frame ascending and therefore consecutive
    order = np.lexsort((phone, frame))
    phone, frame, overlap = phone[order], frame[order], overlap[order]
    group_start = np.flatnonzero(np.r_[True, frame[1:] != frame[:-1]])
    group_size = np.diff(np.r_[group_start, len(frame)])

    # the longest phoneme of each frame, by its position inside the group
    group_max = np.repeat(np.maximum.reduceat(overlap, group_start), group_size)
    position = ragged_arange(group_size)
    best_at = np.minimum.reduceat(
        np.where(overlap == group_max, position, len(frame)), group_start
    )
    best = group_start + best_at

    left = np.where(best_at > 0, overlap[np.maximum(best - 1, 0)], -1)
    right = np.where(
        best_at < group_size - 1, overlap[np.minimum(best + 1, len(frame) - 1)], -1
    )
    alone = group_size == 1
    take_right = right >= left

    first = np.full(num_frames, -1, dtype=np.int64)
    last = np.full(num_frames, -1, dtype=np.int64)
    frames = frame[group_start]
    first[frames] = phone[best] - (~alone & ~take_right)
    last[frames] = phone[best] + (~alone & take_right)
    return first, last


def drop_skipped(first: np.ndarray, last: np.ndarray, n_phonemes: int) -> np.ndarray:
    """
    Keep mask over the phonemes that bounds the per-frame shift to `MAX_SHIFT`.

    Between the last phoneme of a frame and the first of the next there may be phonemes
    no frame holds. All but the last of them go: what is left is a shift of two, which
    the pointer crosses in one step. Phonemes before the first frame's own go as well,
    so the sequence starts where the alignment does.
    """
    keep = np.ones(n_phonemes, dtype=bool)
    keep[: first[0]] = False
    gap = first[1:] - last[:-1]  # >= 0, the frames are in time order
    (over,) = np.nonzero(gap > MAX_SHIFT)
    if len(over):
        n_drop = gap[over] - MAX_SHIFT
        idx = np.repeat(last[over] + 1, n_drop) + ragged_arange(n_drop)
        keep[idx] = False
    return keep


# ---------------------------
# One entry
# ---------------------------


class Entry:
    __slots__ = ("phone_tokens", "word_arrays", "frame_state", "num_speech_frames")

    def __init__(self, phone_tokens, word_arrays, frame_state, num_speech_frames):
        self.phone_tokens = phone_tokens
        self.word_arrays = word_arrays
        self.frame_state = frame_state
        self.num_speech_frames = num_speech_frames


def build_entry(
    segments: np.ndarray, num_speech_frames: int, cfg: Settings, vocab: PhonemeVocab
) -> Optional[Entry]:
    words = collect_words(segments, cfg)
    if not words:
        return None

    num_speech_frames = int(min(num_speech_frames, cfg.num_frames))
    speech_end_ms = num_speech_frames * cfg.frame_ms

    # flatten to phonemes, carrying the word each one belongs to
    starts: List[int] = []
    owners: List[int] = []
    raw: List[str] = []
    for w, word in enumerate(words):
        for phoneme, start_ms in word.phonemes:
            starts.append(start_ms)
            owners.append(w)
            raw.append(phoneme)
    if len(raw) < cfg.min_phonemes:
        return None

    start_ms = np.asarray(starts, dtype=np.int64)
    owner = np.asarray(owners, dtype=np.int64)
    # phonemes run until the next one begins: the first covers the leading silence, the
    # last covers the pause and the trimmed tail that close the recording. Words in the
    # alignment can overlap, so a phoneme has to beat every start before it, not just the
    # one in front of it, or a zero-length phoneme survives and no audio backs it.
    keep_time = np.r_[True, start_ms[1:] > np.maximum.accumulate(start_ms)[:-1]] & (
        start_ms < speech_end_ms
    )
    if keep_time.sum() < cfg.min_phonemes:
        return None
    start_ms, owner = start_ms[keep_time], owner[keep_time]
    raw = [raw[i] for i in np.flatnonzero(keep_time)]
    bounds = np.r_[0, start_ms[1:], speech_end_ms]

    first, last = assign_frames(bounds, num_speech_frames, cfg.frame_ms)
    if (first < 0).any():  # a frame no phoneme reaches, i.e. a broken alignment
        return None

    keep = drop_skipped(first, last, len(raw))
    new_index = np.cumsum(keep) - 1
    first, last = new_index[first], new_index[last]
    owner = owner[keep]
    raw = [raw[i] for i in np.flatnonzero(keep)]
    n_phonemes = len(raw)

    # the shift states are what gets stored, so derive the indices back out of them and
    # build everything else against those: whatever the dataloader reconstructs is then
    # exactly what the word table and the phoneme sequence were written for
    state = indices_to_states(first, last)
    if state.min() < 0 or state.max() > 2 * MAX_SHIFT + 1 or (last - first).max() > 1:
        return None  # not a walk the pointer can make, so not a usable entry
    state = state.astype(np.uint8)
    first, last = states_to_indices(state)

    # five streams
    tokens = np.zeros((n_phonemes, NUM_STREAMS), dtype=np.uint8)
    for i, phoneme in enumerate(raw):
        tokens[i, 0], tokens[i, 1], tokens[i, 2] = vocab.normalize(phoneme)
    # the last phoneme of a word carries its end mark and whatever follows it
    word_last = np.flatnonzero(np.r_[np.diff(owner) > 0, True])
    tokens[word_last, 4] = EOW_SET
    punct = np.array([w.punct for w in words], dtype=np.uint8)
    tokens[word_last, 3] = punct[owner[word_last]]

    # word table: only the words a crop may start at, i.e. whose first phoneme opens a
    # frame of its own; `start_frame` comes from the alignment so that cropping there
    # lands exactly on that phoneme
    phone_start = np.flatnonzero(np.r_[True, np.diff(owner) > 0])
    present = owner[phone_start]
    cand = np.searchsorted(first, phone_start, side="left")
    usable = (cand < num_speech_frames) & (
        first[np.minimum(cand, len(first) - 1)] == phone_start
    )
    if not usable.any():
        return None
    phone_start, present, start_frame = (
        phone_start[usable],
        present[usable],
        cand[usable],
    )

    # The silence around a word is measured against its neighbours in the full word list,
    # not against the previous row of this table: a word that failed `usable` is still
    # spoken, and treating it as part of a pause would let a crop end on [SIL] + [EOS]
    # while the audio runs straight on into the next word.
    # counted in whole frames rather than milliseconds, so that a pause of n frames
    # really is n frames holding no speech: a word owns every frame its audio touches
    all_first = np.array([w.start_ms for w in words], dtype=np.int64) // cfg.frame_ms
    all_last = -(
        -np.array([w.end_ms for w in words], dtype=np.int64) // cfg.frame_ms
    )  # ceil
    # words in the alignment can overlap and are not strictly ordered, so a gap is
    # measured against every word before and after it, not against its two neighbours:
    # the span between the latest end so far and the earliest start still to come is the
    # only interval guaranteed to hold no speech at all
    latest_end = np.maximum.accumulate(all_last)
    earliest_start = np.minimum.accumulate(all_first[::-1])[::-1]
    next_start = np.r_[earliest_start[1:], num_speech_frames][present]
    prev_end = np.r_[0, latest_end[:-1]][present]

    # the pauses are measured against the frames actually stored, so that
    # `end_frame + pause_after` is exactly where the next word's audio begins
    end_frame = np.clip(latest_end[present], start_frame + 1, num_speech_frames)

    def to_u8(f):
        return np.clip(f, 0, 255).astype(np.uint8)

    pause_after = to_u8(next_start - end_frame)
    pause_before = to_u8(start_frame - prev_end)

    flags = np.zeros(len(present), dtype=np.uint8)
    w_punct = punct[present]
    flags |= np.where(w_punct != PUNCT_PAD, WORD_ANY_PUNCT, 0).astype(np.uint8)
    flags |= np.where(
        np.isin(w_punct, list(TERMINAL_PUNCT_TOKENS)), WORD_TERMINAL_PUNCT, 0
    ).astype(np.uint8)
    flags |= np.where(
        [words[w].utterance_end for w in present], WORD_UTTERANCE_END, 0
    ).astype(np.uint8)

    frame_state = np.zeros(cfg.num_frames, dtype=np.uint8)
    frame_state[:num_speech_frames] = state

    return Entry(
        phone_tokens=tokens,
        word_arrays=(
            phone_start.astype(np.uint16),
            start_frame.astype(np.uint16),
            end_frame.astype(np.uint16),
            pause_after,
            pause_before,
            flags,
        ),
        frame_state=frame_state,
        num_speech_frames=num_speech_frames,
    )


# ---------------------------
# Workers
# ---------------------------


def build_shard(job: Tuple[int, int]) -> Tuple[int, dict]:
    """Build the entries of clusters `[start, stop)`; returns the shard index and arrays."""
    start, stop = job
    phone_blocks, word_blocks = [], []
    phone_lengths, word_lengths = [], []
    frame_states, speech_frames, spk_embs, cluster_ids = [], [], [], []

    for i in range(start, stop):
        cluster_id, plan_start, plan_count, num_speech_frames = CLUSTERS[i]
        segments = PLAN[plan_start : plan_start + plan_count]
        entry = build_entry(segments, num_speech_frames, CFG, VOCAB)
        if entry is None:
            continue

        # the median over the utterances of the cluster, skipping the zero rows the
        # speaker encoder leaves behind for files it could not read
        emb = np.asarray(SPK_EMB[segments[:, 0]], dtype=np.float32)
        emb = emb[np.linalg.norm(emb, axis=1) > 0.5]
        if not len(emb):
            continue
        emb = np.median(emb, axis=0)
        norm = np.linalg.norm(emb)
        if not np.isfinite(norm) or norm <= 0:
            continue

        phone_blocks.append(entry.phone_tokens)
        phone_lengths.append(len(entry.phone_tokens))
        word_blocks.append(entry.word_arrays)
        word_lengths.append(len(entry.word_arrays[0]))
        frame_states.append(entry.frame_state)
        speech_frames.append(entry.num_speech_frames)
        spk_embs.append((emb / norm).astype(np.float16))
        cluster_ids.append(cluster_id)

    if not cluster_ids:
        return start, {}
    return start, {
        "phone_tokens": np.concatenate(phone_blocks),
        "phone_lengths": np.asarray(phone_lengths, dtype=np.int64),
        "word_phone_start": np.concatenate([b[0] for b in word_blocks]),
        "word_start_frame": np.concatenate([b[1] for b in word_blocks]),
        "word_end_frame": np.concatenate([b[2] for b in word_blocks]),
        "word_pause_after": np.concatenate([b[3] for b in word_blocks]),
        "word_pause_before": np.concatenate([b[4] for b in word_blocks]),
        "word_flags": np.concatenate([b[5] for b in word_blocks]),
        "word_lengths": np.asarray(word_lengths, dtype=np.int64),
        "frame_state": np.stack(frame_states),
        "num_speech_frames": np.asarray(speech_frames, dtype=np.uint16),
        "spk_emb": np.stack(spk_embs),
        "cluster_id": np.asarray(cluster_ids, dtype=np.int32),
    }


# ---------------------------
# Streaming .npy writer
# ---------------------------

_HEADER_LEN = 128


class NpyWriter:
    """Append rows to a .npy whose row count is only known at the end."""

    def __init__(self, path: Path, dtype: np.dtype, row_shape: Tuple[int, ...] = ()):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.row_shape = tuple(row_shape)
        self.rows = 0
        self._header(0)  # fail now rather than after the data is on disk
        self.f = open(path, "wb")
        self.f.write(b"\x00" * _HEADER_LEN)

    def append(self, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array, dtype=self.dtype)
        if array.shape[1:] != self.row_shape:
            raise ValueError(
                f"{self.path} takes rows of {self.row_shape}, got {array.shape[1:]}"
            )
        self.f.write(array.tobytes())
        self.rows += len(array) if array.ndim else 1

    def _header(self, rows: int) -> bytes:
        header = (
            "{'descr': %r, 'fortran_order': False, 'shape': %r, }"
            % (self.dtype.str, (rows,) + self.row_shape)
        ).encode("latin1")
        pad = _HEADER_LEN - 10 - len(header)
        if pad < 1:
            raise ValueError(f"header too long for {self.path}")
        return (
            b"\x93NUMPY\x01\x00"
            + (len(header) + pad).to_bytes(2, "little")
            + header
            + b" " * (pad - 1)
            + b"\n"
        )

    def close(self) -> None:
        blob = self._header(self.rows)
        self.f.seek(0)
        self.f.write(blob)
        self.f.close()


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
        help="protocol parquet the plan rows point into",
    )
    p.add_argument(
        "-p",
        "--plan-dir",
        required=True,
        help="crop plan of `emilia_crop_plan.py`",
    )
    p.add_argument(
        "-e",
        "--spk-emb-path",
        required=True,
        help="L2-normalized speaker embeddings, row-aligned with the protocol",
    )
    p.add_argument(
        "-v",
        "--vocab-path",
        required=True,
        help="`{root: count}` of the normalized phoneme roots",
    )
    p.add_argument(
        "-d",
        "--data-dir",
        required=True,
        help="root of the `{parent}/{name}.json` alignment tree",
    )
    p.add_argument("-o", "--out-dir", required=True, help="where the store goes")
    p.add_argument(
        "--min-phonemes",
        type=int,
        default=200,
        help="drop an entry with fewer phonemes than this",
    )
    p.add_argument("-nj", "--num-jobs", type=int, default=32)
    p.add_argument(
        "--shard-size", type=int, default=2000, help="clusters per worker task"
    )
    p.add_argument(
        "-n", "--limit", type=int, default=None, help="only N clusters (test mode)"
    )
    args = p.parse_args()

    global PATHS, SPK_EMB, VOCAB, PLAN, CLUSTERS, CFG
    plan, clusters, meta = load_plan(Path(args.plan_dir))
    PLAN = plan[["row", "src_start", "src_end", "dst_start"]].to_numpy(np.int64)
    CLUSTERS = clusters[
        ["cluster_id", "plan_start", "plan_count", "num_speech_frames"]
    ].to_numpy(np.int64)
    PATHS = (
        pq.read_table(args.meta_path, columns=["path"]).column("path").combine_chunks()
    )
    check_protocol(PATHS[0].as_py())
    SPK_EMB = np.load(args.spk_emb_path, mmap_mode="r")
    VOCAB = PhonemeVocab.from_counts(Path(args.vocab_path))
    CFG = Settings(args, meta)

    n_clusters = min(len(CLUSTERS), args.limit or len(CLUSTERS))
    print(
        f"{n_clusters:,} entries, {CFG.num_frames} frames each, "
        f"{VOCAB.root_vocab_size} roots",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "phone_tokens": NpyWriter(
            out_dir / "phone_tokens.values.npy", np.uint8, (PHONE_STREAMS,)
        ),
        "word_phone_start": NpyWriter(
            out_dir / "word_phone_start.values.npy", np.uint16
        ),
        "word_start_frame": NpyWriter(
            out_dir / "word_start_frame.values.npy", np.uint16
        ),
        "word_end_frame": NpyWriter(out_dir / "word_end_frame.values.npy", np.uint16),
        "word_pause_after": NpyWriter(
            out_dir / "word_pause_after.values.npy", np.uint8
        ),
        "word_pause_before": NpyWriter(
            out_dir / "word_pause_before.values.npy", np.uint8
        ),
        "word_flags": NpyWriter(out_dir / "word_flags.values.npy", np.uint8),
        "frame_state": NpyWriter(
            out_dir / "frame_state.npy", np.uint8, (CFG.num_frames,)
        ),
        "num_speech_frames": NpyWriter(out_dir / "num_speech_frames.npy", np.uint16),
        "spk_emb": NpyWriter(out_dir / "spk_emb.npy", np.float16, (SPK_EMB.shape[1],)),
        "cluster_id": NpyWriter(out_dir / "cluster_id.npy", np.int32),
    }
    phone_lengths: List[np.ndarray] = []
    word_lengths: List[np.ndarray] = []

    jobs = [
        (i, min(i + args.shard_size, n_clusters))
        for i in range(0, n_clusters, args.shard_size)
    ]
    n_entries = 0
    with (
        mp.get_context("fork").Pool(args.num_jobs) as pool,
        tqdm(total=n_clusters, unit="entry", unit_scale=True) as bar,
    ):
        for start, shard in pool.imap(build_shard, jobs):
            if shard:
                for name, writer in writers.items():
                    writer.append(shard[name])
                phone_lengths.append(shard["phone_lengths"])
                word_lengths.append(shard["word_lengths"])
                n_entries += len(shard["cluster_id"])
            bar.update(min(args.shard_size, n_clusters - start))

    if not n_entries:
        for writer in writers.values():
            writer.close()
        print("no entries were built; check the plan and the alignment tree")
        return

    for writer in writers.values():
        writer.close()
    np.save(
        out_dir / "phone_tokens.offsets.npy",
        build_offsets(np.concatenate(phone_lengths)),
    )
    np.save(out_dir / "words.offsets.npy", build_offsets(np.concatenate(word_lengths)))
    VOCAB.save(out_dir / "phoneme_vocab.json")

    n_phonemes = int(np.concatenate(phone_lengths).sum())
    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "num_entries": n_entries,
                "num_frames": CFG.num_frames,
                "mimi_tps": meta["mimi_tps"],
                "sample_rate": CFG.sr,
                "num_phonemes": n_phonemes,
                "phone_streams": PHONE_STREAMS,
                "vocab_sizes": list(VOCAB.vocab_sizes),
                "spk_emb_dim": int(SPK_EMB.shape[1]),
                # the name only: a store is shared without the machine it was built on
                "plan_dir": Path(args.plan_dir).resolve().name,
            },
            f,
            indent=2,
        )
    print(
        f"{n_entries:,} entries of {n_clusters:,} clusters, "
        f"{n_phonemes:,} phonemes ({n_phonemes / max(n_entries, 1):.0f} per entry)"
    )


if __name__ == "__main__":
    main()
