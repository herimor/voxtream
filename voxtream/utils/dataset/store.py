"""
On-disk layout of the training artifacts, and the reader the dataloader uses.

Every dataset (Emilia, HiFiTTS-2, ...) is its own store directory. A store holds
one training entry per speaker cluster, all of them cropped to the same number
of Mimi frames, so everything that is per frame is a single fixed-shape array
and everything that is per phoneme or per word is a CSR pair of a flat `values`
array and an `offsets` array of length `N + 1`:

    meta.json                     counts, frame rate, vocabulary sizes
    phone_tokens.values.npy       uint8  [total_phonemes, 5]  root/prefix/suffix/punct/eow
    phone_tokens.offsets.npy      int64  [N + 1]
    word_phone_start.values.npy   uint16 [total_words]   first phoneme of the word
    word_start_frame.values.npy   uint16 [total_words]
    word_end_frame.values.npy     uint16 [total_words]   exclusive, pause not included
    word_pause_after.values.npy   uint8  [total_words]   real silence after it, frames
    word_pause_before.values.npy  uint8  [total_words]   real silence before it, frames
    word_flags.values.npy         uint8  [total_words]   see WORD_* below
    words.offsets.npy             int64  [N + 1]
    frame_state.npy               uint8  [N, num_frames] phoneme shift state per frame
    silence_tokens.json           which codebook-0 tokens mean silence (optional)
    num_speech_frames.npy         uint16 [N]             frames before the silence pad
    spk_emb.npy                   float16[N, spk_dim]    cluster median, L2-normalized
    cluster_id.npy                int32  [N]             provenance
    mimi_codes.npy                uint16 [N, 16, num_frames]  written by the Mimi stage
    mimi_codes_16.npy             uint16 [N, 16, num_frames]  codebooks 16.. , if extracted

The Mimi codes are split into shards of 16 codebooks, each file named after the
index of its first codebook, so a run that trains on 16 never opens - and never
pages in - the codebooks above them. `read_codes` stops as soon as it has the
number asked for, and concatenates only when it has to.

No object arrays: a forked dataloader worker that touches a Python object in a
shared array bumps its refcount and copies the page it lives on, so an object
array is paid for once per worker. Flat arrays are opened with `mmap_mode="r"`
and cost nothing to fork.

Stores are never concatenated either. `MultiStore` keeps them side by side and
maps a global index onto `(store, local index)`, which keeps the CSR offsets of
each store valid as they were written.
"""

import json
from bisect import bisect_right
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, overload

import numpy as np

# word_flags bits
WORD_TERMINAL_PUNCT = 1 << 0  # the word is followed by "." / "?" / "!"
WORD_ANY_PUNCT = 1 << 1  # the word is followed by any punctuation
WORD_UTTERANCE_END = 1 << 2  # the word closes one of the concatenated utterances

PHONE_STREAMS = 5


class Ragged:
    """A CSR view: `self[i]` is `values[offsets[i]:offsets[i + 1]]`."""

    __slots__ = ("values", "offsets")

    def __init__(self, values: np.ndarray, offsets: np.ndarray) -> None:
        self.values = values
        self.offsets = offsets

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self.values[self.offsets[i] : self.offsets[i + 1]]

    def span(self, i: int) -> Tuple[int, int]:
        return int(self.offsets[i]), int(self.offsets[i + 1])


def states_to_indices(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    First and last phoneme of every frame, from the stored shift states.

    A state is `2 * shift + (n - 1)`: how far the pointer moves into the phoneme sequence
    at this frame and how many phonemes the frame holds. The absolute index is the
    running sum, which starts at phoneme 0 because the first frame always has shift 0.
    """
    shift = (state >> 1).astype(np.int64)
    n = (state & 1).astype(np.int64) + 1
    last = np.cumsum(shift + n - 1)
    return last - (n - 1), last


def indices_to_states(first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """
    Inverse of `states_to_indices`; the first frame is taken to start at phoneme 0.

    Returns int64 and may fall outside `0..2 * MAX_SHIFT + 1`, which means the frames do
    not form a walk the pointer can make: the caller checks before narrowing to uint8,
    where a negative shift would wrap into a valid-looking state.
    """
    shift = np.r_[0, first[1:] - last[:-1]]
    return 2 * shift + (last - first)


def build_offsets(lengths: Sequence[int]) -> np.ndarray:
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
    return offsets


class DatasetStore:
    """One dataset's artifacts, memory-mapped."""

    def __init__(self, root: Path, load_mimi: bool = True) -> None:
        self.root = Path(root)
        with open(self.root / "meta.json") as f:
            self.meta = json.load(f)

        self.num_frames: int = self.meta["num_frames"]
        self.mimi_tps: float = self.meta["mimi_tps"]

        self.phone_tokens = self._ragged("phone_tokens", "phone_tokens")
        self.word_phone_start = self._ragged("words", "word_phone_start")
        self.word_start_frame = self._ragged("words", "word_start_frame")
        self.word_end_frame = self._ragged("words", "word_end_frame")
        # measured against the neighbouring words of the alignment, not the neighbouring
        # rows of this table, which lists only the words a crop may start at
        self.word_pause_after = self._ragged("words", "word_pause_after")
        self.word_pause_before = self._ragged("words", "word_pause_before")
        self.word_flags = self._ragged("words", "word_flags")

        self.frame_state = self._load("frame_state.npy")
        # `silence_tokens.py` calibrates these; a crop needs them to place `[EOS]`
        # on frames that are really silent, which the alignment cannot tell it
        self.silence_tokens: Optional[np.ndarray] = None
        path = self.root / "silence_tokens.json"
        if path.exists():
            with open(path) as f:
                mask = np.zeros(2048, dtype=bool)
                mask[json.load(f)["tokens"]] = True
            self.silence_tokens = mask
        self.num_speech_frames = self._load("num_speech_frames.npy")
        self.spk_emb = self._load("spk_emb.npy")
        self.cluster_id = self._load("cluster_id.npy", required=False)

        # shard k starts at codebook `offset`, and is named after it; the chain ends
        # at the first name that is not there
        self.mimi_shards: List[np.ndarray] = []
        if load_mimi:
            offset = 0
            while True:
                name = "mimi_codes.npy" if not offset else f"mimi_codes_{offset}.npy"
                shard = self._load(name, required=False)
                if shard is None:
                    break
                self.mimi_shards.append(shard)
                offset += shard.shape[1]
        self.num_mimi_codebooks: int = sum(s.shape[1] for s in self.mimi_shards)

    def read_codes(
        self, i: int, num_codebooks: int, start: int, stop: int
    ) -> np.ndarray:
        """`[num_codebooks, stop - start]` of entry `i`, across as few shards as needed."""
        blocks, taken = [], 0
        for shard in self.mimi_shards:
            if taken >= num_codebooks:
                break  # the shards past what was asked for are never touched
            want = min(shard.shape[1], num_codebooks - taken)
            blocks.append(np.asarray(shard[i, :want, start:stop]))
            taken += want
        if taken < num_codebooks:
            raise ValueError(
                f"{self.root} holds {taken} Mimi codebooks, {num_codebooks} were asked for"
            )
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)

    @overload
    def _load(self, name: str, required: Literal[True] = ...) -> np.ndarray: ...
    @overload
    def _load(self, name: str, required: bool) -> Optional[np.ndarray]: ...
    def _load(self, name: str, required: bool = True) -> Optional[np.ndarray]:
        path = self.root / name
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return None
        return np.load(path, mmap_mode="r")

    def _ragged(self, group: str, name: str) -> Ragged:
        # the offsets are small and are read often, so they stay in RAM
        offsets = np.load(self.root / f"{group}.offsets.npy")
        values = np.load(self.root / f"{name}.values.npy", mmap_mode="r")
        return Ragged(values, offsets)

    def __len__(self) -> int:
        return len(self.frame_state)


class MultiStore:
    """Several stores addressed by one global index."""

    def __init__(self, stores: List[DatasetStore]) -> None:
        if not stores:
            raise ValueError("no dataset stores given")
        self.stores = stores
        self.bounds: List[int] = [0]
        for store in stores:
            self.bounds.append(self.bounds[-1] + len(store))

        first = stores[0]
        for store in stores[1:]:
            if store.num_frames != first.num_frames:
                raise ValueError(
                    f"{store.root} holds {store.num_frames}-frame entries, "
                    f"{first.root} holds {first.num_frames}"
                )
        self.num_frames = first.num_frames
        self.mimi_tps = first.mimi_tps

    def __len__(self) -> int:
        return self.bounds[-1]

    def locate(self, index: int) -> Tuple[DatasetStore, int]:
        which = bisect_right(self.bounds, index) - 1
        return self.stores[which], index - self.bounds[which]
