"""
Training dataset of the factored phoneme vocabulary (`DATASET_POOL["aligned"]`): a
word-aligned crop out of an 80 s speaker cluster.

Each store entry (`utils/dataset/store.py`) is one speaker cluster laid out on a single
timeline and padded with silence to a fixed number of frames. A training sample is a
window of `crop_frames` cut out of it, and both ends of that window are placed on word
borders taken from the alignment, never in the middle of one:

  - The end is drawn from the word borders near the end of the entry, which is what gives
    a long-form model varied endings instead of one fixed position. Two kinds are drawn
    with equal probability:
      * a sentence end, a word with terminal punctuation followed by a real pause. The
        release is held on the last phoneme, the way every other pause in the entry is,
        and then the pointer steps onto the `[EOS]` appended behind it and stands there
        through the rest of the silence - exactly the shape the generator builds at
        inference. This is the sample that teaches the model to stop, and the step onto
        `[EOS]` is supervised rather than left to the model to invent: a pause held on
        the last phoneme is what every sentence boundary inside an entry looks like, and
        those are all followed by more speech, so a model never shown the step reads the
        end of the text as one of them and opens a sentence the text never asked for.
      * an open end, any word border with a short pause and no `[EOS]`. The next
        phonemes of the cluster are appended past the end of the window, unaligned, so
        the phoneme transformer has the look-ahead it gets mid-utterance at inference.
        Without these, every late frame position in training would mean "the text is
        about to run out", and the model would race the pointer whenever generation ran
        longer than a training crop.

  - The start is the first word border at or after `end - crop_frames`, preceded by a
    pause. Whatever is left over at the end of the window is padding and is masked out of
    the loss, along with the silence that pads the entry itself.

  - The acoustic prompt is the head of the window, its text masked with `[PROMPT]`. Its
    length is drawn in `prompt_sec` and then snapped to a word border, so the text the
    model is actually asked to speak starts at a word, as it does at inference.

Punctuation and word borders are two of the five phoneme streams rather than tokens of
their own, and a share of them is dropped on the fly: all of them for a few samples, the
punctuation alone for a few phonemes, and both at the open end of a crop, where a
streaming frontend would not know yet what follows the last word.

Sampling cannot fail. `_candidates` works out, per entry, every crop end that is certain
to yield a complete sample - the end fits, a word border can open the window, a prompt
border leaves enough content behind it, and a sentence end really holds its pause - and
the draw only ever sees those. Entries that offer none are found once at construction and
left out of the index, so a training step never retries, never silently substitutes a
neighbouring entry, and never raises halfway through a run. That check costs one pass over
the store and is cached beside it, keyed by the crop settings that decide the answer.
"""

import hashlib
import json
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from voxtream.utils.dataset.phoneme_vocab import (
    EOW_PAD,
    NUM_STREAMS,
    PUNCT_PAD,
    ROOT_BOS,
    ROOT_EOS,
    ROOT_PAD,
    ROOT_PROMPT,
)
from voxtream.utils.dataset.store import (
    WORD_TERMINAL_PUNCT,
    DatasetStore,
    MultiStore,
    states_to_indices,
)

PUNCT_STREAM, EOW_STREAM = 3, 4
_VALIDATION_SHARD = 20_000  # entries per validation task
PROMPT_STATE = 2  # (shift 1, one phoneme): the pointer walks the prompt one per frame


@dataclass
class Entry:
    """One store entry, with the frame-to-phoneme pointer reconstructed."""

    num_speech: int
    first: np.ndarray
    last: np.ndarray
    start_frame: np.ndarray
    end_frame: np.ndarray
    phone_start: np.ndarray
    flags: np.ndarray
    pause_after: np.ndarray
    pause_before: np.ndarray
    silent: np.ndarray  # per frame, from codebook 0; see `silence_tokens.py`


@dataclass
class Candidates:
    """Crop ends of an entry that are guaranteed to produce a complete sample."""

    eos_words: np.ndarray
    eos_ends: np.ndarray
    eos_starts: np.ndarray
    eos_holds: np.ndarray  # frames of the closing pause spent on `[EOS]`
    open_words: np.ndarray
    open_ends: np.ndarray
    open_starts: np.ndarray

    @classmethod
    def empty(cls) -> "Candidates":
        z = np.zeros(0, dtype=np.int64)
        return cls(z, z, z, z, z, z, z)

    def __bool__(self) -> bool:
        return bool(len(self.eos_words) or len(self.open_words))


# set before the validation workers are forked
_VALIDATE: Optional["AlignedDataset"] = None


def _validate_shard(span: Tuple[int, int]) -> Tuple[int, np.ndarray]:
    """Which entries of `[lo, hi)` can be cropped under the current config."""
    lo, hi = span
    out = np.zeros(hi - lo, dtype=bool)
    for k in range(lo, hi):
        store, i = _VALIDATE.store.locate(k)
        out[k - lo] = bool(_VALIDATE._candidates(_VALIDATE._entry(store, i)))
    return lo, out


def collate_fn(batch, phone_pad_token: int = ROOT_PAD) -> Dict[str, torch.Tensor]:
    """Pad the phoneme sequences of a batch; everything else is already fixed-shape."""
    (
        phone_tokens,
        phone_emb_indices,
        audio_codes,
        depth_audio_codes,
        semantic_labels,
        audio_labels,
        loss_mask,
        audio_mask,
        prompt_lens,
        spk_templates,
    ) = zip(*batch, strict=False)

    phone_tokens = pad_sequence(
        [torch.from_numpy(seq) for seq in phone_tokens],
        batch_first=True,
        padding_value=phone_pad_token,
    )

    def stack(arrays, dtype=torch.int64):
        return torch.as_tensor(np.array(arrays)).to(dtype)

    return {
        "phone_tokens": phone_tokens.to(torch.int64),
        "phone_emb_indices": stack(phone_emb_indices),
        "audio_tokens": stack(audio_codes),
        "depth_audio_tokens": stack(depth_audio_codes),
        "semantic_labels": stack(semantic_labels),
        "audio_labels": stack(audio_labels),
        "loss_mask": stack(loss_mask, torch.bool),
        "audio_mask": stack(audio_mask, torch.bool),
        "prompt_lens": stack(prompt_lens),
        "spk_embeddings": (
            None if spk_templates[0] is None else stack(spk_templates, torch.float32)
        ),
    }


@dataclass
class CropConfig:
    """Everything the crop algorithm is allowed to decide."""

    crop_frames: int = 750  # 60 s of the 80 s entry
    prompt_length_sec: Tuple[int, int] = (3, 10)
    num_phones_per_frame: int = 2
    # end of the crop
    eos_prob: float = 0.5  # share of samples that close on a held pause + [EOS]
    end_slack_frames: int = (
        125  # how far back from the fullest window ends may be drawn
    )
    min_content_frames: int = 250  # must exceed max_end_pause_frames
    min_end_pause_frames: int = 2  # 160 ms; Emilia runs most sentences together
    max_end_pause_frames: int = 4  # 320 ms of it is held on the last phoneme
    # frames of the closing pause the pointer spends on `[EOS]` instead, which is
    # what supervises the step generation ends on. At least one is required of a
    # sentence end, so `min_end_pause_frames` has to stay >= 2
    eos_hold_frames: int = 2
    short_pause_frames: int = 1  # 80 ms, enough for an open end
    # start of the crop
    min_start_pause_frames: int = 1
    # Unaligned look-ahead appended past an open end, drawn per sample. A fixed 32
    # would give every crop more text past its last phoneme than a streaming frontend
    # usually has; the small values reproduce a text stream that is barely ahead of the
    # pointer. The largest should stay >= the phoneme transformer's receptive field
    # (~max_look_ahead x max_look_ahead phonemes), which is where its representations
    # settle and a KV cache could commit them.
    extra_phonemes: Tuple[int, ...] = (8, 16, 32)
    # label dropout, for a streaming frontend that does not know the text yet
    drop_all_labels_prob: float = 0.05
    drop_punct_prob: float = 0.1
    drop_open_end_prob: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.extra_phonemes, int):
            self.extra_phonemes = (self.extra_phonemes,)
        self.extra_phonemes = tuple(int(n) for n in self.extra_phonemes)
        if not self.extra_phonemes or min(self.extra_phonemes) < 0:
            raise ValueError("extra_phonemes must be one or more non-negative counts")
        self.prompt_length_sec = tuple(self.prompt_length_sec)


class AlignedDataset(Dataset):
    """Word-aligned crops of the speaker clusters of one or more stores."""

    collate_fn = staticmethod(collate_fn)

    def __init__(
        self,
        base_dir: Path,
        model: dict,
        datasets: Dict[str, str],
        audio_delay_frames: int,
        pad_len: int,
        dtype: str,
        cfg_prob: float,
        mimi_tps: float,
        crop: Optional[dict] = None,
        prompt_loss: bool = True,
        load_mimi: bool = True,
        validation_jobs: int = 16,
    ) -> None:
        self.store = MultiStore(
            [
                DatasetStore(Path(base_dir) / path, load_mimi=load_mimi)
                for path in datasets.values()
            ]
        )
        audio_vocab_size = model["audio_vocab_size"]
        audio_pad_size = model["audio_pad_size"]
        num_codebooks = model["num_codebooks"]
        self.crop = CropConfig(**(crop or {}))
        self.audio_pad_token = audio_vocab_size - 1
        self.audio_unk_token = audio_vocab_size - 2
        self.semantic_stride = audio_vocab_size + audio_pad_size
        self.num_codebooks = num_codebooks
        self.audio_delay_frames = audio_delay_frames
        self.pad_len = pad_len
        self.audio_shift = audio_delay_frames + pad_len - 1
        self.dtype = dtype
        self.cfg_prob = cfg_prob
        self.mimi_tps = mimi_tps
        self.prompt_loss = prompt_loss

        self.min_prompt_frames = round(self.crop.prompt_length_sec[0] * mimi_tps)
        self.max_prompt_frames = round(self.crop.prompt_length_sec[1] * mimi_tps)
        # without the calibration a crop cannot tell silence from the next recording
        # starting, and every sentence end would quietly fall through to an open one,
        # leaving nothing in the run that teaches the model to stop
        uncalibrated = [
            str(s.root) for s in self.store.stores if s.silence_tokens is None
        ]
        if self.crop.eos_hold_frames > 0 and uncalibrated:
            raise FileNotFoundError(
                "silence_tokens.json is missing from "
                + ", ".join(uncalibrated)
                + "; run `python -m voxtream.utils.dataset.silence_tokens -d <store> -g 0` "
                "for each of them, or set dataset.crop.eos_hold_frames=0 to train "
                "without a supervised ending"
            )
        missing = [
            str(s.root) for s in self.store.stores if load_mimi and not s.mimi_shards
        ]
        if missing:
            raise FileNotFoundError(
                "mimi_codes.npy is missing from "
                + ", ".join(missing)
                + "; run the Mimi stage against the crop plan first"
            )
        short = [
            f"{s.root} has {s.num_mimi_codebooks}"
            for s in self.store.stores
            if load_mimi and s.num_mimi_codebooks < self.num_codebooks
        ]
        if short:
            raise ValueError(
                f"the model wants {self.num_codebooks} Mimi codebooks but "
                + ", ".join(short)
                + "; extract the missing shard or lower num_codebooks"
            )
        if self.store.num_frames < self.crop.crop_frames:
            raise ValueError(
                f"entries are {self.store.num_frames} frames, "
                f"shorter than the {self.crop.crop_frames}-frame crop"
            )
        if self.crop.min_content_frames <= self.crop.max_end_pause_frames:
            raise ValueError(
                "min_content_frames must exceed max_end_pause_frames, or a sentence end "
                "could consume the whole content of a crop"
            )
        self.index = self._usable_index(validation_jobs)

    # -- validation ---------------------------------------------------------

    def _config_key(self) -> str:
        """Hash of everything that decides whether an entry can be cropped."""
        cfg = self.crop
        payload = json.dumps(
            {
                "crop_frames": cfg.crop_frames,
                "prompt_length_sec": list(cfg.prompt_length_sec),
                "min_content_frames": cfg.min_content_frames,
                "min_end_pause_frames": cfg.min_end_pause_frames,
                "max_end_pause_frames": cfg.max_end_pause_frames,
                "eos_hold_frames": cfg.eos_hold_frames,
                "short_pause_frames": cfg.short_pause_frames,
                "min_start_pause_frames": cfg.min_start_pause_frames,
                "mimi_tps": self.mimi_tps,
                "num_frames": self.store.num_frames,
                # the calibration decides which sentence ends are silent enough to
                # close a crop, so a mask cached under an older one is not this one's
                "silence_tokens": [
                    (
                        None
                        if s.silence_tokens is None
                        else hashlib.sha1(
                            np.flatnonzero(s.silence_tokens).astype(np.int64).tobytes()
                        ).hexdigest()[:12]
                    )
                    for s in self.store.stores
                ],
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def _usable_index(self, jobs: int) -> np.ndarray:
        """
        Global indices of the entries a crop can actually be taken from.

        Every entry is checked once, here, rather than discovered to be unusable in the
        middle of a training run: an entry that cannot serve the configured crop is left
        out of the index, so `__getitem__` never has to skip, retry or raise.
        """
        key = self._config_key()
        masks, missing = [], []
        for n, store in enumerate(self.store.stores):
            cached = store.root / f".crop_index_{key}.npy"
            if cached.exists():
                masks.append(np.load(cached))
            else:
                masks.append(None)
                missing.append(n)

        if missing:
            global _VALIDATE
            _VALIDATE = self
            spans = [
                (lo, min(lo + _VALIDATION_SHARD, len(self.store)))
                for lo in range(0, len(self.store), _VALIDATION_SHARD)
            ]
            whole = np.zeros(len(self.store), dtype=bool)
            print(
                f"checking {len(self.store):,} entries against the crop config "
                f"(cached afterwards as .crop_index_{key}.npy)",
                flush=True,
            )
            with mp.get_context("fork").Pool(min(jobs, len(spans))) as pool:
                for lo, part in pool.imap_unordered(_validate_shard, spans):
                    whole[lo : lo + len(part)] = part
            _VALIDATE = None
            for n in missing:
                lo, hi = self.store.bounds[n], self.store.bounds[n + 1]
                masks[n] = whole[lo:hi]
                self._cache_mask(
                    self.store.stores[n].root / f".crop_index_{key}.npy", masks[n]
                )

        keep = np.concatenate(masks)
        if not keep.any():
            raise ValueError(
                "no entry in the store can serve this crop config; "
                f"crop_frames={self.crop.crop_frames} and "
                f"min_content_frames={self.crop.min_content_frames} are too large for "
                f"{self.store.num_frames}-frame entries"
            )
        dropped = int((~keep).sum())
        if dropped:
            print(
                f"{dropped:,} of {len(keep):,} entries ({dropped / len(keep):.2%}) "
                "cannot serve this crop config and are left out",
                flush=True,
            )
        return np.flatnonzero(keep).astype(np.int64)

    @staticmethod
    def _cache_mask(path: Path, mask: np.ndarray) -> None:
        """Write the mask beside the store, atomically; a read-only store just skips it."""
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "wb") as f:  # a handle, so np.save keeps the name as given
                np.save(f, mask)
            tmp.replace(path)
        except OSError as exc:  # a read-only store just revalidates next time
            print(f"could not cache the crop index at {path}: {exc}", flush=True)
            tmp.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self.index)

    # -- which entries can be cropped at all --------------------------------

    def _entry(self, store: DatasetStore, i: int) -> "Entry":
        """The arrays of one store entry, with the frame pointer reconstructed."""
        num_speech = int(store.num_speech_frames[i])
        state = np.asarray(store.frame_state[i][:num_speech], dtype=np.int64)
        first, last = states_to_indices(state)
        return Entry(
            num_speech=num_speech,
            first=first,
            last=last,
            start_frame=np.asarray(store.word_start_frame[i], dtype=np.int64),
            end_frame=np.asarray(store.word_end_frame[i], dtype=np.int64),
            phone_start=np.asarray(store.word_phone_start[i], dtype=np.int64),
            flags=np.asarray(store.word_flags[i]),
            pause_after=np.asarray(store.word_pause_after[i], dtype=np.int64),
            pause_before=np.asarray(store.word_pause_before[i], dtype=np.int64),
            silent=self._silent(store, i, num_speech),
        )

    def _silent(self, store: DatasetStore, i: int, num_speech: int) -> np.ndarray:
        """Which frames of the entry hold silence, read off Mimi's first codebook."""
        if store.silence_tokens is None:
            return np.zeros(num_speech, dtype=bool)
        codes = np.asarray(store.read_codes(i, 1, 0, num_speech))[0]
        return store.silence_tokens[np.clip(codes, 0, 2047)]

    def _crop_starts(self, e: "Entry", ends: np.ndarray) -> np.ndarray:
        """
        Where the window starts for each candidate end, as `_pick_start` would place it.

        The first word border at or after `end - crop_frames` that a pause precedes, the
        first border at all if there is none, and frame 0 when the window reaches the
        head of the entry.
        """
        target = np.maximum(ends - self.crop.crop_frames, 0)
        paused = e.start_frame[e.pause_before >= self.crop.min_start_pause_frames]
        out = target.copy()
        for borders in (paused, e.start_frame):
            if not len(borders):
                continue
            j = np.searchsorted(borders, out, side="left")
            hit = j < len(borders)
            out = np.where(hit, borders[np.minimum(j, len(borders) - 1)], out)
            if borders is paused:
                # words without a pause before them are the fallback, so only the
                # candidates that found no paused border fall through to them
                out = np.where(hit, out, target)
        return np.where(target <= 0, 0, out)

    def _candidates(self, e: "Entry") -> "Candidates":
        """
        Every crop end that is guaranteed to produce a complete sample.

        A candidate has to survive all of it: the end fits in the entry, a word border
        can open the window far enough back, a prompt border sits inside `prompt_sec`
        while still leaving `min_content_frames` of text, and - for a sentence end - the
        pause really holds without the pointer moving on. Filtering here
        rather than at sampling time is what makes the draw total: `_crop` can then never
        come back empty, so a training step never depends on a retry.
        """
        cfg = self.crop
        if not len(e.start_frame):
            return Candidates.empty()

        # a sentence end keeps only the pause frames the pointer stands still through,
        # so the frames that carry the release cannot already belong to the next word
        span = cfg.max_end_pause_frames + cfg.eos_hold_frames
        kept = np.minimum(e.pause_after, span)
        held = np.zeros(len(e.end_frame), dtype=np.int64)
        running = np.ones(len(e.end_frame), dtype=bool)
        base = e.last[np.clip(e.end_frame - 1, 0, e.num_speech - 1)]
        for j in range(span):
            at = e.end_frame + j
            ok = running & (j < kept) & (at < e.num_speech)
            ok &= e.last[np.clip(at, 0, e.num_speech - 1)] == base
            held += ok
            running &= ok

        # that pause is split: the release stays on the last phoneme, as every other
        # pause in the entry does, and what is left closes the crop on `[EOS]`. One
        # frame of each is the minimum, so a sentence end needs the pointer to stand
        # still for two
        release = np.minimum(np.maximum(held - 1, 0), cfg.max_end_pause_frames)
        eos_hold = np.minimum(held - release, cfg.eos_hold_frames)

        def croppable(ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            starts = self._crop_starts(e, ends)
            lo = starts + self.min_prompt_frames
            hi = np.minimum(
                starts + self.max_prompt_frames, ends - cfg.min_content_frames
            )
            j = np.searchsorted(e.start_frame, lo, side="left")
            has_prompt = (j < len(e.start_frame)) & (
                e.start_frame[np.minimum(j, len(e.start_frame) - 1)] <= hi
            )
            return (ends <= e.num_speech) & (ends > 0) & has_prompt, starts

        # and `[EOS]` has to land on real silence. The alignment cannot promise it:
        # an entry concatenates utterances that were trimmed before they were joined, so
        # at a join the gap the word timings report is the next recording starting. On
        # Emilia that is 73% of the sentence ends and only 16% of them are silent.
        quiet = np.ones(len(e.end_frame), dtype=bool)
        for j in range(cfg.eos_hold_frames):
            at = e.end_frame + release + j
            inside = j < eos_hold
            reached = at < e.num_speech
            quiet &= ~inside | (reached & e.silent[np.clip(at, 0, e.num_speech - 1)])

        sentence_ends = e.end_frame + release + eos_hold
        ok_eos, starts_eos = croppable(sentence_ends)
        ok_eos &= (e.flags & WORD_TERMINAL_PUNCT) != 0
        ok_eos &= e.pause_after >= cfg.min_end_pause_frames
        ok_eos &= eos_hold >= 1
        ok_eos &= quiet

        # an open end keeps the pause on the last phoneme and stops there
        open_ends = e.end_frame + np.minimum(e.pause_after, cfg.max_end_pause_frames)
        ok_open, starts_open = croppable(open_ends)
        ok_open &= e.pause_after >= cfg.short_pause_frames

        return Candidates(
            eos_words=np.flatnonzero(ok_eos),
            eos_ends=sentence_ends,
            eos_starts=starts_eos,
            eos_holds=eos_hold,
            open_words=np.flatnonzero(ok_open),
            open_ends=open_ends,
            open_starts=starts_open,
        )

    # -- crop ---------------------------------------------------------------

    def _pick_end(
        self, e: "Entry", cand: "Candidates"
    ) -> Tuple[int, int, int, bool, int]:
        """
        Draw a crop end among the candidates, preferring the roomy ones.

        Returns
        -------
        Tuple[int, int, int, bool, int]
            (word, end frame, crop start, whether the sample closes on `[EOS]`,
            how many of its last frames sit on `[EOS]`)
        """
        cfg = self.crop
        floor = max(
            cfg.min_content_frames,
            min(cfg.crop_frames, e.num_speech) - cfg.end_slack_frames,
        )
        wants_eos = random.random() < cfg.eos_prob
        order = [(True, cand.eos_words), (False, cand.open_words)]
        if not wants_eos:
            order.reverse()

        for eos, words in order:
            if not len(words):
                continue
            ends = cand.eos_ends if eos else cand.open_ends
            roomy = words[ends[words] >= floor]
            if not len(roomy):
                roomy = words[-1:]  # nothing roomy left, take the last border
            w = int(roomy[random.randrange(len(roomy))])
            starts = cand.eos_starts if eos else cand.open_starts
            hold = int(cand.eos_holds[w]) if eos else 0
            return w, int(ends[w]), int(starts[w]), eos, hold
        return -1, 0, 0, False, 0

    def _pick_prompt(self, e: "Entry", crop_start: int, end: int) -> int:
        """Prompt length in frames, always on a word border (`_candidates` ensured one)."""
        lo = crop_start + self.min_prompt_frames
        hi = min(
            crop_start + self.max_prompt_frames, end - self.crop.min_content_frames
        )
        borders = e.start_frame[(e.start_frame >= lo) & (e.start_frame <= hi)]
        return int(borders[random.randrange(len(borders))]) - crop_start

    # -- phoneme streams ----------------------------------------------------

    def _drop_labels(self, tokens: np.ndarray, open_end_from: Optional[int]) -> None:
        """Blank punctuation and word borders the way a text stream would lack them."""
        cfg = self.crop
        if random.random() < cfg.drop_all_labels_prob:
            tokens[:, PUNCT_STREAM] = PUNCT_PAD
            tokens[:, EOW_STREAM] = EOW_PAD
            return
        if cfg.drop_punct_prob > 0:
            marked = np.flatnonzero(tokens[:, PUNCT_STREAM] != PUNCT_PAD)
            if len(marked):
                drop = np.random.random(len(marked)) < cfg.drop_punct_prob
                tokens[marked[drop], PUNCT_STREAM] = PUNCT_PAD
        # past the last word of an open crop a streaming frontend knows nothing yet
        if open_end_from is not None and random.random() < cfg.drop_open_end_prob:
            tokens[open_end_from:, PUNCT_STREAM] = PUNCT_PAD
            tokens[open_end_from:, EOW_STREAM] = EOW_PAD

    # -- sample -------------------------------------------------------------

    def __getitem__(self, idx: int):
        store, i = self.store.locate(int(self.index[idx]))
        return self._crop(store, i)

    def _crop(self, store: DatasetStore, i: int):
        cfg = self.crop
        window = cfg.crop_frames

        e = self._entry(store, i)
        first, last = e.first, e.last
        phone_start = e.phone_start

        w_end, end, crop_start, wants_eos, eos_hold = self._pick_end(
            e, self._candidates(e)
        )
        if w_end < 0:
            raise RuntimeError(
                f"entry {i} of {store.root} passed validation but yields no crop; "
                "the store and the crop config have gone out of step"
            )
        prompt_len = self._pick_prompt(e, crop_start, end)
        content_start = crop_start + prompt_len
        content_len = end - content_start

        # phonemes the content frames point at
        p0 = int(first[content_start])
        p1 = int(last[end - 1])
        tokens = np.asarray(store.phone_tokens[i][p0 : p1 + 1], dtype=np.int64)

        content_first = first[content_start:end] - p0
        content_n = last[content_start:end] - first[content_start:end] + 1
        if wants_eos:
            # The release of the last phoneme is held on it, the way every other pause
            # in the entry is, and then the pointer steps onto `[EOS]` and stands there
            # through the silence that closes the utterance. That step is the one
            # generation ends on, so it is supervised here rather than left to the
            # model: holding the last phoneme is what a pause before the next sentence
            # looks like everywhere else in the entry, and a model that has never been
            # shown the step off it reads the end of the text as one of those pauses
            # and starts a sentence the text never asked for.
            extra = np.array([[ROOT_EOS]], dtype=np.int64)
            tokens = np.concatenate([tokens, _specials(extra)])
            content_first[-eos_hold:] = (p1 - p0) + 1
            content_n[-eos_hold:] = 1
            open_end_from = None
        else:
            # unaligned look-ahead, wrapping to the head of the entry when it runs out
            all_tokens = np.asarray(store.phone_tokens[i], dtype=np.int64)
            n_extra = random.choice(cfg.extra_phonemes)
            tail = np.arange(p1 + 1, p1 + 1 + n_extra) % len(all_tokens)
            tokens = np.concatenate([tokens, all_tokens[tail]])
            open_end_from = max(0, int(phone_start[w_end]) - p0)

        self._drop_labels(tokens, open_end_from)

        # the sequence: [BOS] [PROMPT] x prompt_len, then the phonemes of the content
        head = _specials(
            np.array([[ROOT_BOS]] + [[ROOT_PROMPT]] * prompt_len, dtype=np.int64)
        )
        phone_tokens = np.concatenate([head, tokens]).astype(np.uint8)
        offset = prompt_len + 1

        # frame -> phoneme, and the shift state that reproduces it
        n_frames = window + 1
        emb_index = np.zeros((n_frames, cfg.num_phones_per_frame), dtype=np.int64)
        label_state = np.zeros(n_frames, dtype=np.int64)
        emb_index[:offset, :] = np.arange(offset)[:, None]
        label_state[:prompt_len] = PROMPT_STATE

        content = slice(offset, offset + content_len)
        starts = content_first + offset
        emb_index[content, 0] = starts
        emb_index[content, 1:] = (starts + content_n - 1)[:, None]
        shift = np.r_[
            starts[0] - prompt_len, starts[1:] - (starts[:-1] + content_n[:-1] - 1)
        ]
        label_state[prompt_len : prompt_len + content_len] = 2 * shift + content_n - 1

        # the masked tail holds the pointer on the last phoneme, shift 0 like `label_state`
        used = offset + content_len
        emb_index[used:] = emb_index[used - 1, 1]

        # audio, loss mask
        audio_code = self._crop_audio(store, i, crop_start, window)
        # The acoustic prompt is masked on the input of the temporal transformer only.
        # The depth transformer keeps the real tokens: its unconditional input is the
        # temporal transformer output of the masked prompt, followed by real tokens.
        depth_audio_code = audio_code.copy()
        if random.random() < self.cfg_prob:
            audio_code[:, :prompt_len] = self.audio_unk_token
        audio_code = self._pad_audio(audio_code)
        depth_audio_code = self._pad_audio(depth_audio_code)
        semantic_label = (
            label_state * self.semantic_stride + audio_code[0, 1:]
        ).astype(self.dtype)
        audio_label = depth_audio_code[1:, 1:]

        loss_mask = np.zeros(n_frames, dtype=bool)
        loss_mask[: prompt_len + content_len] = True
        if not self.prompt_loss:
            loss_mask[:prompt_len] = False
        # `audio_label[t]` is the acoustic frame `t - audio_shift`, so its mask lags the
        # semantic one; without the shift the first position trains a PAD target and the
        # last real acoustic frame is never supervised
        audio_mask = np.r_[
            np.zeros(self.audio_shift, dtype=bool),
            loss_mask[: n_frames - self.audio_shift],
        ]

        spk_template = np.asarray(store.spk_emb[i], dtype=np.float32)
        if random.random() < self.cfg_prob:
            spk_template = np.zeros_like(spk_template)

        return (
            phone_tokens,
            emb_index,
            audio_code,
            depth_audio_code,
            semantic_label,
            audio_label,
            loss_mask,
            audio_mask,
            prompt_len,
            spk_template,
        )

    # -- audio --------------------------------------------------------------

    def _crop_audio(
        self, store: DatasetStore, i: int, start: int, window: int
    ) -> np.ndarray:
        if not store.mimi_shards:  # the Mimi stage has not run yet
            return np.full(
                (self.num_codebooks, window), self.audio_pad_token, dtype=self.dtype
            )
        codes = np.asarray(
            store.read_codes(i, self.num_codebooks, start, start + window),
            dtype=self.dtype,
        )
        if codes.shape[1] < window:
            codes = np.pad(
                codes,
                ((0, 0), (0, window - codes.shape[1])),
                constant_values=self.audio_pad_token,
            )
        return codes

    def _pad_audio(self, audio_code: np.ndarray) -> np.ndarray:
        """Delay the acoustic codebooks against the semantic one and pad both ends."""
        padded = np.zeros(
            (
                self.num_codebooks,
                audio_code.shape[1] + self.audio_delay_frames + self.pad_len,
            ),
            dtype=self.dtype,
        )
        padded[0, :] = np.concatenate(
            [
                [self.audio_pad_token],
                audio_code[0, :],
                [self.audio_pad_token] * self.audio_delay_frames,
            ]
        )
        prefix = np.full(
            (self.num_codebooks - 1, self.audio_delay_frames + self.pad_len),
            self.audio_pad_token,
        )
        padded[1:, :] = np.concatenate([prefix, audio_code[1:]], axis=1)
        return padded


def _specials(roots: np.ndarray) -> np.ndarray:
    """Turn a column of root tokens into full five-stream tokens, PAD elsewhere."""
    out = np.zeros((len(roots), NUM_STREAMS), dtype=np.int64)
    out[:, 0] = roots[:, 0]
    return out
