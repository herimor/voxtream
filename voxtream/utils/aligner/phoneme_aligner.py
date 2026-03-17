from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


class PhonemeAligner:
    def __init__(
        self,
        phones_per_frame: int,
        phone_to_idx: Dict[str, int],
        window_size: int = 80,
        max_shift: int = 1,
        sec_to_ms: int = 1000,
        num_frames: int = None,
        silence_phone: str = "sil",
        drop_silence: bool = True,
    ):
        self.phones_per_frame = phones_per_frame
        self.window_size = window_size
        self.sec_to_ms = sec_to_ms
        self.max_shift = max_shift
        self.phone_to_idx = phone_to_idx
        self.num_frames = num_frames
        self.silence_phone = silence_phone
        self.drop_silence = drop_silence

        self.label_shift_map = {
            2: {
                1: {(0, 1): 0, (0, 2): 1, (1, 1): 2, (1, 2): 3},
                2: {
                    (0, 1): 0,
                    (0, 2): 1,
                    (1, 1): 2,
                    (1, 2): 3,
                    (2, 1): 4,
                    (2, 2): 5,
                },
            },
        }

    @staticmethod
    def pad_last_phoneme(
        phones: List[Tuple[str, float, float]], max_len_sec: float
    ) -> None:
        """Extend the last phoneme to match max file length if needed."""
        last_ph_end = phones[-1][-1]
        if last_ph_end < max_len_sec:
            ph, start, _ = phones.pop()
            phones.append((ph, start, max_len_sec))

    def build_phone_indices(
        self,
        phones: List[Tuple[str, float, float]],
    ) -> np.ndarray:
        """Map time (ms) to phoneme indices."""
        file_len_ms = self.num_frames * self.window_size
        indices = np.full(file_len_ms, -1, dtype=np.int16)
        for i, (_, start, end) in enumerate(phones):
            indices[int(start * self.sec_to_ms) : int(end * self.sec_to_ms)] = i
        assert np.all(indices > -1), "Missed phoneme indices"
        return indices

    def assign_phones_to_frames(
        self,
        phone_indices: np.ndarray,
    ) -> Tuple[np.ndarray, List[int]]:
        """Assign phoneme indices to each frame with overlap handling."""
        drop_shift, drop_indices = 0, []
        emb_indices = np.full(
            (self.num_frames, self.phones_per_frame), -1, dtype=np.int16
        )
        label_shifts = np.empty((self.num_frames,), dtype=np.uint8)

        for i in range(self.num_frames):
            start, end = i * self.window_size, (i + 1) * self.window_size
            phone_idx = phone_indices[start:end]

            # Select most frequent phones in this frame
            frame_phones = [
                ph[0] - drop_shift
                for ph in Counter(phone_idx).most_common()[: self.phones_per_frame]
            ]
            frame_phones = sorted(frame_phones)
            emb_indices[i, : len(frame_phones)] = frame_phones

            if i > 0:
                cur_start = emb_indices[i][0]
                prev_end = max(emb_indices[i - 1])
                shift = cur_start - prev_end

                while shift > self.max_shift:
                    drop_shift += 1
                    drop_indices.append(prev_end + 1)
                    emb_indices[i, : len(frame_phones)] -= 1
                    cur_start = emb_indices[i][0]
                    shift = cur_start - prev_end
            else:
                shift = 0

            label_shift_key = (int(shift), len(frame_phones))
            label_shifts[i] = self.label_shift_map[self.max_shift][len(frame_phones)][
                label_shift_key
            ]

        return emb_indices, drop_indices, label_shifts

    def finalize_embedding_indices(
        self,
        emb_indices: np.ndarray,
    ) -> np.ndarray:
        """Ensure no padding remains and format depending on phones_per_frame."""
        mask = emb_indices == -1
        vals = emb_indices.max(axis=1)
        emb_indices[mask] = np.broadcast_to(vals[:, None], mask.shape)[mask]
        assert np.all(emb_indices >= 0), "Negative phone indices found"
        return emb_indices

    def align(
        self,
        phoneme_alignment: List[Tuple[float, float, str]],
        num_frames: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Aligns phoneme annotations with frame-based acoustic features.

        Returns:
            phone_tokens: Array of phoneme indices (uint8).
            phone_emb_indices: Frame-to-phoneme index mapping (int16).
        """
        if num_frames is not None:
            self.num_frames = num_frames
        else:
            assert self.num_frames is not None, "num_frames must be specified"

        max_len_sec = self.num_frames * self.window_size / self.sec_to_ms

        phones: List[Tuple[str, float, float]] = []
        prev_end = 0.0

        # Process phoneme alignment
        for start, end, ph in phoneme_alignment:
            if start >= max_len_sec:
                break
            if ph == self.silence_phone and self.drop_silence:
                continue

            if start > prev_end and phones:
                # Adjust previous phoneme boundary
                last_ph, last_start, _ = phones.pop()
                phones.append((last_ph, last_start, start))
            elif start > prev_end:
                start = prev_end

            phones.append((ph, start, end))
            prev_end = end

        self.pad_last_phoneme(phones, max_len_sec)

        # Convert phones to tokens
        phone_tokens = []
        for ph, _, _ in phones:
            if ph not in self.phone_to_idx:
                # TODO: Add better handling for unknown phonemes
                ph = "unk"
            phone_tokens.append(self.phone_to_idx[ph])
        phone_tokens = np.array(phone_tokens, dtype=np.uint8)

        # Map phonemes to frames
        phone_indices = self.build_phone_indices(phones)
        phone_emb_indices, drop_indices, label_shifts = self.assign_phones_to_frames(
            phone_indices
        )

        # Adjust tokens to match embedding indices
        diff = (len(phone_tokens) - len(drop_indices)) - max(phone_emb_indices[-1]) - 1
        assert diff >= 0, "Phoneme index out of range"
        if diff > 0:
            for i in range(diff):
                drop_indices.append(len(phone_tokens) - 1 - i)
        phone_tokens = np.delete(phone_tokens, drop_indices)

        phone_emb_indices = self.finalize_embedding_indices(phone_emb_indices)

        return phone_tokens, phone_emb_indices, label_shifts, drop_indices

    @staticmethod
    def merge_alignments(
        phones: List[np.ndarray[str]],
        alignments: List[np.ndarray[float]],
        punct_ins_indices: List[np.ndarray[int]],
        punct_syms: List[np.ndarray[str]],
        durations: List[float],
        phone_del_indices: List[np.ndarray[int]] = None,
        punct_del_indices: List[np.ndarray[int]] = None,
    ) -> Tuple[List[Tuple[float, float, str]], np.ndarray[int], np.ndarray[str]]:

        if phone_del_indices is not None and punct_del_indices is not None:
            for i in range(len(phones)):
                if len(phone_del_indices[i]) > 0:
                    phones[i] = np.delete(phones[i], phone_del_indices[i])
                    alignments[i] = np.delete(alignments[i], phone_del_indices[i])

                if len(punct_del_indices[i]) > 0:
                    punct_syms[i] = np.delete(punct_syms[i], punct_del_indices[i])

                    removed_vals = punct_ins_indices[i][punct_del_indices[i]]
                    remaining = np.delete(punct_ins_indices[i], punct_del_indices[i])
                    adjusted = remaining - np.searchsorted(
                        removed_vals, remaining, side="right"
                    )
                    punct_ins_indices[i] = adjusted

        all_phones = np.concatenate(phones)
        all_punct_syms = np.concatenate(punct_syms)

        dur_shift, punct_shift = 0, 0
        all_alignments, all_punct_indices = [], []
        for i, alignment in enumerate(alignments):
            alignment = alignment + dur_shift
            all_alignments.append(alignment)
            dur_shift += durations[i]

            all_punct_indices.append(punct_ins_indices[i] + punct_shift)
            punct_shift += len(phones[i])
        all_alignments = np.concatenate(all_alignments)
        all_punct_indices = np.concatenate(all_punct_indices)

        phoneme_alignment = []
        for i in range(len(all_phones) - 1):
            entry = (all_alignments[i], all_alignments[i + 1], all_phones[i])
            phoneme_alignment.append(entry)

        # process final phoneme alignment
        idx = len(all_phones) - 1
        entry = (all_alignments[idx], dur_shift, all_phones[idx])
        phoneme_alignment.append(entry)

        return phoneme_alignment, all_punct_indices, all_punct_syms

    @staticmethod
    def align_insert_indices(
        insert_idx, insert_elems, delete_idx, length, side: str = "left"
    ):
        """
        Map original insert indices (for an array of length `length`) to positions
        after deleting `delete_idx`, with the extra rule:
        - If a delete immediately precedes an insert boundary, drop that boundary.

        insert_idx : 1D array-like of ints (typically in [1..length-1])
        insert_elems: 1D array-like of elements to insert at insert_idx
        delete_idx : 1D array-like of ints in [0..length-1]
        Returns:
        new_idx  : remapped insert indices after deletions
        new_elems: elements to insert at new_idx
        """
        insert_idx = np.asarray(insert_idx, dtype=np.int64)
        insert_elems = np.asarray(insert_elems)
        del_idx = np.unique(np.asarray(delete_idx, dtype=np.int64))

        # -- drop rule (vectorized) --
        # drop i when (i-1) was deleted.
        del_set = set(del_idx.tolist())
        drop = np.fromiter(
            ((i - 1) in del_set for i in insert_idx), count=insert_idx.size, dtype=bool
        )
        keep_mask = ~drop
        kept_inserts = insert_idx[keep_mask]
        kept_elems = insert_elems[keep_mask]

        # -- shift surviving boundaries by number of deletions before them --
        shifts = np.searchsorted(del_idx, kept_inserts, side=side)
        new_idx = kept_inserts - shifts

        mask = new_idx <= length
        new_idx = new_idx[mask]
        kept_elems = kept_elems[mask]

        return new_idx, kept_elems

    def process(
        self,
        group=None,
        phones: List[np.ndarray[str]] = None,
        alignments: List[np.ndarray[float]] = None,
        durations: List[float] = None,
        punct_ins_indices: List[np.ndarray[int]] = None,
        punct_syms: List[np.ndarray[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Merge multiple utterances and align phoneme annotations with frame-based acoustic features.
        """
        if group is not None:
            if group.shape[1] == 6:
                phone_del_indices, punct_del_indices = None, None
                (
                    phones,
                    alignments,
                    durations,
                    punct_ins_indices,
                    punct_syms,
                    speaker,
                ) = group.values.T
            else:
                (
                    phones,
                    alignments,
                    durations,
                    punct_ins_indices,
                    punct_syms,
                    speaker,
                    phone_del_indices,
                    punct_del_indices,
                ) = group.values.T
        else:
            assert phones is not None, "Phones must be provided"
            assert alignments is not None, "Alignments must be provided"
            assert durations is not None, "Durations must be provided"
            assert (
                punct_ins_indices is not None
            ), "Punctuation insert indices must be provided"
            assert punct_syms is not None, "Punctuation symbols must be provided"

        # merge
        phoneme_alignment, punct_ins_indices, punct_syms = self.merge_alignments(
            phones,
            alignments,
            punct_ins_indices,
            punct_syms,
            durations,
            phone_del_indices,
            punct_del_indices,
        )

        # pad with silence
        max_len_sec = self.num_frames * self.window_size / self.sec_to_ms
        file_end = phoneme_alignment[-1][1]
        if file_end < max_len_sec:
            phoneme_alignment.append([file_end, max_len_sec, self.silence_phone])

        # align
        phone_tokens, phone_emb_indices, label_shifts, drop_indices = self.align(
            phoneme_alignment
        )

        # remap + drop invalid insert boundaries
        punct_ins_indices, punct_syms = self.align_insert_indices(
            punct_ins_indices,
            punct_syms,
            drop_indices,
            len(phone_tokens),
        )
        punct_del_indices = punct_ins_indices + np.arange(len(punct_ins_indices))

        # convert punct to tokens
        punct_syms = np.array(
            [self.phone_to_idx[ph] for ph in punct_syms], dtype=np.uint8
        )

        return (
            phone_tokens,
            phone_emb_indices,
            label_shifts,
            punct_ins_indices,
            punct_del_indices,
            punct_syms,
        )
