"""
Training dataset of the flat phoneme vocabulary (`DATASET_POOL["flat"]`), the one
VoXtream2 and VoXtream2.1 were trained on (Hugging Face `herimor/voxtream2-train`).

Every entry holds the Mimi codes of an utterance, its phoneme tokens, the phoneme rows
each frame reads and the phoneme state (shift) of each frame, plus the punctuation of the
utterance as tokens to insert into the phoneme sequence. A sample is a random crop of
`audio_window_size` frames whose head is the acoustic prompt, its text masked with
`[UNK]`.
"""

import random
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate(batch, phone_pad_token: int) -> Dict[str, torch.Tensor]:
    """Pad the phoneme sequences and the punctuation rows of a batch."""
    (
        phoneme_sequences,
        phoneme_embedding_indices,
        audio_codes,
        semantic_labels,
        audio_labels,
        punct_del_indices,
        spk_templates,
    ) = zip(*batch, strict=False)

    def stack(arrays):
        return torch.as_tensor(np.array(arrays))

    return {
        "phone_tokens": pad_sequence(
            [torch.tensor(seq) for seq in phoneme_sequences],
            batch_first=True,
            padding_value=phone_pad_token,
        ),
        "phone_emb_indices": stack(phoneme_embedding_indices),
        "audio_tokens": stack(audio_codes),
        "semantic_labels": stack(semantic_labels),
        "audio_labels": stack(audio_labels),
        "punct_del_indices": pad_sequence(
            [torch.tensor(seq) for seq in punct_del_indices],
            batch_first=True,
            padding_value=-1,
        ),
        "spk_embeddings": None if spk_templates[0] is None else stack(spk_templates),
    }


class FlatDataset(Dataset):
    """Random crops of the utterances of the flat-vocabulary arrays."""

    def __init__(
        self,
        base_dir: Path,
        model: dict,
        datasets: Dict[str, Dict[str, str]],
        audio_delay_frames: int,
        dtype: str,
        pad_len: int,
        semantic_label_pad: int,
        num_phones_per_frame: int,
        phoneme_index_map: Dict[str, int],
        cfg_prob: float,
        prompt_length_sec: List[int],
        mimi_tps: float,
    ):
        # Load data
        data = {}
        for name, dataset in datasets.items():
            for key, path in dataset.items():
                if path is None:
                    data[key] = None
                    continue

                val = np.load(Path(base_dir) / name / path, allow_pickle=True)[()]
                if key not in data:
                    data[key] = val
                else:
                    data[key] = np.concatenate((data[key], val), axis=0)

        # Store loaded arrays as attributes
        for key, val in data.items():
            setattr(self, key, val)

        phone_vocab_size = model["phone_vocab_size"]
        self.phone_bos_token = phone_vocab_size - 2  # <BOS>
        self.phone_eos_token = phone_vocab_size - 1  # <EOS>
        self.phone_unk_token = phone_vocab_size - 3  # <UNK>
        self.audio_pad_token = model["audio_vocab_size"] - 1  # <PAD>
        self.audio_unk_token = model["audio_vocab_size"] - 2  # <UNK>
        self.audio_pad_size = model["audio_pad_size"]
        self.num_codebooks = model["num_codebooks"]
        self.audio_window_size = model["audio_window_size"]
        self.audio_delay_frames = audio_delay_frames
        self.dtype = dtype
        self.pad_len = pad_len
        self.semantic_label_pad = semantic_label_pad
        self.phoneme_index_map = phoneme_index_map
        self.num_phones_per_frame = num_phones_per_frame
        self.cfg_prob = cfg_prob
        self.prompt_length_sec = prompt_length_sec
        self.mimi_tps = mimi_tps
        # the phoneme sequences are padded with <EOS>
        self.collate_fn = partial(collate, phone_pad_token=self.phone_eos_token)

    def __len__(self) -> int:
        return len(self.audio_codes)

    def _pad_audio(self, audio_code: np.ndarray) -> np.ndarray:
        """Apply padding and delay to audio codes."""

        padded_audio = np.zeros(
            (
                self.num_codebooks,
                audio_code.shape[1] + self.audio_delay_frames + self.pad_len,
            )
        ).astype(self.dtype)

        pad_semantic_code = [
            [self.audio_pad_token],
            audio_code[0, :],
            [self.audio_pad_token for _ in range(self.audio_delay_frames)],
        ]
        pad_semantic_code = np.concatenate(pad_semantic_code)

        pad_prefix = np.full(
            (self.num_codebooks - 1, self.audio_delay_frames + self.pad_len),
            fill_value=self.audio_pad_token,
        )
        pad_audio_code = np.concatenate([pad_prefix, audio_code[1:]], axis=1).astype(
            self.dtype
        )

        padded_audio[0, :] = pad_semantic_code
        padded_audio[1:, :] = pad_audio_code

        return padded_audio

    def __getitem__(self, idx: int) -> tuple[np.ndarray, ...]:
        """Preprocess single training sample."""

        audio_code = self.audio_codes[idx].astype(self.dtype)  # [num_cb, T]
        phoneme_embedding_index = self.phoneme_embedding_indices[idx].astype(
            self.dtype
        )  # [T, N]
        semantic_label_shift = self.semantic_label_shifts[idx].astype(self.dtype)  # [T]
        phoneme_sequence = self.phoneme_sequence_map[idx].astype(
            self.dtype
        )  # [num_phonemes]

        spk_template = None
        if self.spk_templates is not None:
            spk_template = self.spk_templates[idx]  # [spk_template_dim,]

        # Crop
        start = np.random.randint(0, audio_code.shape[1] - self.audio_window_size)
        end = start + self.audio_window_size

        # Audio
        audio_code = audio_code[: self.num_codebooks, start:end]

        # Prompt text masking
        prompt_sec = np.random.randint(*self.prompt_length_sec)
        prompt_length = int(prompt_sec * self.mimi_tps)
        start += prompt_length

        # Phoneme embedding index
        phoneme_embedding_index = phoneme_embedding_index[start:end]
        min_idx = phoneme_embedding_index[0, 0]
        max_idx = max(phoneme_embedding_index[-1])
        phoneme_embedding_index -= min_idx - 1

        # Phoneme sequence
        phoneme_sequence = phoneme_sequence[min_idx : max_idx + 1]

        # Semantic label shift
        semantic_label_shift = semantic_label_shift[start:end]

        # Punctuation
        ins_idx, punct_del_idx, tokens = self.punctuation[idx]
        ins_idx = np.array(ins_idx).astype(self.dtype)
        punct_del_idx = np.array(punct_del_idx).astype(self.dtype)
        tokens = np.array(tokens).astype(self.dtype)

        mask = (ins_idx > min_idx) & (ins_idx <= max_idx + 1)
        ins_in_slice = ins_idx[mask] - min_idx
        phoneme_sequence = np.insert(phoneme_sequence, ins_in_slice, tokens[mask])
        punct_del_idx = (ins_in_slice + np.arange(mask.sum())).astype(
            self.dtype
        ) + 1  # <BOS>

        # Add prompt <UNK> phone tokens
        phoneme_sequence = np.concatenate(
            [
                np.full(
                    (prompt_length,),
                    self.phone_unk_token,
                    self.dtype,
                ),
                phoneme_sequence,
            ]
        )
        punct_del_idx += prompt_length

        prompt_phone_emb_idx = np.repeat(
            np.expand_dims(np.arange(1, prompt_length + 1, dtype=self.dtype), axis=1),
            self.num_phones_per_frame,
            axis=1,
        )
        phoneme_embedding_index = np.concatenate(
            [prompt_phone_emb_idx, phoneme_embedding_index + prompt_length]
        )

        if random.random() < self.cfg_prob:
            spk_template = np.zeros_like(spk_template)

        if random.random() < self.cfg_prob:
            audio_code[:, :prompt_length] = self.audio_unk_token  # <UNK>

        # Padding
        # Phoneme sequence
        phoneme_sequence = np.concatenate(
            [
                [self.phone_bos_token],  # <BOS>
                phoneme_sequence,
                [self.phone_eos_token],  # <EOS>
            ]
        ).astype(self.dtype)

        # Phoneme embedding index
        bos_tokens = [0] * self.num_phones_per_frame
        pad_tokens = [
            len(phoneme_sequence) - len(punct_del_idx) - 1
        ] * self.num_phones_per_frame

        phoneme_embedding_index = np.concatenate(
            [
                [bos_tokens],  # <BOS>
                phoneme_embedding_index,
                np.full(
                    (max(0, self.audio_delay_frames - 1), self.num_phones_per_frame),
                    pad_tokens,
                    self.dtype,
                ),  # <PAD> audio_delayed audio tokens
            ]
        )

        # Audio
        audio_code = self._pad_audio(audio_code)
        audio_label = audio_code[1:, 1:]

        # Semantic label shift
        semantic_label_shift[0] = self.phoneme_index_map[str(semantic_label_shift[0])]
        semantic_label_shift = np.concatenate(
            [
                semantic_label_shift,
                [
                    self.semantic_label_pad
                    for _ in range(max(self.audio_delay_frames, self.pad_len))
                ],
            ]
        ).astype(self.dtype)

        semantic_label_shift = np.concatenate(
            [
                np.full(
                    (prompt_length,),
                    self.semantic_label_pad,
                    self.dtype,
                ),
                semantic_label_shift,
            ]
        )

        # Update semantic label
        semantic_label = audio_code[0, 1:]
        semantic_label = (
            semantic_label_shift
            * (self.audio_pad_token + self.pad_len + self.audio_pad_size)
            + semantic_label
        )

        return (
            phoneme_sequence,
            phoneme_embedding_index,
            audio_code,
            semantic_label,
            audio_label,
            punct_del_idx,
            spk_template,
        )
