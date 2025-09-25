from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_fn(
    batch,
    phone_pad_token: int,
) -> Tuple[torch.Tensor, ...]:
    """Collate function for batching phoneme and audio data.

    Args:
        batch: List of tuples containing
            (phoneme_sequence, phoneme_embedding_index,
             audio_code, semantic_label, audio_label, spk_template).
        phone_pad_token: Padding token used for phoneme sequences.

    Returns:
        A tuple of batched tensors:
        (phoneme_sequences, phoneme_embedding_indices,
         audio_codes, semantic_labels, audio_labels, spk_templates).
    """
    (
        phoneme_sequences,
        phoneme_embedding_indices,
        audio_codes,
        semantic_labels,
        audio_labels,
        _spk_templates,
    ) = zip(*batch, strict=False)

    # Pad phoneme sequences
    phoneme_sequences = pad_sequence(
        [torch.tensor(seq) for seq in phoneme_sequences],
        batch_first=True,
        padding_value=phone_pad_token,
    )

    # Convert to tensors
    phoneme_embedding_indices = torch.as_tensor(np.array(phoneme_embedding_indices))
    audio_codes = torch.as_tensor(np.array(audio_codes))
    semantic_labels = torch.as_tensor(np.array(semantic_labels))
    audio_labels = torch.as_tensor(np.array(audio_labels))

    # Handle optional speaker templates
    if _spk_templates[0] is None:
        spk_templates = None
    else:
        spk_templates = torch.as_tensor(np.array(_spk_templates))

    return (
        phoneme_sequences,
        phoneme_embedding_indices,
        audio_codes,
        semantic_labels,
        audio_labels,
        spk_templates,
    )


class TrainDataset(Dataset):
    """Custom training dataset for phoneme based TTS model."""

    def __init__(
        self,
        base_dir: Path,
        datasets: Dict[str, Dict[str, str]],
        phone_vocab_size: int,
        audio_vocab_size: int,
        num_codebooks: int,
        audio_delay_frames: int,
        dtype: str,
        audio_window_size: int,
        pad_len: int,
        semantic_label_pad: int,
        num_phones_per_frame: int,
        phoneme_index_map: Dict[str, int],
    ):
        # Load data
        data = {}
        for name, dataset in datasets.items():
            for key, path in dataset.items():
                if path is None:
                    data[key] = None
                    continue

                val = np.load(base_dir / name / path, allow_pickle=True)[()]
                if key not in data:
                    data[key] = val
                else:
                    data[key] = np.concatenate((data[key], val), axis=0)

        # Store loaded arrays as attributes
        for key, val in data.items():
            setattr(self, key, val)

        self.phone_vocab_size = phone_vocab_size - 2  # <BOS> and <EOS>
        self.phone_pad_token = phone_vocab_size - 1
        self.audio_pad_token = audio_vocab_size - 1  # <PAD>
        self.num_codebooks = num_codebooks
        self.audio_delay_frames = audio_delay_frames
        self.dtype = dtype
        self.audio_window_size = audio_window_size
        self.pad_len = pad_len
        self.semantic_label_pad = semantic_label_pad
        self.phoneme_index_map = phoneme_index_map
        self.num_phones_per_frame = num_phones_per_frame

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

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ...]:
        """Preprocess single training sample."""

        audio_code = self.audio_codes[idx].astype(self.dtype)  # [num_cb, T]
        phoneme_embedding_index = self.phoneme_embedding_indices[idx].astype(
            self.dtype
        )  # [T, 2]
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

        # Phoneme embedding index
        phoneme_embedding_index = phoneme_embedding_index[start:end]
        min_idx = phoneme_embedding_index[0, 0]
        max_idx = max(phoneme_embedding_index[-1])
        phoneme_embedding_index -= min_idx - 1

        # Phoneme sequence
        phoneme_sequence = phoneme_sequence[min_idx : max_idx + 1]

        # Semantic label shift
        semantic_label_shift = semantic_label_shift[start:end]

        # Padding
        # Phoneme sequence
        phoneme_sequence = np.concatenate(
            [
                [self.phone_vocab_size],  # <BOS>
                phoneme_sequence,
                [self.phone_vocab_size + 1],  # <EOS>
            ]
        ).astype(self.dtype)

        # Phoneme embedding index
        pad_tokens = [len(phoneme_sequence) - 1] * self.num_phones_per_frame
        phoneme_embedding_index = np.concatenate(
            [
                [[0] * self.num_phones_per_frame],  # <BOS>
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

        # Update semantic label
        semantic_label = audio_code[0, 1:]
        semantic_label = (
            semantic_label_shift * (self.audio_pad_token + self.pad_len)
            + semantic_label
        )

        return (
            phoneme_sequence,
            phoneme_embedding_index,
            audio_code,
            semantic_label,
            audio_label,
            spk_template,
        )
