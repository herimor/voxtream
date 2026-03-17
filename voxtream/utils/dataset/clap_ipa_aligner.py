import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from tqdm.auto import tqdm

from voxtream.utils.aligner.clap_ipa import ClapIPA


def collate_varlen(
    batch: List[Tuple[np.ndarray, str]],
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Collate that returns:
      - list of numpy arrays (variable length)
      - list of phoneme strings
    """
    audios, phonemes = zip(*batch)  # tuples -> two tuples
    return list(audios), list(phonemes)


class AudioPhonemeDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        sample_rate: int = 16000,
        pad_token: str = "[SIL]",
    ):
        self.meta = meta.values
        self.sample_rate = sample_rate
        self.pad_token = pad_token

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        phonemes, file_path = self.meta[idx]
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        waveform = waveform.mean(dim=0)  # Convert to mono
        waveform = waveform.numpy()
        phonemes = [self.pad_token] + phonemes.tolist() + [self.pad_token]

        return waveform, phonemes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--meta-path",
        type=str,
        help="Path to metadata file",
    )
    parser.add_argument(
        "-sr", "--sample-rate", type=int, help="Sample rate", default=16000
    )
    parser.add_argument("-bs", "--batch-size", type=int, help="Batch size", default=32)
    parser.add_argument("-g", "--gpu", type=int, help="GPU ID", default=0)
    parser.add_argument(
        "-nw", "--num-workers", type=int, help="Number of workers", default=4
    )
    parser.add_argument("--test", action="store_true", help="Enable test mode")
    args = parser.parse_args()

    meta = pd.read_parquet(args.meta_path)
    columns = ["phones", "full_path"]
    dataset = AudioPhonemeDataset(
        meta=meta[columns],
        sample_rate=args.sample_rate,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_varlen,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    aligner = ClapIPA()

    all_alignments = []
    for waveforms, phonemes in tqdm(dataloader):
        try:
            alignment = aligner(waveforms, phonemes)
        except Exception:
            alignment = [None] * len(waveforms)
        all_alignments.extend(alignment)

    meta["alignment"] = all_alignments
    meta.to_parquet(args.meta_path, index=None)
