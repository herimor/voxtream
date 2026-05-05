import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


class AudioDataset(Dataset):
    def __init__(
        self,
        meta_path: str,
        sample_rate: int = 16000,
        target_length_sec: int = 3,
        sep: str = "^",
    ):
        self.meta = pd.read_parquet(meta_path).paths.values
        self.sample_rate = sample_rate
        self.target_length = target_length_sec * sample_rate
        self.sep = sep

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        paths = self.meta[idx]
        path = paths.split(self.sep)[0]

        waveform, orig_sr = torchaudio.load(path)

        # Resample if necessary
        if orig_sr != self.sample_rate:
            resampler = Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)

        # Tile signal to target length
        if waveform.shape[1] < self.target_length:
            zero_pad = torch.zeros((1, self.target_length - waveform.shape[1]))
            waveform = torch.cat((waveform, zero_pad), dim=1)

        # Crop to match target length
        waveform = waveform[:, : self.target_length]

        return waveform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--meta-path",
        type=str,
        help="Path to metadata file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Path to output directory",
    )
    parser.add_argument(
        "-s", "--sep", type=str, help="Separator used in file names", default="^"
    )
    parser.add_argument("-bs", "--batch-size", type=int, help="Batch size", default=64)
    parser.add_argument(
        "-w", "--num-workers", type=int, help="Number of workers", default=4
    )
    parser.add_argument(
        "-tl",
        "--target-length-sec",
        type=int,
        help="Target length in seconds",
        default=3,
    )
    parser.add_argument("-g", "--gpu", type=int, help="GPU ID", default=0)
    parser.add_argument("-ch", "--chunk", type=int, help="Chunk ID", default=None)
    args = parser.parse_args()

    meta_path = args.meta_path
    filename = "spk_templates.npy"
    if args.chunk is not None:
        meta_path = meta_path.replace(".parquet", f"_chunk{args.chunk}.parquet")
        filename = filename.replace(".npy", f"_chunk{args.chunk}.npy")

    dataset = AudioDataset(
        meta_path=meta_path,
        target_length_sec=args.target_length_sec,
        sep=args.sep,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model = (
        torch.hub.load(
            "IDRnD/ReDimNet",
            "ReDimNet",
            model_name="M",
            train_type="ft_mix",
            dataset="vb2+vox2+cnc",
            trust_repo=True,
            verbose=False,
        )
        .to(device)
        .half()
    )
    model.spec.float()
    model.bn.float()
    model.eval()

    all_embeddings = []
    for waveforms in tqdm(dataloader):
        waveforms = waveforms.to(device).half()
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
            embs = model(waveforms)
        all_embeddings.append(embs.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_embeddings /= np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    assert (
        np.dot(all_embeddings[:10], all_embeddings[:10].T).max() < 1.02
    ), "Embeddings are not normalized properly."

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    np.save(output_dir / filename, all_embeddings)
