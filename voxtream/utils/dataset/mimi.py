import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


class AudioDataset(Dataset):
    def __init__(
        self,
        meta_path: str,
        sample_rate: int = 24000,
        target_length_sec: int = 55,
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

        all_waveforms = []
        for path in paths.split(self.sep):
            waveform, orig_sr = torchaudio.load(path)

            # Resample if necessary
            if orig_sr != self.sample_rate:
                resampler = Resample(orig_sr, self.sample_rate)
                waveform = resampler(waveform)

            all_waveforms.append(waveform)
        waveform = torch.cat(all_waveforms, dim=1)

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
    parser.add_argument("-bs", "--batch-size", type=int, help="Batch size", default=32)
    parser.add_argument(
        "-w", "--num-workers", type=int, help="Number of workers", default=4
    )
    parser.add_argument(
        "-cb", "--num-codebooks", type=int, help="Number of codebooks", default=16
    )
    parser.add_argument(
        "-tl",
        "--target-length-sec",
        type=int,
        help="Target length in seconds",
        default=55,
    )
    parser.add_argument(
        "-s", "--sep", type=str, help="Separator used in file paths", default="^"
    )
    parser.add_argument("-g", "--gpu", type=int, help="GPU ID", default=0)
    parser.add_argument("-ch", "--chunk", type=int, help="Chunk ID", default=None)
    args = parser.parse_args()

    meta_path = args.meta_path
    filename = f"mimi_codes_{args.num_codebooks}cb.npy"
    if args.chunk is not None:
        meta_path = meta_path.replace(".parquet", f"_chunk{args.chunk}.parquet")
        filename = filename.replace(".npy", f"_chunk{args.chunk}.npy")

    dataset = AudioDataset(
        meta_path=meta_path,
        target_length_sec=args.target_length_sec,
        sep=args.sep,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    mimi_weight = hf_hub_download(
        repo_id="kyutai/moshiko-pytorch-bf16",
        filename="tokenizer-e351c8d8-checkpoint125.safetensors",
    )
    model = (
        loaders.get_mimi(
            filename=mimi_weight, device=device, num_codebooks=args.num_codebooks
        )
        .eval()
        .half()
    )

    all_audio_codes = []
    for batch in tqdm(dataloader):
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
            encoder_outputs = model.encode(batch.to(device).half())

        audio_codes = (
            encoder_outputs[:, : args.num_codebooks, :].cpu().numpy().astype("uint16")
        )
        all_audio_codes.append(audio_codes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    np.save(output_dir / filename, np.concatenate(all_audio_codes, axis=0))
