import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


def main(
    meta_path: Path,
    dataset_dir: Path,
    file_ext: str = "flac",
    model_name: str = "utmos22_strong",
    repo_id: str = "tarepan/SpeechMOS:v1.2.0",
):
    save_path = dataset_dir / "utmos.txt"
    if save_path.exists():
        with open(save_path) as f:
            avg_score = f.read().strip()
        print(avg_score)
        return

    if meta_path.suffix == ".parquet":
        meta = pd.read_parquet(meta_path)
    else:
        meta = pd.read_csv(meta_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(repo_id, model_name, trust_repo=True, verbose=False)
    model.to(device)
    model.eval()

    scores = []
    for path in tqdm(meta.path, desc="Evaluating with UTMOS"):
        try:
            signal, sr = sf.read(
                (dataset_dir / path).with_suffix(f".{file_ext}"), dtype="float32"
            )
            with torch.no_grad():
                score = model(torch.from_numpy(signal).unsqueeze(0).to(device), sr)
            scores.append(score.cpu().numpy()[0])
        except Exception:
            print(f"Error processing {path}, skipping...")
            continue

    np.save(dataset_dir / "utmos_scores.npy", scores)
    result = f"UTMOS: {round(float(np.mean(scores)), 2)}"
    print(result)
    with open(save_path, "w") as f:
        f.write(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--meta-path", type=str, help="Path to metadata file")
    parser.add_argument(
        "-d", "--dataset-dir", type=str, help="Path to dataset directory"
    )
    args = parser.parse_args()

    main(args.meta_path, Path(args.dataset_dir))
