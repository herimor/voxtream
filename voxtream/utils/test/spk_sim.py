import argparse
import os
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)  # intra-op
torch.set_num_interop_threads(1)  # inter-op

from voxtream.utils.test.ecapa_tdnn import init_ecapa


def init_redimnet(
    model_path: str = None,
    repo_or_dir: str = "IDRnD/ReDimNet",
    model: str = "ReDimNet",
    model_name: str = "M",
    train_type: str = "ft_mix",
    dataset: str = "vb2+vox2+cnc",
    device: str = "cuda",
):
    model = (
        torch.hub.load(
            repo_or_dir=repo_or_dir,
            model=model,
            model_name=model_name,
            train_type=train_type,
            dataset=dataset,
            trust_repo=True,
            verbose=False,
        )
        .to(device)
        .half()
    )
    model.spec.float()
    model.bn.float()
    model.eval()

    return model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_POOL = {
    "redimnet": (init_redimnet, "rdn.npy", None),
    "wavlm-ecapa": (
        init_ecapa,
        "wlm.npy",
        str(PROJECT_ROOT / "wavlm-large/wavlm_large_finetune.pth"),
    ),
}


def process_file(
    path: Path, model: torch.nn.Module, sample_rate: int, emb_ext: str, device: str
):
    save_path = path.parent / f"{path.stem}.{emb_ext}"
    if save_path.exists():
        return

    signal, _ = librosa.load(path, sr=sample_rate)
    signal = np.expand_dims(signal, 0)

    signal = torch.from_numpy(signal).to(device).half()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
        emb = model(signal)

    save_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(save_path, emb.squeeze().cpu())


def main(
    model_name: str,
    dataset_dir: Path,
    reference_dir: Path | None = None,
    meta: pd.DataFrame | str | None = None,
    sample_rate: int = 16000,
    file_ext: str = "flac",
):
    save_path = dataset_dir / "spk_sim.txt"
    if save_path.exists():
        with open(save_path) as f:
            avg_score = f.read().strip()
        print(avg_score)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_fn, emb_ext, model_path = MODEL_POOL[model_name]
    model = model_fn(model_path, device)

    # STEP 1: Extract templates
    for path in tqdm(
        list(dataset_dir.rglob(f"*.{file_ext}")), desc="Extracting speaker templates"
    ):
        try:
            process_file(
                path=path,
                model=model,
                sample_rate=sample_rate,
                emb_ext=emb_ext,
                device=device,
            )
        except Exception:
            print(f"Error processing {path}, skipping...")
            continue

    # STEP 2: Calculate SPK-SIM
    if reference_dir is not None and meta is not None:
        if isinstance(meta, str):
            if meta.endswith(".parquet"):
                meta_df = pd.read_parquet(meta)
            else:
                meta_df = pd.read_csv(meta)
        else:
            meta_df = meta

        templates_map: dict[str, list[Any]] = {"ref": [], "file": []}
        for path, prompt_path in meta_df[["path", "file_name"]].values:
            audio_prompt_path = (reference_dir / prompt_path).with_suffix(
                f".{file_ext}"
            )
            prompt_path = (reference_dir / prompt_path).with_suffix(f".{emb_ext}")

            if not prompt_path.exists():
                process_file(
                    path=audio_prompt_path,
                    model=model,
                    sample_rate=sample_rate,
                    emb_ext=emb_ext,
                    device=device,
                )

            file_path = (dataset_dir / path).with_suffix(f".{emb_ext}")
            if file_path.exists():
                ref_template = np.load(prompt_path)
                file_template = np.load(file_path)

                templates_map["ref"].append(ref_template)
                templates_map["file"].append(file_template)

        # L2-normalize templates
        for key, val in templates_map.items():
            templates_map[key] = np.array(val)
            templates_map[key] /= np.linalg.norm(
                templates_map[key], axis=1, keepdims=True
            )

        sim_scores = np.dot(templates_map["ref"], templates_map["file"].T)
        avg_score = round(float(np.diag(sim_scores).mean()), 3)
        print(f"SPK-SIM: {avg_score}")
        with open(save_path, "w") as f:
            f.write(f"SPK-SIM: {avg_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        help="Model name",
        choices=MODEL_POOL.keys(),
        default="wavlm-ecapa",
    )
    parser.add_argument(
        "-d", "--dataset-dir", type=str, help="Path to dataset directory"
    )
    parser.add_argument(
        "-r",
        "--reference-dir",
        type=str,
        help="Path to reference directory",
        default=None,
    )
    parser.add_argument(
        "-mp", "--meta-path", type=str, help="Path to metadata", default=None
    )
    parser.add_argument(
        "-e", "--file-ext", type=str, help="File extension", default="flac"
    )

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        dataset_dir=Path(args.dataset_dir),
        reference_dir=Path(args.reference_dir) if args.reference_dir else None,
        meta=args.meta_path,
        file_ext=args.file_ext,
    )
