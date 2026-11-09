"""
Export a trained checkpoint as a model directory laid out like the Hugging Face repo of a
release: `model.safetensors`, `config.json` and the phoneme vocabulary.

    python -m voxtream.utils.export_model \
        --checkpoint experiments/train_v451/lightning_logs/version_0/epoch=8.ckpt \
        --model-config experiments/train_v451/config.json \
        --phoneme-vocab /data/stores/emilia/phoneme_vocab.json \
        --out-dir models/voxtream2.5

The weights are loaded into the model of the config before they are written, so a
checkpoint that does not fit it fails here rather than at inference. A generator config
with `model_repo` set to the directory runs it as it would run the Hub repo.
"""

import argparse
import json
import shutil
from dataclasses import fields
from pathlib import Path

from safetensors.torch import save_file

from voxtream.model import Model, ModelConfig
from voxtream.utils.model import load_model_weights


def load_model_config(path: Path) -> ModelConfig:
    """The model config of a training run, the fields of other versions dropped."""
    with open(path) as f:
        params = json.load(f)
    names = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in params.items() if k in names})


def export_model(
    checkpoint: Path, model_config: Path, phoneme_vocab: Path, out_dir: Path
) -> Path:
    config = load_model_config(model_config)
    model = Model(config)
    model.load_state_dict(load_model_weights(str(checkpoint)), strict=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    params = {f.name: getattr(config, f.name) for f in fields(ModelConfig)}
    with open(out_dir / "config.json", "w") as f:
        json.dump({k: v for k, v in params.items() if v is not None}, f, indent=4)
        f.write("\n")
    shutil.copy(phoneme_vocab, out_dir / Path(phoneme_vocab).name)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, required=True, help="config.json of the run"
    )
    parser.add_argument(
        "--phoneme-vocab",
        type=Path,
        required=True,
        help="phoneme vocabulary the store was built with (phoneme_vocab.json of the "
        "store, the same as that of herimor/voxtream2.5)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = export_model(
        args.checkpoint, args.model_config, args.phoneme_vocab, args.out_dir
    )
    for path in sorted(out_dir.iterdir()):
        print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
