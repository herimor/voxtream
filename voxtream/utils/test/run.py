import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from voxtream.generator import SpeechGenerator, SpeechGeneratorConfig
from voxtream.utils.generator import interpolate_speaking_rate_params, set_seed
from voxtream.utils.test.asr import MODEL_POOL as ASR_MODELS
from voxtream.utils.test.asr import main as wer
from voxtream.utils.test.spk_sim import main as spk_sim
from voxtream.utils.test.utmos import main as utmos


def load_emilia_spk_rate_test(prompt_ext: str = "flac"):
    dataset_dir = snapshot_download("herimor/voxtream2-test", repo_type="dataset")
    dataset_dir = Path(dataset_dir) / "test"
    meta_path = dataset_dir / "metadata.csv"
    protocol = pd.read_csv(meta_path)

    return dataset_dir, protocol, prompt_ext


def load_seedtts_en_test(dataset_dir: Path, prompt_ext: str = "wav"):
    protocol = pd.read_csv(
        filepath_or_buffer=dataset_dir / "meta.lst",
        sep="|",
        header=None,
        names=["path", "prompt_text", "file_name", "text"],
    )

    return dataset_dir, protocol, prompt_ext


def main(
    dataset: str,
    output_dir: Path,
    speaking_rate: float,
    spk_sim_model: str,
    asr_model_name: str,
    file_ext: str,
    dataset_dir: Path = None,
):
    set_seed()
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    with open(PROJECT_ROOT / "configs/generator.json") as f:
        config = SpeechGeneratorConfig(**json.load(f))

    with open(PROJECT_ROOT / "configs/speaking_rate.json") as f:
        speaking_rate_config = json.load(f)

    speech_generator = SpeechGenerator(config)

    if dataset == "voxtream2-test":
        dataset_dir, protocol, prompt_ext = load_emilia_spk_rate_test()
    elif dataset == "seedtts-en":
        dataset_dir, protocol, prompt_ext = load_seedtts_en_test(dataset_dir, file_ext)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for _, row in tqdm(
        protocol.iterrows(), total=len(protocol), desc="Generating speech"
    ):
        save_path = (output_dir / row.path).with_suffix(f".{file_ext}")
        if save_path.exists():
            continue
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        duration_state, weight, cfg_gamma = None, None, None
        if speaking_rate is not None:
            duration_state, weight, cfg_gamma = interpolate_speaking_rate_params(
                speaking_rate_config, speaking_rate, logger=speech_generator.logger
            )

        speech_stream = speech_generator.generate_stream(
            prompt_audio_path=(dataset_dir / row.file_name).with_suffix(
                f".{prompt_ext}"
            ),
            text=row.text,
            target_spk_rate_cnt=duration_state,
            spk_rate_weight=weight,
            cfg_gamma=cfg_gamma,
        )

        audio_frames = [audio_frame for audio_frame, _ in speech_stream]
        sf.write(save_path, np.concatenate(audio_frames), config.mimi_sr)

    # Objective evaluation
    utmos(
        meta=protocol,
        dataset_dir=output_dir,
        file_ext=file_ext,
    )
    spk_sim(
        model_name=spk_sim_model,
        meta=protocol,
        dataset_dir=output_dir,
        reference_dir=dataset_dir,
        file_ext=file_ext,
    )
    wer(
        meta=protocol,
        dataset_dir=output_dir,
        asr_model_name=asr_model_name,
        file_ext=file_ext,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Dataset name",
        default="voxtream2-test",
        choices=["voxtream2-test", "seedtts-en"],
    )
    parser.add_argument(
        "-dd",
        "--dataset-dir",
        type=str,
        help="Dataset directory",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "-src",
        "--speaking-rate",
        type=float,
        help="Speaking rate control",
        default=None,
    )
    parser.add_argument(
        "-spk",
        "--spk-sim-model",
        type=str,
        help="Speaker similarity model",
        choices=["redimnet", "wavlm-ecapa"],
        default="wavlm-ecapa",
    )
    parser.add_argument(
        "-asr",
        "--asr-model-name",
        type=str,
        help="ASR model name",
        choices=ASR_MODELS.keys(),
        default="openai/whisper-large-v3",
    )
    parser.add_argument(
        "-e", "--file-ext", type=str, help="Audio file extension", default="flac"
    )
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        output_dir=Path(args.output_dir),
        speaking_rate=args.speaking_rate,
        spk_sim_model=args.spk_sim_model,
        asr_model_name=args.asr_model_name,
        file_ext=args.file_ext,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
    )
