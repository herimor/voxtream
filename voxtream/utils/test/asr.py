import argparse
import random
import string
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from jiwer import wer
from scipy.signal import resample
from tqdm.auto import tqdm
from transformers import (
    HubertForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from whisper.normalizers import EnglishTextNormalizer

warnings.filterwarnings("ignore")

FILLER_WORDS = set(
    [
        "uh",
        "uhm",
        "yeah",
        "oh",
        "yeaah",
        "um",
        "yea",
        "ah",
        "eh",
        "huh",
        "ooh",
        "hmm",
        "uuuh",
        "mmm",
        "ehm",
        "mm",
        "hm",
        "ehh",
    ]
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For full determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_whisper_model(
    asr_model_name: str = "openai/whisper-large-v3", device: str = "cuda"
):
    processor = WhisperProcessor.from_pretrained(asr_model_name)
    model = WhisperForConditionalGeneration.from_pretrained(asr_model_name).to(device)

    def transcribe(processor, model, wav, sr: int = 16000, device: str = "cuda"):
        processed = processor(
            wav, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
        )
        input_features = processed.input_features
        attention_mask = processed.get("attention_mask")
        input_features = input_features.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids,
        )
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[
            0
        ]

        return transcription

    return processor, model, transcribe


def init_hubert_model(
    asr_model_name: str = "facebook/hubert-large-ls960-ft", device: str = "cuda"
):
    processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
    model = HubertForCTC.from_pretrained(asr_model_name).to(device)

    def transcribe(processor, model, wav, sr: int = 16000, device: str = "cuda"):
        input_values = processor(
            wav, sampling_rate=sr, return_tensors="pt"
        ).input_values
        input_values = input_values.to(device)
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        return transcription

    return processor, model, transcribe


MODEL_POOL = {
    "openai/whisper-large-v3": init_whisper_model,
    "facebook/hubert-large-ls960-ft": init_hubert_model,
}


def prepare_sentence(text):
    # remove punctuation
    for x in string.punctuation:
        if x == "'":
            continue
        text = text.replace(x, "")

    # drop extra spaces
    text = text.replace("  ", " ")

    # lower case
    text = text.lower()

    return text


def remove_fillers(text):
    text_tokens = []
    for word in text.split():
        if prepare_sentence(word) not in FILLER_WORDS:
            text_tokens.append(word)

    return " ".join(text_tokens)


def load_audio(path: Path, sample_rate: int):
    audio, sr = sf.read(path)
    if sr != sample_rate:
        audio = resample(audio, int(len(audio) * sample_rate / sr))

    return audio


def main(
    meta: pd.DataFrame | str,
    dataset_dir: Path,
    asr_model_name: str,
    sample_rate: int = 16000,
    file_ext: str = "flac",
    min_audio_length_sec: float = 0.2,
):
    scores_file_name = "wer_whisper" if "whisper" in asr_model_name else "wer_hubert"
    if (dataset_dir / f"{scores_file_name}.txt").exists():
        with open(dataset_dir / f"{scores_file_name}.txt") as f:
            avg_score = f.read().strip()
        print(avg_score)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model, transcribe = MODEL_POOL[asr_model_name](asr_model_name, device)

    if isinstance(meta, str):
        if meta.endswith(".parquet"):
            meta = pd.read_parquet(meta)
        elif meta.endswith(".csv"):
            meta = pd.read_csv(meta)
        else:
            raise ValueError(f"Unsupported metadata file format: {meta}")
    elif not isinstance(meta, pd.DataFrame):
        raise TypeError("Invalid metadata type")

    normalizer = EnglishTextNormalizer()
    wer_scores: dict[str, list[Any]] = {
        "path": [],
        "wer": [],
        "wer_norm": [],
        "wer_wo_fillers": [],
        "text": [],
        "pred_text": [],
    }

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Evaluating with ASR"):
        audio_path = (dataset_dir / row.path).with_suffix(f".{file_ext}")
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            continue

        try:
            audio = load_audio(audio_path, sample_rate)
            if len(audio) < min_audio_length_sec * sample_rate:
                print(f"Audio file is too short: {audio_path}")
                continue
        except Exception:
            print(f"Audio file is broken: {audio_path}")
            continue

        pred_text = transcribe(processor, model, audio, sr=sample_rate, device=device)

        # Original transcription
        wer_score = wer(prepare_sentence(row.text), prepare_sentence(pred_text))

        # remove fillers
        text = remove_fillers(row.text)
        pred_text = remove_fillers(pred_text)
        wer_score_wo_fillers = wer(prepare_sentence(text), prepare_sentence(pred_text))

        # Whisper normalization + removed fillers
        wer_score_norm = wer(normalizer(text), normalizer(pred_text))

        wer_scores["path"].append(row.path)
        wer_scores["text"].append(row.text)
        wer_scores["pred_text"].append(pred_text)
        wer_scores["wer"].append(wer_score)
        wer_scores["wer_norm"].append(wer_score_norm)
        wer_scores["wer_wo_fillers"].append(wer_score_wo_fillers)

    wer_scores = pd.DataFrame.from_dict(wer_scores)
    wer_scores.to_csv(dataset_dir / f"{scores_file_name}.csv", index=None)
    avg_score = round(wer_scores.wer.mean() * 100.0, 2)
    avg_score_norm = round(wer_scores.wer_norm.mean() * 100.0, 2)
    avg_score_wo_fillers = round(wer_scores.wer_wo_fillers.mean() * 100.0, 2)

    result = (
        f"WER: {avg_score}%\n"
        f"WER-norm: {avg_score_norm}%\n"
        f"WER-wo-fillers: {avg_score_wo_fillers}%"
    )
    print(result)
    with open(dataset_dir / f"{scores_file_name}.txt", "w") as f:
        f.write(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-path",
        type=str,
        help="Path to metadata",
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        type=str,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "-asr",
        "--asr-model",
        type=str,
        help="ASR model name",
        choices=MODEL_POOL.keys(),
        default="openai/whisper-large-v3",
    )
    parser.add_argument(
        "-e", "--file-ext", type=str, help="File extension", default="flac"
    )

    args = parser.parse_args()

    set_seed()
    main(
        meta=args.meta_path,
        dataset_dir=Path(args.dataset_dir),
        asr_model_name=args.asr_model,
        file_ext=args.file_ext,
    )
