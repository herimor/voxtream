import argparse
import random
import re
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


# Whisper's encoder sees a fixed 30 s window. Audio up to this length is
# transcribed in one pass (short-form); anything longer is transcribed with the
# sequential long-form algorithm of the Whisper paper (predicted segment
# timestamps decide where the next 30 s window starts), which the default
# feature extractor would otherwise silently truncate to the first 30 s.
WHISPER_WINDOW_SEC = 30.0

# Whisper's temperature fallback, restricted to its compression-ratio check: a
# decode whose tokens compress too well (a repetition loop such as "dr- dr- dr-"
# running to the token limit) is re-run at the next temperature. Any other
# output is the plain greedy decode, so non-degenerate transcripts are unchanged.
WHISPER_FALLBACK = dict(
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold=1.35,  # HF token-level counterpart of openai-whisper's 2.4
)

# Whisper decides for itself when an utterance is over and sometimes ends one early,
# returning two words for ten seconds of speech. The compression-ratio fallback above
# does not catch it - a short transcript compresses fine - so the guard is the density
# of the transcript against the audio. Read speech runs at 2.5-3 words per second in
# both test sets (p1 = 1.4), and this is set below even that: a decode that merely
# looks thin is left alone, because a second reading of a sound file that is simply
# quiet or slow brings back inventions rather than words ("Yeah.", "Chow.").
MIN_WORDS_PER_SEC = 1.0
MIN_RETRY_SEC = 2.0

# Window the retry decodes the audio in, when timestamps have not recovered it either.
# Each window is decoded on its own, so none of them can end the utterance for the
# others. They do not overlap: a seam costs at most the word it cuts, while an overlap
# repeats whatever both sides heard.
RETRY_WINDOW_SEC = 4.0

# A rescued decode has to bring back a good deal more than it had, or it is not a
# rescue. The collapses this is for return a fraction of the utterance - one word of
# twenty-six - while a sound transcript that trips the density guard gains a word or
# two of noise on a second reading, which this rejects. The ratio alone is not enough
# at the short end, where a four-word transcript reaches it by picking up a "Thank
# you.", so a rescue also has to be worth whole words.
MIN_RETRY_GAIN = 1.5
MIN_RETRY_WORDS = 4


def init_whisper_model(
    asr_model_name: str = "openai/whisper-large-v3",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    processor = WhisperProcessor.from_pretrained(asr_model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        asr_model_name, dtype=dtype
    ).to(device)

    def transcribe(processor, model, wav, sr: int = 16000, device: str = "cuda"):
        long_form = len(wav) > WHISPER_WINDOW_SEC * sr
        if long_form:
            # Full-length features (> 3000 frames) switch `generate` into
            # sequential long-form decoding. Timestamps are required for it;
            # `condition_on_prev_tokens=False` avoids repetition loops.
            processed = processor(
                wav,
                sampling_rate=sr,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_attention_mask=True,
            )
        else:
            processed = processor(
                wav, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
            )
        input_features = processed.input_features.to(device, dtype=dtype)
        attention_mask = processed.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        generate_kwargs = dict(
            language="english", task="transcribe", **WHISPER_FALLBACK
        )
        if long_form:
            generate_kwargs.update(
                return_timestamps=True, condition_on_prev_tokens=False
            )
        # Fallback samples at T > 0: seed per file so a transcript does not depend
        # on how many files were decoded before it.
        set_seed()
        predicted_ids = model.generate(
            input_features, attention_mask=attention_mask, **generate_kwargs
        )
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[
            0
        ]

        return transcription

    def decode(wav, sr, **kwargs):
        """One `generate` over `wav`, with the defaults of the harness."""
        processed = processor(
            wav, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
        )
        attention_mask = processed.get("attention_mask")
        set_seed()
        predicted_ids = model.generate(
            processed.input_features.to(device, dtype=dtype),
            attention_mask=(
                None if attention_mask is None else attention_mask.to(device)
            ),
            language="english",
            task="transcribe",
            **WHISPER_FALLBACK,
            **kwargs,
        )
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[
            0
        ].strip()

    def transcribe_dense(processor, model, wav, sr: int = 16000, device: str = "cuda"):
        """
        `transcribe`, with a second look at a decode that stopped early.

        A transcript far too short for the audio is re-decoded with timestamps, which
        is Whisper's own way of being told the utterance continues, and failing that in
        windows. A rereading is kept only when it brings back `MIN_RETRY_GAIN` times the
        words: each is a reading of the same audio, so one that finds far more of them
        is one that gave up later, while one that finds a couple more has only decorated
        what was already there. A file this never fires on is scored exactly as before.
        """
        best = transcribe(processor, model, wav, sr=sr, device=device)
        duration = len(wav) / sr
        if duration < MIN_RETRY_SEC or duration > WHISPER_WINDOW_SEC:
            return best

        def dense(text):
            return len(text.split()) / duration >= MIN_WORDS_PER_SEC

        if dense(best):
            return best

        def gained(candidate):
            found, had = len(candidate.split()), len(best.split())
            return (
                found >= MIN_RETRY_GAIN * max(1, had) and found - had >= MIN_RETRY_WORDS
            )

        retry = decode(wav, sr, return_timestamps=True, condition_on_prev_tokens=False)
        if gained(retry):
            best = retry
        if dense(best):
            return best

        window = int(RETRY_WINDOW_SEC * sr)
        parts = [
            decode(wav[i : i + window], sr)
            for i in range(0, len(wav), window)
            if len(wav[i : i + window]) > 0.3 * sr
        ]
        windowed = " ".join(p for p in parts if p)
        return windowed if gained(windowed) else best

    return processor, model, transcribe_dense


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


# An apostrophe inside a word is part of it (don't, Jane's); one on the outside
# is a quotation mark. The references write quoted speech with double quotes, which
# this strips, while Whisper writes it with single ones, so keeping those turned every
# quoted word into a substitution - 38% of the raw errors on librispeech-pc.
QUOTE_APOSTROPHE = re.compile(r"(?<![A-Za-z])'|'(?![A-Za-z])")

# The Whisper normalizer deletes bracketed text outright. A reference carries it as an
# aside the speaker still reads aloud ("(the figure is obliterated)"), so the model says
# those words and the normalized reference no longer has them: every one is an
# insertion. Dropping the brackets before normalizing keeps the words on both sides.
BRACKETS = re.compile(r"[()\[\]]")


def prepare_sentence(text):
    # replace ’ with ' before the quote apostrophes are picked out
    text = text.replace("’", "'")

    # remove punctuation, apart from the apostrophes that spell a word
    text = QUOTE_APOSTROPHE.sub("", text)
    for x in string.punctuation:
        if x == "'":
            continue
        text = text.replace(x, "")

    # drop extra spaces
    text = text.replace("  ", " ")

    # lower case
    text = text.lower()

    return text


def normalize_sentence(normalizer, text):
    """Whisper normalization, with the brackets taken off first (see `BRACKETS`)."""
    return normalizer(BRACKETS.sub(" ", text))


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
        "pred_text_wo_fillers": [],
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

        if "start" in row and "end" in row:
            start_sample = int(row.start * sample_rate)
            end_sample = int(row.end * sample_rate)
            audio = audio[start_sample:end_sample]

        pred_text = transcribe(processor, model, audio, sr=sample_rate, device=device)

        # Original transcription
        wer_score = wer(prepare_sentence(row.text), prepare_sentence(pred_text))

        # remove fillers (kept separate so `pred_text` stays the raw transcript)
        text_wo_fillers = remove_fillers(row.text)
        pred_text_wo_fillers = remove_fillers(pred_text)
        wer_score_wo_fillers = wer(
            prepare_sentence(text_wo_fillers),
            prepare_sentence(pred_text_wo_fillers),
        )

        # Whisper normalization + removed fillers
        wer_score_norm = wer(
            normalize_sentence(normalizer, text_wo_fillers),
            normalize_sentence(normalizer, pred_text_wo_fillers),
        )

        wer_scores["path"].append(row.path)
        wer_scores["text"].append(row.text)
        wer_scores["pred_text"].append(pred_text)
        wer_scores["pred_text_wo_fillers"].append(pred_text_wo_fillers)
        wer_scores["wer"].append(wer_score)
        wer_scores["wer_norm"].append(wer_score_norm)
        wer_scores["wer_wo_fillers"].append(wer_score_wo_fillers)

    wer_scores = pd.DataFrame.from_dict(wer_scores)
    avg_score = round(wer_scores.wer.mean() * 100.0, 2)
    avg_score_norm = round(wer_scores.wer_norm.mean() * 100.0, 2)
    avg_score_wo_fillers = round(wer_scores.wer_wo_fillers.mean() * 100.0, 2)

    result = (
        f"WER: {avg_score}%\n"
        f"WER-norm: {avg_score_norm}%\n"
        f"WER-wo-fillers: {avg_score_wo_fillers}%"
    )
    print(result)
    wer_scores.to_csv(dataset_dir / f"{scores_file_name}.csv", index=None)
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
