import argparse
import json
import random
import re
import shlex
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import inflect
import nltk
import numpy as np
import torch
from g2p_en import G2p

num_conv = inflect.engine()
currency_map = {
    "$": " dollar",
    "€": " euro",
    "£": " pound",
    "¥": " yen",
    "₹": " rupee",
    "₩": " won",
    "₽": " ruble",
    "₴": " hryvnia",
    "₺": " lira",
    "₦": " naira",
    "%": " percent",
}
# Regex pattern to match common currency symbols
currency_pattern = r"[%$€£¥₹₩₽₴₺₦]"


def normalize_text(text: str) -> str:
    text = text.lower()
    # OOV fix
    # convert 'mr' -> 'mister'
    text = text.replace("mr", "mister")
    # convert 'mrs' -> 'missis'
    text = text.replace("mrs", "missis")

    # convert $1 -> 1$
    upd_words = []
    for word in text.split():
        if word.startswith("$"):
            upd_words.append((word, word[1:] + "$"))
    for word, rep_word in upd_words:
        text = text.replace(word, rep_word)

    # find all numbers
    nums = sorted(re.findall(r"\d+", text), key=len, reverse=True)
    # convert numbers to words
    for num in nums:
        text = text.replace(num, num_conv.number_to_words(num))
    # remove added ['—,–'] special and [-,.?!] symbols
    text = re.sub(r"[—–.,-?!]", " ", text)
    # convert all currency characters to words
    for cur_ch in re.findall(currency_pattern, text):
        text = text.replace(cur_ch, currency_map[cur_ch])

    # remove extra space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def text_generator(text: str) -> Generator[str, None, None]:
    for word in normalize_text(text).split():
        yield word


def existing_file(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def ensure_nltk_resource(resource: str):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1], quiet=True, raise_on_error=True)


def mfa_align(
    audio_file: str,
    text: str,
    dict_path: str = "english_us_arpa",
    model_path: str = "english_us_arpa",
    dict_default_path: str = "Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict",
) -> Dict[str, Any]:
    """
    Runs MFA `align_one`.

    Args:
        audio_file (str): Path to the input file
        text (str): Transcript to audio file
        dict_path (str): Path to pronunciation dictionary (or model name like 'english_us_arpa')
        model_path (str): Path to acoustic model (or model name)

    Returns:
        subprocess.CompletedProcess: Contains stdout/stderr and return code
    """
    # Download models if not present
    if not (Path.home() / dict_default_path).exists():
        print("Downloading MFA models...")
        commands = [
            "mfa model download acoustic english_us_arpa",
            "mfa model download dictionary english_us_arpa",
        ]
        for cmd in commands:
            subprocess.run(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        print("Done!")

    temp_dir = tempfile.TemporaryDirectory()
    output_path = Path(temp_dir.name) / "mfa.json"
    text_file = Path(temp_dir.name) / "transcript.txt"
    with open(text_file, "w") as f:
        f.write(text)

    command = f"""
    mfa align_one --output_format json --clean -q \
    "{audio_file}" "{text_file}" "{dict_path}" "{model_path}" "{output_path}"
    """

    # Use shlex.split to safely parse the command into a list
    process = subprocess.run(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # ensures output is returned as strings, not bytes
    )

    if process.returncode != 0:
        raise RuntimeError(f"Error running MFA align_one: {process.stderr.strip()}\n")

    with open(output_path) as f:
        phoneme_alignment = json.load(f)
    temp_dir.cleanup()

    return phoneme_alignment


# ------------------------
# Prompt alignment
# ------------------------


# ------------------------
# Helper functions
# ------------------------
def build_word_search_map(phoneme_alignment: Dict[str, Any]) -> np.ndarray:
    """Precompute word boundaries for alignment."""
    return np.array(
        [
            start + end
            for start, end, _ in phoneme_alignment["tiers"]["words"]["entries"]
        ]
    )


def expand_unknown_phone(
    start: float,
    end: float,
    word_key: float,
    words_search_map: np.ndarray,
    phoneme_alignment: Dict[str, Any],
    g2p: G2p,
) -> List[Tuple[str, float, float]]:
    """Expand unknown phone into G2P-generated phonemes."""
    idx = np.argmin(np.abs(words_search_map - word_key))
    word = phoneme_alignment["tiers"]["words"]["entries"][idx][-1]
    sub_phones = [_ph for _ph in g2p(word) if _ph not in (" ", "'")]
    ph_len = (end - start) / len(sub_phones)
    return [
        (ph, start + i * ph_len, start + (i + 1) * ph_len)
        for i, ph in enumerate(sub_phones)
    ]


def pad_last_phoneme(
    phones: List[Tuple[str, float, float]], max_len_sec: float
) -> None:
    """Extend the last phoneme to match max file length if needed."""
    last_ph_end = phones[-1][-1]
    if last_ph_end < max_len_sec:
        ph, start, _ = phones.pop()
        phones.append((ph, start, max_len_sec))


def build_phone_indices(
    phones: List[Tuple[str, float, float]],
    num_frames: int,
    window_size: int,
    sec_to_ms: int,
) -> np.ndarray:
    """Map time (ms) to phoneme indices."""
    file_len_ms = num_frames * window_size
    indices = np.full(file_len_ms, -1, dtype=np.int16)
    for i, (_, start, end) in enumerate(phones):
        indices[int(start * sec_to_ms) : int(end * sec_to_ms)] = i
    assert np.all(indices > -1), "Missed phoneme indices"
    return indices


def assign_phones_to_frames(
    phone_indices: np.ndarray,
    num_frames: int,
    window_size: int,
    phones_per_frame: int,
    max_shift: int,
) -> Tuple[np.ndarray, List[int]]:
    """Assign phoneme indices to each frame with overlap handling."""
    drop_shift, drop_indices = 0, []
    emb_indices = np.full((num_frames, phones_per_frame), -1, dtype=np.int16)

    for i in range(num_frames):
        start, end = i * window_size, (i + 1) * window_size
        phone_idx = phone_indices[start:end]

        # Select most frequent phones in this frame
        frame_phones = [
            ph[0] - drop_shift
            for ph in Counter(phone_idx).most_common()[:phones_per_frame]
        ]
        frame_phones = sorted(frame_phones)
        emb_indices[i, : len(frame_phones)] = frame_phones

        if i > 0:
            cur_start = emb_indices[i][0]
            prev_end = max(emb_indices[i - 1])
            shift = cur_start - prev_end

            while shift > max_shift:
                drop_shift += 1
                drop_indices.append(prev_end + 1)
                emb_indices[i, : len(frame_phones)] -= 1
                cur_start = emb_indices[i][0]
                shift = cur_start - prev_end

    return emb_indices, drop_indices


def finalize_embedding_indices(
    emb_indices: np.ndarray, phones_per_frame: int
) -> np.ndarray:
    """Ensure no padding remains and format depending on phones_per_frame."""
    if phones_per_frame == 3:
        return emb_indices + 1  # shift -1 padding to 0
    else:
        # Duplicate primary phone where secondary is missing
        for row in emb_indices:
            if row[1] == -1:
                row[1] = row[0]
        assert np.all(emb_indices >= 0), "Negative phone indices found"
        return emb_indices


# ------------------------
# Main function
# ------------------------
def align_prompt(
    phoneme_alignment: Dict[str, Any],
    num_frames: int,
    phone_to_idx: Dict[str, int],
    g2p: G2p,
    phones_per_frame: int = 2,
    max_shift: int = 1,
    sec_to_ms: int = 1000,
    window_size: int = 80,
    silence_phone: str = "sil",
    unknown_phone: str = "spn",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns phoneme annotations with frame-based acoustic features.

    Returns:
        phone_tokens: Array of phoneme indices (uint8).
        phone_emb_indices: Frame-to-phoneme index mapping (int16).
    """

    words_search_map = build_word_search_map(phoneme_alignment)
    max_len_sec = num_frames * window_size / sec_to_ms

    phones: List[Tuple[str, float, float]] = []
    prev_end = 0.0

    # Process phoneme alignment
    for start, end, ph in phoneme_alignment["tiers"]["phones"]["entries"]:
        if start >= max_len_sec:
            break
        if ph == silence_phone:
            continue

        if start > prev_end and phones:
            # Adjust previous phoneme boundary
            last_ph, last_start, _ = phones.pop()
            phones.append((last_ph, last_start, start))
        elif start > prev_end:
            start = prev_end

        if ph == unknown_phone:
            phones.extend(
                expand_unknown_phone(
                    start, end, start + end, words_search_map, phoneme_alignment, g2p
                )
            )
            end = phones[-1][-1]
        else:
            phones.append((ph, start, end))

        prev_end = end

    pad_last_phoneme(phones, max_len_sec)

    # Convert phones to tokens
    phone_tokens = np.array([phone_to_idx[ph] for ph, _, _ in phones], dtype=np.uint8)

    # Map phonemes to frames
    phone_indices = build_phone_indices(phones, num_frames, window_size, sec_to_ms)
    phone_emb_indices, drop_indices = assign_phones_to_frames(
        phone_indices, num_frames, window_size, phones_per_frame, max_shift
    )

    # Adjust tokens to match embedding indices
    phone_tokens = np.delete(phone_tokens, drop_indices)
    diff = len(phone_tokens) - max(phone_emb_indices[-1]) - 1
    assert diff >= 0, "Phoneme index out of range"
    if diff > 0:
        phone_tokens = phone_tokens[:-diff]

    phone_emb_indices = finalize_embedding_indices(phone_emb_indices, phones_per_frame)

    return phone_tokens, phone_emb_indices
