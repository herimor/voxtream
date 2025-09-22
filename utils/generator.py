import re
import json
import nltk
import shlex
import torch
import random
import inflect
import tempfile
import argparse
import subprocess
import numpy as np
from g2p_en import G2p
from typing import Dict
from pathlib import Path
from typing import Generator
from collections import Counter


num_conv = inflect.engine()
currency_map = {
    '$': ' dollar',
    '€': ' euro',
    '£': ' pound',
    '¥': ' yen',
    '₹': ' rupee',
    '₩': ' won',
    '₽': ' ruble',
    '₴': ' hryvnia',
    '₺': ' lira',
    '₦': ' naira',
    '%': ' percent'
}
# Regex pattern to match common currency symbols
currency_pattern = r'[%$€£¥₹₩₽₴₺₦]'


def normalize_text(text: str) -> str:
    text = text.lower()
    # OOV fix
    # convert 'mr' -> 'mister'
    text = text.replace('mr', 'mister')
    # convert 'mrs' -> 'missis'
    text = text.replace('mrs', 'missis')

    # convert $1 -> 1$
    upd_words = []
    for word in text.split():
        if word.startswith('$'):
            upd_words.append((word, word[1:] + '$'))
    for word, rep_word in upd_words:
        text = text.replace(word, rep_word)
    
    # find all numbers
    nums = sorted(re.findall(r'\d+', text), key=len, reverse=True)
    # convert numbers to words
    for num in nums:
        text = text.replace(num, num_conv.number_to_words(num))
    # remove added ['—,–'] special and [-,.?!] symbols
    text = re.sub(r'[—–.,-?!]', ' ', text)
    # convert all currency characters to words
    for cur_ch in re.findall(currency_pattern, text):
        text = text.replace(cur_ch, currency_map[cur_ch])
    
    # remove extra space
    text = re.sub(r'\s+', ' ', text).strip()
    
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
        nltk.download(resource.split("/")[-1])


def mfa_align(
    audio_file: str,
    text: str,
    dict_path: str = 'english_us_arpa',
    model_path: str = 'english_us_arpa',
    dict_default_path: str = 'Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict'
) -> Dict[str, any]:
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
        print('Downloading MFA models...')
        commands = [
            'mfa model download acoustic english_us_arpa',
            'mfa model download dictionary english_us_arpa'
        ]
        for cmd in commands:
            subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print('Done!')
    
    temp_dir = tempfile.TemporaryDirectory()
    output_path = Path(temp_dir.name) / 'mfa.json'
    text_file = Path(temp_dir.name) / 'transcript.txt'
    with open(text_file, 'w') as f:
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
        text=True  # ensures output is returned as strings, not bytes
    )

    if process.returncode != 0:
        raise RuntimeError(f"Error running MFA align_one: {process.stderr.strip()}\n")
    
    with open(output_path) as f:
        phoneme_alignment = json.load(f)
    temp_dir.cleanup()

    return phoneme_alignment


def align_prompt(
    row,
    phone_to_idx: Dict,
    g2p: G2p,
    phones_per_frame: int = 2,
    max_shift: int = 1,
    sec_to_ms: int = 1000,
    window_size: int = 80,
    silence_phone: str = 'sil',
    unknown_phone: str = 'spn',
):
    path, num_frames = row
    if isinstance(path, str):
        with open(path) as f:
            data = json.load(f)
    else:
        data = path

    words_search_map = []
    for start, end, word in data['tiers']['words']['entries']:
        words_search_map.append(start + end)
    words_search_map = np.array(words_search_map)

    prev_end = 0
    phones, phone_tokens = [], []
    max_len_sec = num_frames * window_size / sec_to_ms
    for start, end, ph in data['tiers']['phones']['entries']:
        if start >= max_len_sec:
            break

        if ph == silence_phone:
            continue

        word_key = start + end
        if start > prev_end:
            if len(phones) > 0:
                _ph, _start, _end = phones.pop()
                _end = start
                phones.append([_ph, _start, _end])
            else:
                start = prev_end

        if ph == unknown_phone:
            idx = np.argmin(np.abs(words_search_map - word_key))
            word = data['tiers']['words']['entries'][idx][-1]
            _phs = [_ph for _ph in g2p(word) if _ph not in (' ', "'")]
            ph_len = (end - start) / len(_phs)
            for _ph in _phs:
                phones.append([_ph, start, start + ph_len])
                start += ph_len
            end = start
        else:
            phones.append([ph, start, end])
        prev_end = end

    # pad last phoneme if needed
    file_end = phones[-1][-1]
    if file_end < max_len_sec:
        ph, start, end = phones.pop()
        end = max_len_sec
        phones.append([ph, start, end])

    for ph, *_ in phones:
        phone_tokens.append(phone_to_idx[ph])
    phone_tokens = np.array(phone_tokens, dtype=np.uint8)
    
    # map phonemes idx to time in ms
    file_len_ms = num_frames * window_size
    phone_indices = np.full((file_len_ms,), -1, dtype=np.int16)
    for i, (ph, start, end) in enumerate(phones):
        phone_indices[int(start * sec_to_ms): int(end * sec_to_ms)] = i
    assert np.all(phone_indices > -1), 'Missed phoneme indices'

    drop_shift, drop_indices = 0, []
    phone_emb_indices = np.full((num_frames, phones_per_frame), -1, dtype=np.int16)
    
    for i in range(num_frames):
        start = i * window_size
        end = start + window_size
        phone_idx = phone_indices[start: end]

        target_phones = Counter(phone_idx).most_common()[:phones_per_frame]
        target_phones = sorted([ph[0] - drop_shift for ph in target_phones])
        phone_emb_indices[i, :len(target_phones)] = target_phones

        if i > 0:
            cur_start = phone_emb_indices[i][0]
            prev_end = max(phone_emb_indices[i - 1])
            shift = cur_start - prev_end

            while shift > max_shift:
                drop_shift += 1
                drop_indices.append(prev_end + 1)
                
                phone_emb_indices[i, :len(target_phones)] -= 1
                cur_start = phone_emb_indices[i][0]
                shift = cur_start - prev_end
        else:
            shift = 0
    
    phone_tokens = np.delete(phone_tokens, drop_indices)
    diff = len(phone_tokens) - max(phone_emb_indices[-1]) - 1
    assert diff >= 0, 'Phoneme index out of range'
    if diff > 0:
        phone_tokens = phone_tokens[:-diff]  # remove last token if not used
    
    if phones_per_frame == 3:
        phone_emb_indices += 1 # convert PAD from -1 to 0
    else:
        # replace padding with the same token per frame
        for i in range(len(phone_emb_indices)):
            if phone_emb_indices[i][1] == -1:
                phone_emb_indices[i][1] = phone_emb_indices[i][0]
        assert np.all(phone_emb_indices >= 0), 'Negative phone indices found'

    return phone_tokens, phone_emb_indices
