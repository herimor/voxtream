import argparse
from functools import partial
from pathlib import Path

import pandas as pd

from voxtream.utils.dataset.align import mp_func
from voxtream.utils.text.phonemizer import ESpeak


def process(text: str, phonemizer, punctuation, sep="|"):
    phones, punct_symbs, punct_pos, words_pos = [], [], [], []
    try:
        phones_seq = phonemizer.phonemize(text)

        prev_end = 0
        for word in phones_seq.split():
            for ph in word.split(sep):
                if ph == "":
                    continue
                phones.append(ph)

            words_pos.append((prev_end, len(phones) - 1))
            prev_end = len(phones)

            if phones[-1].endswith(punctuation):
                phone = phones.pop()
                symb = phone[-1]
                if phone[:-1] != "":
                    phones.append(phone[:-1])
                punct_symbs.append(symb)
                punct_pos.append(len(phones))
    except Exception:
        pass

    return phones, punct_symbs, punct_pos, words_pos


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
        "-np",
        "--num-proc",
        type=str,
        help="Number of processes for parallel processing",
    )
    args = parser.parse_args()

    meta_path = Path(args.meta_path)
    meta = pd.read_parquet(meta_path)

    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    phonemizer = ESpeak("en")
    punctuation = (".", ",", "?", "!")
    proc_fn = partial(process, phonemizer=phonemizer, punctuation=punctuation)

    results = mp_func(meta.text_norm.tolist(), proc_fn, args.num_proc)
    results = pd.DataFrame(
        data=results, columns=["phones", "punct_symbs", "punct_pos", "words_pos"]
    )
    results.to_parquet(save_dir / "phonemes.parquet", index=None)
