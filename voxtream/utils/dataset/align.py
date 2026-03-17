import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from voxtream.utils.aligner import PhonemeAligner


def mp_func(
    data,
    func,
    num_proc: int = mp.cpu_count(),
    total: int = None,
):
    results = []
    with mp.Pool(processes=num_proc) as p, tqdm(total=total or len(data)) as pbar:
        for result in p.imap(func, data):
            results.append(result)
            pbar.update()

    return results


def grouped_iter(df, groups):
    for idx in groups:
        yield df.loc[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--meta",
        type=str,
        help="Path to metadata file",
    )
    parser.add_argument(
        "-g",
        "--group-meta",
        type=str,
        help="Path to groupped metadata file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Path to output directory",
    )
    parser.add_argument(
        "-ph", "--num-phones", type=int, help="Number of phonemes per frames", default=2
    )
    parser.add_argument(
        "-nf", "--num-frames", type=int, help="Number of Mimi frames", default=688
    )
    parser.add_argument(
        "-s", "--shift", type=int, help="Maximum phoneme shift", default=2
    )
    parser.add_argument(
        "-np",
        "--num-proc",
        type=int,
        help="Number of processes",
        default=mp.cpu_count(),
    )
    parser.add_argument(
        "-fm",
        "--filler-meta",
        type=str,
        help="Path to fillers metadata file",
        default=None,
    )
    args = parser.parse_args()

    phoneme_dict_path = hf_hub_download("herimor/voxtream2", "phoneme_to_token.json")
    with open(phoneme_dict_path) as f:
        phone_to_idx = json.loads(f.read())

    aligner = PhonemeAligner(
        phones_per_frame=args.num_phones,
        num_frames=args.num_frames,
        phone_to_idx=phone_to_idx,
        max_shift=args.shift,
        drop_silence=False,  # drop only for Charsiu aligner
    )

    print("Loading metadata...")
    # [phones, alignments, durations, punct_insertions, punct_symbols, speaker]
    meta = pd.read_parquet(args.meta)
    # [indices]
    group_meta = pd.read_parquet(args.group_meta)

    if args.filler_meta is not None:
        fillers = pd.read_parquet(args.filler_meta)
        meta = pd.concat([meta, fillers], axis=1)
    print("Metadata loaded.")

    # # debug single process
    # results = []
    # for group in tqdm(grouped_iter(meta, group_meta.indices), total=len(group_meta)):
    #     results.append(aligner.process(group))

    results = mp_func(
        data=grouped_iter(meta, group_meta.indices),
        func=aligner.process,
        num_proc=args.num_proc,
        total=len(group_meta),
    )

    all_punct = np.empty((len(results),), dtype=object)
    all_phone_tokens = np.empty((len(results),), dtype=object)
    all_label_shifts = np.empty((len(results), args.num_frames), dtype=np.uint8)
    all_phone_indices = np.empty(
        (len(results), args.num_frames, args.num_phones), dtype=np.uint16
    )

    for i, (
        phone_tokens,
        phone_indices,
        label_shifts,
        punct_ins_indices,
        punct_del_indices,
        punct_syms,
    ) in enumerate(tqdm(results)):
        all_phone_tokens[i] = phone_tokens
        all_phone_indices[i] = phone_indices
        all_label_shifts[i] = label_shifts

        punct = [
            punct_ins_indices,
            punct_del_indices,
            punct_syms,
        ]
        all_punct[i] = punct

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Saving results...")
    np.save(output_dir / "phone_tokens.npy", all_phone_tokens)
    np.save(output_dir / "sem_label_shifts.npy", all_label_shifts)
    np.save(output_dir / "phone_emb_indices.npy", all_phone_indices)
    np.save(output_dir / "punctuation.npy", all_punct)
