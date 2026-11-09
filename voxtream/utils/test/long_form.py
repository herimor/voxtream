"""
Long-form evaluation over fixed windows of the generated audio, after Lim et al.
(arXiv:2610.19553): WER, speaker similarity to the prompt and UTMOS of every window of
`--windows` seconds, pooled over the files of the protocol by segment of generated time.

WER: the Whisper transcript (the long-form decoding of `asr.py`), its words timestamped
by CTC forced alignment on the generated audio (torchaudio MMS_FA), edit-distance
aligned to the target text; an error lands in the window of its hypothesis word, a
deletion between the neighbouring hypothesis words. The window WER pools reference
words and errors over all files. SPK-SIM: cosine similarity of the WavLM-ECAPA
embedding of a window to that of the prompt. UTMOS: on the window.

Everything is cached per file in the output directory, so the script can run while
files are still being generated and only scores the new ones on the next run. It also
writes `wer_whisper.csv/txt` as `asr.main` does, from the same transcripts.
"""

import argparse
import json
import re
import time
from pathlib import Path

import jiwer
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm
from whisper.normalizers import EnglishTextNormalizer

from voxtream.utils.test.asr import (
    init_whisper_model,
    load_audio,
    prepare_sentence,
    remove_fillers,
    set_seed,
)
from voxtream.utils.test.ecapa_tdnn import init_ecapa
from voxtream.utils.text.number_norm import normalize_numbers

SR = 16000
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ECAPA_CKPT = PROJECT_ROOT / "wavlm-large/wavlm_large_finetune.pth"


class WordAligner:
    """Word timestamps of a transcript on its audio, by CTC forced alignment."""

    def __init__(self, device: str):
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model(with_star=True).to(device).eval()
        self.dict = bundle.get_dict(star="*")
        self.device = device

    def letters(self, word: str) -> str:
        # the Whisper normalizer writes numbers in digits; the aligner reads letters
        try:
            word = normalize_numbers(word)
        except Exception:
            pass
        return re.sub(r"[^a-z']", "", word.lower())

    @torch.inference_mode()
    def emission(self, audio: np.ndarray, chunk_sec: float = 60.0) -> torch.Tensor:
        """CTC log-probabilities (frames, labels), the audio run in chunks: the acoustic
        model holds the activations of a whole 5 min file only on an empty GPU."""
        out = []
        step = int(chunk_sec * SR)
        for a in range(0, len(audio), step):
            x = torch.from_numpy(audio[a : a + step]).unsqueeze(0).to(self.device)
            em, _ = self.model(x)
            out.append(torch.log_softmax(em[0], dim=-1).cpu())
        return torch.cat(out)

    def __call__(
        self, audio: np.ndarray, words: list[str]
    ) -> list[tuple[float, float]]:
        units = []
        for w in words:
            toks = [self.dict[c] for c in self.letters(w)]
            units.append(toks or [self.dict["*"]])
        targets = torch.tensor([[t for u in units for t in u]], dtype=torch.int32)
        emission = self.emission(audio).unsqueeze(0)
        # the alignment itself is cheap and runs on the CPU, whatever the GPU holds
        ali, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
        spans = torchaudio.functional.merge_tokens(ali[0], scores[0].exp())
        ratio = len(audio) / emission.size(1) / SR
        times, j = [], 0
        for u in units:
            sp = spans[j : j + len(u)]
            j += len(u)
            times.append((sp[0].start * ratio, sp[-1].end * ratio))
        return times


def word_errors(
    ref: str, hyp: str, hyp_times: list[tuple[float, float]], duration: float
) -> list[tuple[str, float]]:
    """
    Reference words and errors of the alignment of `hyp` to `ref`, each with the time
    it lands on: ("hit"|"sub"|"ins"|"del", seconds).
    """
    ref_words, hyp_words = ref.split(), hyp.split()
    if not ref_words or not hyp_words:
        events = [("del", 0.0)] * len(ref_words)
        events += [("ins", (s + e) / 2) for s, e in hyp_times[: len(hyp_words)]]
        return events
    out = jiwer.process_words(ref, hyp)
    mid = [(s + e) / 2 for s, e in hyp_times]
    events = []
    for ch in out.alignments[0]:
        if ch.type in ("equal", "substitute"):
            kind = "hit" if ch.type == "equal" else "sub"
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                events.append((kind, mid[ch.hyp_start_idx + k]))
        elif ch.type == "insert":
            for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                events.append(("ins", mid[j]))
        elif ch.type == "delete":
            prev_end = hyp_times[ch.hyp_start_idx - 1][1] if ch.hyp_start_idx else 0.0
            next_start = (
                hyp_times[ch.hyp_start_idx][0]
                if ch.hyp_start_idx < len(hyp_times)
                else duration
            )
            events += [("del", 0.5 * (prev_end + next_start))] * (
                ch.ref_end_idx - ch.ref_start_idx
            )
    return events


@torch.no_grad()
def spk_embedding(model, audio: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(audio).unsqueeze(0).to(device).half()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        emb = model(x).float().squeeze().cpu().numpy()
    return emb / np.linalg.norm(emb)


@torch.no_grad()
def utmos_score(model, audio: np.ndarray, device: str) -> float:
    x = torch.from_numpy(audio).unsqueeze(0).to(device)
    return float(model(x, SR).cpu().numpy()[0])


def segment_label(w: int, window: float) -> str:
    return f"{int(w * window)}-{int((w + 1) * window)}"


def ready_files(
    protocol: pd.DataFrame, dataset_dir: Path, file_ext: str, settle_sec: float = 30.0
):
    """Rows whose generated file exists and was not written in the last `settle_sec`."""
    now = time.time()
    rows = []
    for _, row in protocol.iterrows():
        path = (dataset_dir / row.path).with_suffix(f".{file_ext}")
        if path.exists() and now - path.stat().st_mtime > settle_sec:
            rows.append(row)
    return rows


def main(
    meta: Path,
    dataset_dir: Path,
    reference_dir: Path,
    windows: list[float],
    file_ext: str = "flac",
    asr_model_name: str = "openai/whisper-large-v3",
    min_window_sec: float = 2.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    protocol = pd.read_csv(meta)
    rows = ready_files(protocol, dataset_dir, file_ext)
    print(f"{len(rows)} of {len(protocol)} files ready")
    if not rows:
        return
    normalizer = EnglishTextNormalizer()

    # --- transcripts and word timestamps, cached per file -------------------------
    align_path = dataset_dir / "long_form_align.json"
    aligned = json.loads(align_path.read_text()) if align_path.exists() else {}
    todo = [row for row in rows if row.path not in aligned]
    if todo:
        # two passes, so Whisper and the aligner never hold the GPU together; the
        # transcripts are cached as they come, in case the run is cut short
        transcript_path = dataset_dir / "long_form_transcripts.json"
        transcripts = (
            json.loads(transcript_path.read_text()) if transcript_path.exists() else {}
        )
        pending = [row for row in todo if row.path not in transcripts]
        if pending:
            processor, whisper, transcribe = init_whisper_model(
                asr_model_name, device, dtype=torch.float16
            )
            for row in tqdm(pending, desc="Transcribing"):
                audio = load_audio(
                    (dataset_dir / row.path).with_suffix(f".{file_ext}"), SR
                )
                transcripts[row.path] = (
                    transcribe(processor, whisper, audio, sr=SR, device=device),
                    len(audio) / SR,
                )
                transcript_path.write_text(json.dumps(transcripts))
            del whisper, processor
        torch.cuda.empty_cache()
        aligner = WordAligner(device)
        cpu_aligner = None
        for row in tqdm(todo, desc="Aligning"):
            pred_text, duration = transcripts[row.path]
            hyp = normalizer(remove_fillers(pred_text))
            words = hyp.split()
            times = []
            if words:
                audio = load_audio(
                    (dataset_dir / row.path).with_suffix(f".{file_ext}"), SR
                ).astype(np.float32)
                try:
                    times = aligner(audio, words)
                except (torch.OutOfMemoryError, RuntimeError) as err:
                    if "out of memory" not in str(err):
                        raise
                    # a long file on a full GPU: align it on the CPU rather than fail
                    torch.cuda.empty_cache()
                    if cpu_aligner is None:
                        cpu_aligner = WordAligner("cpu")
                    times = cpu_aligner(audio, words)
            aligned[row.path] = dict(
                pred_text=pred_text, hyp=hyp, times=times, duration=duration
            )
            align_path.write_text(json.dumps(aligned))
        del aligner, cpu_aligner
        torch.cuda.empty_cache()

    # --- file-level WER, as `asr.main` writes it -----------------------------------
    wer_rows = []
    for row in rows:
        rec = aligned[row.path]
        text_wo, pred_wo = remove_fillers(row.text), remove_fillers(rec["pred_text"])
        wer_rows.append(
            dict(
                path=row.path,
                wer=jiwer.wer(
                    prepare_sentence(row.text), prepare_sentence(rec["pred_text"])
                ),
                wer_norm=jiwer.wer(normalizer(text_wo), normalizer(pred_wo)),
                wer_wo_fillers=jiwer.wer(
                    prepare_sentence(text_wo), prepare_sentence(pred_wo)
                ),
                text=row.text,
                pred_text=rec["pred_text"],
                pred_text_wo_fillers=pred_wo,
            )
        )
    wer_df = pd.DataFrame(wer_rows)
    wer_df.to_csv(dataset_dir / "wer_whisper.csv", index=False)
    wer_txt = (
        f"WER: {round(wer_df.wer.mean() * 100, 2)}%\n"
        f"WER-norm: {round(wer_df.wer_norm.mean() * 100, 2)}%\n"
        f"WER-wo-fillers: {round(wer_df.wer_wo_fillers.mean() * 100, 2)}%"
    )
    (dataset_dir / "wer_whisper.txt").write_text(wer_txt)

    # --- word events: the reference words and errors of a file with their time -------
    events, files = [], []
    for row in rows:
        rec = aligned[row.path]
        ref = normalizer(remove_fillers(str(row.text)))
        ev = word_errors(
            ref, rec["hyp"], [tuple(t) for t in rec["times"]], rec["duration"]
        )
        events += [dict(path=row.path, kind=k, t=t) for k, t in ev]
        n_ref = sum(k != "ins" for k, _ in ev)
        n_err = sum(k != "hit" for k, _ in ev)
        files.append(
            dict(
                path=row.path,
                gen_duration=rec["duration"],
                target_duration=row.get("target_duration", np.nan),
                n_ref=n_ref,
                n_err=n_err,
                wer=n_err / max(n_ref, 1),
            )
        )
    events = pd.DataFrame(events)
    files = pd.DataFrame(files).set_index("path")
    events.to_csv(dataset_dir / "long_form_words.csv", index=False)

    # --- speaker similarity and UTMOS per window, cached per file ----------------------
    score_path = dataset_dir / "long_form_scores.json"
    scores = json.loads(score_path.read_text()) if score_path.exists() else {}
    todo = [
        row for row in rows if any(f"{row.path}|{w}" not in scores for w in windows)
    ]
    if todo:
        ecapa = init_ecapa(str(ECAPA_CKPT), device)
        utmos = (
            torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0",
                "utmos22_strong",
                trust_repo=True,
                verbose=False,
            )
            .to(device)
            .eval()
        )
        for row in tqdm(todo, desc="Scoring windows"):
            audio, _ = librosa.load(
                (dataset_dir / row.path).with_suffix(f".{file_ext}"), sr=SR
            )
            prompt, _ = librosa.load(
                (reference_dir / row.file_name).with_suffix(f".{file_ext}"), sr=SR
            )
            prompt_emb = spk_embedding(ecapa, prompt, device)
            duration = len(audio) / SR
            for window in windows:
                key = f"{row.path}|{window}"
                if key in scores:
                    continue
                recs, embs = [], []
                for w in range(int(np.ceil(duration / window))):
                    chunk = audio[int(w * window * SR) : int((w + 1) * window * SR)]
                    rec = dict(
                        window=w, audio_sec=len(chunk) / SR, spk_sim=None, utmos=None
                    )
                    if len(chunk) >= min_window_sec * SR:
                        emb = spk_embedding(ecapa, chunk, device)
                        embs.append(emb)
                        rec["spk_sim"] = float(prompt_emb @ emb)
                        rec["utmos"] = utmos_score(utmos, chunk, device)
                    recs.append(rec)
                mean_emb = np.mean(embs, axis=0) if embs else None
                scores[key] = dict(
                    windows=recs,
                    file_spk_sim=(
                        float(prompt_emb @ (mean_emb / np.linalg.norm(mean_emb)))
                        if mean_emb is not None
                        else None
                    ),
                )
            score_path.write_text(json.dumps(scores))
        del ecapa, utmos
        torch.cuda.empty_cache()

    # --- tables --------------------------------------------------------------------
    rows_out = []
    for row in rows:
        for window in windows:
            rec = scores[f"{row.path}|{window}"]
            ev = events[events.path == row.path] if len(events) else events
            for r in rec["windows"]:
                w = r["window"]
                e = ev[(ev.t // window) == w] if len(ev) else ev
                rows_out.append(
                    dict(
                        path=row.path,
                        window_sec=window,
                        window=w,
                        audio_sec=r["audio_sec"],
                        n_ref=int((e.kind != "ins").sum()) if len(e) else 0,
                        n_err=int((e.kind != "hit").sum()) if len(e) else 0,
                        n_sub=int((e.kind == "sub").sum()) if len(e) else 0,
                        n_del=int((e.kind == "del").sum()) if len(e) else 0,
                        n_ins=int((e.kind == "ins").sum()) if len(e) else 0,
                        spk_sim=np.nan if r["spk_sim"] is None else r["spk_sim"],
                        utmos=np.nan if r["utmos"] is None else r["utmos"],
                    )
                )
            if window == windows[0]:
                files.loc[row.path, "spk_sim"] = rec["file_spk_sim"]
                files.loc[row.path, "utmos"] = np.nanmean(
                    [r["utmos"] for r in rec["windows"] if r["utmos"] is not None]
                    or [np.nan]
                )
    win_df = pd.DataFrame(rows_out)
    win_df.to_csv(dataset_dir / "long_form_windows.csv", index=False)
    files.to_csv(dataset_dir / "long_form_files.csv")

    lines = [
        f"files: {len(files)} of {len(protocol)}, generated {files.gen_duration.sum() / 3600:.2f} h "
        f"(target {files.target_duration.sum() / 3600:.2f} h), "
        f"mean gen/target duration {np.nanmean(files.gen_duration / files.target_duration):.3f}",
        wer_txt.replace("\n", "  "),
        f"file-level: WER-norm {100 * files.n_err.sum() / files.n_ref.sum():.2f}% (pooled); "
        f"SPK-SIM {files.spk_sim.mean():.3f} (mean window embedding); UTMOS {files.utmos.mean():.2f} (mean of windows)",
    ]
    for window in windows:
        r = win_df[win_df.window_sec == window]
        lines.append(f"\nper {int(window)} s segment of generated audio:")
        lines.append(
            f"{'segment':>10} {'files':>6} {'ref words':>10} {'WER%':>7} {'SPK-SIM':>8} {'UTMOS':>6}"
        )
        for w, g in r.groupby("window"):
            lines.append(
                f"{segment_label(w, window):>10} {len(g):>6} {int(g.n_ref.sum()):>10} "
                f"{100 * g.n_err.sum() / max(g.n_ref.sum(), 1):>7.2f} "
                f"{g.spk_sim.mean():>8.3f} {g.utmos.mean():>6.2f}"
            )
        lines.append(
            f"{'overall':>10} {len(files):>6} {int(r.n_ref.sum()):>10} "
            f"{100 * r.n_err.sum() / max(r.n_ref.sum(), 1):>7.2f} "
            f"{r.spk_sim.mean():>8.3f} {r.utmos.mean():>6.2f}"
        )
    result = "\n".join(lines)
    print(result)
    (dataset_dir / "long_form.txt").write_text(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--meta-path", type=str, required=True, help="Protocol csv"
    )
    parser.add_argument(
        "-d", "--dataset-dir", type=str, required=True, help="Generated audio"
    )
    parser.add_argument(
        "-r", "--reference-dir", type=str, required=True, help="Root of the prompts"
    )
    parser.add_argument("-w", "--windows", type=float, nargs="+", default=[30.0, 10.0])
    parser.add_argument("-e", "--file-ext", type=str, default="flac")
    args = parser.parse_args()
    set_seed()
    main(
        meta=Path(args.meta_path),
        dataset_dir=Path(args.dataset_dir),
        reference_dir=Path(args.reference_dir),
        windows=args.windows,
        file_ext=args.file_ext,
    )
