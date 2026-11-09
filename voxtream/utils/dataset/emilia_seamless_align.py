"""Word-level forced alignment for Emilia and HiFiTTS-2 with the Seamless UnitY2 aligner.

Reads a protocol parquet of (path, duration, text), `path` being relative to
`--audio-dir` and without an extension (`{audio_dir}/{path}.mp3`, see `paths.py`), keeps
the rows between `--min-duration` and `--max-duration`, and writes one JSON per audio
file holding the word timings:

    {out_root}/{parent}/{name}.seamless.json
    {
      "duration": 19.964,
      "words": [
        {"word": "Disney",  "start": 0.02, "end": 0.62, "score": 0.494},
        {"word": "planes.", "start": 0.78, "end": 1.74, "score": 0.136}
      ]
    }

Word text is exactly what the aligner returns -- punctuation stays attached, and
the word count matches the whitespace token count of the transcript. `score` is
the Viterbi geometric-mean probability per aligned frame, in [0, 1]; low values
flag words the aligner was unsure about. The audio path is not stored: readers take it from
the protocol parquet and find the JSON by the audio file's parent directory and stem.

Work is sharded over `--procs-per-gpu` worker processes on each of `--gpus`.
Seamless saturates one RTX 3090 at roughly 60x realtime with two workers, so
more than two per GPU buys nothing. The pure-python monotonic alignment search
in the vendored seamless_communication is ~42% of the runtime; `--numba` swaps
in a jitted, bit-identical replacement and is on by default.

CPU is the scarce resource: each worker is capped to `--cpu-budget / n_workers`
torch threads so the whole job stays inside the budget.

Already-written outputs are skipped, so the job is resumable: re-running picks
up where it stopped.

Example (6 GPUs x 2 procs, 31 cores, files >= 16 s):
    python -m voxtream.utils.dataset.emilia_seamless_align \
        -m emilia/metadata.parquet -a /data/emilia -o /data/alignment/emilia \
        --aligner-root /path/to/forced-aligners-bench \
        --gpus 0,2,3,5,6,7 --procs-per-gpu 2 --cpu-budget 31 --min-duration 16
"""

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

from tqdm.auto import tqdm

from voxtream.utils.dataset.paths import (
    SEAMLESS_SUFFIX,
    alignment_path,
    audio_path,
    check_protocol,
)

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# numba monotonic alignment search
# --------------------------------------------------------------------------- #
def _install_numba_mas() -> bool:
    """Replace seamless's pure-python MAS with a jitted, bit-identical version.

    `viterbi_decode` looks `_monotonic_alignment_search` up in its module globals
    at call time, so rebinding the module attribute is enough.
    """
    import numpy as np
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _mas(log_prob, T_feat, T_text):  # log_prob: (T_text, T_feat)
        Q = np.full((T_text, T_feat), -np.inf)
        s = 0.0
        for j in range(T_feat):
            s += log_prob[0, j]
            Q[0, j] = s
        for j in range(1, T_feat):
            hi = min(j + 1, T_text)
            for i in range(1, hi):
                a = Q[i - 1, j - 1]
                b = Q[i, j - 1]
                Q[i, j] = (a if a > b else b) + log_prob[i, j]
        A = np.full(T_feat, T_text - 1, dtype=np.int64)
        for j in range(T_feat - 2, -1, -1):
            i_b = A[j + 1]
            if i_b == 0:
                A[j] = 0
            else:
                A[j] = i_b - 1 if Q[i_b - 1, j] >= Q[i_b, j] else i_b
        return A

    def monotonic_alignment_search(attn_lprob):
        lp = np.ascontiguousarray(attn_lprob.transpose(1, 0).astype(np.float64))
        return _mas(lp, lp.shape[1], lp.shape[0])

    import seamless_communication.models.aligner.model as M

    M._monotonic_alignment_search = monotonic_alignment_search
    monotonic_alignment_search(np.log(np.full((40, 8), 1 / 8)))  # warm the JIT
    return True


def align_with_retry(aligner, torch, row, args, wid):
    """Align one file, retrying past the OOM two workers on a card provoke.

    A 30 s clip with a long transcript needs a single multi-GB distance tensor in
    UnitY2AlignmentEncoder. When both workers on a GPU reach one at the same
    moment the second loses. Dropping this worker's cached blocks and waiting for
    the sibling to move on clears it; the wait is jittered so two workers that
    collide do not retry in lockstep.
    """
    import random

    for attempt in range(args.oom_retries + 1):
        try:
            df, _ = aligner(
                audio=audio_path(row["path"], args.audio_dir), transcript=row["text"]
            )
            return df
        except torch.OutOfMemoryError:
            if attempt == args.oom_retries:
                raise
            torch.cuda.empty_cache()
            time.sleep((2**attempt) * (0.5 + random.random()))


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def worker(wid, gpu, rows, args, ready, q):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Two workers share a card and the aligner's distance tensor is a single
    # multi-GB allocation for a 30 s clip, so the default allocator fragments
    # badly and OOMs with GBs reserved-but-unused.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(args.threads_per_worker)

    import torch

    torch.set_num_threads(args.threads_per_worker)
    sys.path.insert(0, str(Path(args.aligner_root) / "aligners/seamless"))
    sys.path.insert(
        0,
        str(Path(args.aligner_root) / "aligners/seamless/seamless_unity_aligner/deps"),
    )

    from seamless_unity_aligner import SeamlessUnityAligner

    if args.numba:
        _install_numba_mas()

    out_root = Path(args.out_root)
    fail_dir = out_root / "_failed"
    fail_dir.mkdir(parents=True, exist_ok=True)
    faillog = open(fail_dir / f"worker{wid:02d}.tsv", "a")
    aligner = SeamlessUnityAligner(device=torch.device("cuda"))
    ready.put(wid)

    done = skipped = failed = 0
    audio_s = 0.0
    t0 = time.perf_counter()
    last = t0
    # Counts since the last tick, batched so 2M files do not mean 2M queue puts.
    tick = dict(n=0, done=0, skipped=0, failed=0, audio_s=0.0)

    def flush_tick():
        if tick["n"]:
            q.put(dict(t="tick", **tick))
            tick.update(n=0, done=0, skipped=0, failed=0, audio_s=0.0)

    for r in rows:
        dst = alignment_path(r["path"], out_root, SEAMLESS_SUFFIX)
        if not args.overwrite and dst.exists():
            skipped += 1
            tick["n"] += 1
            tick["skipped"] += 1
            if tick["n"] >= args.tick_every:
                flush_tick()
            continue
        try:
            df = align_with_retry(aligner, torch, r, args, wid)
            words = [
                dict(
                    word=w.word,
                    start=round(float(w.start), 3),
                    end=round(float(w.end), 3),
                    score=round(float(w.score), 4),
                )
                for w in df.itertuples()
            ]
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".json.tmp")
            with open(tmp, "w") as fh:
                json.dump(
                    dict(duration=r["duration"], words=words),
                    fh,
                    ensure_ascii=False,
                )
            os.replace(tmp, dst)  # atomic: a killed job never leaves half a file
            done += 1
            audio_s += r["duration"]
            tick["done"] += 1
            tick["audio_s"] += r["duration"]
            if r["duration"] >= args.empty_cache_above:
                torch.cuda.empty_cache()
        except Exception:
            failed += 1
            tick["failed"] += 1
            # Every failure is recorded -- at 2M files the first-5 stderr cap
            # would hide the rest, and re-running only retries what has no output.
            print(
                f"{r['path']}\t{traceback.format_exc(limit=0).strip()}",
                file=faillog,
                flush=True,
            )
            if failed <= 5:
                print(
                    f"[w{wid}] FAILED {r['path']}\n{traceback.format_exc()}",
                    file=sys.stderr,
                    flush=True,
                )

        tick["n"] += 1
        now = time.perf_counter()
        if tick["n"] >= args.tick_every or now - last >= 5.0:
            flush_tick()
            last = now

    flush_tick()
    faillog.close()
    q.put(
        dict(
            t="done",
            wid=wid,
            gpu=gpu,
            done=done,
            skipped=skipped,
            failed=failed,
            audio_s=audio_s,
            sec=time.perf_counter() - t0,
        )
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def load_rows(meta, min_duration, max_duration, limit=None):
    import pyarrow.parquet as pq

    rows = []
    pf = pq.ParquetFile(meta)
    for b in pf.iter_batches(batch_size=500_000, columns=["path", "duration", "text"]):
        d = b.to_pydict()
        if not rows and d["path"]:
            check_protocol(d["path"][0])
        for p, du, t in zip(d["path"], d["duration"], d["text"]):
            if min_duration <= du <= max_duration:
                rows.append(dict(path=p, duration=du, text=t))
        if limit and len(rows) >= limit:
            return rows[:limit]
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "-m",
        "--meta",
        required=True,
        help="protocol parquet (path, duration, text) with relative paths",
    )
    ap.add_argument(
        "-a",
        "--audio-dir",
        required=True,
        help="absolute root of the audio: row `path` is `{audio_dir}/{path}.mp3`",
    )
    ap.add_argument(
        "-o", "--out-root", required=True, help="root of the alignment tree"
    )
    ap.add_argument("--gpus", default="4,5,6,7", help="comma-separated GPU ids")
    ap.add_argument("-p", "--procs-per-gpu", type=int, default=2)
    ap.add_argument(
        "--cpu-budget",
        type=int,
        default=24,
        help="total CPU threads across all workers",
    )
    ap.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
        help="keep files with duration >= this many seconds",
    )
    ap.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="keep files with duration <= this many seconds",
    )
    ap.add_argument(
        "--aligner-root",
        required=True,
        help="checkout of forced-aligners-bench that holds aligners/seamless",
    )
    ap.add_argument(
        "--numba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="jitted monotonic alignment search (bit-identical, ~1.5x end to end)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="re-align files that already have output",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N eligible rows (smoke test)",
    )
    ap.add_argument(
        "--log-every",
        type=float,
        default=60.0,
        help="progress-bar refresh interval when stderr is not a TTY (nohup)",
    )
    ap.add_argument(
        "--oom-retries",
        type=int,
        default=3,
        help="retries when two workers on one GPU collide on a long file",
    )
    ap.add_argument(
        "--empty-cache-above",
        type=float,
        default=25.0,
        help="release cached VRAM after a file at least this long, seconds",
    )
    ap.add_argument(
        "--tick-every",
        type=int,
        default=20,
        help="files a worker processes between progress reports",
    )
    args = ap.parse_args()

    args.audio_dir = os.path.abspath(args.audio_dir)
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    n_workers = len(gpus) * args.procs_per_gpu
    args.threads_per_worker = max(1, args.cpu_budget // n_workers)

    rows = load_rows(args.meta, args.min_duration, args.max_duration, args.limit)
    if not rows:
        print("no rows matched the duration filter", file=sys.stderr)
        return 1

    total_h = sum(r["duration"] for r in rows) / 3600
    print(f"aligner   : seamless (numba={args.numba}) @ {args.aligner_root}")
    print(f"input     : {args.meta}")
    print(
        f"filter    : {args.min_duration}s <= duration <= {args.max_duration}s -> {len(rows):,} files, {total_h:,.1f} h"
    )
    print(f"audio     : {args.audio_dir}/<path>.mp3")
    print(f"output    : {args.out_root}/<parent>/<name>.seamless.json")
    print(
        f"workers   : {n_workers} ({len(gpus)} gpus {gpus} x {args.procs_per_gpu}), "
        f"{args.threads_per_worker} threads each = {n_workers * args.threads_per_worker} "
        f"of {args.cpu_budget} cores",
        flush=True,
    )

    # Interleave so every worker gets a mix of durations rather than a
    # contiguous run of long files -- keeps the shards finishing together.
    shards = [rows[i::n_workers] for i in range(n_workers)]

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    ready, q = ctx.Queue(), ctx.Queue()
    procs = []
    for i in range(n_workers):
        gpu = gpus[i % len(gpus)]
        p = ctx.Process(
            target=worker, args=(i, gpu, shards[i], args, ready, q), daemon=False
        )
        p.start()
        procs.append(p)

    # Liveness-aware barrier: a worker that dies during model load must abort the
    # run instead of leaving the parent waiting forever.
    seen = 0
    while seen < n_workers:
        try:
            ready.get(timeout=5)
            seen += 1
        except Exception:
            if any(not p.is_alive() and p.exitcode not in (None, 0) for p in procs):
                print("a worker died during startup -- aborting", file=sys.stderr)
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    p.join()
                return 1
    print(f"all {n_workers} workers up, aligning...", flush=True)

    t0 = time.perf_counter()
    res = []
    # One bar for the whole job, fed by the workers. On a TTY it refreshes
    # continuously; under nohup tqdm still writes lines, so throttle it to
    # --log-every seconds to keep a 20 h logfile readable.
    is_tty = sys.stderr.isatty()
    pbar = tqdm(
        total=len(rows),
        unit="file",
        unit_scale=True,
        smoothing=0.05,
        dynamic_ncols=True,
        mininterval=0.5 if is_tty else args.log_every,
    )
    agg = dict(done=0, skipped=0, failed=0, audio_s=0.0)
    while len(res) < n_workers:
        m = q.get()
        if m["t"] == "tick":
            pbar.update(m["n"])
        else:
            res.append(m)
            continue
        for k in ("done", "skipped", "failed"):
            agg[k] += m[k]
        agg["audio_s"] += m["audio_s"]
        el = time.perf_counter() - t0
        pbar.set_postfix_str(
            f"{agg['audio_s'] / max(el, 1e-9):.0f}xRT "
            f"done={agg['done']:,} skip={agg['skipped']:,} fail={agg['failed']:,}",
            refresh=False,
        )
    pbar.close()
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0

    done = sum(r["done"] for r in res)
    skipped = sum(r["skipped"] for r in res)
    failed = sum(r["failed"] for r in res)
    audio_s = sum(r["audio_s"] for r in res)
    print(f"\naligned {done:,} | skipped {skipped:,} | failed {failed:,}")
    print(
        f"wall {wall / 3600:.2f} h | {audio_s / 3600:,.1f} audio-h | "
        f"{audio_s / wall:.1f}xRT aggregate"
    )
    return 1 if failed and not done else 0


if __name__ == "__main__":
    sys.exit(main())
