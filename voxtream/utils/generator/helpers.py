import argparse
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Generator

import nltk
import numpy as np
import torch

DTYPE_MAP = {
    "cpu": torch.float32,
    "cuda": torch.bfloat16,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def text_generator(
    text: str, window_size: int = 1, delay_ms: int = None
) -> Generator[str | None, None, None]:
    words = text.split()
    assert window_size >= 1, "window_size must be at least 1"

    if delay_ms is None or delay_ms <= 0:
        for i in range(0, len(words), window_size):
            window_words = words[i : i + window_size]
            yield " ".join(window_words)
        return

    delay_ns = delay_ms * 1_000_000
    next_ready_ns = time.monotonic_ns()
    i = 0
    total = len(words)
    while i < total:
        now_ns = time.monotonic_ns()
        if i > 0 and now_ns < next_ready_ns:
            yield None
            continue
        window_words = words[i : i + window_size]
        i += window_size
        yield " ".join(window_words)
        next_ready_ns = now_ns + delay_ns


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


def autocast_ctx(device: str, dtype: torch.dtype):
    if device == "cuda":
        return torch.autocast(device_type=device, dtype=dtype)
    return nullcontext()


def configure_cpu_threads(
    torch_threads: int | None = None,
    inter_op_threads: int | None = None,
) -> tuple[int, int]:
    cpu_threads = torch_threads or (os.cpu_count() or 1)
    if cpu_threads <= 0:
        raise ValueError("torch_threads must be a positive integer")

    if inter_op_threads is None:
        inter_op_threads = cpu_threads
    if inter_op_threads <= 0:
        raise ValueError("inter_op_threads must be a positive integer")

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(inter_op_threads)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)

    return cpu_threads, inter_op_threads


def interpolate_speaking_rate_params(
    speaking_rate_config: Dict[str, Dict[str, list | float]],
    speaking_rate: float,
    logger=None,
):
    parsed = [
        (float(rate_key), config) for rate_key, config in speaking_rate_config.items()
    ]
    parsed.sort(key=lambda item: item[0])

    min_rate = parsed[0][0]
    max_rate = parsed[-1][0]
    if speaking_rate < min_rate:
        if logger:
            logger.warning(
                f"Speaking rate {speaking_rate} is below the minimum configured rate {min_rate}. "
                f"Using parameters for the minimum rate."
            )
        base = parsed[0][1]
        return (
            list(base["duration_state"]),
            float(base["weight"]),
            float(base["cfg_gamma"]),
        )
    if speaking_rate > max_rate:
        if logger:
            logger.warning(
                f"Speaking rate {speaking_rate} is above the maximum configured rate {max_rate}. "
                f"Using parameters for the maximum rate."
            )
        base = parsed[-1][1]
        return (
            list(base["duration_state"]),
            float(base["weight"]),
            float(base["cfg_gamma"]),
        )

    lower_rate, lower_state = parsed[0]
    upper_rate, upper_state = parsed[-1]
    for idx in range(1, len(parsed)):
        candidate_rate, candidate_state = parsed[idx]
        if candidate_rate > speaking_rate:
            lower_rate, lower_state = parsed[idx - 1]
            upper_rate, upper_state = candidate_rate, candidate_state
            break

    if lower_rate == upper_rate:
        base = lower_state
        return (
            list(base["duration_state"]),
            float(base["weight"]),
            float(base["cfg_gamma"]),
        )

    def _normalize_duration_state(values):
        total = float(sum(values))
        if total <= 0:
            raise ValueError("duration_state sum must be positive")
        return [value / total for value in values]

    alpha = (speaking_rate - lower_rate) / (upper_rate - lower_rate)

    lower_duration_state = _normalize_duration_state(lower_state["duration_state"])
    upper_duration_state = _normalize_duration_state(upper_state["duration_state"])
    lower_total = float(sum(lower_state["duration_state"]))
    upper_total = float(sum(upper_state["duration_state"]))
    target_total = lower_total + alpha * (upper_total - lower_total)
    interpolated_duration_state = [
        max(
            1,
            int(round(target_total * (low + alpha * (high - low)))),
        )
        for low, high in zip(lower_duration_state, upper_duration_state)
    ]

    interpolated_weight = lower_state["weight"] + alpha * (
        upper_state["weight"] - lower_state["weight"]
    )
    interpolated_cfg_gamma = lower_state["cfg_gamma"] + alpha * (
        upper_state["cfg_gamma"] - lower_state["cfg_gamma"]
    )

    return (
        interpolated_duration_state,
        float(interpolated_weight),
        float(interpolated_cfg_gamma),
    )
