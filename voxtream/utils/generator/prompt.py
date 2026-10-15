from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from moshi.models import MimiModel
from silero_vad import get_speech_timestamps
from torchaudio.transforms import Resample

from voxtream.config import SpeechGeneratorConfig
from voxtream.utils.generator.helpers import autocast_ctx

_LOUDNESS_METERS: Dict[int, pyln.Meter] = {}


def normalize_prompt_loudness(
    waveform: torch.Tensor,
    sample_rate: int,
    target_lufs: float,
    logger=None,
) -> torch.Tensor:
    """Normalize the acoustic prompt to a fixed integrated loudness (ITU-R BS.1770)."""
    if sample_rate not in _LOUDNESS_METERS:
        _LOUDNESS_METERS[sample_rate] = pyln.Meter(sample_rate)
    meter = _LOUDNESS_METERS[sample_rate]

    if waveform.shape[1] < int(meter.block_size * sample_rate):
        if logger:
            logger.warning(
                f"Prompt audio is shorter than the {meter.block_size:.1f} s loudness "
                f"block size; skipping loudness normalization."
            )
        return waveform

    signal = waveform.squeeze(0).to(dtype=torch.float32).cpu().numpy()
    loudness = meter.integrated_loudness(signal)
    if not np.isfinite(loudness):
        if logger:
            logger.warning(
                "Prompt audio loudness is not measurable (silent audio); "
                "skipping loudness normalization."
            )
        return waveform

    signal = pyln.normalize.loudness(signal, loudness, target_lufs)

    peak = float(np.max(np.abs(signal)))
    if peak > 1.0:
        if logger:
            logger.debug(
                f"Loudness-normalized prompt peaks at {peak:.2f}; "
                f"rescaling to avoid clipping."
            )
        signal /= peak

    return torch.from_numpy(signal).unsqueeze(0).to(dtype=waveform.dtype)


def encode_audio_prompt(
    waveform: torch.Tensor,
    orig_sr: int,
    mimi_prompt: MimiModel,
    mimi_sr: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if orig_sr != mimi_sr:
        resampler = Resample(orig_sr, mimi_sr)
        waveform = resampler(waveform)

    with (
        torch.no_grad(),
        autocast_ctx(device=device, dtype=dtype),
    ):
        audio_tokens = mimi_prompt.encode(waveform.unsqueeze(0).to(device, dtype=dtype))

    return audio_tokens


def delay_audio_tokens(
    audio_tokens: torch.Tensor,
    audio_pad_token: int,
    audio_delay_frames: int,
    device: str,
) -> torch.Tensor:
    delayed_tokens = torch.full(
        (
            1,
            audio_tokens.shape[1],
            audio_tokens.shape[2] + 1,
        ),  # [bs, cb, time]
        fill_value=audio_pad_token,
        dtype=torch.int64,
        device=device,
    )

    delayed_tokens[:, 0, 1:] = audio_tokens[:, 0]
    delayed_tokens[:, 1:, audio_delay_frames + 1 :] = audio_tokens[
        :, 1:, : audio_tokens.shape[2] - audio_delay_frames
    ]

    return delayed_tokens


def extract_speaker_template(
    waveform: torch.Tensor,
    orig_sr: int,
    spk_enc: torch.nn.Module,
    spk_enc_sr: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if orig_sr != spk_enc_sr:
        resampler = Resample(orig_sr, spk_enc_sr)
        waveform = resampler(waveform)

    with (
        torch.no_grad(),
        autocast_ctx(device=device, dtype=dtype),
    ):
        spk_embedding = spk_enc(waveform.to(device, dtype=dtype))

    spk_embedding /= spk_embedding.norm(keepdim=True)
    return spk_embedding


def extract_speech_frames(
    waveform_16k: torch.Tensor,
    waveform: torch.Tensor,
    vad: torch.nn.Module,
    orig_sr: int,
    vad_sr: int,
    min_speech_seg_sec: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    speech_timestamps = get_speech_timestamps(
        waveform_16k,
        vad,
        return_seconds=True,
    )

    last_end = 0
    valid_segments, valid_segments_16k = [], []
    for i, segment in enumerate(speech_timestamps):
        start = max(last_end, segment["start"] - min_speech_seg_sec)
        end = segment["end"]
        if i == len(speech_timestamps) - 1:
            end += min_speech_seg_sec
        last_end = end

        valid_segments.append(waveform[:, int(start * orig_sr) : int(end * orig_sr)])
        valid_segments_16k.append(
            waveform_16k[:, int(start * vad_sr) : int(end * vad_sr)]
        )

    waveform_vad = torch.cat(valid_segments, dim=1)
    waveform_vad_16k = torch.cat(valid_segments_16k, dim=1)
    return waveform_vad, waveform_vad_16k


def prepare_prompt(
    prompt_audio_path: Path,
    config: SpeechGeneratorConfig,
    logger,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    mimi_prompt: MimiModel,
    spk_enc: torch.nn.Module,
    sidon_se,
    vad: torch.nn.Module,
    enhance_prompt: bool = None,
    apply_vad: bool = None,
    normalize_loudness: bool = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_path = prompt_audio_path.parent / f"{prompt_audio_path.stem}.prompt.npy"
    if config.cache_prompt and prompt_path.exists():
        prompt_data = np.load(prompt_path, allow_pickle=True).item()
        audio_tokens = torch.from_numpy(prompt_data["audio_tokens"]).to(device)
        spk_embedding = torch.from_numpy(prompt_data["spk_embedding"]).to(
            device, dtype=dtype
        )
    else:
        waveform, orig_sr = torchaudio.load(str(prompt_audio_path))
        if waveform.shape[0] != 1:
            logger.warning(
                f"Prompt audio has {waveform.shape[0]} channels; converting to mono by averaging."
            )
            waveform = waveform.mean(dim=0, keepdim=True)

        if normalize_loudness:
            waveform = normalize_prompt_loudness(
                waveform,
                sample_rate=orig_sr,
                target_lufs=config.prompt_loudness_lufs,
                logger=logger,
            )

        waveform_16k = None
        if apply_vad:
            waveform_16k = torchaudio.functional.resample(
                waveform, orig_sr, config.spk_enc_sr
            )
            waveform, waveform_16k = extract_speech_frames(
                waveform_16k,
                waveform,
                vad,
                orig_sr=orig_sr,
                vad_sr=config.spk_enc_sr,
                min_speech_seg_sec=config.min_speech_seg_sec,
            )

        assert (
            waveform.shape[1] >= orig_sr * config.min_prompt_sec
        ), f"Acoustic prompt is too short ({waveform.shape[1] / orig_sr:.2f} seconds); should be at least {config.min_prompt_sec} seconds."
        if waveform.shape[1] > orig_sr * config.max_prompt_sec:
            logger.warning(
                f"Prompt audio is longer than {config.max_prompt_sec} seconds; trimming."
            )
            waveform = waveform[:, : orig_sr * config.max_prompt_sec]

        waveform_mimi = waveform
        sample_rate = orig_sr

        if enhance_prompt:
            if waveform_16k is None:
                waveform_16k = torchaudio.functional.resample(
                    waveform, orig_sr, config.spk_enc_sr
                )
            waveform_mimi, sample_rate = sidon_se(waveform_16k, config.spk_enc_sr)

        audio_tokens = encode_audio_prompt(
            waveform_mimi,
            sample_rate,
            mimi_prompt,
            config.mimi_sr,
            device,
            dtype,
        )

        if waveform_16k is None:
            waveform_16k = waveform
            spk_sr = orig_sr
        else:
            spk_sr = config.spk_enc_sr
        spk_embedding = extract_speaker_template(
            waveform_16k,
            spk_sr,
            spk_enc,
            config.spk_enc_sr,
            device,
            dtype,
        )

        if config.cache_prompt:
            np.save(
                prompt_path,
                {
                    "audio_tokens": audio_tokens.cpu().numpy(),
                    "spk_embedding": spk_embedding.to(
                        device="cpu", dtype=torch.float16
                    ).numpy(),
                },
            )

    audio_tokens = delay_audio_tokens(
        audio_tokens=audio_tokens,
        audio_pad_token=config.audio_pad_token,
        audio_delay_frames=config.audio_delay_frames,
        device=device,
    )

    prompt_phone_tokens = np.full(
        (audio_tokens.shape[2] - 1,),
        fill_value=config.unk_token,
        dtype=np.int64,
    )
    prompt_phone_tokens[-1] = config.eop_token

    phone_emb_indices = np.repeat(
        np.expand_dims(np.arange(audio_tokens.shape[2]), axis=1),
        config.num_phones_per_frame,
        axis=1,
    )
    punct_del_indices = np.array([])

    prompt_phone_tokens = torch.from_numpy(
        np.expand_dims(
            np.concatenate([[config.bos_token], prompt_phone_tokens]), axis=0
        ),
    ).to(dtype=torch.int64, device=device)

    phone_emb_indices = torch.from_numpy(np.expand_dims(phone_emb_indices, axis=0)).to(
        dtype=torch.int64, device=device
    )

    punct_del_indices = torch.from_numpy(np.expand_dims(punct_del_indices, axis=0)).to(
        dtype=torch.int64, device=device
    )

    if config.cfg_gamma is not None:
        audio_tokens = audio_tokens.repeat(batch_size, 1, 1)
        mask = audio_tokens[1] != config.audio_pad_token
        audio_tokens[1, mask] = config.mimi_vocab_size

        spk_embedding = torch.cat(
            [spk_embedding, torch.zeros_like(spk_embedding)], dim=0
        )

        prompt_phone_tokens = prompt_phone_tokens.repeat(batch_size, 1)
        phone_emb_indices = phone_emb_indices.repeat(batch_size, 1, 1)
        punct_del_indices = punct_del_indices.repeat(batch_size, 1)

    return (
        audio_tokens,
        spk_embedding,
        prompt_phone_tokens,
        phone_emb_indices,
        punct_del_indices,
    )
