"""
Sidon Speech Enhancement — standalone module.
Original code: https://github.com/sarulab-speech/Sidon
Paper: https://arxiv.org/pdf/2509.17052

Uses a pure-PyTorch re-implementation of the SeamlessM4T / w2v-bert-2.0
feature extractor (originally from w2vb2.py) instead of the slow
HuggingFace numpy-based pipeline.
Thanks @vanIvan for implementing.
"""

import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from transformers.audio_utils import mel_filter_bank, window_function

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-extraction building blocks (ported from w2vb2.py)
# ---------------------------------------------------------------------------


class DC_Filter(nn.Module):
    """Remove DC component from framed signal via average pooling."""

    def __init__(self, winlen: int = 400):
        super().__init__()
        self.mean_filter = nn.AvgPool1d(winlen, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, WinLen, NumFrames)
        signal_T = x.transpose(2, 1)
        means = self.mean_filter(signal_T)
        return x - means.transpose(2, 1)


class Wav2Frames(nn.Module):
    """Slice a waveform into overlapping frames via strided convolution."""

    def __init__(self, winlen: int = 400, winstep: int = 160):
        super().__init__()
        self.winstep = winstep
        self.window_filt_weights = nn.Parameter(
            torch.eye(winlen, dtype=torch.float32).unsqueeze(1),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)  →  (B, WinLen, NumFrames)
        return F.conv1d(x, self.window_filt_weights, stride=self.winstep)


class PreEmphasis(nn.Module):
    """First-order pre-emphasis filter: y[k] = x[k] - k·x[k-1]."""

    def __init__(self, k: float = 0.97, requires_grad: bool = False):
        super().__init__()
        pr_filt = torch.tensor([[-k], [1.0]], dtype=torch.float32)
        self.pr_filt = nn.Parameter(
            pr_filt.unsqueeze(0).unsqueeze(0), requires_grad=requires_grad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, NumFrames, WinLen)
        windows = torch.cat([x[:, 0:1, :], x], 1)
        windows = windows.unsqueeze(1)
        return F.conv2d(windows, self.pr_filt, stride=1, padding=0).squeeze(1)


class SpectrogramTorch(nn.Module):
    """Pure-PyTorch mel-spectrogram matching the HuggingFace numpy pipeline."""

    def __init__(
        self,
        window: torch.Tensor,
        frame_length: int,
        hop_length: int,
        fft_length: Optional[int] = None,
        power: Optional[float] = 1.0,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        dither: float = 0.0,
        preemphasis: Optional[float] = None,
        mel_filters: Optional[torch.Tensor] = None,
        mel_floor: float = 1e-10,
        log_mel: Optional[str] = None,
        reference: float = 1.0,
        min_value: float = 1e-10,
        db_range: Optional[float] = None,
        remove_dc_offset: bool = False,
    ):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window_length = len(window)
        self.fft_length = fft_length if fft_length is not None else frame_length

        if self.frame_length > self.fft_length:
            raise ValueError(
                f"frame_length ({self.frame_length}) may not be larger "
                f"than fft_length ({self.fft_length})"
            )
        if self.window_length != self.frame_length:
            raise ValueError(
                f"Length of the window ({self.window_length}) must equal "
                f"frame_length ({self.frame_length})"
            )

        self.power = power
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.dither = dither
        self.mel_floor = mel_floor
        self.log_mel = log_mel
        self.reference = reference
        self.min_value = min_value
        self.db_range = db_range
        self.remove_dc_offset = remove_dc_offset

        self.register_buffer("window", window.float())

        if mel_filters is not None:
            self.register_buffer("mel_filters", mel_filters.float())
        else:
            self.mel_filters = None

        self.framer = Wav2Frames(winlen=self.frame_length, winstep=self.hop_length)
        if self.remove_dc_offset:
            self.dc_remover = DC_Filter(winlen=self.frame_length)
        if preemphasis is not None:
            self.pre_emphasizer = PreEmphasis(k=preemphasis)
        else:
            self.pre_emphasizer = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) or (T,).
        Returns:
            spectrogram: (B, MelBins, NumFrames)
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.window.dtype)

        # 1. Center padding
        if self.center:
            pad = int(self.frame_length // 2)
            if self.pad_mode == "reflect":
                waveform = F.pad(
                    waveform.unsqueeze(1), (pad, pad), mode=self.pad_mode
                ).squeeze(1)
            else:
                waveform = F.pad(waveform, (pad, pad), mode=self.pad_mode)

        # 2. Framing
        frames = self.framer(waveform.unsqueeze(1))  # (B, WinLen, NumFrames)

        # 3. Dither
        if self.dither != 0.0:
            frames = frames + self.dither * torch.randn_like(frames)

        # 4. DC removal
        if self.remove_dc_offset:
            frames = self.dc_remover(frames)

        # 5. Pre-emphasis
        if self.pre_emphasizer is not None:
            frames = self.pre_emphasizer(frames)

        # 6. Windowing
        frames = frames * self.window.view(1, -1, 1)

        # 7. Transpose for FFT → (B, NumFrames, WinLen)
        frames = frames.transpose(1, 2)
        if self.fft_length > self.frame_length:
            frames = F.pad(frames, (0, self.fft_length - self.frame_length))

        # 8. FFT
        if self.onesided:
            spectrogram = torch.fft.rfft(frames, n=self.fft_length, dim=-1)
        else:
            spectrogram = torch.fft.fft(frames, n=self.fft_length, dim=-1)

        # 9. Power / magnitude
        if self.power is not None:
            spectrogram = torch.abs(spectrogram).pow(self.power)

        # Back to (B, FreqBins, NumFrames)
        spectrogram = spectrogram.transpose(1, 2)

        # 10. Mel filterbank
        if self.mel_filters is not None:
            spectrogram = torch.matmul(self.mel_filters.T, spectrogram)
            spectrogram = torch.maximum(
                torch.tensor(self.mel_floor, device=spectrogram.device),
                spectrogram,
            )

        # 11. Log
        if self.power is not None and self.log_mel is not None:
            if self.log_mel == "log":
                spectrogram = torch.log(spectrogram)
            elif self.log_mel == "log10":
                spectrogram = torch.log10(spectrogram)
            else:
                raise ValueError(f"Unknown log_mel option: {self.log_mel}")

        return spectrogram


# ---------------------------------------------------------------------------
# Pure-PyTorch SeamlessM4T feature extractor (replaces HF numpy version)
# ---------------------------------------------------------------------------


class SeamlessM4TFeatureExtractor(nn.Module):
    """
    Drop-in replacement for
    ``transformers.SeamlessM4TFeatureExtractor`` that runs entirely on
    GPU tensors and avoids the slow numpy round-trip.
    """

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        num_mel_bins: int = 80,
        padding_value: float = 0.0,
        stride: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        self.padding_value = padding_value
        self.stride = stride
        self.return_attention_mask = True

        mel_filters_np = mel_filter_bank(
            num_frequency_bins=257,
            num_mel_filters=num_mel_bins,
            min_frequency=20,
            max_frequency=sampling_rate // 2,
            sampling_rate=sampling_rate,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )
        window_np = window_function(400, "povey", periodic=False)

        self.spectrogram = SpectrogramTorch(
            window=torch.from_numpy(window_np),
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            center=False,
            preemphasis=0.97,
            mel_filters=torch.from_numpy(mel_filters_np),
            log_mel="log",
            mel_floor=1.192092955078125e-07,
            remove_dc_offset=True,
        )

    def _compute_output_length(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(input_lengths - 400, 160, rounding_mode="floor") + 1

    def forward(
        self,
        raw_speech: Union[torch.Tensor, np.ndarray],
        padding: bool = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = 2,
        return_attention_mask: Optional[bool] = None,
        do_normalize_per_mel_bins: bool = True,
    ) -> dict:
        """
        Args:
            raw_speech: (B, T) or (T,) tensor/ndarray of float samples in [-1, 1].
        Returns:
            dict with ``"input_features"`` (B, T', 160) and optionally
            ``"attention_mask"`` (B, T').
        """
        dev = self.spectrogram.window.device

        if not torch.is_tensor(raw_speech):
            raw_speech = torch.tensor(raw_speech, device=dev)
        if raw_speech.ndim == 1:
            raw_speech = raw_speech.unsqueeze(0)

        raw_speech = raw_speech.to(dev).float()
        input_lengths = torch.tensor(
            [raw_speech.shape[1]] * raw_speech.shape[0], device=dev
        )

        # Kaldi-style 16-bit scaling
        raw_speech = raw_speech * (2**15)

        # Mel spectrogram → (B, MelBins, NumFrames)
        features = self.spectrogram(raw_speech)
        # → (B, T, C)
        features = features.transpose(1, 2).contiguous()

        valid_feat_lengths = self._compute_output_length(input_lengths).clamp(min=1)
        B, T, C = features.shape
        mask = torch.arange(T, device=dev).unsqueeze(0) < valid_feat_lengths.unsqueeze(
            1
        )

        # Per-mel-bin zero-mean / unit-variance normalization
        if do_normalize_per_mel_bins:
            mask_float = mask.float().unsqueeze(-1)
            features = features * mask_float

            seq_len = mask_float.sum(dim=1)
            mean = features.sum(dim=1) / (seq_len + 1e-7)

            diffs_sq = ((features - mean.unsqueeze(1)) ** 2) * mask_float
            var = diffs_sq.sum(dim=1) / (seq_len + 1e-7)
            std = torch.sqrt(var + 1e-7)

            features = (features - mean.unsqueeze(1)) / std.unsqueeze(1)
            features = features * mask_float

        # Pad to multiple
        target_len = T
        if pad_to_multiple_of and pad_to_multiple_of > 0:
            if target_len % pad_to_multiple_of != 0:
                target_len = (
                    (target_len // pad_to_multiple_of) + 1
                ) * pad_to_multiple_of
        if max_length is not None and max_length > target_len:
            target_len = max_length
        if target_len > T:
            diff = target_len - T
            features = F.pad(features, (0, 0, 0, diff), value=self.padding_value)
            mask = F.pad(mask, (0, diff), value=False)
            T = target_len

        # Stride / stacking (default stride=2 → 160-dim features)
        if self.stride > 1:
            remainder = T % self.stride
            if remainder != 0:
                features = features[:, : T - remainder, :]
                mask = mask[:, : T - remainder]
                T = T - remainder
            features = features.view(B, T // self.stride, self.stride * C)
            mask = mask[:, 1 :: self.stride]

        return {
            "input_features": features,
            "attention_mask": (
                mask.long()
                if return_attention_mask or self.return_attention_mask
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Sidon speech-enhancement wrapper
# ---------------------------------------------------------------------------


class SidonSE:
    """
    Sidon v0.1 speech-enhancement inference wrapper.

    Usage::

        se = SidonSE("cuda")
        enhanced = se(samples_16k, sr=16000)   # torch.Tensor, 48 kHz float32
        # or from file:
        enhanced = se.enhance_file("noisy.wav")

    The model internally resamples everything to 16 kHz for feature
    extraction and produces output at **48 kHz**.

    **Chunked processing** — The w2v-bert-2.0 encoder has a practical
    receptive-field / memory limit. The original Sidon demo processes
    audio in 96-second windows (16 000 x 96 = 1 536 000 samples) and
    carries forward the last encoder frame as a "feature cache" so that
    the decoder sees continuous context across chunk boundaries.  The 960
    samples trimmed from each decoded chunk (``[:-960]``) compensate for
    the decoder's look-ahead / padding artefact at the tail of each
    block.
    """

    OUTPUT_SR = 48_000
    INTERNAL_SR = 16_000
    CHUNK_SECONDS = 96
    CHUNK_SAMPLES = INTERNAL_SR * CHUNK_SECONDS  # 1 536 000
    EDGE_PAD = 160  # symmetric pad around each chunk (encoder conv edge)
    TAIL_TRIM = 960  # decoder tail artefact trimmed per chunk
    TAIL_PAD_16K = 24_000  # silence appended so last chunk is fully covered
    HP_CUTOFF = 50.0  # high-pass filter cutoff (Hz)

    def __init__(
        self,
        device: Union[str, torch.device, None] = None,
        repo_id: str = "sarulab-speech/sidon-v0.1",
        reload_model: bool = False,
    ):
        self.repo_id = repo_id
        self.reload_model = reload_model

        if not reload_model:
            self.load_model(device=device)

    def load_model(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.debug("Building Sidon SE (%s) on device: %s", self.repo_id, self.device)

        # --- Sidon encoder / decoder (TorchScript) ---
        suffix = "cuda" if self.device.type == "cuda" else "cpu"
        fe_path = hf_hub_download(
            self.repo_id, filename=f"feature_extractor_{suffix}.pt"
        )
        dec_path = hf_hub_download(self.repo_id, filename=f"decoder_{suffix}.pt")
        self.fe = torch.jit.load(fe_path, map_location=self.device).eval()
        self.dec = torch.jit.load(dec_path, map_location=self.device).eval()

        # --- Pure-torch feature extractor (replaces HF numpy pipeline) ---
        self.preproc = SeamlessM4TFeatureExtractor().to(self.device).eval()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def __call__(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        sr: int = 16_000,
    ) -> np.ndarray:
        """
        Enhance a waveform.

        Args:
            samples: 1-D float waveform (any sample-rate).
            sr:      Sample-rate of *samples*.

        Returns:
            Enhanced waveform at 48 kHz, float32 numpy array (shape ``[T]``).
            Sample-rate (48 kHz)
        """
        # Reload model at each call on HF ZeroGPU
        if self.reload_model:
            self.load_model()

        if isinstance(samples, np.ndarray):
            wav = torch.from_numpy(samples).float()
        else:
            wav = samples.float()

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # (1, T)
        elif wav.ndim > 2:
            raise ValueError(f"Expected 1-D or 2-D input, got ndim={wav.ndim}")

        # Mono
        wav = wav.mean(dim=0, keepdim=True)

        # Normalize to ≈ [-0.9, 0.9]
        wav = 0.9 * (wav / wav.abs().max().clamp_min(1e-8))

        # High-pass & resample to 16 kHz for the encoder
        wav = torchaudio.functional.highpass_biquad(wav, sr, self.HP_CUTOFF)
        wav16 = torchaudio.functional.resample(wav, sr, self.INTERNAL_SR)  # (1, T16)

        # Target output length at 48 kHz
        target_n = int(self.OUTPUT_SR / sr * wav.shape[1])

        # Tail-pad so the last chunk has enough context
        wav16 = F.pad(wav16, (0, self.TAIL_PAD_16K))

        # --- Chunked processing ---
        restored_chunks: list[torch.Tensor] = []
        feature_cache: Optional[torch.Tensor] = None

        for chunk in wav16.view(-1).split(self.CHUNK_SAMPLES):
            # Symmetric edge pad (compensates for encoder conv edges)
            chunk = F.pad(chunk, (self.EDGE_PAD, self.EDGE_PAD))

            # Feature extraction (pure PyTorch, on-device)
            inputs = self.preproc(chunk)
            feats = self.fe(inputs["input_features"].to(self.device))
            feats = feats["last_hidden_state"]  # (1, T_enc, D)

            # Prepend cached frame for cross-chunk continuity
            if feature_cache is not None:
                feats = torch.cat([feature_cache, feats], dim=1)

            # Decode: expects (B, D, T)
            y = self.dec(feats.transpose(1, 2)).view(-1)
            # Trim decoder tail artefact
            y = y[: -self.TAIL_TRIM] if self.TAIL_TRIM else y
            restored_chunks.append(y)

            # Cache last encoder frame for next iteration
            feature_cache = feats[:, -1:]

        restored = torch.cat(restored_chunks, dim=0)
        out = restored[:target_n].detach().cpu().float()
        out = torch.clamp(out.unsqueeze(0), -1.0, 1.0)

        return out, self.OUTPUT_SR

    @torch.inference_mode()
    def enhance_file(self, audio_path: str) -> np.ndarray:
        """
        Convenience: load a file and enhance it.

        Returns:
            Enhanced waveform at 48 kHz, float32 numpy array.
        """
        wav, sr = torchaudio.load(audio_path)
        return self(wav, sr=sr)


# ---------------------------------------------------------------------------
# Backwards-compatible functional API
# ---------------------------------------------------------------------------


def enhance_audio(audio_path: str, device: Optional[str] = None) -> np.ndarray:
    """
    Drop-in replacement for the original ``enhance_audio`` function.

    Returns enhanced waveform at 48 kHz as float32 numpy (shape ``[T]``).
    """
    se = SidonSE(device=device)
    return se.enhance_file(audio_path)
