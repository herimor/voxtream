"""
Clap-IPA — standalone phoneme aligner (GPU-resident fast path).

Same public API and same outputs as the original implementation, but the
per-utterance CPU work has been moved onto the GPU / vectorised:

  * log-mel front-end computed with `torch.stft` on the GPU instead of
    `WhisperFeatureExtractor` (which spends ~99% of its time marshalling
    numpy arrays, not computing the STFT),
  * one batched phone-encoder forward instead of one bs=1 forward per item,
  * phone pooling via scatter_add instead of a [K, P_sub] mask matmul,
  * `librosa.sequence.dtw` replaced by a batched Viterbi DP on the GPU that
    reproduces its output exactly for `step_sizes_sigma=[[1, 1], [0, 1]]`,
  * the identity sliding-window matmul short-circuited when win_len == shift == 1,
  * phoneme -> subword-id memoisation (the IPA inventory is a few hundred symbols).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample
from transformers import (
    AutoProcessor,
    BertModel,
    BertPreTrainedModel,
    DebertaV2Tokenizer,
    WhisperConfig,
    WhisperPreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

from voxtream.utils.aligner.base import BaseAligner


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = src_len if tgt_len is None else tgt_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inv = 1.0 - expanded_mask
    return inv.masked_fill(inv.to(torch.bool), torch.finfo(dtype).min)


class WhisperEncoder(WhisperPreTrainedModel):
    """Minimal Whisper encoder: conv -> pos + Transformer -> LN"""

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        x = F.gelu(self.conv1(input_features))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        pos = self.embed_positions.weight[: x.size(1)]
        hidden_states = F.dropout(x + pos, p=self.dropout, training=self.training)

        if attention_mask is not None:
            dtype = hidden_states.dtype
            attention_mask = attention_mask[:, ::2].to(dtype=dtype)
            attn = _expand_mask(attention_mask, dtype, tgt_len=hidden_states.size(1))
        else:
            attn = None

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attn,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=False,
            )[0]

        hidden_states = self.layer_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class PhoneEncoder(BertPreTrainedModel):
    """BERT -> Linear projector (no pooling; we need per-token features)."""

    _keys_to_ignore_on_load_unexpected = [r"^t_prime$", r"^b$"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.projector = nn.Linear(config.hidden_size, config.proj_size)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
            return_dict=True,
        )
        return BaseModelOutput(last_hidden_state=self.projector(out.last_hidden_state))


class SpeechEncoder(WhisperPreTrainedModel):
    """Whisper encoder + projector (no pooling; we need per-frame features)."""

    def __init__(self, config):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)
        self.projector = nn.Linear(config.hidden_size, config.proj_size)
        self.post_init()

    def forward(self, input_features, attention_mask=None, **kwargs):
        enc = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return BaseModelOutput(last_hidden_state=self.projector(enc.last_hidden_state))


# ---------------------------
# Batched forced alignment (replaces librosa.sequence.dtw)
# ---------------------------


@torch.no_grad()
def batched_forced_align(
    sim: torch.Tensor, lens_t: np.ndarray, lens_p: np.ndarray
) -> List[np.ndarray]:
    """
    Batched Viterbi forced alignment on the GPU.

    Reproduces ``librosa.sequence.dtw(C=-sim.T, step_sizes_sigma=[[1, 1], [0, 1]])``
    followed by "last time index of every phone row on the warping path", which is
    what the original implementation extracts from the warping path.

    With that step pattern the DP is a linear HMM with self-loops::

        A[i, j] = C[i, j] + min(A[i - 1, j - 1], A[i, j - 1])

    so the whole batch advances one time step at a time, vectorised over phones.
    Ties go to the diagonal step, matching librosa's strict ``<`` update.

    Parameters
    ----------
    sim : torch.Tensor
        [B, T, P] cosine similarities (padded).
    lens_t : np.ndarray
        [B] real number of speech frames per item.
    lens_p : np.ndarray
        [B] real number of phones per item.

    Returns
    -------
    List[np.ndarray]
        Per item, the last frame index assigned to every phone (length lens_p[b]).
    """
    B, T, P = sim.shape
    # librosa accumulates in float64; keep the same dtype so that near-ties in the
    # accumulated cost break the same way and the warping path is bit-identical
    cost = (-sim).transpose(1, 2).double().contiguous()  # [B, P, T], rows = phones

    acc = cost.new_full((B, P), float("inf"))
    acc[:, 0] = cost[:, 0, 0]

    steps = torch.zeros((B, P, T), dtype=torch.uint8, device=sim.device)
    pad_col = cost.new_full((B, 1), float("inf"))

    for j in range(1, T):
        diag = torch.cat([pad_col, acc[:, :-1]], dim=1)
        c = cost[:, :, j]
        # librosa compares the *sums* (diagonal first, strict `<`), so ties -> diagonal
        steps[:, :, j] = (acc + c < diag + c).to(torch.uint8)
        acc = torch.minimum(diag + c, acc + c)

    # ----- backtracking, vectorised over the batch
    S = steps.cpu().numpy()
    row = lens_p.astype(np.int64) - 1
    col = lens_t.astype(np.int64) - 1
    ends = np.full((B, P), -1, dtype=np.int64)
    idx_all = np.arange(B)
    ends[idx_all, row] = col

    active = row > 0
    while active.any():
        idx = np.nonzero(active)[0]
        r, c = row[idx], col[idx]
        took_diag = S[idx, r, c] == 0
        di = idx[took_diag]
        ends[di, row[di] - 1] = col[di] - 1
        row[di] -= 1
        col[idx] -= 1
        active = (row > 0) & (col >= 0)

    return [ends[b, : lens_p[b]] for b in range(B)]


class ClapIPA(nn.Module, BaseAligner):
    def __init__(
        self,
        speech_encoder: str = "anyspeech/ipa-align-base-speech",
        phone_encoder: str = "anyspeech/ipa-align-base-phone",
        tokenizer: str = "charsiu/IPATokenizer",
        processor: str = "openai/whisper-base",
        sample_rate: int = 16000,
        win_len: int = 1,
        frame_shift: int = 1,
        hop_len: float = 0.02,
        pad_token: str = "[SIL]",
        revision: str = "refs/pr/1",
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # transformers>=4.50 refuses `pytorch_model.bin` on torch<2.6; `refs/pr/1` is the
        # Hub's safetensors conversion of the same checkpoint (bit-identical weights)
        self.speech_encoder = (
            SpeechEncoder.from_pretrained(speech_encoder, revision=revision)
            .eval()
            .half()
            .to(self.device)
        )
        self.phone_encoder = (
            PhoneEncoder.from_pretrained(phone_encoder, revision=revision)
            .eval()
            .half()
            .to(self.device)
        )

        self.tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer)
        self.processor = AutoProcessor.from_pretrained(processor)

        self.win_len = win_len
        self.frame_shift = frame_shift
        self.hop_seconds = hop_len * frame_shift
        self.sample_rate = sample_rate
        self.pad_token = pad_token

        # ----- log-mel front-end, GPU resident (mirrors WhisperFeatureExtractor)
        fe = getattr(self.processor, "feature_extractor", self.processor)
        self.n_fft = fe.n_fft
        self.hop_length = fe.hop_length
        self.register_buffer(
            "mel_filters",
            torch.from_numpy(fe.mel_filters).float().to(self.device),
            persistent=False,
        )
        self.register_buffer(
            "stft_window",
            torch.hann_window(self.n_fft, device=self.device),
            persistent=False,
        )

        self._tok_cache: Dict[str, List[int]] = {}
        self._resamplers: Dict[int, Resample] = {}

        # Reused pinned staging buffer. Allocating (and first-touching) a fresh
        # multi-MB numpy array per batch exposes the process to direct memory
        # compaction stalls on a busy, fragmented host - we measured isolated
        # 25-50 s stalls on a single np.zeros((64, 80000)). Keeping one buffer
        # around and growing it monotonically removes that exposure, and pinning
        # it makes the host-to-device copy ~3x faster.
        self._stage_t: torch.Tensor = torch.empty(0, dtype=torch.float32)

    def _staging(self, batch: int, samples: int) -> np.ndarray:
        """A reused [batch, samples] pinned host buffer, grown on demand."""
        need = batch * samples
        if self._stage_t.numel() < need:
            self._stage_t = torch.empty(
                int(need * 1.25), dtype=torch.float32
            ).pin_memory()
        return self._stage_t[:need].view(batch, samples).numpy()

    # ---------------------------
    # Front-end
    # ---------------------------

    @torch.no_grad()
    def log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        [B, T] float32 waveform on the GPU -> [B, n_mels, T // hop] log-mel.

        Numerically identical to `WhisperFeatureExtractor._torch_extract_fbank_features`.
        """
        stft = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            window=self.stft_window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_val = log_spec.amax(dim=(1, 2), keepdim=True)
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        return (log_spec + 4.0) / 4.0

    def _tokenize(self, phones: List[str]) -> Tuple[List[int], List[int]]:
        """Phone strings -> flat subword ids + per-phone subword counts (memoised)."""
        cache = self._tok_cache
        missing = [p for p in phones if p not in cache]
        if missing:
            enc = self.tokenizer(
                missing,
                add_special_tokens=False,
                return_attention_mask=False,
                return_length=False,
            )["input_ids"]
            for p, ids in zip(missing, enc):
                cache[p] = ids

        flat, lengths = [], []
        for p in phones:
            ids = cache[p]
            flat.extend(ids)
            lengths.append(len(ids))
        return flat, lengths

    # ---------------------------
    # Alignment
    # ---------------------------

    def _encode_phones(
        self, ids: torch.Tensor, mask: torch.Tensor, overlap: int = 64
    ) -> torch.Tensor:
        """
        Run the phone encoder over `ids`, in blocks if the sequence is too long.

        The phone encoder is a BERT with 514 position embeddings, so a phoneme sequence
        whose subword expansion exceeds that raises inside the embedding lookup. At the
        IPA tokenizer's ~1.4 subwords per phone that is ~367 phones in one call - a 30 s
        Emilia file already reaches 494. Rather than lose the utterance, long sequences
        are encoded in overlapping blocks and only each block's core is kept. Sequences
        that fit take the single-forward path and are unaffected.
        """
        limit = int(self.phone_encoder.config.max_position_embeddings) - 2
        if ids.shape[1] <= limit:
            return self.phone_encoder(
                input_ids=ids, attention_mask=mask
            ).last_hidden_state.float()

        P_sub = ids.shape[1]
        stride = limit - 2 * overlap
        out: Optional[torch.Tensor] = None
        for start in range(0, P_sub, stride):
            end = min(start + limit, P_sub)
            blk = self.phone_encoder(
                input_ids=ids[:, start:end], attention_mask=mask[:, start:end]
            ).last_hidden_state.float()
            if out is None:
                out = blk.new_zeros((ids.shape[0], P_sub, blk.shape[-1]))
            # keep the block's core, so every token is read with context on both sides
            lo = start if start == 0 else start + overlap
            hi = end if end == P_sub else end - overlap
            out[:, lo:hi] = blk[:, lo - start : hi - start]
            if end == P_sub:
                break
        return out

    def _gpu_resampler(self, orig_sr: int) -> Resample:
        """Cached `Resample` living on the GPU."""
        key = -orig_sr  # keep GPU and CPU resamplers apart in the same cache
        if key not in self._resamplers:
            self._resamplers[key] = Resample(orig_sr, self.sample_rate).to(self.device)
        return self._resamplers[key]

    @torch.no_grad()
    def forward(
        self,
        audios: List[np.ndarray] | torch.Tensor,
        phonemes: List[List[str]],
        audio_lengths: Optional[np.ndarray] = None,
        orig_sr: Optional[int] = None,
    ) -> List[List[float] | None]:
        """
        Returns per-utterance list of timestamps (seconds) aligned to each phoneme.

        `audios` may also be a pre-batched [B, T] float32 tensor together with
        `audio_lengths`, which lets the collate function do the padding in a
        dataloader worker.

        Pass `orig_sr` to hand over un-resampled audio: the resampling then runs on
        the GPU. Resampling is ~2/3 of the CPU cost of preparing a batch (decode +
        resample), so for a dataloader-bound job this roughly triples the number of
        audio-hours a machine can push through.
        """
        B = len(phonemes)
        in_sr = self.sample_rate if orig_sr is None else orig_sr

        # ----- batch the waveforms (contiguous copy; never go through numpy object lists)
        if isinstance(audios, torch.Tensor):
            wav = audios.to(self.device, torch.float32, non_blocking=True)
            lens = (
                audio_lengths
                if audio_lengths is not None
                else np.full(B, wav.shape[1], dtype=np.int64)
            )
        else:
            lens = np.array([len(a) for a in audios], dtype=np.int64)
            # round the batch up to a whole number of hops so that the number of
            # mel frames equals ceil(max_len / hop), like the HF feature extractor
            hop_in = self.hop_length * in_sr // self.sample_rate
            T_pad = int(math.ceil(int(lens.max()) / hop_in) * hop_in)
            buf = self._staging(B, T_pad)
            buf.fill(0.0)
            for i, a in enumerate(audios):
                buf[i, : len(a)] = a
            wav = (
                self._stage_t[: B * T_pad]
                .view(B, T_pad)
                .to(self.device, non_blocking=True)
            )

        # ----- resample on the GPU if the caller handed over native-rate audio
        if in_sr != self.sample_rate:
            wav = self._gpu_resampler(in_sr)(wav)
            lens = (lens.astype(np.float64) * self.sample_rate / in_sr).astype(np.int64)
            lens = np.minimum(lens, wav.shape[1])

        # ----- log-mel on the GPU
        feats = self.log_mel(wav).half()  # [B, n_mels, N]
        n_frames = feats.shape[-1]
        mel_lens = np.minimum(-(-lens // self.hop_length), n_frames)
        frame_idx = torch.arange(n_frames, device=self.device)
        attn = (
            frame_idx[None, :] < torch.as_tensor(mel_lens, device=self.device)[:, None]
        ).half()

        # ----- speech encoder
        speech = self.speech_encoder(
            input_features=feats, attention_mask=attn
        ).last_hidden_state
        speech = F.normalize(speech, dim=-1)

        enc_T = speech.shape[1]
        # encoder frames per item (conv2 stride 2), matches attention_mask[:, ::2].sum()
        lens_t = np.minimum((mel_lens + 1) // 2, enc_T).astype(np.int64)

        if self.win_len != 1 or self.frame_shift != 1:
            speech, lens_t = self._sliding_window(speech, lens_t)

        # ----- phone encoder, one batched forward
        flat_ids, sub_lens, owners = [], [], []
        for ph in phonemes:
            flat, lengths = self._tokenize(ph)
            flat_ids.append(flat)
            sub_lens.append(len(flat))
            owners.append(
                np.repeat(np.arange(len(ph), dtype=np.int64), np.array(lengths))
            )

        P_sub = max(sub_lens)
        K_max = max(len(ph) for ph in phonemes)
        ids = np.zeros((B, P_sub), dtype=np.int64)
        ids_mask = np.zeros((B, P_sub), dtype=np.int64)
        owner = np.full((B, P_sub), K_max, dtype=np.int64)  # padding -> bucket K_max
        for i, (flat, own) in enumerate(zip(flat_ids, owners)):
            ids[i, : len(flat)] = flat
            ids_mask[i, : len(flat)] = 1
            owner[i, : len(own)] = own

        ids_t = torch.from_numpy(ids).to(self.device)
        mask_t = torch.from_numpy(ids_mask).to(self.device)
        owner_t = torch.from_numpy(owner).to(self.device)

        phone_seq = self._encode_phones(ids_t, mask_t)  # [B, P_sub, D]

        # ----- mean-pool subwords back to phones
        D = phone_seq.shape[-1]
        pooled = phone_seq.new_zeros((B, K_max + 1, D))
        pooled.scatter_add_(1, owner_t.unsqueeze(-1).expand(-1, -1, D), phone_seq)
        counts = phone_seq.new_zeros((B, K_max + 1))
        counts.scatter_add_(1, owner_t, mask_t.to(phone_seq.dtype))
        pooled = pooled[:, :K_max] / counts[:, :K_max].clamp(min=1.0).unsqueeze(-1)
        pooled = F.normalize(pooled, dim=-1)

        # ----- similarities and forced alignment
        sim = torch.bmm(speech.float(), pooled.transpose(1, 2))  # [B, T', K]
        lens_p = np.array([len(ph) for ph in phonemes], dtype=np.int64)
        ends = batched_forced_align(sim, lens_t, lens_p)

        results: List[List[float] | None] = []
        for b in range(B):
            e = ends[b]
            n = min(len(e) - 1, len(phonemes[b]) - 2)
            results.append([round(float(t * self.hop_seconds), 3) for t in e[:n]])
        return results

    def _sliding_window(self, speech: torch.Tensor, lens_t: np.ndarray):
        """Average `win_len` frames every `frame_shift` frames (generic path)."""
        B, T, D = speech.shape
        num_win = int(math.ceil(T / self.frame_shift))
        pooled = F.avg_pool1d(
            speech.transpose(1, 2),
            kernel_size=self.win_len,
            stride=self.frame_shift,
            ceil_mode=True,
            count_include_pad=False,
        ).transpose(1, 2)[:, :num_win]
        new_lens = np.ceil(lens_t / self.frame_shift).astype(np.int64)
        return F.normalize(pooled, dim=-1), np.minimum(new_lens, pooled.shape[1])

    def align(
        self, audio: torch.Tensor, orig_sr: int, phonemes: List[str], *args, **kwargs
    ) -> List[Tuple[float, float, str]]:
        """
        Perform forced alignment.

        Parameters
        ----------
        audio : torch.Tensor
            Waveform.
        orig_sr : int
            Original sampling rate.
        phonemes : List[str]
            List of phonemes.

        Returns
        -------
        List[Tuple[float, float, str]]
            Phoneme alignment.
        """
        if orig_sr != self.sample_rate:
            if orig_sr not in self._resamplers:
                self._resamplers[orig_sr] = Resample(orig_sr, self.sample_rate)
            audio = self._resamplers[orig_sr](audio)

        alignment = self.forward(
            audios=list(audio.numpy()),
            phonemes=[[self.pad_token] + phonemes + [self.pad_token]],
        )[0]

        phoneme_alignment = []
        for i in range(len(phonemes) - 1):
            phoneme_alignment.append((alignment[i], alignment[i + 1], phonemes[i]))

        idx = len(phonemes) - 1
        duration = audio.shape[1] / self.sample_rate
        phoneme_alignment.append((alignment[idx], duration, phonemes[idx]))

        return phoneme_alignment


# ---------------------------
# Example main
# ---------------------------


def main():
    aligner = ClapIPA()

    phones_batch = [
        [
            "[SIL]",
            "\u0259",
            "m",
            "i",
            "l",
            "i",
            "\u0259",
            "t",
            "i",
            "\u02d0",
            "v",
            "\u0259",
            "[SIL]",
        ],
        ["[SIL]", "h", "\u025b", "l", "o\u028a", "[SIL]"],
    ]
    audios_batch = [
        np.zeros(16000, dtype=np.float32),
        np.zeros(24000, dtype=np.float32),
    ]

    alignment = aligner(audios_batch, phones_batch)
    for i, times in enumerate(alignment):
        print(f"Utterance {i}: {times}")


if __name__ == "__main__":
    main()
