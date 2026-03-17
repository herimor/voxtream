"""
Clap-IPA — standalone phoneme aligner.
Original code: https://github.com/lingjzhu/clap-ipa
Paper: https://arxiv.org/pdf/2311.08323

Optimized for batch inference on GPU.
"""

import math
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.sequence import dtw
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

# ---------------------------
# Small utils
# ---------------------------


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = src_len if tgt_len is None else tgt_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inv = 1.0 - expanded_mask
    return inv.masked_fill(inv.to(torch.bool), torch.finfo(dtype).min)


def build_phone_pool_mask(
    lengths: list[int], device, dtype=torch.float32
) -> torch.Tensor:
    """
    lengths: list of segment lengths that sum to total subword length P_sub.
    Returns [K, P_sub] mask whose rows average each contiguous segment.
    """
    K = len(lengths)
    P_sub = int(sum(lengths))
    M = torch.zeros((K, P_sub), device=device, dtype=dtype)
    pos = 0
    for k, length in enumerate(lengths):
        if length > 0:
            M[k, pos : pos + length] = 1.0 / float(length)
        pos += length
    return M


# ---------------------------
# Whisper encoder (trimmed)
# ---------------------------


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
        output_attentions = (
            bool(output_attentions) if output_attentions is not None else False
        )
        output_hidden_states = (
            bool(output_hidden_states) if output_hidden_states is not None else False
        )
        return_dict = True if return_dict is None else bool(return_dict)

        x = F.gelu(self.conv1(input_features))
        x = F.gelu(self.conv2(x))  # [B, C, T/2]
        x = x.permute(0, 2, 1)  # [B, T', C]
        pos = self.embed_positions.weight[: x.size(1)]
        hidden_states = F.dropout(x + pos, p=self.dropout, training=self.training)

        if attention_mask is not None:
            dtype = hidden_states.dtype
            # downsampled by 2 in conv2 stride
            attention_mask = attention_mask[:, ::2].to(dtype=dtype)
            attn = _expand_mask(attention_mask, dtype, tgt_len=hidden_states.size(1))
        else:
            attn = None

        for idx, layer in enumerate(self.layers):
            if self.training and torch.rand(()) < self.layerdrop:
                continue
            hidden_states = layer(
                hidden_states,
                attn,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions,
            )[0]

        hidden_states = self.layer_norm(hidden_states)

        if not return_dict:
            return (hidden_states,)
        return BaseModelOutput(last_hidden_state=hidden_states)


# ---------------------------
# Phone & Speech encoders (trimmed)
# ---------------------------


class PhoneEncoder(BertPreTrainedModel):
    """BERT -> Linear projector (no pooling; we need per-token features)."""

    _keys_to_ignore_on_load_unexpected = [r"^t_prime$", r"^b$"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.projector = nn.Linear(config.hidden_size, config.proj_size)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
            return_dict=True,
        )
        seq = self.projector(out.last_hidden_state)  # [B, P, D]
        return BaseModelOutput(last_hidden_state=seq)


class SpeechEncoder(WhisperPreTrainedModel):
    """Whisper encoder + projector (no pooling; we need per-frame features)."""

    def __init__(self, config):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)
        self.projector = nn.Linear(config.hidden_size, config.proj_size)
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        enc = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hid = self.projector(enc.last_hidden_state)  # [B, T', D]
        return BaseModelOutput(last_hidden_state=hid)


# ---------------------------
# Sliding window builder (batched)
# ---------------------------


def build_sliding_window(
    L: int, win_len: int = 10, shift: int = 5, device=None, dtype=torch.float32
):
    """
    Returns a [Tw, L] binary window mask, Tw = ceil(L / shift).
    """
    num_win = int(math.ceil(L / shift))
    W = torch.zeros((num_win, L), device=device, dtype=dtype)
    for n in range(num_win):
        s, e = n * shift, min(n * shift + win_len, L)
        if s < e:
            W[n, s:e] = 1.0
    return W


# ---------------------------
# Torch DTW (diag or horizontal steps)
# ---------------------------


def forced_align_librosa_from_sim(
    sim_tw_p: torch.Tensor, phone_list: list[str]
) -> list[tuple[int, str]]:
    """
    sim_tw_p: [Tw, P] cosine similarity (float32 on CPU/NPU/GPU ok; we'll move to CPU for librosa).
    Returns list of (time_index, phone) pairs using your original logic.
    """
    # original: forced_align(-pairwise_cos_sim); inside: dtw(C=cost.T, steps=[[1,1],[0,1]])
    C = (-sim_tw_p).detach().cpu().numpy().astype(np.float32)
    _, path = dtw(
        C=C.T, step_sizes_sigma=np.array([[1, 1], [0, 1]])
    )  # rows=phones, cols=time

    max_row = int(path[:, 0].max()) + 1
    align_seq = [-1] * max_row
    for r, c in path:
        if align_seq[r] < c:
            align_seq[r] = c

    align_id = list(align_seq)
    return [(i, p) for i, p in zip(align_id[:-1], phone_list[1:-1])]


# ---------------------------
# Alignment (batched)
# ---------------------------


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
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.speech_encoder = (
            SpeechEncoder.from_pretrained(speech_encoder).eval().half().to(self.device)
        )
        self.phone_encoder = (
            PhoneEncoder.from_pretrained(phone_encoder).eval().half().to(self.device)
        )

        self.tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer)
        self.processor = AutoProcessor.from_pretrained(processor)

        self.win_len = win_len
        self.frame_shift = frame_shift
        self.hop_seconds = hop_len * frame_shift
        self.sample_rate = sample_rate
        self.pad_token = pad_token

    @torch.no_grad()
    def forward(
        self,
        audios: List[np.ndarray],
        phonemes: List[List[str]],
    ) -> List[List[float] | None]:
        """
        Returns per-utterance list of timestamps (seconds) aligned to each phoneme.
        """
        # ----- Batch audio -> features
        batch = self.processor(
            audios,
            sampling_rate=self.sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
            padding=True,
        )
        # trim to max real length (saves compute)
        max_len = batch["attention_mask"].sum(dim=1).max().item()
        batch["attention_mask"] = batch["attention_mask"][:, :max_len]
        batch["input_features"] = batch["input_features"][:, :, :max_len]
        batch = {k: v.half().to(self.device) for k, v in batch.items()}

        # ----- Encode
        speech_feats = self.speech_encoder(**batch).last_hidden_state  # [B, T', D]

        # ----- Normalize for cosine similarity
        speech_feats = F.normalize(speech_feats, dim=-1)

        # ----- Sliding windows per utterance (on downsampled T')
        results_sec = []
        enc_T_all = speech_feats.shape[1]

        for b in range(len(audios)):
            try:
                # Per-item real length from its own attention mask (downsampled)
                L_mask = int(batch["attention_mask"][b, ::2].sum().item())

                # Use the smaller one to be 100% safe against off-by-one
                L = min(enc_T_all, L_mask)

                # ----- windowed speech [Tw, L] @ [L, D] -> [Tw, D]
                W = build_sliding_window(
                    L=L,
                    win_len=self.win_len,
                    shift=self.frame_shift,
                    device=speech_feats.device,
                    dtype=speech_feats.dtype,
                )
                sw = W @ speech_feats[b, :L, :]  # shapes now always match
                sw = F.normalize(sw, dim=-1)

                # ----- phone pooling back to phone level (exactly like original)
                tok_single = self.tokenizer(
                    phonemes[b],
                    add_special_tokens=False,
                    return_length=True,
                    return_attention_mask=False,
                    is_split_into_words=False,
                )
                flat_ids = (
                    torch.tensor(list(chain.from_iterable(tok_single["input_ids"])))
                    .long()
                    .unsqueeze(0)
                    .to(self.device)
                )

                phone_seq = self.phone_encoder(
                    input_ids=flat_ids
                ).last_hidden_state.squeeze(
                    0
                )  # [P_sub, D]

                M = build_phone_pool_mask(
                    tok_single["length"], device=phone_seq.device, dtype=phone_seq.dtype
                )  # [K, P_sub]
                phone_pooled = M @ phone_seq  # [K, D]
                phone_pooled = F.normalize(phone_pooled, dim=-1)

                # ----- similarities in float32 for stability
                sim = sw.float() @ phone_pooled.float().T  # [Tw, K]

                # ----- librosa DTW (phones × time) exactly like the original
                pairs = forced_align_librosa_from_sim(
                    sim, phonemes[b]
                )  # [(time_idx, phone), ...]
                times = [round(float(t * self.hop_seconds), 3) for (t, _) in pairs]
                results_sec.append(times)
            except Exception:
                results_sec.append(None)

        return results_sec

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
            resampler = Resample(orig_sr, self.sample_rate)
            audio = resampler(audio)

        # predict start time for each phoneme
        alignment = self.forward(
            audios=audio.numpy(),
            phonemes=[[self.pad_token] + phonemes + [self.pad_token]],
        )[0]

        # process phoneme alignment
        phoneme_alignment = []
        for i in range(len(phonemes) - 1):
            entry = (alignment[i], alignment[i + 1], phonemes[i])
            phoneme_alignment.append(entry)

        # process final phoneme alignment
        idx = len(phonemes) - 1
        duration = audio.shape[1] / self.sample_rate
        entry = (alignment[idx], duration, phonemes[idx])
        phoneme_alignment.append(entry)

        return phoneme_alignment


# ---------------------------
# Example main
# ---------------------------


def main():
    aligner = ClapIPA()

    phones_batch = [
        ["[SIL]", "ə", "m", "i", "l", "i", "ə", "t", "i", "ː", "v", "ə", "[SIL]"],
        ["[SIL]", "h", "ɛ", "l", "oʊ", "[SIL]"],
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
