# The code is partially borrowed from https://github.com/lingjzhu/charsiu
# Charsiu: A transformer-based phonetic aligner: https://arxiv.org/pdf/2110.03876

import json
import re
import unicodedata
from itertools import chain, groupby
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from g2p_en import G2p
from g2p_en.expand import normalize_numbers
from huggingface_hub import hf_hub_download
from librosa.sequence import dtw
from nltk.tokenize import word_tokenize
from torchaudio.transforms import Resample
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from transformers.modeling_outputs import CausalLMOutput

from voxtream.utils.generator import ensure_nltk_resource


def forced_align(cost: np.ndarray, phone_ids: list[int]) -> list[int]:
    """
    Force align a phone sequence with an acoustic cost matrix.

    Parameters
    ----------
    cost : np.ndarray
        Acoustic model costs (frames x phones).
    phone_ids : list[int]
        Sequence of phone IDs.

    Returns
    -------
    list[int]
        Alignment indices mapping frames to phone IDs.
    """
    D, align = dtw(C=-cost[:, phone_ids], step_sizes_sigma=np.array([[1, 1], [1, 0]]))

    align_seq = [-1 for _ in range(max(align[:, 0]) + 1)]
    for i in align:
        if align_seq[i[0]] < i[1]:
            align_seq[i[0]] = i[1]

    return list(align_seq)


def seq2duration(
    phones: list[str], resolution: float = 0.01
) -> list[tuple[float, float, str]]:
    """
    Convert a sequence of phones into durations.

    Parameters
    ----------
    phones : list[str]
        Sequence of phones.
    resolution : float, optional
        Time resolution in seconds per frame (default=0.01).

    Returns
    -------
    list[tuple[float, float, str]]
        Each entry is (start_time, end_time, phone).
    """
    counter = 0
    durations = []
    for phone, group in groupby(phones):
        length = len(list(group))
        durations.append(
            (
                round(counter * resolution, 2),
                round((counter + length) * resolution, 2),
                phone,
            )
        )
        counter += length
    return durations


class Wav2Vec2ForFrameClassification(Wav2Vec2ForCTC):
    """Wav2Vec2 model adapted for frame-level classification."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        cfg_path = hf_hub_download(
            pretrained_model_name_or_path, filename="config.json"
        )
        if Path(cfg_path).exists():
            with open(cfg_path) as f:
                config = json.load(f)
                if "gradient_checkpointing" in config:
                    del config["gradient_checkpointing"]

                config = Wav2Vec2Config(**config)

            # Load the model normally (ignoring gradient_checkpointing)
            model = super().from_pretrained(
                pretrained_model_name_or_path, *args, config=config, **kwargs
            )
        else:
            model = super().from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )

        return model

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = self.dropout(outputs[0])
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(
                    f"Label values must be <= vocab_size ({self.config.vocab_size})"
                )

            if attention_mask is None:
                attention_mask = torch.ones_like(input_values, dtype=torch.long)

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(2)),
                labels.flatten(),
                reduction="mean",
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CharsiuPreprocessor:
    """English G2P-based Charsiu preprocessor."""

    def __init__(self):
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("charsiu/tokenizer_en_cmu")
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )
        self.g2p = G2p()
        self.sil = "[SIL]"
        self.sil_idx = self.mapping_phone2id(self.sil)
        self.punctuation = set()

    def mapping_phone2id(self, phone: str) -> int:
        """Convert a phone to its numerical ID."""
        return self.processor.tokenizer.convert_tokens_to_ids(phone)

    def mapping_id2phone(self, idx: int) -> str:
        """Convert a numerical ID to its phone symbol."""
        return self.processor.tokenizer.convert_ids_to_tokens(idx)

    def audio_preprocess(
        self, audio: torch.Tensor, orig_sr: int, sr: int = 16000
    ) -> torch.Tensor:
        """
        Load and normalize audio for model input.

        Parameters
        ----------
        audio : torch.Tensor
            Raw waveform tensor.
        orig_sr : int
            Original sampling rate.
        sr : int, optional
            Target sampling rate (default=16000).

        Returns
        -------
        torch.Tensor
            Audio features ready for model input.
        """
        if orig_sr != sr:
            audio = Resample(orig_sr, sr)(audio)

        return self.processor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_values.squeeze()

    def get_phones_and_words(self, sentence: str) -> tuple[list[tuple[str]], list[str]]:
        """
        Convert text into phones and words.

        Parameters
        ----------
        sentence : str
            Input sentence.

        Returns
        -------
        tuple
            (phones, words), aligned at word level.
        """
        phones = self.g2p(sentence)
        words = self._get_words(sentence)

        phones = [tuple(g) for k, g in groupby(phones, key=lambda x: x != " ") if k]

        aligned_phones, aligned_words = [], []
        for p, w in zip(phones, words, strict=False):
            if re.search(r"\w+\d?", p[0]):
                aligned_phones.append(p)
                aligned_words.append(w)
            elif p in self.punctuation:
                aligned_phones.append((self.sil,))
                aligned_words.append(self.sil)

        assert len(aligned_words) == len(aligned_phones)
        return aligned_phones, aligned_words

    def get_phone_ids(
        self, phones: list[tuple[str]], append_silence: bool = True
    ) -> list[int]:
        """
        Convert phone sequence to IDs, with optional silence padding.

        Parameters
        ----------
        phones : list[tuple[str]]
            Sequence of phones grouped by word.
        append_silence : bool, optional
            Whether to add silence at the start and end (default=True).

        Returns
        -------
        list[int]
            Phone ID sequence.
        """
        flat_phones = list(chain.from_iterable(phones))
        ids = [self.mapping_phone2id(re.sub(r"\d", "", p)) for p in flat_phones]

        if append_silence:
            if ids[0] != self.sil_idx:
                ids.insert(0, self.sil_idx)
            if ids[-1] != self.sil_idx:
                ids.append(self.sil_idx)
        return ids

    def _get_words(self, text: str) -> list[str]:
        """Normalize and tokenize text into words."""
        text = str(text)
        text = normalize_numbers(text)
        text = "".join(
            char
            for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        ).lower()
        text = re.sub(r"[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is").replace("e.g.", "for example")

        return word_tokenize(text)

    def align_words(
        self,
        preds: list[tuple[float, float, str]],
        phones: list[tuple[str]],
        words: list[str],
    ) -> list[tuple[float, float, str]]:
        """
        Align phone durations with words.

        Parameters
        ----------
        preds : list
            Predicted phone durations.
        phones : list
            Original phone sequence.
        words : list
            Original word sequence.

        Returns
        -------
        list[tuple[float, float, str]]
            Word-level alignment.
        """
        words_rep = [w for ph, w in zip(phones, words, strict=False) for _ in ph]
        phones_rep = [re.sub(r"\d", "", p) for ph in phones for p in ph]
        assert len(words_rep) == len(phones_rep)

        word_dur, count = [], 0
        for dur in preds:
            if dur[-1] == self.sil:
                word_dur.append((dur, self.sil))
            else:
                while dur[-1] != phones_rep[count]:
                    count += 1
                word_dur.append((dur, words_rep[count]))

        merged = []
        for key, _group in groupby(word_dur, lambda x: x[-1]):
            group = list(_group)
            merged.append((group[0][0][0], group[-1][0][1], key))
        return merged


class CharsiuForcedAligner:
    """Forced alignment using Wav2Vec2 acoustic model."""

    def __init__(
        self,
        aligner: str,
        sil_threshold: int = 4,
        sampling_rate: int = 16000,
        device: str | None = None,
        resolution: float = 0.01,
    ):
        self.resolution = resolution
        self.sr = sampling_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = CharsiuPreprocessor()
        self.aligner = Wav2Vec2ForFrameClassification.from_pretrained(aligner)

        ensure_nltk_resource("punkt_tab")

        self.sil_threshold = sil_threshold
        self.sil = "[SIL]"
        self.vowels_vocab = {
            "AA",
            "AE",
            "AH",
            "AO",
            "AW",
            "AY",
            "EH",
            "ER",
            "EY",
            "IH",
            "IY",
            "OW",
            "OY",
            "UH",
            "UW",
        }
        self._freeze_model()

    def _freeze_model(self):
        self.aligner.eval().to(self.device)

    def _merge_silence(
        self, aligned_phones: list[str], sil_mask: list[int]
    ) -> list[str]:
        """Merge silence and non-silence intervals."""
        result, count = [], 0
        for i in sil_mask:
            if i == self.processor.sil_idx:
                result.append(self.sil)
            else:
                result.append(aligned_phones[count])
                count += 1
        assert len(result) == len(sil_mask)
        return result

    def _get_sil_mask(self, cost: np.ndarray) -> np.ndarray:
        """Compute silence mask based on posterior probabilities."""
        preds = np.argmax(cost, axis=-1)
        mask = []
        for key, _group in groupby(preds):
            group = list(_group)
            if key == self.processor.sil_idx and len(group) < self.sil_threshold:
                mask.extend([-1] * len(group))
            else:
                mask.extend(group)
        return np.array(mask)

    def to_mfa_format(
        self,
        pred_phones: list[tuple[float, float, str]],
        pred_words: list[tuple[float, float, str]],
        orig_phones: list[tuple[str]],
    ) -> dict:
        """Convert alignments to MFA (Montreal Forced Aligner) format."""
        vowels = [ph for ph in chain.from_iterable(orig_phones) if ph[-1].isdigit()]
        alignment: Dict[str, Any] = {
            "tiers": {"phones": {"entries": []}, "words": {"entries": []}}
        }

        for start, end, word in pred_words:
            if word not in ("[PAD]", "[UNK]", self.sil):
                alignment["tiers"]["words"]["entries"].append([start, end, word])

        for start, end, ph in pred_phones:
            if ph in ("[PAD]", "[UNK]"):
                continue
            if ph == self.sil:
                ph = "sil"
            elif ph in self.vowels_vocab:
                found = False
                while vowels:
                    if vowels[0][:-1] == ph:
                        ph = vowels.pop(0)
                        found = True
                        break
                    vowels.pop(0)
                if not found:
                    ph = f"{ph}0"
            alignment["tiers"]["phones"]["entries"].append([start, end, ph])

        assert len(vowels) <= 1, "Vowel count mismatch!"

        return alignment

    def align(self, audio: torch.Tensor, orig_sr: int, text: str) -> dict:
        """
        Perform forced alignment.

        Parameters
        ----------
        audio : torch.Tensor
            Waveform.
        orig_sr : int
            Original sampling rate.
        text : str
            Transcription.

        Returns
        -------
        dict
            MFA-style phoneme alignment.
        """
        audio = self.processor.audio_preprocess(audio, orig_sr=orig_sr, sr=self.sr)
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)

        phones, words = self.processor.get_phones_and_words(text)
        phone_ids = self.processor.get_phone_ids(phones)

        with torch.no_grad():
            out = self.aligner(audio)
        cost = torch.softmax(out.logits, dim=-1).cpu().numpy().squeeze()

        sil_mask = self._get_sil_mask(cost)
        nonsil_idx = np.argwhere(sil_mask != self.processor.sil_idx).squeeze()
        if nonsil_idx is None or nonsil_idx.size == 0:
            raise RuntimeError("No speech detected! Please check the audio file.")

        aligned_phone_ids = forced_align(cost[nonsil_idx, :], phone_ids[1:-1])
        aligned_phones = [
            self.processor.mapping_id2phone(phone_ids[1:-1][i])
            for i in aligned_phone_ids
        ]

        _pred_phones = self._merge_silence(aligned_phones, sil_mask)
        pred_phones = seq2duration(_pred_phones, resolution=self.resolution)
        pred_words = self.processor.align_words(pred_phones, phones, words)

        return self.to_mfa_format(pred_phones, pred_words, phones)
