import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import torch
import torchaudio
from g2p_en import G2p
from huggingface_hub import hf_hub_download
from moshi.models import MimiModel, loaders
from safetensors.torch import load_file
from torchaudio.transforms import Resample

from voxtream.model import Model, ModelConfig
from voxtream.utils.aligner import CharsiuForcedAligner
from voxtream.utils.generator import align_prompt, ensure_nltk_resource, normalize_text


@dataclass
class SpeechGeneratorConfig:
    sil_token: int
    bos_token: int
    eos_token: int
    end_pad: int
    num_codebooks: int
    num_phones_per_frame: int
    audio_delay_frames: int
    temperature: float
    topk: int
    max_audio_length_ms: int
    model_repo: str
    model_name: str
    model_config_name: str
    mimi_sr: int
    mimi_vocab_size: int
    mimi_frame_ms: int
    mimi_repo: str
    mimi_name: str
    spk_enc_sr: int
    spk_enc_repo: str
    spk_enc_model: str
    spk_enc_model_name: str
    spk_enc_train_type: str
    spk_enc_dataset: str
    phoneme_index_map: Dict
    phoneme_dict_name: str
    nltk_resource: str
    aligner: str
    max_prompt_sec: int
    max_prompt_chars: int
    max_phone_tokens: int
    cache_prompt: bool


class SpeechGenerator:
    def __init__(self, config: SpeechGeneratorConfig, compile: bool = False):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger("voxtream")

        # Model
        model, phone_to_token = self.get_model(config)
        self.model = model
        self.phone_to_token = phone_to_token

        # Mimi and speaker encoder
        self.mimi = self.get_mimi(config, num_codebooks=config.num_codebooks)
        self.spk_enc = self.get_spk_enc(config)

        # G2P
        self.g2p = G2p()
        ensure_nltk_resource(config.nltk_resource)

        # Phoneme aligner
        self.aligner = CharsiuForcedAligner(aligner=config.aligner)

        self.decoder_fn = (
            torch.compile(
                model=self.model.generate_frame, dynamic=False, mode="max-autotune"
            )
            if compile
            else self.model.generate_frame
        )

    def get_model(self, config: SpeechGeneratorConfig) -> Tuple[Model, Dict]:
        model_weight_path = hf_hub_download(config.model_repo, config.model_name)
        model_config_path = hf_hub_download(config.model_repo, config.model_config_name)
        phoneme_dict_path = hf_hub_download(config.model_repo, config.phoneme_dict_name)

        with open(phoneme_dict_path) as f:
            phone_to_token = json.load(f)

        with open(model_config_path) as f:
            model_config = ModelConfig(**json.load(f))

        model = Model(model_config)
        state_dict = load_file(model_weight_path)
        model.load_state_dict(state_dict)

        model = model.eval().half().to(self.device)
        model.setup_caches(max_batch_size=1, dtype=torch.float16)

        return model, phone_to_token

    def get_mimi(self, config: SpeechGeneratorConfig, num_codebooks: int) -> MimiModel:
        mimi_weight = hf_hub_download(config.mimi_repo, config.mimi_name)
        mimi = (
            loaders.get_mimi(
                filename=mimi_weight, device=self.device, num_codebooks=num_codebooks
            )
            .eval()
            .half()
        )

        return mimi

    def get_spk_enc(self, config: SpeechGeneratorConfig) -> torch.nn.Module:
        model = (
            torch.hub.load(
                config.spk_enc_repo,
                config.spk_enc_model,
                model_name=config.spk_enc_model_name,
                train_type=config.spk_enc_train_type,
                dataset=config.spk_enc_dataset,
                trust_repo=True,
                verbose=False,
            )
            .to(self.device)
            .half()
        )
        model.spec.float()
        model.bn.float()
        model.eval()

        return model

    def encode_audio_prompt(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr != self.config.mimi_sr:
            resampler = Resample(orig_sr, self.config.mimi_sr)
            waveform = resampler(waveform)

        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device, dtype=torch.float16),
        ):
            mimi_codes = self.mimi.encode(waveform.unsqueeze(0).to(self.device).half())

        padded_tokens = torch.full(
            (1, mimi_codes.shape[1], mimi_codes.shape[2] + 1),  # [bs, cb, time]
            fill_value=self.config.mimi_vocab_size,
            dtype=torch.int64,
            device=self.device,
        )

        padded_tokens[:, 0, 1:] = mimi_codes[:, 0]
        padded_tokens[:, 1:, self.config.audio_delay_frames + 1 :] = mimi_codes[
            :, 1:, : mimi_codes.shape[2] - self.config.audio_delay_frames
        ]

        return padded_tokens

    def extract_speaker_template(
        self, waveform: torch.Tensor, orig_sr: int
    ) -> torch.Tensor:
        if orig_sr != self.config.spk_enc_sr:
            resampler = Resample(orig_sr, self.config.spk_enc_sr)
            waveform = resampler(waveform)

        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device, dtype=torch.float16),
        ):
            spk_embedding = self.spk_enc(waveform.to(self.device).half())

        # L2-normalization
        spk_embedding /= spk_embedding.norm(keepdim=True)

        return spk_embedding

    def text_to_phone_tokens(self, text: str) -> List[int]:
        phone_tokens = []
        for ph in self.g2p(normalize_text(text)):
            if ph in (" ", "'"):
                continue
            phone_tokens.append(self.phone_to_token[ph])
        return phone_tokens

    def prepare_prompt(
        self, prompt_audio_path: Path, prompt_text: str
    ) -> Tuple[torch.Tensor, ...]:
        prompt_path = prompt_audio_path.parent / f"{prompt_audio_path.stem}.prompt.npy"
        if prompt_path.exists():
            prompt_data = np.load(prompt_path, allow_pickle=True).item()
            mimi_codes = torch.from_numpy(prompt_data["mimi_codes"]).to(self.device)
            spk_embedding = torch.from_numpy(prompt_data["spk_embedding"]).to(
                self.device
            )
            prompt_phone_tokens = torch.from_numpy(prompt_data["phone_tokens"]).to(
                self.device
            )
            phone_emb_indices = torch.from_numpy(prompt_data["phone_emb_indices"]).to(
                self.device
            )
        else:
            waveform, orig_sr = torchaudio.load(prompt_audio_path)
            if waveform.shape[0] != 1:
                # Convert to mono
                self.logger.warning(
                    f"Prompt audio has {waveform.shape[0]} channels; converting to mono by averaging."
                )
                waveform = waveform.mean(dim=0, keepdim=True)
            if waveform.shape[1] > orig_sr * self.config.max_prompt_sec:
                # Trim to max_prompt_sec seconds
                self.logger.warning(
                    f"Prompt audio is longer than {self.config.max_prompt_sec} seconds; trimming."
                )
                waveform = waveform[:, : orig_sr * self.config.max_prompt_sec]

            # 1. Extract mimi codes
            mimi_codes = self.encode_audio_prompt(waveform, orig_sr)
            # 2. Extract speaker embedding
            spk_embedding = self.extract_speaker_template(waveform, orig_sr)

            # 3. Align prompt text to audio
            # TODO: Add Whisper transcription if prompt_text is not provided
            prompt_text = normalize_text(prompt_text)
            if len(prompt_text) > self.config.max_prompt_chars:
                self.logger.warning(
                    f"Prompt text is longer than {self.config.max_prompt_chars} characters; trimming."
                )
                prompt_text = prompt_text[: self.config.max_prompt_chars]
            phoneme_alignment = self.aligner.align(
                audio=waveform, orig_sr=orig_sr, text=prompt_text
            )

            # 4. Convert alignment to phone tokens and phone embedding indices
            prompt_phone_tokens, phone_emb_indices = align_prompt(
                phoneme_alignment,
                num_frames=mimi_codes.shape[-1]
                - 1,  # -1 because the first token is <PAD>
                phones_per_frame=self.config.num_phones_per_frame,
                phone_to_idx=self.phone_to_token,
                g2p=self.g2p,
            )

            prompt_phone_tokens = torch.from_numpy(
                np.expand_dims(
                    np.concatenate([[self.config.bos_token], prompt_phone_tokens]),
                    axis=0,
                ),
            ).to(dtype=torch.int64, device=self.device)
            phone_emb_indices = torch.from_numpy(
                np.expand_dims(
                    np.concatenate(
                        [
                            [[0] * self.config.num_phones_per_frame],
                            phone_emb_indices + 1,
                        ]
                    ),
                    axis=0,
                ),
            ).to(dtype=torch.int64, device=self.device)

            if self.config.cache_prompt:
                np.save(
                    prompt_path,
                    {
                        "mimi_codes": mimi_codes.cpu().numpy(),
                        "spk_embedding": spk_embedding.cpu().numpy(),
                        "phone_tokens": prompt_phone_tokens.cpu().numpy(),
                        "phone_emb_indices": phone_emb_indices.cpu().numpy(),
                    },
                )

        return mimi_codes, spk_embedding, prompt_phone_tokens, phone_emb_indices

    def prepare_non_streaming_text(
        self, text: str, prompt_phone_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Prepare phone embeddings for non-streaming mode."""
        phonemes_to_gen = self.text_to_phone_tokens(text)
        max_len = self.config.max_phone_tokens - prompt_phone_tokens.shape[1]
        if len(phonemes_to_gen) > max_len:
            self.logger.warning(
                f"Input text exceeds max_phone_tokens ({self.config.max_phone_tokens}); trimming."
            )
            phonemes_to_gen = phonemes_to_gen[:max_len]
        phonemes_to_gen.extend([self.config.sil_token, self.config.eos_token])
        phone_tokens = torch.cat(
            [
                prompt_phone_tokens,
                torch.tensor([phonemes_to_gen], device=self.device, dtype=torch.long),
            ],
            dim=1,
        )
        phone_emb = self.model.extract_phoneme_embeddings(phone_tokens)
        phone_seq_len = phone_emb.shape[1] - 2  # ignore <silence> and <EOS> tokens
        return phone_tokens, phone_emb, phone_seq_len

    def prepare_streaming_text(
        self,
        text_gen: Generator[str, None, None],
        phone_tokens: torch.Tensor,
        empty_text_stream: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        """Fetch next text chunk from generator and update phone tokens/embeddings."""
        try:
            text_chunk = next(text_gen)
        except StopIteration:
            empty_text_stream = True

        phonemes_to_gen = []
        if empty_text_stream:
            phonemes_to_gen.extend([self.config.sil_token, self.config.eos_token])
        else:
            phonemes_to_gen.extend(self.text_to_phone_tokens(text_chunk))

        max_len = self.config.max_phone_tokens - phone_tokens.shape[1]
        if len(phonemes_to_gen) > max_len and not empty_text_stream:
            self.logger.warning(
                f"Input text stream is too long; trimming to fit max_phone_tokens ({self.config.max_phone_tokens})."
            )
            phonemes_to_gen = phonemes_to_gen[:max_len]
            phonemes_to_gen.extend([self.config.sil_token, self.config.eos_token])
            empty_text_stream = True

        phonemes_to_gen = torch.tensor(
            [phonemes_to_gen], device=self.device, dtype=torch.long
        )
        phone_tokens = torch.cat([phone_tokens, phonemes_to_gen], dim=1)
        phone_emb = self.model.extract_phoneme_embeddings(phone_tokens)

        phone_seq_len = phone_emb.shape[1]
        if empty_text_stream:
            phone_seq_len -= 2  # ignore <silence> and <EOS> tokens

        return phone_tokens, phone_emb, phone_seq_len, empty_text_stream

    def decode_audio_frame(
        self, frame: torch.Tensor, sem_code: torch.Tensor, start_time: float
    ) -> Tuple[np.ndarray, float, torch.Tensor, float]:
        """Decode predicted frame into audio and compute generation time."""
        audio_frame = torch.cat([sem_code, frame[:, 1:]], dim=1)
        audio_frame = torch.clamp(
            audio_frame, 0, int(self.config.mimi_vocab_size - 1)
        ).to(torch.long)
        sem_code = frame[:, :1]

        audio_frame = self.mimi.decode(audio_frame.unsqueeze(-1)).squeeze()
        gen_time = time.time() - start_time
        return (
            audio_frame.cpu().numpy().astype(np.float32),
            gen_time,
            sem_code,
            time.time(),
        )

    def update_indices_and_tokens(
        self,
        pred_shift: torch.Tensor,
        phone_emb_indices: torch.Tensor,
        phone_seq_len: int,
        idx: int,
        eos_idx: int,
        frame: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Update phone embedding indices, audio tokens, and EOS logic."""
        pred_shift_int = int(pred_shift.item())
        shift, num_tokens = self.config.phoneme_index_map[str(pred_shift_int)]

        start = max(phone_emb_indices[0][-1]) + shift
        if start == phone_seq_len:
            val = [start] * self.config.num_phones_per_frame
            if eos_idx == int(
                self.config.max_audio_length_ms / self.config.mimi_frame_ms
            ):
                eos_idx = idx + self.config.end_pad
        elif start > phone_seq_len:
            val = [phone_seq_len + 1] * self.config.num_phones_per_frame
            eos_idx = idx + 1
        else:
            val = list(range(int(start), int(start + num_tokens)))
            while len(val) < self.config.num_phones_per_frame:
                val.append(val[-1])

        phone_emb_indices = torch.tensor(
            [[val]], device=phone_emb_indices.device, dtype=torch.long
        )
        mimi_codes = frame.unsqueeze(dim=2)

        return phone_emb_indices, mimi_codes, eos_idx

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_text: str,
        prompt_audio_path: Path,
        text: str | Generator[str, None, None],
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate speech stream from text.

        Args:
            prompt_text (str): The text prompt for the model.
            prompt_audio_path (Path): The file path for the prompt audio.
            text (str | Generator): The text input for audio generation. Pass a text generator to enable input streaming.
        """
        mimi_codes, spk_embedding, prompt_phone_tokens, phone_emb_indices = (
            self.prepare_prompt(
                prompt_audio_path=prompt_audio_path, prompt_text=prompt_text
            )
        )

        self.model.reset_caches()
        spk_embedding = self.model.spk_emb_proj(spk_embedding)

        # Prepare phoneme embeddings depending on mode
        if isinstance(text, str):
            phone_tokens, phone_emb, phone_seq_len = self.prepare_non_streaming_text(
                text, prompt_phone_tokens
            )
        else:
            phone_emb, phone_seq_len = None, 0
            phone_tokens = prompt_phone_tokens

        max_seq_len = int(self.config.max_audio_length_ms / self.config.mimi_frame_ms)
        curr_pos = (
            torch.arange(0, mimi_codes.size(2)).unsqueeze(0).long().to(self.device)
        )

        sem_code: torch.Tensor = None
        eos_idx = max_seq_len
        empty_text_stream = False

        start_time = time.time()
        with self.mimi.streaming(batch_size=1):
            for idx in range(max_seq_len):
                # Handle streaming input
                if isinstance(text, Generator) and not empty_text_stream:
                    phone_tokens, phone_emb, phone_seq_len, empty_text_stream = (
                        self.prepare_streaming_text(
                            text, phone_tokens, empty_text_stream
                        )
                    )

                phone_emb_chunk = self.model.reorder_phone_emb(
                    phone_emb, phone_emb_indices
                )

                gen_fn = self.model.generate_frame if idx == 0 else self.decoder_fn
                frame, pred_shift = gen_fn(
                    phone_emb=phone_emb_chunk,
                    audio_tokens=mimi_codes,
                    input_pos=curr_pos,
                    spk_embeddings=spk_embedding,
                    temperature=self.config.temperature,
                    topk=self.config.topk,
                )

                # Skip first padding frames
                if idx >= self.config.audio_delay_frames:
                    audio_frame, gen_time, sem_code, start_time = (
                        self.decode_audio_frame(frame, sem_code, start_time)
                    )
                    yield audio_frame, gen_time
                else:
                    sem_code = frame[:, :1]

                if eos_idx <= idx:
                    break  # <EOS> reached

                # Update phoneme embedding indices & audio tokens
                phone_emb_indices, mimi_codes, eos_idx = self.update_indices_and_tokens(
                    pred_shift, phone_emb_indices, phone_seq_len, idx, eos_idx, frame
                )

                # Position tracking
                curr_pos = curr_pos[:, -1:] + 1
