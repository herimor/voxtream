# The code is partially borrowed from https://github.com/SesameAILabs/csm/blob/main/models.py

import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from voxtream.config import SpeechGeneratorConfig
from voxtream.utils.model import (
    MODEL_POOL,
    create_dynamic_la_masks,
    create_mask,
    get_mask,
    index_causal_mask,
    prepare_transformer,
    remove_punctuation,
    sample_acoustic_token,
    sample_semantic_token,
    sink_mask,
)
from voxtream.utils.mog import MoGHead, RVQCodebooks, residual_quantize
from voxtream.utils.sink_attention import (
    PhoneSinkAttention,
    PhoneSinkAttentionConfig,
    SinkAttention,
    SinkAttentionConfig,
)


@dataclass
class ModelConfig:
    phone_former: str
    temp_former: str
    dep_former: str
    audio_vocab_size: int
    audio_pad_size: int
    embedding_dim: int
    spk_embedding_dim: int
    num_codebooks: int
    num_phone_states: int
    amortization_divisor: int
    max_look_ahead: int
    audio_window_size: int
    phone_window_size: int
    # The phoneme vocabulary is one of two kinds.
    # Flat (VoXtream2, VoXtream2.1): one token per phoneme with its stress and length,
    # punctuation as tokens of their own, inserted into the sequence and removed from
    # the output of the phoneme transformer.
    phone_vocab_size: int = None
    # Factored: a root stream plus the prefix, suffix, punctuation and end-of-word
    # streams, whose embeddings are summed onto the root embedding.
    phone_root_vocab_size: int = None
    phone_attr_vocab_sizes: List[int] = None
    # sampling weights for the look-ahead masks 1..max_look_ahead during training.
    # Uniform when left out, which over-trains the small values: the phoneme stack
    # compounds a mask of `la` into a receptive field of roughly `5 * la` phonemes, so
    # `la` should be weighted by how often that much text is really available ahead of
    # the pointer - heavily at both ends, thinly in between.
    phone_la_weights: Optional[List[float]] = None
    # keep the acoustic prompt attendable past the window, as the sink cache does at
    # inference; without it a crop longer than the window trains without its own prompt
    train_prompt_sink: bool = True
    # Depth transformer head: "ar" predicts the acoustic codebooks one by one. "mog"
    # predicts the first `mog_ar_levels` codebooks one by one with an AR head and the
    # sum of all remaining codebooks in a single step with a Mixture of Gaussians head,
    # whose sample is quantized into those codebooks.
    dep_former_head: str = "ar"
    mog_codebook_size: int = 2048
    mog_latent_dim: int = 256
    mog_num_components: int = 1024
    mog_low_rank: int = 64
    mog_num_layers: int = 3
    mog_intermediate_dim: int = 4096
    mog_min_log_std: float = -4.0
    mog_label_smoothing: float = 0.01
    # Leading acoustic codebooks of the AR head. The MoG step reads the depth
    # transformer output of the last AR step, the projected temporal transformer output
    # (the input of the depth transformer, with the speaker embedding) and the latent of
    # the AR codebooks, so a frame takes `mog_ar_levels` depth transformer steps.
    mog_ar_levels: int = 3

    @property
    def flat_phone_vocab(self) -> bool:
        return self.phone_vocab_size is not None


class Model(nn.Module):
    def __init__(self, config: ModelConfig, compile_forward: bool = False):
        super().__init__()
        self.config = config
        if config.flat_phone_vocab == (config.phone_root_vocab_size is not None):
            raise ValueError(
                "the model config needs either phone_vocab_size (flat vocabulary) or "
                "phone_root_vocab_size and phone_attr_vocab_sizes (factored)"
            )
        self.sink_attention = SinkAttention(
            SinkAttentionConfig(audio_window_size=config.audio_window_size)
        )
        self._dep_former_init = self._dep_former

        self.phone_former, phone_former_dim = prepare_transformer(
            MODEL_POOL[config.phone_former]()
        )
        self.temp_former, temp_former_dim = prepare_transformer(
            MODEL_POOL[config.temp_former]()
        )
        self.dep_former, dep_former_dim = prepare_transformer(
            MODEL_POOL[config.dep_former]()
        )

        self.phone_sink_attention = None
        if config.flat_phone_vocab:
            self.phone_embeddings = nn.Embedding(
                config.phone_vocab_size, phone_former_dim
            )
        else:
            # KV-cached phoneme transformer of the generator (`encode_phonemes`), fixed
            # to the largest look-ahead mask; the caches are installed by `setup_caches`
            self.phone_sink_attention = PhoneSinkAttention(
                PhoneSinkAttentionConfig(
                    phone_window_size=config.phone_window_size,
                    look_ahead=config.max_look_ahead,
                    num_layers=len(self.phone_former.layers),
                )
            )
            self.phone_embeddings = nn.Embedding(
                config.phone_root_vocab_size, phone_former_dim
            )
            # one table for every attribute stream, laid end to end; starting at zero
            # means a checkpoint trained without them keeps its behaviour until they
            # learn something
            self.phone_attr_embeddings = nn.Embedding(
                sum(config.phone_attr_vocab_sizes), phone_former_dim
            )
            nn.init.zeros_(self.phone_attr_embeddings.weight)
            self.register_buffer(
                "phone_attr_offsets",
                torch.tensor(
                    [
                        0,
                        *torch.tensor(config.phone_attr_vocab_sizes)
                        .cumsum(0)
                        .tolist()[:-1],
                    ]
                ).view(1, 1, -1),
                persistent=False,
            )
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.num_codebooks, config.embedding_dim
        )

        self.spk_emb_proj = nn.Linear(
            config.spk_embedding_dim, temp_former_dim, bias=False
        )
        self.sem_head = nn.Linear(
            temp_former_dim,
            (config.audio_vocab_size + config.audio_pad_size) * config.num_phone_states,
            bias=False,
        )
        if config.dep_former_head == "ar":
            # zero, as the AR head of the MoG model, unless a pre-trained one is loaded
            self.audio_head = nn.Parameter(
                torch.zeros(
                    config.num_codebooks - 1, dep_former_dim, config.audio_vocab_size
                )
            )
        elif config.dep_former_head == "mog":
            num_ar = config.mog_ar_levels
            if not 0 < num_ar < config.num_codebooks - 1:
                raise ValueError(
                    f"mog_ar_levels {num_ar} must leave at least one codebook to the AR "
                    "head and one to the MoG step"
                )
            self.mog_head = MoGHead(
                hidden_dim=dep_former_dim,
                latent_dim=config.mog_latent_dim,
                num_components=config.mog_num_components,
                low_rank=config.mog_low_rank,
                num_layers=config.mog_num_layers,
                intermediate_dim=config.mog_intermediate_dim,
                min_log_std=config.mog_min_log_std,
            )
            # Embedding of the MoG step, added to its input
            self.mog_step_embedding = nn.Parameter(torch.zeros(dep_former_dim))
            self.mog_ar_head = nn.Parameter(
                torch.zeros(num_ar, dep_former_dim, config.audio_vocab_size)
            )
            # Inputs of the MoG step next to the output of the last AR step: the latent
            # of the AR codebooks and the temporal transformer output
            self.mog_prefix_proj = nn.Linear(
                config.mog_latent_dim, dep_former_dim, bias=False
            )
            self.mog_temp_proj = nn.Linear(temp_former_dim, dep_former_dim, bias=False)
            nn.init.zeros_(self.mog_prefix_proj.weight)
            nn.init.zeros_(self.mog_temp_proj.weight)
            # Acoustic RVQ codebooks of Mimi, scaled to a unit latent RMS
            self.register_buffer(
                "mog_codebooks",
                torch.zeros(
                    config.num_codebooks - 1,
                    config.mog_codebook_size,
                    config.mog_latent_dim,
                ),
            )
        else:
            raise ValueError(f"Unknown dep_former_head: {config.dep_former_head}")
        self._generate_acoustic_mog = None
        # codebooks of the MoG step, see `prepare_mog_decoding`
        self.mog_rvq_codebooks = None

        self.register_buffer(
            "audio_shifts",
            (config.audio_vocab_size * torch.arange(config.num_codebooks)).view(
                1, -1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "phone_former_mask",
            create_dynamic_la_masks(
                config.max_look_ahead,
                config.phone_window_size,
                self.phone_former.max_seq_len,
            ),
            persistent=False,
        )
        la_weights = config.phone_la_weights or [1.0] * config.max_look_ahead
        if len(la_weights) != config.max_look_ahead:
            raise ValueError(
                f"phone_la_weights has {len(la_weights)} entries but max_look_ahead is "
                f"{config.max_look_ahead}; give one weight per look-ahead mask"
            )
        if min(la_weights) < 0 or sum(la_weights) <= 0:
            raise ValueError("phone_la_weights must be non-negative and not all zero")
        self.phone_la_weights = [float(w) for w in la_weights]

        self.register_buffer(
            "temp_former_causal_mask",
            create_mask(self.temp_former.max_seq_len, config.audio_window_size),
            persistent=False,
        )
        self.register_buffer(
            "dep_former_causal_mask",
            create_mask(self.config.num_codebooks, config.audio_window_size),
            persistent=False,
        )

        if compile_forward:
            self.forward_audio_fn = torch.compile(
                model=self.forward_audio, dynamic=False, mode="default"
            )
        else:
            self.forward_audio_fn = self.forward_audio

    def reorder_phone_emb(
        self, phone_emb: torch.Tensor, phoneme_embedding_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            phone_emb: (batch_size, phone_seq_len, phone_dim)
            phoneme_embedding_indices: (batch_size, phone_seq_len, num_phones)

        Returns:
            (batch_size, seq_len, dim) generated tokens
        """
        bs, seq_len, num_ph = phoneme_embedding_indices.shape
        batch_indices = torch.arange(bs).view(bs, 1, 1).expand(bs, seq_len, num_ph)
        phone_emb = phone_emb[batch_indices, phoneme_embedding_indices]
        phone_emb = phone_emb.sum(dim=2)

        return phone_emb

    def setup_caches(
        self,
        max_batch_size: int,
        dtype: torch.dtype,
        phone_cache_len: int = None,
        max_phone_rows: int = None,
    ) -> None:
        """
        Setup KV caches

        Args:
            phone_cache_len: slots of the phoneme transformer cache, see
                `PhoneSinkAttentionConfig.cache_len`
            max_phone_rows: rows of a phoneme stream the encoder buffers, at least the
                position range of the phoneme transformer
        """
        device = next(self.parameters()).device

        with device:
            self.temp_former.setup_caches(max_batch_size, dtype)
            self.dep_former.setup_caches(
                max_batch_size, dtype, decoder_max_seq_len=self.config.num_codebooks
            )
            if self.phone_sink_attention is not None:
                self.phone_sink_attention.setup(
                    self.phone_former,
                    max_batch_size,
                    dtype,
                    cache_len=phone_cache_len,
                    max_rows=max_phone_rows,
                    num_streams=1 + len(self.config.phone_attr_vocab_sizes),
                )

    def extract_phoneme_embeddings(
        self,
        phone_tokens: torch.Tensor,
        input_pos: torch.Tensor = None,
        phoneme_embedding_indices: torch.Tensor = None,
        prompt_len: int = None,
        look_ahead_phones: int = None,
        punct_del_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            phone_tokens: (batch_size, phone_seq_len, 5) root and attribute streams, or
                (batch_size, phone_seq_len) tokens of the flat vocabulary
            phoneme_embedding_indices: (batch_size, seq_len, 2)
            look_ahead_phones: phonemes available past the pointer. The look-ahead mask
                is picked to match it, because a mask of `la` compounds through the stack
                into a receptive field of roughly `max_look_ahead * la` phonemes: asking
                for more than the text stream has only means the recent phonemes are
                encoded against a sequence that is about to change. Falls back to the
                sequence length when the caller does not know. A flat vocabulary picks
                the mask from the text after the prompt, as it was released with.
            punct_del_indices: (batch_size, num_punct) rows of the punctuation tokens of
                a flat vocabulary, padded with -1, removed from the output

        Returns:
            (batch_size, seq_len, dim)
        """
        emb = self._embed_phone_tokens(phone_tokens)
        if input_pos is None:
            input_pos = torch.arange(0, emb.shape[1]).unsqueeze(0).long().to(emb.device)

        # look-ahead: sampled over `phone_la_weights` while training, matched to the
        # text actually available at inference
        max_la = self.config.max_look_ahead
        if self.training:
            idx = random.choices(
                range(self.phone_former_mask.size(0)), weights=self.phone_la_weights
            )[0]
        elif self.config.flat_phone_vocab:
            seq_len = phone_tokens.size(1)
            if prompt_len is not None:
                seq_len -= prompt_len
            idx = min(seq_len // max_la, max_la - 1)
        else:
            if look_ahead_phones is None:
                look_ahead_phones = phone_tokens.size(1)
                if prompt_len is not None:
                    look_ahead_phones -= prompt_len
            idx = max(1, min(int(look_ahead_phones) // max_la, max_la)) - 1
        phone_former_mask = self.phone_former_mask[idx]

        mask = get_mask(phone_former_mask, input_pos).to(emb.device)
        # a full-sequence pass, whether or not the generator installed the caches
        with (
            self.phone_sink_attention.disabled()
            if self.phone_sink_attention is not None
            else nullcontext()
        ):
            phone_emb = self.phone_former(emb, input_pos=input_pos, mask=mask).to(
                dtype=emb.dtype
            )

        if punct_del_indices is not None:
            phone_emb = remove_punctuation(phone_emb, punct_del_indices)

        if phoneme_embedding_indices is not None:
            phone_emb = self.reorder_phone_emb(phone_emb, phoneme_embedding_indices)

        return phone_emb

    def encode_phonemes(
        self,
        phone_tokens: torch.Tensor,
        prompt_len: int = None,
        look_ahead_phones: int = None,
    ) -> torch.Tensor:
        """
        Streaming counterpart of `extract_phoneme_embeddings` for the generator, with
        the factored vocabulary.

        Encodes the rows of `phone_tokens` that are new since the last call, together
        with the rows before them whose output can still change (the look-ahead
        receptive field), on the KV cache of the phoneme transformer, and returns the
        embeddings of the whole sequence. The look-ahead mask is the largest one
        (`max_look_ahead`), whatever text is available past the pointer: a row near the
        end of the text is then encoded against the rows that exist, exactly as the end
        of a training crop is, and is re-encoded once more text arrives.

        Args:
            phone_tokens: (batch_size, phone_seq_len, 5) the sequence so far
            prompt_len, look_ahead_phones: accepted for the signature of
                `extract_phoneme_embeddings`, not used

        Returns:
            (batch_size, phone_seq_len, dim) a view of the encoder's buffer, kept up to
            date by later calls
        """
        return self.phone_sink_attention.extend(phone_tokens, self._embed_phone_tokens)

    def _embed_phone_tokens(self, phone_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phone_tokens: (batch_size, phone_seq_len, 5) root and attribute streams, or
                (batch_size, phone_seq_len) tokens of the flat vocabulary

        Returns:
            (batch_size, phone_seq_len, dim) input of the phoneme transformer
        """
        if self.config.flat_phone_vocab:
            return self.phone_embeddings(phone_tokens)
        return self.phone_embeddings(phone_tokens[..., 0]) + self.phone_attr_embeddings(
            phone_tokens[..., 1:] + self.phone_attr_offsets
        ).sum(dim=-2)

    def generate_frame(
        self,
        config: SpeechGeneratorConfig,
        phone_emb: torch.Tensor,
        audio_tokens: torch.Tensor,
        input_pos: torch.Tensor,
        spk_embeddings: torch.Tensor,
        cfg_gamma: float,
        spk_rate_weight: float,
        cur_spk_rate_cnt: torch.Tensor,
        target_spk_rate_cnt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            phone_emb: (batch_size, seq_len, dim)
            audio_tokens: (batch_size, num_codebooks, seq_len)
            input_pos: (batch_size, seq_len) positions for each token
            spk_embeddings: (batch_size, 1, spk_emb_dim)

        Returns:
            frame: (batch_size, num_codebooks)
            pred_shift: (batch_size)
        """
        assert (
            self.temp_former.caches_are_enabled()
        ), "temp_former caches are not enabled"

        local_input_pos, curr_temp_former_mask = self.sink_attention.get_context(
            input_pos=input_pos,
            phone_emb=phone_emb,
            audio_tokens=audio_tokens,
            temp_former=self.temp_former,
            temp_former_causal_mask=self.temp_former_causal_mask,
            embed_audio_tokens=self._embed_audio_tokens,
        )

        audio_emb = self._embed_audio_tokens(audio_tokens)
        dtype = audio_emb.dtype

        emb = torch.cat([phone_emb.unsqueeze(1), audio_emb], dim=1)
        h = emb.sum(dim=1, dtype=dtype)

        if h.size(1) == 1:
            h = self._temp_former(h, local_input_pos, curr_temp_former_mask).to(
                dtype=dtype
            )
        else:
            h = self.temp_former(
                h, input_pos=local_input_pos, mask=curr_temp_former_mask
            ).to(dtype=dtype)

        last_h = h[:, -1, :]
        c0_logits = self.sem_head(last_h)

        c0_sample, pred_shift, cur_spk_rate_cnt = sample_semantic_token(
            config=config,
            logits=c0_logits,
            cfg_gamma=cfg_gamma,
            cur_spk_rate_cnt=cur_spk_rate_cnt,
            target_spk_rate_cnt=target_spk_rate_cnt,
            spk_rate_weight=spk_rate_weight,
            num_states=self.config.num_phone_states,
            codebook_size=self.config.audio_vocab_size + self.config.audio_pad_size,
        )

        c0_embed = self._embed_audio(0, c0_sample)
        c0_embed = c0_embed.repeat(last_h.shape[0], 1, 1)

        if spk_embeddings is not None:
            last_h = last_h + spk_embeddings

        if self.config.dep_former_head == "mog":
            if self._generate_acoustic_mog is not None:  # CUDA graph of the generator
                acoustic_tokens = self._generate_acoustic_mog(last_h, c0_embed)
            else:
                cfg = c0_sample.size(0) != last_h.size(0)
                acoustic_tokens = self.generate_acoustic_mog(
                    last_h, c0_embed, dtype, **self.mog_decoding_kwargs(config, cfg)
                )
        else:
            acoustic_tokens = self.generate_acoustic_ar(
                last_h=last_h,
                c0_embed=c0_embed,
                dtype=dtype,
                temperature=config.temperature,
                cfg_ac_gamma=config.cfg_ac_gamma,
            )
        frame = torch.cat([c0_sample, acoustic_tokens], dim=1)

        if input_pos.shape[1] == 1:
            self.sink_attention.append_step(phone_emb, audio_tokens)

        return frame, pred_shift, cur_spk_rate_cnt

    def generate_acoustic_ar(
        self,
        last_h: torch.Tensor,
        c0_embed: torch.Tensor,
        dtype: torch.dtype,
        temperature: float,
        cfg_ac_gamma: float = None,
    ) -> torch.Tensor:
        """
        Predict acoustic codebooks one by one with the depth transformer.

        Args:
            last_h: (batch_size, dim) temporal transformer output
            c0_embed: (batch_size, 1, dim) semantic token embedding

        Returns:
            (batch_size', num_codebooks - 1) acoustic tokens, batch_size' is 1 with CFG
        """
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0)
            .repeat(curr_h.size(0), 1)
        )
        tokens = []

        # Depth transfomer caches must be reset every frame.
        self.dep_former.reset_caches()
        for i in range(1, self.config.num_codebooks):
            curr_dep_former_mask = index_causal_mask(
                self.dep_former_causal_mask, curr_pos
            )

            if curr_h.size(1) == 1:
                dep_former_h = self._dep_former(
                    curr_h, curr_pos, curr_dep_former_mask
                ).to(dtype=dtype)
            else:
                dep_former_h = self._dep_former_init(
                    curr_h, curr_pos, curr_dep_former_mask
                ).to(dtype=dtype)

            ci_logits = torch.mm(dep_former_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_acoustic_token(
                logits=ci_logits,
                temperature=temperature,
                cfg_ac_gamma=cfg_ac_gamma,
            )

            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed.repeat(last_h.shape[0] // ci_embed.shape[0], 1, 1)

            tokens.append(ci_sample)
            curr_pos = curr_pos[:, -1:] + 1

        return torch.cat(tokens, dim=1)

    def generate_acoustic_mog(
        self,
        last_h: torch.Tensor,
        c0_embed: torch.Tensor,
        dtype: torch.dtype,
        cfg_gamma: float = None,
        ar_cfg_gamma: float = None,
        ar_temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict the first `mog_ar_levels` acoustic codebooks one by one with the AR head
        of the depth transformer, then all remaining codebooks in a single MoG step: the
        head predicts the sum of their codebook vectors from the output of the last AR
        step, the projected temporal transformer output and the latent of the AR
        codebooks, and the mean of its most likely component is quantized greedily into
        the codebooks.

        Args:
            last_h: (batch_size, dim) temporal transformer output with the speaker
                embedding
            c0_embed: (batch_size, 1, dim) semantic token embedding
            cfg_gamma: classifier-free guidance of the MoG step, see `MoGHead.sample`
            ar_cfg_gamma, ar_temperature: see `sample_acoustic_token`, AR codebooks only

        Returns:
            (batch_size', num_codebooks - 1) acoustic tokens, batch_size' is 1 with CFG
        """
        num_ar = self.config.mog_ar_levels
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0)
            .repeat(curr_h.size(0), 1)
        )
        tokens = []
        prefix = None  # latent of the AR codebooks

        # Depth transfomer caches must be reset every frame.
        self.dep_former.reset_caches()
        for level in range(num_ar):
            curr_dep_former_mask = index_causal_mask(
                self.dep_former_causal_mask, curr_pos
            )
            h = self._dep_former(curr_h, curr_pos, curr_dep_former_mask)
            h = h[:, -1, :].to(dtype=dtype)

            logits = h @ self.mog_ar_head[level].to(dtype)
            logits = logits[:, : self.config.mog_codebook_size].float()
            step_tokens = sample_acoustic_token(
                logits, temperature=ar_temperature, cfg_ac_gamma=ar_cfg_gamma
            )
            latent = self.mog_codebooks[level, step_tokens.squeeze(-1)]
            prefix = latent if prefix is None else prefix + latent
            tokens.append(step_tokens)

            if level + 1 < num_ar:
                # input of the next depth transformer step
                ci_embed = self._embed_audio(level + 1, step_tokens)
                curr_h = ci_embed.repeat(last_h.shape[0] // ci_embed.shape[0], 1, 1)
                curr_pos = curr_pos[:, -1:] + 1

        prefix = prefix.repeat(h.size(0) // prefix.size(0), 1)
        z = self.mog_head.sample(
            self._mog_input(h, last_h, prefix), cfg_gamma=cfg_gamma
        )
        tokens.append(residual_quantize(z, self.mog_rvq_codebooks))

        return torch.cat(tokens, dim=1)

    @staticmethod
    def mog_decoding_kwargs(config: SpeechGeneratorConfig, cfg: bool) -> dict:
        """
        Arguments of `generate_acoustic_mog` from the generator config.

        Args:
            cfg: the batch holds the conditional and the unconditional rows
        """
        cfg_gamma, ar_cfg_gamma = None, None
        if cfg:
            cfg_gamma = config.mog_cfg_gamma
            ar_cfg_gamma = config.cfg_ac_gamma

        return dict(
            cfg_gamma=cfg_gamma,
            ar_cfg_gamma=ar_cfg_gamma,
            ar_temperature=config.temperature,
        )

    def prepare_mog_decoding(self) -> None:
        """
        Lay out the codebooks of the MoG step for `residual_quantize`. Called once the
        weights are loaded and on their device, before `generate_acoustic_mog`.
        """
        levels = self.mog_codebooks[self.config.mog_ar_levels :]
        self.mog_rvq_codebooks = RVQCodebooks(levels)

    def _mog_input(
        self, h: torch.Tensor, temp_h: torch.Tensor, prefix: torch.Tensor
    ) -> torch.Tensor:
        """
        Input of the MoG head.

        Args:
            h: (..., dim) depth transformer output of the last AR step
            temp_h: (..., temp_dim) temporal transformer output with the speaker
                embedding
            prefix: (..., mog_latent_dim) latent of the AR codebooks

        Returns:
            (..., dim)
        """
        x = h + self.mog_step_embedding.to(h.dtype)
        x = x + self.mog_temp_proj(temp_h.to(x.dtype))
        return x + self.mog_prefix_proj(prefix.to(x.dtype))

    def _temp_former(
        self,
        h: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.temp_former(h, input_pos=input_pos, mask=mask).to(dtype=h.dtype)
        return h

    def _dep_former(
        self,
        curr_h: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.dep_former(curr_h, input_pos=input_pos, mask=mask)
        return h

    def forward_temp_former(
        self,
        phone_emb: torch.Tensor,
        audio_tokens: torch.Tensor,
        spk_embeddings: torch.Tensor = None,
        prompt_lens: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            phone_emb: (batch_size, seq_len, dim)
            audio_tokens: (batch_size, num_codebooks, seq_len)
            spk_embeddings: (batch_size, spk_emb_dim)
            prompt_lens: (batch_size,) frames of acoustic prompt at the head of the crop

        Returns:
            sem_logits: (batch_size, dim, seq_len)
            h: (batch_size, audio_seq_len, dim) depth transformer conditioning
            audio_emb: (batch_size, num_codebooks, audio_seq_len, dim) embeddings of the
                tokens predicted by the depth transformer at every step
        """
        audio_emb = self._embed_audio_tokens(audio_tokens)

        # exclude last pad audio token
        emb = torch.cat([phone_emb.unsqueeze(1), audio_emb[:, :, :-1]], dim=1)
        emb = emb.sum(dim=1, dtype=phone_emb.dtype)

        input_pos = (
            torch.arange(0, emb.shape[1]).unsqueeze(0).long().to(audio_emb.device)
        )
        if self.config.train_prompt_sink and prompt_lens is not None:
            curr_temp_former_mask = sink_mask(
                emb.shape[1], self.config.audio_window_size, prompt_lens
            )
        else:
            curr_temp_former_mask = get_mask(
                self.temp_former_causal_mask, input_pos
            ).to(emb.device)

        h = self.temp_former(emb, input_pos=input_pos, mask=curr_temp_former_mask).to(
            emb.dtype
        )

        sem_logits = self.sem_head(h).permute(0, 2, 1)

        # exclude first pad audio token
        audio_emb = audio_emb[:, :, 1:]

        # speaker embedding
        if spk_embeddings is not None:
            spk_embeddings = self.spk_emb_proj(spk_embeddings).unsqueeze(1)
            h = h + spk_embeddings

        return sem_logits, h, audio_emb

    def forward_audio(
        self,
        phone_emb: torch.Tensor,
        audio_tokens: torch.Tensor,
        spk_embeddings: torch.Tensor = None,
        prompt_lens: torch.Tensor = None,
        depth_audio_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            phone_emb: (batch_size, seq_len, dim)
            audio_tokens: (batch_size, num_codebooks, seq_len)
            spk_embeddings: (batch_size, spk_emb_dim)
            prompt_lens: (batch_size,) frames of acoustic prompt at the head of the crop
            depth_audio_tokens: (batch_size, num_codebooks, seq_len) tokens of the depth
                transformer, the acoustic prompt is only masked in `audio_tokens`

        Returns:
            sem_logits: (batch_size, dim, seq_len)
            audio_logits: (batch_size, dim, num_codebooks - 1, audio_seq_len), or for the
                MoG head the inputs of `mog_loss`: the depth transformer outputs of the
                AR steps (batch_size, audio_seq_len, mog_ar_levels, dim) and the temporal
                output of the same frames (batch_size, audio_seq_len, temp_dim)
            rand_idx: (amortized_seq_len,)
        """
        sem_logits, h, audio_emb = self.forward_temp_former(
            phone_emb, audio_tokens, spk_embeddings, prompt_lens
        )
        if depth_audio_tokens is not None:
            audio_emb = self._embed_audio_tokens(depth_audio_tokens[:, :, 1:])

        # Depth transformer inputs: the temporal output and the codebooks of each step,
        # for the MoG head the AR codebooks only
        mog = self.config.dep_former_head == "mog"
        num_positions = self.config.num_codebooks
        if mog:
            num_positions = self.config.mog_ar_levels + 1
        temp_h = h
        h = torch.cat([h.unsqueeze(1), audio_emb[:, : num_positions - 1]], dim=1)

        bs = h.shape[0]
        h = h.permute((0, 2, 1, 3))
        rand_idx = self._amortized_frames(h.shape[1], h.device)
        h = h[:, rand_idx]
        h = h.reshape((bs * len(rand_idx), num_positions, -1))

        input_pos = (
            torch.arange(0, num_positions).unsqueeze(0).long().to(audio_emb.device)
        )
        curr_dep_former_causal_mask = get_mask(
            self.dep_former_causal_mask, input_pos
        ).to(audio_emb.device)

        dep_former_h = self.dep_former(
            h, input_pos=input_pos, mask=curr_dep_former_causal_mask
        ).to(h.dtype)

        # remove prediction for the first codebook
        dep_former_h = dep_former_h[:, 1:]
        dep_former_h = dep_former_h.reshape((bs, len(rand_idx), num_positions - 1, -1))

        if mog:
            return sem_logits, (dep_former_h, temp_h[:, rand_idx]), rand_idx

        return sem_logits, self.audio_logits(dep_former_h), rand_idx

    def _amortized_frames(self, num_frames: int, device: torch.device) -> torch.Tensor:
        """
        Depth transformer amortization: a random subset of the frames.
        See Compute amortization in https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice

        Returns:
            (num_frames // amortization_divisor,) sorted frame indices
        """
        return torch.randperm(num_frames, device=device)[
            : int(num_frames // self.config.amortization_divisor)
        ].sort()[0]

    def audio_logits(self, dep_former_h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dep_former_h: (batch_size, seq_len, num_codebooks - 1, dim)

        Returns:
            (batch_size, audio_vocab_size, num_codebooks - 1, seq_len)
        """
        return torch.einsum(
            "b s c d, c d k -> b k c s", dep_former_h, self.audio_head
        ).contiguous()

    def mog_loss(
        self,
        audio_out: tuple[torch.Tensor, torch.Tensor],
        audio_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Loss of the MoG head: the cross-entropy of the AR codebooks and the MoG loss of
        the step after them, which has seen the AR codebooks and predicts the sum of the
        codebook vectors of all remaining codebooks.

        Args:
            audio_out: the depth transformer outputs of the AR steps
                (batch_size, seq_len, mog_ar_levels, dim) and the temporal output
                (batch_size, seq_len, temp_dim), from `forward_audio`
            audio_labels: (batch_size, num_codebooks - 1, seq_len) acoustic tokens

        Returns:
            loss, c_loss, k_loss: losses normalized by the number of predicted tokens
            log_std: average predicted log std
            rel_err: relative squared error of the mean of the most likely component
            ar_loss, ar_acc, ar_acc_c1: cross-entropy and top-1 accuracy of the AR head
        """
        tokens = audio_labels.transpose(1, 2)
        num_levels = tokens.size(-1)
        num_ar = self.config.mog_ar_levels

        # frames with special tokens (padding, CFG dropout) have no latent
        valid = (tokens < self.config.mog_codebook_size).all(dim=-1)
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)

        dep_former_h, temp_h = audio_out
        prefix, target = self.mog_targets(tokens)
        ar_losses = self._mog_ar_losses(dep_former_h, tokens, valid)

        x = self._mog_input(dep_former_h[:, :, -1], temp_h, prefix)
        weight = valid.to(target.dtype)

        losses = self.mog_head.loss(
            x.flatten(0, 1),
            target.flatten(0, 1),
            weight.flatten(),
            label_smoothing=self.config.mog_label_smoothing,
        )

        num_tokens = ((num_levels - num_ar) * weight).sum().clamp_min(1)
        num_targets = weight.sum().clamp_min(1)
        c_loss = losses["c_loss"] / num_tokens
        k_loss = losses["k_loss"] / num_tokens

        return {
            "loss": c_loss + k_loss + ar_losses["ar_loss"],
            "c_loss": c_loss,
            "k_loss": k_loss,
            "log_std": losses["log_std"] / num_targets,
            "rel_err": losses["sq_err"] / losses["energy"].clamp_min(1e-8),
            **ar_losses,
        }

    def _mog_ar_losses(
        self, h: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Cross-entropy and top-1 accuracy of the AR head (`mog_ar_levels`).

        Args:
            h: (batch_size, seq_len, mog_ar_levels, dim) features of the AR steps
            tokens: (batch_size, seq_len, num_codebooks - 1) target acoustic tokens
            valid: (batch_size, seq_len) frames with a latent

        Returns:
            ar_loss, ar_acc (mean over the AR codebooks), ar_acc_c1 (first codebook)
        """
        num_ar = self.config.mog_ar_levels
        ar_logits = torch.einsum(
            "b s c d, c d k -> b k c s", h, self.mog_ar_head.to(h.dtype)
        ).float()
        ar_labels = tokens[..., :num_ar].masked_fill(~valid.unsqueeze(-1), -100)
        ar_labels = ar_labels.transpose(1, 2)
        ar_loss = F.cross_entropy(ar_logits, ar_labels, ignore_index=-100)
        hits = (ar_logits.argmax(1) == ar_labels).float().sum((0, 2))
        ar_acc = hits / valid.sum().clamp_min(1)
        return {"ar_loss": ar_loss, "ar_acc_c1": ar_acc[0], "ar_acc": ar_acc.mean()}

    @torch.no_grad()
    def mog_targets(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (..., num_codebooks - 1) acoustic tokens without special tokens

        Returns:
            (..., mog_latent_dim) latent of the AR codebooks, the sum of their codebook
                vectors, the input of the MoG step
            (..., mog_latent_dim) latent of the remaining codebooks, its target
        """
        levels = torch.arange(tokens.size(-1), device=tokens.device)
        emb = self.mog_codebooks.float()[levels, tokens]
        num_ar = self.config.mog_ar_levels
        return emb[..., :num_ar, :].sum(-2), emb[..., num_ar:, :].sum(-2)

    @torch.no_grad()
    def set_mog_codebooks(self, codebooks: torch.Tensor) -> None:
        """
        Args:
            codebooks: (num_codebooks - 1, mog_codebook_size, mog_latent_dim)
        """
        self.mog_codebooks.copy_(codebooks)

    def forward(
        self,
        phone_tokens: torch.Tensor,
        phoneme_embedding_indices: torch.Tensor,
        audio_tokens: torch.Tensor,
        prompt_lens: torch.Tensor = None,
        spk_embeddings: torch.Tensor = None,
        depth_audio_tokens: torch.Tensor = None,
        punct_del_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            phone_tokens: (batch_size, phone_seq_len, 5), or (batch_size, phone_seq_len)
                with the flat vocabulary
            phoneme_embedding_indices: (batch_size, seq_len, N)
            audio_tokens: (batch_size, num_codebooks, seq_len)
            prompt_lens: (batch_size,) frames of acoustic prompt at the head of the crop
            spk_embeddings: (batch_size, spk_emb_dim)
            depth_audio_tokens: (batch_size, num_codebooks, seq_len) tokens of the depth
                transformer, defaults to `audio_tokens`
            punct_del_indices: (batch_size, num_punct) rows of the punctuation tokens of
                the flat vocabulary, padded with -1

        Returns:
            sem_logits: (batch_size, dim, seq_len)
            audio_logits: (batch_size, dim, num_codebooks - 1, audio_seq_len)
            rand_idx: (amortized_seq_len,)
        """
        phone_emb = self.extract_phoneme_embeddings(
            phone_tokens, punct_del_indices=punct_del_indices
        )
        phone_emb = self.reorder_phone_emb(phone_emb, phoneme_embedding_indices)

        sem_logits, audio_logits, rand_idx = self.forward_audio_fn(
            phone_emb, audio_tokens, spk_embeddings, prompt_lens, depth_audio_tokens
        )

        return sem_logits, audio_logits, rand_idx

    def reset_caches(self):
        self.temp_former.reset_caches()
        self.dep_former.reset_caches()
        self.sink_attention.reset()
        if self.phone_sink_attention is not None:
            self.phone_sink_attention.reset()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_audio_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        audio_tokens = tokens + self.audio_shifts.to(tokens.device)
        audio_embeds = self.audio_embeddings(audio_tokens)

        return audio_embeds
