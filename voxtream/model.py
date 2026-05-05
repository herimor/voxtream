# The code is partially borrowed from https://github.com/SesameAILabs/csm/blob/main/models.py

from dataclasses import dataclass

import torch
import torch.nn as nn

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
)
from voxtream.utils.sink_attention import SinkAttention, SinkAttentionConfig


@dataclass
class ModelConfig:
    phone_former: str
    temp_former: str
    dep_former: str
    phone_vocab_size: int
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


class Model(nn.Module):
    def __init__(self, config: ModelConfig, compile_forward: bool = False):
        super().__init__()
        self.config = config
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

        self.phone_embeddings = nn.Embedding(config.phone_vocab_size, phone_former_dim)
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
        self.audio_head = nn.Parameter(
            torch.empty(
                config.num_codebooks - 1, dep_former_dim, config.audio_vocab_size
            )
        )

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
        self.register_buffer(
            "phone_pad_embedding",
            torch.zeros(1, self.phone_embeddings.embedding_dim),
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

    def setup_caches(self, max_batch_size: int, dtype: torch.dtype) -> None:
        """Setup KV caches"""
        device = next(self.parameters()).device

        with device:
            self.temp_former.setup_caches(max_batch_size, dtype)
            self.dep_former.setup_caches(
                max_batch_size, dtype, decoder_max_seq_len=self.config.num_codebooks
            )

    def extract_phoneme_embeddings(
        self,
        phone_tokens: torch.Tensor,
        input_pos: torch.Tensor = None,
        phoneme_embedding_indices: torch.Tensor = None,
        prompt_len: int = None,
    ) -> torch.Tensor:
        """
        Args:
            phone_tokens: (batch_size, phone_seq_len)
            phoneme_embedding_indices: (batch_size, seq_len, 2)

        Returns:
            (batch_size, seq_len, dim)
        """
        emb = self.phone_embeddings(phone_tokens)
        if input_pos is None:
            input_pos = torch.arange(0, emb.shape[1]).unsqueeze(0).long().to(emb.device)

        # random look-ahead
        if self.training:
            perm = torch.randperm(self.phone_former_mask.size(0))
            phone_former_mask = self.phone_former_mask[perm[0]]
        else:
            # select based on the length of phone tokens
            seq_len = phone_tokens.size(1)
            if prompt_len is not None:
                seq_len -= prompt_len

            idx = min(
                seq_len // self.config.max_look_ahead, self.config.max_look_ahead - 1
            )
            phone_former_mask = self.phone_former_mask[idx]

        mask = get_mask(phone_former_mask, input_pos).to(emb.device)
        phone_emb = self.phone_former(emb, input_pos=input_pos, mask=mask).to(
            dtype=emb.dtype
        )

        if phoneme_embedding_indices is not None:
            phone_emb = self.reorder_phone_emb(phone_emb, phoneme_embedding_indices)

        return phone_emb

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
            last_h = last_h + spk_embeddings * config.spk_proj_weight

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        frame = c0_sample.clone()
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0)
            .repeat(curr_h.size(0), 1)
        )

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
                temperature=config.temperature,
                cfg_ac_gamma=config.cfg_ac_gamma,
            )

            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed.repeat(last_h.shape[0], 1, 1)

            frame = torch.cat([frame, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        if input_pos.shape[1] == 1:
            self.sink_attention.append_step(phone_emb, audio_tokens)

        return frame, pred_shift, cur_spk_rate_cnt

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

    def forward_audio(
        self,
        phone_emb: torch.Tensor,
        audio_tokens: torch.Tensor,
        spk_embeddings: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            phone_emb: (batch_size, seq_len, dim)
            audio_tokens: (batch_size, num_codebooks, seq_len)
            spk_embeddings: (batch_size, spk_emb_dim)

        Returns:
            sem_logits: (batch_size, dim, seq_len)
            audio_logits: (batch_size, dim, num_codebooks - 1, audio_seq_len)
            rand_idx: (amortized_seq_len,)
        """
        audio_emb = self._embed_audio_tokens(audio_tokens)

        # exclude last pad audio token
        emb = torch.cat([phone_emb.unsqueeze(1), audio_emb[:, :, :-1]], dim=1)
        emb = emb.sum(dim=1, dtype=phone_emb.dtype)

        input_pos = (
            torch.arange(0, emb.shape[1]).unsqueeze(0).long().to(audio_emb.device)
        )
        curr_temp_former_mask = get_mask(self.temp_former_causal_mask, input_pos).to(
            emb.device
        )

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

        h = torch.cat([h.unsqueeze(1), audio_emb[:, :-1]], dim=1)

        # Depth transformer amortization.
        # See Compute amortization in https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice
        bs = h.shape[0]
        h = h.permute((0, 2, 1, 3))
        rand_idx = torch.randperm(h.shape[1], device=h.device)[
            : int(h.shape[1] // self.config.amortization_divisor)
        ].sort()[0]
        h = h[:, rand_idx]
        h = h.reshape((bs * len(rand_idx), self.config.num_codebooks, -1))

        input_pos = (
            torch.arange(0, self.config.num_codebooks)
            .unsqueeze(0)
            .long()
            .to(audio_emb.device)
        )
        curr_dep_former_causal_mask = get_mask(
            self.dep_former_causal_mask, input_pos
        ).to(audio_emb.device)

        dep_former_h = self.dep_former(
            h, input_pos=input_pos, mask=curr_dep_former_causal_mask
        ).to(h.dtype)

        # remove prediction for the first codebook
        dep_former_h = dep_former_h[:, 1:]
        dep_former_h = dep_former_h.reshape(
            (bs, len(rand_idx), self.config.num_codebooks - 1, -1)
        )

        # head
        audio_logits = torch.einsum(
            "b s c d, c d k -> b k c s", dep_former_h, self.audio_head
        ).contiguous()

        return sem_logits, audio_logits, rand_idx

    def forward(
        self,
        phone_tokens: torch.Tensor,
        phoneme_embedding_indices: torch.Tensor,
        audio_tokens: torch.Tensor,
        punct_del_indices: torch.Tensor,
        spk_embeddings: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            phone_tokens: (batch_size, phone_seq_len)
            phoneme_embedding_indices: (batch_size, seq_len, N)
            audio_tokens: (batch_size, num_codebooks, seq_len)
            punct_del_indices: (batch_size, punct_seq_len)
            spk_embeddings: (batch_size, spk_emb_dim)

        Returns:
            sem_logits: (batch_size, dim, seq_len)
            audio_logits: (batch_size, dim, num_codebooks - 1, audio_seq_len)
            rand_idx: (amortized_seq_len,)
        """
        phone_emb = self.extract_phoneme_embeddings(phone_tokens)
        phone_emb = remove_punctuation(phone_emb, punct_del_indices)
        phone_emb = self.reorder_phone_emb(phone_emb, phoneme_embedding_indices)

        sem_logits, audio_logits, rand_idx = self.forward_audio_fn(
            phone_emb, audio_tokens, spk_embeddings
        )

        return sem_logits, audio_logits, rand_idx

    def reset_caches(self):
        self.temp_former.reset_caches()
        self.dep_former.reset_caches()
        self.sink_attention.reset()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_audio_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        audio_tokens = tokens + self.audio_shifts.to(tokens.device)
        audio_embeds = self.audio_embeddings(audio_tokens)

        return audio_embeds
