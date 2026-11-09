import contextlib
import io
import logging

import torch
import torch.nn as nn
from safetensors.torch import load_file

from voxtream.config import SpeechGeneratorConfig
from voxtream.utils.sampling import sample_top_k, sample_top_p

# torchtune imports torchao, which prints "import error: No module named 'triton'"
# where triton is missing (macOS, CPU-only installs), and through it
# torch.distributed, whose elastic module warns that redirects are not supported on
# macOS and Windows. Neither concerns inference.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(
    logging.ERROR
)
with contextlib.redirect_stdout(io.StringIO()):
    import torchtune
    from torchtune.models import llama3_2


def load_model_weights(path: str) -> dict[str, torch.Tensor]:
    """Load model weights from a safetensors file or a Lightning checkpoint."""
    if str(path).endswith(".safetensors"):
        return load_file(path)

    weights = torch.load(path, map_location="cpu")
    state_dict = {}
    # Remove torch lightning 'model.' prefix
    for k, v in weights["state_dict"].items():
        k = k.replace("model.", "")
        state_dict[k] = v
    return state_dict


def get_llama3_2(
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    vocab_size: int = 128_256,
    max_seq_len: int = 2048,
) -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        intermediate_dim=intermediate_dim,
    )


# factories, not instances: `prepare_transformer` swaps a model's embedding and output
# layers for Identity, so a shared instance can only ever be built into one Model
MODEL_POOL = {
    "phone_former": lambda: get_llama3_2(
        num_layers=6, num_heads=8, num_kv_heads=2, embed_dim=1024, intermediate_dim=4096
    ),
    "temp_former": lambda: get_llama3_2(
        num_layers=12,
        num_heads=16,
        num_kv_heads=4,
        embed_dim=1024,
        intermediate_dim=4096,
    ),
    "dep_former_csm": lambda: get_llama3_2(
        num_layers=4, num_heads=8, num_kv_heads=2, embed_dim=1024, intermediate_dim=8192
    ),
    # half the depth of the pre-trained `dep_former_csm` at half the MLP width
    "dep_former": lambda: get_llama3_2(
        num_layers=2, num_heads=8, num_kv_heads=2, embed_dim=1024, intermediate_dim=4096
    ),
}


def prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def sample_acoustic_token(
    logits: torch.Tensor,
    temperature: float,
    cfg_ac_gamma: float = None,
) -> torch.Tensor:
    # CFG
    if cfg_ac_gamma is not None:
        assert logits.size(0) == 2, "CFG requires batch size of 2 (cond + uncond)"
        logits_cond, logits_uncond = torch.split(logits, 1, dim=0)
        logits = logits_cond * cfg_ac_gamma + (1 - cfg_ac_gamma) * logits_uncond

    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
    sampled_token = torch.argmax(probs, dim=-1, keepdim=True)

    return sampled_token


def sample_semantic_token(
    config: SpeechGeneratorConfig,
    logits: torch.Tensor,
    num_states: int,
    codebook_size: int,
    cfg_gamma: float = None,
    cur_spk_rate_cnt: torch.Tensor = None,
    target_spk_rate_cnt: torch.Tensor = None,
    spk_rate_weight: float = None,
) -> torch.Tensor:
    # CFG
    if cfg_gamma is not None:
        assert logits.size(0) == 2, "CFG requires batch size of 2 (cond + uncond)"
        logits_cond, logits_uncond = torch.split(logits, 1, dim=0)
        logits = logits_cond * cfg_gamma + (1 - cfg_gamma) * logits_uncond

        joint_logits_2d = logits_cond.view(-1, num_states, codebook_size)
        joint_logits_2d_cfg = logits.view(-1, num_states, codebook_size)
    else:
        joint_logits_2d = logits.view(-1, num_states, codebook_size)
        joint_logits_2d_cfg = joint_logits_2d

    spk_rate_logits = torch.logsumexp(joint_logits_2d, dim=-1)
    if target_spk_rate_cnt is not None:
        target_spk_rate_dist = torch.log(
            target_spk_rate_cnt / target_spk_rate_cnt.sum()
        )
        cur_spk_rate_dist = torch.log(cur_spk_rate_cnt / cur_spk_rate_cnt.sum())
        spk_rate_guided = (
            spk_rate_logits
            + spk_rate_weight * target_spk_rate_dist
            - spk_rate_weight * cur_spk_rate_dist
        )
    else:
        spk_rate_guided = spk_rate_logits

    spk_rate_probs = torch.nn.functional.softmax(
        spk_rate_guided / config.temperature, dim=-1
    )

    spk_rate_token = sample_top_p(spk_rate_probs, config.top_p)
    if cur_spk_rate_cnt is not None:
        cur_spk_rate_cnt[spk_rate_token.squeeze()] += 1

    spk_rate_index = spk_rate_token.unsqueeze(-1).expand(
        -1, -1, joint_logits_2d.size(-1)
    )
    semantic_logits = torch.gather(
        joint_logits_2d_cfg, dim=1, index=spk_rate_index
    ).squeeze(1)
    semantic_probs = torch.nn.functional.softmax(
        semantic_logits / config.temperature, dim=-1
    )
    semantic_token = sample_top_k(semantic_probs, config.topk)

    return semantic_token, spk_rate_token.squeeze(1), cur_spk_rate_cnt


def create_mask(seq_len: int, window_size: int, look_ahead: int = 0) -> torch.Tensor:
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        end = i + look_ahead + 1
        if look_ahead == -1:
            start = max(0, i - window_size // 2 + 1)
            end = i + window_size // 2 + 1
        mask[i, start:end] = True

    return mask


def create_dynamic_la_masks(
    max_look_ahead: int,
    window_size: int,
    max_seq_len: int,
):
    masks = []
    for la in range(1, max_look_ahead + 1):
        mask = create_mask(
            max_seq_len,
            window_size,
            la,
        )
        masks.append(mask)

    return torch.stack(masks, dim=0)  # (num_look_aheads, max_seq_len, max_seq_len)


def get_mask(mask: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, seq_len)
    """
    r = mask[input_pos, : len(input_pos[0])]
    return r


def sink_mask(
    seq_len: int,
    window_size: int,
    prompt_lens: torch.Tensor,
    min_recent_window: int = 1,
) -> torch.Tensor:
    """
    Causal mask that keeps the acoustic prompt attendable however far generation runs.

    This is the attention the generator actually has: `SinkAttention` rebuilds the cache
    out of the prompt frames and the most recent `window_size` ones, so a query past the
    window sees the prompt and its own recent past, nothing in between. Training the same
    shape is what makes a crop longer than the window behave like generation past a cache
    reset, instead of silently losing the prompt the speaker identity comes from.

    The prompt is part of the budget, not on top of it: the cache holds `window_size`
    frames in all, so the recent window is what the prompt leaves over. Keeping the full
    window as well would train on more context than generation ever has.

    Args:
        seq_len: length of the temporal transformer input
        window_size: total frames the cache holds, prompt included
        prompt_lens: (batch_size,) prompt frames at the head of every sample
        min_recent_window: floor on the recent window, as in `SinkAttentionConfig`

    Returns:
        (batch_size, seq_len, seq_len) boolean mask
    """
    pos = torch.arange(seq_len, device=prompt_lens.device)
    query, key = pos.view(1, -1, 1), pos.view(1, 1, -1)
    # the BOS frame is prefilled with the prompt, so the prompt owns keys 0..prompt_len
    prompt_keys = prompt_lens.view(-1, 1, 1) + 1
    window = torch.clamp(window_size - prompt_keys, min=min_recent_window)
    start = torch.maximum(query - (window - 1), prompt_keys)
    return (key <= query) & ((key < prompt_keys) | (key >= start))


def index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


def remove_punctuation(
    data: torch.Tensor, indices_to_remove: torch.Tensor
) -> torch.Tensor:
    """
    Remove the rows of the punctuation tokens of a flat phoneme vocabulary from the
    output of the phoneme transformer.

    Args:
        data: (batch_size, seq_len, dim)
        indices_to_remove: (batch_size, num_rows) padded with -1; may include invalid
            indices or duplicates

    Returns:
        (batch_size, seq_len, dim), the requested rows removed (per batch) and the tail
        padded with the last row
    """
    B, S, D = data.shape
    device = data.device

    # 1) Build keep_mask [B, S]
    valid = (indices_to_remove >= 0) & (indices_to_remove < S)  # [B, R]
    idx = indices_to_remove.clamp_min(0).clamp_max(S - 1)  # [B, R]
    batch = torch.arange(B, device=device).unsqueeze(1).expand_as(idx)  # [B, R]

    keep_mask = torch.ones((B, S), dtype=torch.bool, device=device)
    keep_mask[batch[valid], idx[valid]] = False  # mark removals

    # 2) Compute target positions for kept tokens (stable order)
    #    For kept positions, pos = cumsum(keep_mask) - 1; undefined for removed.
    pos = keep_mask.cumsum(dim=1) - 1  # [B, S], 0-based
    pos = pos.clamp_min(0)  # avoid negatives where masked

    # 3) Scatter kept rows into compacted positions
    #    We scatter-add the source masked by keep_mask to avoid overwriting problems.
    out = data.new_zeros(B, S, D)  # zeros for now
    src = data * keep_mask.unsqueeze(-1)  # zero out removed rows
    out.scatter_add_(
        dim=1, index=pos.unsqueeze(-1).expand(-1, -1, D), src=src
    )  # [B, S, D]

    # 4) Tail padding: replace positions >= num_kept with the last row of each batch
    num_kept = keep_mask.sum(dim=1)  # [B]
    arangeS = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
    tail_mask = arangeS >= num_kept.unsqueeze(1)  # [B, S]
    pad_row = data[:, -1:, :].expand(B, S, D)  # repeat last row along S
    out = torch.where(tail_mask.unsqueeze(-1), pad_row, out)  # fill tail

    return out


def patch_kv_cache_for_cuda_graph() -> None:
    """Avoid Python-side tensor asserts during CUDA graph capture.

    torchtune's KVCache.update uses a Python assert on a CUDA tensor, which
    triggers a disallowed sync during capture. We skip that assert only while
    capturing and rely on warmup/non-capture calls to catch misuse.
    """
    try:
        from torchtune.modules.kv_cache import KVCache
    except Exception:
        return

    if getattr(KVCache.update, "_voxtream_cuda_graph_patched", False):
        return

    orig_update = KVCache.update
    orig_reset = KVCache.reset

    def _update(self, k_val, v_val):
        if torch.cuda.is_current_stream_capturing():
            bsz, _, seq_len, _ = k_val.shape
            if bsz > self.k_cache.shape[0]:
                raise ValueError(
                    f"The current cache has been setup with a batch size of {self.k_cache.shape[0]}"
                    f", but found new key tensors with batch size {k_val.shape[0]}!"
                )
            k_out = self.k_cache
            v_out = self.v_cache
            k_out[:, :, self.cache_pos[:seq_len]] = k_val
            v_out[:, :, self.cache_pos[:seq_len]] = v_val
            self.cache_pos += seq_len
            return k_out, v_out
        return orig_update(self, k_val, v_val)

    def _reset(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            self.k_cache.zero_()
            self.v_cache.zero_()
            # Avoid .item() on CUDA tensor during capture.
            tmp = self.cache_pos - self.cache_pos[0]
            self.cache_pos.copy_(tmp)
            return
        return orig_reset(self)

    _update._voxtream_cuda_graph_patched = True
    KVCache.update = _update
    KVCache.reset = _reset
