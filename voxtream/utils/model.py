import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2

from voxtream.utils.sampling import sample_top_k


def get_llama3_2(
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    vocab_size: int = 128_256,
    max_seq_len: int = 1024,
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


MODEL_POOL = {
    "phone_former": get_llama3_2(
        num_layers=6, num_heads=8, num_kv_heads=2, embed_dim=1024, intermediate_dim=4096
    ),
    "temp_former": get_llama3_2(
        num_layers=12,
        num_heads=16,
        num_kv_heads=4,
        embed_dim=1024,
        intermediate_dim=4096,
    ),
    "dep_former_csm": get_llama3_2(
        num_layers=4, num_heads=8, num_kv_heads=2, embed_dim=1024, intermediate_dim=8192
    ),
}


def prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def sample_token(
    logits: torch.Tensor,
    topk: int,
    temperature: float,
) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
    sample_token = sample_top_k(probs, topk)

    return sample_token


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
