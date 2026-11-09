# Mixture of Gaussians estimation head (MoGH) for the depth transformer.
#
# The head is adapted from the EAR-TTS model of NVIDIA NeMo
# (https://github.com/NVIDIA-NeMo/Speech, Apache 2.0), used in VoiceChat-TTS
# (https://arxiv.org/abs/2608.13831). It follows ResGen (Kim et al., "Efficient generative
# modeling with residual vector quantization-based tokens", ICML 2025).
#
# Instead of predicting one RVQ codebook per step, the head predicts a mixture of Gaussians
# over the continuous sum of the codebook vectors of all codebooks that are not decoded yet.
# The mean of the most likely component is quantized into those codebooks, so they are all
# decoded in a single step.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class MLPLayer(nn.Module):
    """Residual gated GELU MLP with pre- and post-normalization."""

    def __init__(self, dim: int, intermediate_dim: int, eps: float = 1e-6):
        super().__init__()
        self.pre_norm = RMSNorm(dim, eps=eps)
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)
        self.post_norm = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_norm(x)
        y = F.gelu(self.gate_proj(y), approximate="tanh") * self.up_proj(y)
        return x + self.post_norm(self.down_proj(y))


class MoGHead(nn.Module):
    """Mixture of Gaussians head with low-rank component means.

    Component k is an isotropic Gaussian N(z; sigma * (W_k @ m_k) + mu_res, sigma^2 I), where
    the low-rank coordinates m_k, the shared offset mu_res and the scale sigma are predicted
    from the input and W_k is a learned (latent_dim, low_rank) basis of the component.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        num_components: int,
        low_rank: int,
        num_layers: int,
        intermediate_dim: int,
        min_log_std: float = -4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.low_rank = low_rank
        self.min_log_std = min_log_std

        self.mlp = nn.Sequential(
            *[
                MLPLayer(hidden_dim, intermediate_dim, eps=eps)
                for _ in range(num_layers)
            ],
            RMSNorm(hidden_dim, eps=eps),
        )
        self.proj_logits = nn.Linear(hidden_dim, num_components, bias=False)
        self.proj_mus = nn.Linear(hidden_dim, num_components * low_rank, bias=False)
        self.proj_mu_res = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.proj_log_std = nn.Linear(hidden_dim, 1, bias=False)
        self.low_mat = nn.Parameter(
            torch.randn(num_components, latent_dim, low_rank) * low_rank**-0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Head features: (..., hidden_dim)."""
        return self.mlp(x)

    def _log_std(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj_log_std(h).clamp_min(self.min_log_std)

    def _component_mean(
        self,
        h: torch.Tensor,
        idx: torch.Tensor,
        mu_res: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        """Mean of component `idx` for features `h`: (N, latent_dim)."""
        proj_mus = self.proj_mus.weight.view(self.num_components, self.low_rank, -1)
        mus = (proj_mus[idx] @ h.unsqueeze(-1)).squeeze(-1)  # coordinates of `idx` only
        mean = (self.low_mat[idx].to(mus.dtype) @ mus.unsqueeze(-1)).squeeze(-1)
        return torch.exp(log_std) * mean + mu_res

    def _component_log_prob(
        self,
        h: torch.Tensor,
        mu_res: torch.Tensor,
        log_std: torch.Tensor,
        target: torch.Tensor,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Log-likelihood of the targets under every component: (N, num_components)
            Mean of component `idx` (no gradient): (N, latent_dim)
        """
        mus = self.proj_mus(h).unflatten(-1, (self.num_components, self.low_rank))

        with torch.autocast(device_type=h.device.type, enabled=False):
            mus = mus.float()
            mu_res, log_std = mu_res.float(), log_std.float()
            low_mat = self.low_mat.float()
            y = (target.float() - mu_res) * torch.exp(-log_std)

            # ||W_k m_k - y||^2 = m_k^T (W_k^T W_k) m_k - 2 m_k^T W_k^T y + ||y||^2
            v = torch.einsum("nrs,bns->bnr", low_mat.transpose(-1, -2) @ low_mat, mus)
            v = v - 2 * torch.einsum("bd,ndr->bnr", y, low_mat)
            dist = torch.einsum("bnr,bnr->bn", mus, v) + y.pow(2).sum(-1, keepdim=True)
            dist = dist.abs()  # guard against small negative values from rounding
            logp = (
                -0.5 * math.log(2 * math.pi) - log_std
            ) * self.latent_dim - 0.5 * dist

            with torch.no_grad():
                index = idx[:, None, None].expand(-1, 1, self.low_rank)
                mean = low_mat[idx] @ mus.gather(1, index).transpose(1, 2)
                mean = torch.exp(log_std) * mean.squeeze(-1) + mu_res

        return logp, mean

    def loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        label_smoothing: float = 0.0,
        chunk_size: int = 2048,
    ) -> dict[str, torch.Tensor]:
        """EM-style negative log-likelihood of the targets (as in NeMo EAR-TTS).

        The components are fit with responsibilities computed from their likelihoods, and the
        mixture weights are trained to match these responsibilities with a KL divergence.

        Args:
            x: (N, hidden_dim) inputs
            target: (N, latent_dim) continuous targets
            weight: (N,) weight of each target, zero for ignored targets
            chunk_size: number of targets per chunk of the component likelihoods

        Returns:
            Weighted sums of the component NLL (c_loss), of the mixture weight KL (k_loss), of
            the predicted log std, of the squared error of the mean of the most likely
            component (sq_err) and of the squared norm of the targets (energy).
        """
        h = self.mlp(x)
        logits = self.proj_logits(h)
        mu_res = self.proj_mu_res(h)
        log_std = self._log_std(h)
        idx = logits.argmax(-1)

        logp, mode = [], []
        for i in range(0, h.size(0), chunk_size):
            args = (h, mu_res, log_std, target, idx)
            out = self._component_log_prob(*[arg[i : i + chunk_size] for arg in args])
            logp.append(out[0])
            mode.append(out[1])
        logp, mode = torch.cat(logp), torch.cat(mode)

        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = logits.float()
            q = logp.detach().softmax(-1)
            q = q * (1 - label_smoothing) + label_smoothing / self.num_components
            c_loss = -(q * logp).sum(-1)
            k_loss = (q * (torch.log(q + 1e-8) - logits.log_softmax(-1))).sum(-1)

            target = target.float()
            sq_err = (mode - target).pow(2).sum(-1)

        return {
            "c_loss": (c_loss * weight).sum(),
            "k_loss": (k_loss * weight).sum(),
            "log_std": (log_std.float().squeeze(-1) * weight).sum(),
            "sq_err": (sq_err * weight).sum(),
            "energy": (target.pow(2).sum(-1) * weight).sum(),
        }

    @torch.no_grad()
    def sample(self, x: torch.Tensor, cfg_gamma: float = None) -> torch.Tensor:
        """The mean of the most likely component of the predicted mixture.

        Args:
            x: (batch_size, hidden_dim); with CFG the batch holds the conditional and the
                unconditional rows.
            cfg_gamma: classifier-free guidance weight applied to the head features
                (uncond + cfg_gamma * (cond - uncond)).

        Returns:
            (batch_size, latent_dim) latent, batch_size is 1 with CFG
        """
        h = self.mlp(x)
        if cfg_gamma is not None:
            h_cond, h_uncond = h.chunk(2, dim=0)
            h = h_uncond + cfg_gamma * (h_cond - h_uncond)

        idx = self.proj_logits(h).float().argmax(-1)
        log_std = self._log_std(h)
        return self._component_mean(h, idx, self.proj_mu_res(h), log_std)


class RVQCodebooks:
    """Codebooks of consecutive RVQ levels laid out for `residual_quantize`.

    The distances of a level are one GEMM against the codebook; done against the
    transposed view of a (codebook_size, latent_dim) table, PyTorch on the CPU takes
    about ten times longer than against a contiguous (latent_dim, codebook_size) one.
    The squared norms are also fixed.
    """

    def __init__(self, codebooks: torch.Tensor):
        self.codebooks = codebooks.float().contiguous()  # (num_levels, size, dim)
        self.transposed = self.codebooks.transpose(
            1, 2
        ).contiguous()  # (levels, dim, size)
        self.sq_norms = self.codebooks.pow(2).sum(-1)  # (num_levels, size)

    def __len__(self) -> int:
        return self.codebooks.size(0)


def residual_quantize(z: torch.Tensor, codebooks: RVQCodebooks) -> torch.Tensor:
    """Greedy residual quantization of latents with consecutive RVQ codebooks, as RVQ
    encoders do.

    Args:
        z: (batch_size, latent_dim) latents
        codebooks: codebooks of the levels to decode

    Returns:
        (batch_size, num_levels) codes
    """
    z = z.float()
    cb, cb_t, sq = codebooks.codebooks, codebooks.transposed, codebooks.sq_norms

    codes = []
    for level in range(len(codebooks)):
        # ||c||^2 - 2 z.c, the argmin of the distance
        dist = torch.addmm(sq[level], z, cb_t[level], alpha=-2.0)
        idx = dist.argmin(-1)
        z = z - cb[level][idx]
        codes.append(idx)

    return torch.stack(codes, dim=-1)


def _load_mimi_quantizer(mimi_repo: str, mimi_name: str, mimi_weight_path: str = None):
    from moshi.models import loaders

    if mimi_weight_path is None:
        mimi_weight_path = hf_hub_download(mimi_repo, mimi_name)
    return loaders.get_mimi(mimi_weight_path, device="cpu").quantizer


def load_mimi_acoustic_codebooks(
    mimi_repo: str, mimi_name: str, num_levels: int, mimi_weight_path: str = None
) -> torch.Tensor:
    """Codebooks of the acoustic RVQ of Mimi, scaled to a unit latent RMS per dimension.

    Args:
        mimi_weight_path: local Mimi weights, downloaded from `mimi_repo` if None

    Returns:
        (num_levels, codebook_size, latent_dim)
    """
    quantizer = _load_mimi_quantizer(mimi_repo, mimi_name, mimi_weight_path)
    layers = [layer._codebook for layer in quantizer.rvq_rest.vq.layers]
    return _scale_codebooks(layers[:num_levels])


def _scale_codebooks(layers: list) -> torch.Tensor:
    codebooks = torch.stack([layer.embedding for layer in layers]).float()
    usage = torch.stack([layer.cluster_usage for layer in layers]).float()

    # Usage-weighted energy of the summed latent
    probs = usage / usage.sum(dim=-1, keepdim=True)
    energy = (probs * codebooks.pow(2).sum(-1)).sum()
    scale = (energy / codebooks.size(-1)).sqrt()

    return codebooks / scale
