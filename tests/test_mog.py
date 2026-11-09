import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxtream.utils.mog import MoGHead, RVQCodebooks, residual_quantize


def test_residual_quantize_recovers_codes():
    torch.manual_seed(0)
    # Codebooks with decreasing scales, as in RVQ
    codebooks = (
        torch.randn(4, 64, 16) * torch.tensor([1.0, 0.1, 0.01, 0.001])[:, None, None]
    )
    codes = torch.randint(0, 64, (32, 4))
    z = codebooks[torch.arange(4), codes].sum(1)

    assert torch.equal(residual_quantize(z, RVQCodebooks(codebooks)), codes)


def _tiny_head() -> MoGHead:
    torch.manual_seed(0)
    return MoGHead(
        hidden_dim=32,
        latent_dim=16,
        num_components=8,
        low_rank=4,
        num_layers=1,
        intermediate_dim=64,
    )


def test_mog_head_loss():
    head = _tiny_head()
    x = torch.randn(10, 32)
    target = torch.randn(10, 16)
    weight = torch.ones(10)
    weight[-1] = 0

    losses = head.loss(x, target, weight, label_smoothing=0.01, chunk_size=3)
    loss = losses["c_loss"] + losses["k_loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert head.proj_mus.weight.grad is not None
    assert head.low_mat.grad is not None

    # Component likelihoods match the closed form of the low-rank Gaussian
    with torch.no_grad():
        h = head.mlp(x)
        mu_res, log_std = head.proj_mu_res(h), head._log_std(h)
        idx = torch.zeros(10, dtype=torch.long)
        logp, _ = head._component_log_prob(h, mu_res, log_std, target, idx)
        mus = head.proj_mus(h).view(10, 8, 4)
        means = (
            torch.exp(log_std)[:, None]
            * torch.einsum("kdr,nkr->nkd", head.low_mat, mus)
            + mu_res[:, None]
        )
        dist = torch.distributions.Normal(means, torch.exp(log_std)[:, None])
        expected = dist.log_prob(target[:, None]).sum(-1)
        assert torch.allclose(logp, expected, atol=1e-3, rtol=1e-4)


def test_mog_head_sample_is_the_mode():
    """The sample is the mean of the most likely component, the same on every call."""
    head = _tiny_head().eval()
    x = torch.randn(10, 32)
    z = head.sample(x)
    assert z.shape == (10, 16)
    assert torch.equal(head.sample(x), z)

    with torch.no_grad():
        h = head.mlp(x)
        idx = head.proj_logits(h).argmax(-1)
        log_std = head._log_std(h)
        mus = head.proj_mus(h).view(10, 8, 4)[torch.arange(10), idx]
        mean = torch.einsum("ndr,nr->nd", head.low_mat[idx], mus)
        expected = torch.exp(log_std) * mean + head.proj_mu_res(h)
    assert torch.allclose(z, expected, atol=1e-5)

    # CFG: the conditional and the unconditional rows give one latent
    assert head.sample(x[:2], cfg_gamma=1.5).shape == (1, 16)


def _tiny_config(**config):
    """Config of a tiny MoG model; registers its transformers in the model pool."""
    from voxtream.model import ModelConfig
    from voxtream.utils.model import MODEL_POOL, get_llama3_2

    MODEL_POOL["test_tiny_former"] = lambda: get_llama3_2(
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        embed_dim=32,
        intermediate_dim=64,
        max_seq_len=128,
    )
    params = dict(
        phone_former="test_tiny_former",
        temp_former="test_tiny_former",
        dep_former="test_tiny_former",
        phone_root_vocab_size=20,
        phone_attr_vocab_sizes=[3, 2, 5, 2],
        audio_vocab_size=18,
        audio_pad_size=0,
        embedding_dim=32,
        spk_embedding_dim=8,
        num_codebooks=4,
        num_phone_states=6,
        amortization_divisor=2,
        max_look_ahead=2,
        audio_window_size=64,
        phone_window_size=64,
        dep_former_head="mog",
        mog_codebook_size=16,
        mog_latent_dim=8,
        mog_num_components=4,
        mog_low_rank=2,
        mog_num_layers=1,
        mog_intermediate_dim=64,
        mog_label_smoothing=0,
        mog_ar_levels=2,
    )
    return ModelConfig(**{**params, **config})


def _tiny_model(**config):
    """Tiny MoG model with random weights in the inputs of the MoG step, which start
    at zero."""
    from voxtream.model import Model

    torch.manual_seed(0)
    model = Model(_tiny_config(**config))
    model.set_mog_codebooks(torch.randn_like(model.mog_codebooks) * 0.3)
    model.prepare_mog_decoding()
    with torch.no_grad():
        model.mog_step_embedding.normal_()
        model.mog_prefix_proj.weight.normal_()
        model.mog_temp_proj.weight.normal_()
        model.mog_ar_head.normal_()
    return model


def _tiny_batch(batch_size=2, num_frames=20, num_phones=12):
    g = torch.Generator().manual_seed(1)
    sizes = [20, 3, 2, 5, 2]
    phone_tokens = torch.stack(
        [torch.randint(0, n, (batch_size, num_phones), generator=g) for n in sizes], -1
    )
    emb_idx = torch.randint(0, num_phones, (batch_size, num_frames, 2), generator=g)
    audio = torch.randint(0, 16, (batch_size, 4, num_frames + 1), generator=g)
    audio[:, :, 0] = 17  # PAD frame
    spk = torch.randn(batch_size, 8, generator=g)
    return phone_tokens, emb_idx.sort(1)[0], audio, spk


def test_mog_model_validation():
    from voxtream.model import Model

    for num_ar in (0, 3):  # one codebook each to the AR head and to the MoG step
        with pytest.raises(ValueError):
            Model(_tiny_config(mog_ar_levels=num_ar))
    model = Model(_tiny_config())
    assert model.mog_ar_head.shape == (2, 32, 18)
    # the inputs of the MoG step next to the depth transformer start at zero
    assert not model.mog_temp_proj.weight.any()
    assert not model.mog_prefix_proj.weight.any()


def test_mog_forward_and_loss():
    """The depth transformer runs the AR steps only; the MoG input is the output of the
    last AR step, the temporal output and the latent of the AR codebooks."""
    model = _tiny_model().eval()
    model.config.amortization_divisor = 1  # every frame, in order
    phone_tokens, emb_idx, audio, spk = _tiny_batch()

    sem_logits, out, rand_idx = model(phone_tokens, emb_idx, audio, None, spk, audio)
    dep_h, temp_h = out
    assert dep_h.shape == (2, 20, 2, 32)  # the 2 AR steps
    assert torch.equal(rand_idx, torch.arange(20))
    phone_emb = model.extract_phoneme_embeddings(phone_tokens)
    phone_emb = model.reorder_phone_emb(phone_emb, emb_idx)
    _, h, _ = model.forward_temp_former(phone_emb, audio, spk)
    assert torch.allclose(temp_h, h, atol=1e-5)

    labels = audio[:, 1:, 1:][..., rand_idx]
    labels[0, :, :3] = 17  # padding
    losses = model.mog_loss(out, labels)
    assert torch.isfinite(losses["loss"])
    assert 0 <= losses["ar_acc"] <= 1
    assert torch.allclose(
        losses["loss"], losses["c_loss"] + losses["k_loss"] + losses["ar_loss"]
    )

    # the MoG step predicts the last codebook from this input and the latent of the AR
    # codebooks
    tokens = labels.transpose(1, 2)
    valid = (tokens < 16).all(-1)
    tokens0 = tokens.masked_fill(~valid.unsqueeze(-1), 0)
    prefix, target = model.mog_targets(tokens0)
    latents = model.mog_codebooks[torch.arange(3), tokens0]
    assert torch.allclose(prefix, latents[..., :2, :].sum(-2))
    assert torch.allclose(target, latents[..., 2, :])
    x = dep_h[:, :, -1] + model.mog_step_embedding + model.mog_temp_proj(temp_h)
    x = x + model.mog_prefix_proj(prefix)
    expected = model.mog_head.loss(
        x.flatten(0, 1), target.flatten(0, 1), valid.flatten().float()
    )
    num_tokens = valid.sum()
    assert torch.allclose(losses["c_loss"], expected["c_loss"] / num_tokens, atol=1e-6)

    # the AR head predicts the tokens of its codebooks
    logits = torch.einsum("bscd,cdk->bkcs", dep_h, model.mog_ar_head)
    ar_labels = tokens[..., :2].masked_fill(~valid.unsqueeze(-1), -100)
    expected_ar = torch.nn.functional.cross_entropy(
        logits, ar_labels.transpose(1, 2), ignore_index=-100
    )
    assert torch.allclose(losses["ar_loss"], expected_ar, atol=1e-6)

    # the trainer adds the semantic loss; every parameter gets a gradient (DDP)
    (losses["loss"] + sem_logits.float().mean()).backward()
    unused = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not unused, unused


def test_mog_generation():
    """Two AR codebooks from depth transformer steps, then the MoG step without one."""
    from unittest.mock import patch

    from voxtream.utils.model import index_causal_mask

    model = _tiny_model().eval()
    model.setup_caches(max_batch_size=2, dtype=torch.float32)
    last_h = torch.randn(2, 32)
    c0_embed = model._embed_audio(0, torch.tensor([[5], [7]]))
    with patch.object(
        model.dep_former, "forward", wraps=model.dep_former.forward
    ) as forward:
        tokens = model.generate_acoustic_mog(last_h, c0_embed, torch.float32)
    assert forward.call_count == 2  # the AR steps only
    assert tokens.shape == (2, 3)
    assert tokens.max() < 16

    # step by step
    model.dep_former.reset_caches()
    curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
    pos = torch.arange(2).unsqueeze(0).repeat(2, 1)
    expected, prefix = [], torch.zeros(2, 8)
    for level in range(2):
        mask = index_causal_mask(model.dep_former_causal_mask, pos)
        h = model.dep_former(curr_h, input_pos=pos, mask=mask)[:, -1]
        token = (h @ model.mog_ar_head[level])[:, :16].argmax(-1)
        expected.append(token)
        prefix = prefix + model.mog_codebooks[level, token]
        curr_h = model._embed_audio(level + 1, token).unsqueeze(1)
        pos = pos[:, -1:] + 1
    x = h + model.mog_step_embedding + model.mog_temp_proj(last_h)
    z = model.mog_head.sample(x + model.mog_prefix_proj(prefix))
    codebooks = RVQCodebooks(model.mog_codebooks[2:])
    expected.append(residual_quantize(z, codebooks)[:, 0])
    assert torch.equal(tokens, torch.stack(expected, 1))

    # CFG: the conditional and the unconditional rows share the decoded codebooks
    tokens = model.generate_acoustic_mog(
        last_h,
        c0_embed[:1].repeat(2, 1, 1),
        torch.float32,
        cfg_gamma=1.5,
        ar_cfg_gamma=2.0,
    )
    assert tokens.shape == (1, 3)
    assert tokens.max() < 16


def test_mog_decoding_kwargs():
    from types import SimpleNamespace

    from voxtream.model import Model

    config = SimpleNamespace(mog_cfg_gamma=1.75, cfg_ac_gamma=2.0, temperature=0.9)
    assert Model.mog_decoding_kwargs(config, cfg=True) == dict(
        cfg_gamma=1.75, ar_cfg_gamma=2.0, ar_temperature=0.9
    )
    kwargs = Model.mog_decoding_kwargs(config, cfg=False)
    assert kwargs["cfg_gamma"] is None and kwargs["ar_cfg_gamma"] is None


def test_trainer_loads_pretrained_and_freezes(tmp_path):
    """Pre-trained depth transformer weights are loaded and the parameters named in
    `freeze` keep them, the rest train."""
    from dataclasses import asdict

    from omegaconf import OmegaConf
    from safetensors.torch import save_file

    from voxtream.model import Model
    from voxtream.trainer import Trainer

    torch.manual_seed(0)
    ar = _tiny_config(dep_former_head="ar")
    pretrained = Model(ar)
    with torch.no_grad():
        pretrained.audio_head.normal_()
    weights = {
        k: v.clone()
        for k, v in pretrained.state_dict().items()
        if k.startswith(("dep_former.", "audio_embeddings.", "audio_head"))
    }
    path = tmp_path / "dep_former.safetensors"
    save_file(weights, str(path))

    config = OmegaConf.create(
        dict(
            model=asdict(ar),
            compile_forward=False,
            dep_former_weight_path=str(path),
            dep_former_name=None,
            model_repo=None,
            model_weight_path=None,
            freeze=["audio_embeddings", "audio_head"],
        )
    )
    module = Trainer(config)
    state = module.model.state_dict()
    for k, v in weights.items():
        assert torch.equal(state[k], v), k
    trainable = {n for n, p in module.model.named_parameters() if p.requires_grad}
    assert "audio_head" not in trainable
    assert "audio_embeddings.weight" not in trainable
    assert any(n.startswith("dep_former.") for n in trainable)
    assert "sem_head.weight" in trainable
