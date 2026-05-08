import json
from dataclasses import fields
from typing import Any, Tuple, cast

import torch
from huggingface_hub import hf_hub_download
from moshi.models import MimiModel, loaders
from safetensors.torch import load_file

from voxtream.config import SpeechGeneratorConfig
from voxtream.model import Model, ModelConfig


def load_generator_model(
    config: SpeechGeneratorConfig,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
) -> Tuple[Model, dict[str, Any]]:
    model_config_path = hf_hub_download(
        config.model_repo, config.model_config_name, token=config.hf_token
    )
    phoneme_dict_path = hf_hub_download(
        config.model_repo, config.phoneme_dict_name, token=config.hf_token
    )

    with open(phoneme_dict_path) as f:
        phone_to_token = json.load(f)

    with open(model_config_path) as f:
        model_config_params = json.load(f)

    field_names = {f.name for f in fields(ModelConfig)}
    model_config_params = {
        k: v for k, v in model_config_params.items() if k in field_names
    }
    model_config = ModelConfig(**model_config_params)
    model = Model(model_config)

    model_weight_path = hf_hub_download(
        config.model_repo, config.model_name, token=config.hf_token
    )
    state_dict = load_file(model_weight_path)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert (
        len(missing) == 0 and len(unexpected) == 0
    ), f"Model weights mismatch!\nmissing: {missing}\nunexpected: {unexpected}"

    model = model.eval().to(device, dtype=dtype)
    model.setup_caches(max_batch_size=batch_size, dtype=dtype)

    return model, phone_to_token


def load_mimi_model(
    config: SpeechGeneratorConfig, device: str, dtype: torch.dtype, num_codebooks: int
) -> MimiModel:
    mimi_weight = hf_hub_download(config.mimi_repo, config.mimi_name)

    return (
        loaders.get_mimi(
            filename=mimi_weight, device=device, num_codebooks=num_codebooks
        )
        .eval()
        .to(dtype=dtype)
    )


def load_speaker_encoder(
    config: SpeechGeneratorConfig, device: str, dtype: torch.dtype
) -> torch.nn.Module:
    if config.spk_enc_repo != "IDRnD/ReDimNet":
        raise ValueError(
            "Refusing to load untrusted speaker encoder repo. "
            "Only IDRnD/ReDimNet is allowed by default."
        )
    model = cast(
        torch.nn.Module,
        torch.hub.load(
            config.spk_enc_repo,
            config.spk_enc_model,
            model_name=config.spk_enc_model_name,
            train_type=config.spk_enc_train_type,
            dataset=config.spk_enc_dataset,
            trust_repo=True,
            verbose=False,
        ),
    ).to(device, dtype=dtype)
    model.spec.float()
    model.bn.float()
    model.eval()

    return model
