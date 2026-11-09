import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, Tuple

import torch
from huggingface_hub import hf_hub_download
from moshi.models import MimiModel, loaders

from voxtream.config import SpeechGeneratorConfig
from voxtream.model import Model, ModelConfig
from voxtream.utils.dataset.phoneme_vocab import PhonemeVocab
from voxtream.utils.generator.flat_text import FlatPhonemeVocab
from voxtream.utils.model import load_model_weights


def resolve_model_file(
    config: SpeechGeneratorConfig, name: str, local_path: str = None
) -> str:
    """
    `local_path` if given, else the file `name` of `config.model_repo`: a local
    directory holding the files of the Hugging Face repo, or the repo itself.
    """
    if local_path is not None:
        return local_path
    if Path(config.model_repo).is_dir():
        return str(Path(config.model_repo) / name)
    return hf_hub_download(config.model_repo, name, token=config.hf_token)


def build_generator_model(
    config: SpeechGeneratorConfig,
    weights: bool = True,
) -> Tuple[Model, PhonemeVocab | FlatPhonemeVocab, str]:
    """The model of the config, unloaded, with its phoneme vocabulary and the path
    of its weights (a local checkpoint or the file of the model repo).

    ``weights=False`` skips resolving the weights (the Hub download of
    ``model_name`` when ``model_weight_path`` is None) and returns None in their
    place, for callers that load the model from somewhere else.
    """
    model_config_path = resolve_model_file(
        config, config.model_config_name, config.model_config_path
    )
    with open(model_config_path) as f:
        model_config_params = json.load(f)

    field_names = {f.name for f in fields(ModelConfig)}
    model_config_params = {
        k: v for k, v in model_config_params.items() if k in field_names
    }
    model_config = ModelConfig(**model_config_params)
    if config.audio_window_size is not None:
        model_config.audio_window_size = config.audio_window_size

    vocab_path = resolve_model_file(
        config, config.phoneme_vocab_name, config.phoneme_vocab_path
    )
    if model_config.flat_phone_vocab:
        if config.phone_kv_cache:
            raise ValueError(
                "phone_kv_cache needs the factored phoneme vocabulary; set it to false "
                "for a model with phone_vocab_size"
            )
        vocab = FlatPhonemeVocab.load(vocab_path, model_config.phone_vocab_size)
    else:
        vocab = PhonemeVocab.load(vocab_path)
        # a stale phoneme_vocab.json would quietly embed the wrong rows rather than fail
        trained = [
            model_config.phone_root_vocab_size,
            *model_config.phone_attr_vocab_sizes,
        ]
        assert list(vocab.vocab_sizes) == trained, (
            f"phoneme vocabulary is {list(vocab.vocab_sizes)}, the checkpoint was "
            f"trained with {trained}"
        )
    model = Model(model_config)

    weight_path = None
    if weights:
        weight_path = resolve_model_file(
            config, config.model_name, config.model_weight_path
        )
    return model, vocab, weight_path


def load_generator_model(
    config: SpeechGeneratorConfig,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    gguf: Dict = None,
) -> Tuple[Model, PhonemeVocab | FlatPhonemeVocab]:
    """Build the model, load its weights and set up its caches.

    Args:
        gguf: run the converted transformer stacks through llama.cpp (CPU only):
            ``dir`` (the converted files, see
            ``voxtream-convert-gguf``; default: ``gguf_dir`` of the config, else the
            model repo), ``threads``. The PyTorch
            weights of those stacks are dropped once they are swapped out.
    """
    model, vocab, weight_path = build_generator_model(config)

    state_dict = load_model_weights(weight_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert (
        len(missing) == 0 and len(unexpected) == 0
    ), f"Model weights mismatch!\nmissing: {missing}\nunexpected: {unexpected}"
    del state_dict

    model = model.eval().to(device, dtype=dtype)
    if model.config.dep_former_head == "mog":
        model.prepare_mog_decoding()
    model.sink_attention.config.prompt_bias = config.prompt_attention_bias
    model.sink_attention.config.prompt_bias_dtype = dtype
    model.sink_attention.config.prompt_bias_start = round(
        config.prompt_attention_bias_start_sec * 1000 / config.mimi_frame_ms
    )
    model.sink_attention.config.tail_window_divisor = config.sink_tail_window_divisor

    if gguf is not None:
        from voxtream.utils.gguf.backend import patch_model_with_gguf, resolve_gguf_dir

        assert device == "cpu", "the GGUF backend runs on the CPU"
        patch_model_with_gguf(
            model,
            batch_size=batch_size,
            gguf_dir=resolve_gguf_dir(config, gguf.get("dir")),
            n_threads=gguf.get("threads"),
            n_gpu_layers=gguf.get("n_gpu_layers", 0),
            # positions of the temporal transformer stay inside the audio window
            max_positions=model.config.audio_window_size + 1,
            max_prompt_len=int(config.max_prompt_sec * 1000 / config.mimi_frame_ms) + 1,
            prompt_bias=config.prompt_attention_bias,
        )
    model.setup_caches(
        max_batch_size=batch_size,
        dtype=dtype,
        phone_cache_len=config.phone_cache_len,
        # the text budget plus the `[EOS]` row a stream closes on
        max_phone_rows=config.max_phone_tokens + 1,
    )

    return model, vocab


def load_mimi_model(
    config: SpeechGeneratorConfig,
    device: str,
    dtype: torch.dtype,
    num_codebooks: int,
    gguf: Dict = None,
) -> MimiModel:
    """
    Args:
        gguf: as for `load_generator_model`; the decoder transformer then runs
            through llama.cpp when the converted files hold it
    """
    mimi_weight = hf_hub_download(config.mimi_repo, config.mimi_name)

    mimi = (
        loaders.get_mimi(
            filename=mimi_weight, device=device, num_codebooks=num_codebooks
        )
        .eval()
        .to(dtype=dtype)
    )
    if gguf is not None:
        from voxtream.utils.gguf.backend import patch_mimi_with_gguf, resolve_gguf_dir

        patch_mimi_with_gguf(
            mimi,
            resolve_gguf_dir(config, gguf.get("dir")),
            n_threads=gguf.get("threads"),
            n_gpu_layers=gguf.get("n_gpu_layers", 0),
        )
    return mimi


def load_speaker_encoder(
    config: SpeechGeneratorConfig, device: str, dtype: torch.dtype
) -> torch.nn.Module:
    model = torch.hub.load(
        config.spk_enc_repo,
        config.spk_enc_model,
        model_name=config.spk_enc_model_name,
        train_type=config.spk_enc_train_type,
        dataset=config.spk_enc_dataset,
        trust_repo=True,
        verbose=False,
    ).to(device, dtype=dtype)
    model.spec.float()
    model.bn.float()
    model.eval()

    return model
