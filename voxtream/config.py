import json
from dataclasses import dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass
class SpeechGeneratorConfig:
    sil_token: int
    bos_token: int
    eos_token: int
    unk_token: int
    eop_token: int
    num_codebooks: int
    num_phones_per_frame: int
    audio_delay_frames: int
    temperature: float
    topk: int
    top_p: float
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
    phoneme_index_map: dict[str, list[int]]
    phoneme_dict_name: str
    max_prompt_sec: int
    min_prompt_sec: int
    max_phone_tokens: int
    cache_prompt: bool
    punct_map: dict[str, int]
    phonemizer: str
    spk_rate_window_sec: float
    cfg_gamma: float
    cfg_ac_gamma: float
    text_context: str
    text_context_length: int
    spk_proj_weight: float
    audio_pad_token: int
    enhance_prompt: bool
    sidon_se_reload_model: bool
    reset_streaming_state: bool
    hf_token: str
    apply_vad: bool
    min_speech_seg_sec: float
    min_look_ahead_phones: int
    frame_repeat_counter: int


def resolve_data_path(path: str | Path, package_relative_path: str) -> Path:
    """Resolve a user path, repo checkout path, or packaged data resource."""
    candidate = Path(path)
    if candidate.exists():
        return candidate

    repo_candidate = Path(__file__).resolve().parents[1] / candidate
    if repo_candidate.exists():
        return repo_candidate

    resource = resources.files("voxtream").joinpath(package_relative_path)
    if resource.is_file():
        with resources.as_file(resource) as resource_path:
            return resource_path

    raise FileNotFoundError(
        f"Could not find {path}. Pass an explicit path or reinstall voxtream with package data."
    )


def load_json(path: str | Path, package_relative_path: str) -> object:
    with resolve_data_path(path, package_relative_path).open() as f:
        return json.load(f)


def _json_object(raw: object, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object")
    return raw


def load_generator_config(path: str | Path = "configs/generator.json") -> SpeechGeneratorConfig:
    raw = _json_object(load_json(path, "configs/generator.json"), "Generator config")

    field_names = {field.name for field in fields(SpeechGeneratorConfig)}
    missing = sorted(field_names - raw.keys())
    unknown = sorted(raw.keys() - field_names)
    if missing:
        raise ValueError(f"Generator config missing required fields: {missing}")
    if unknown:
        raise ValueError(f"Generator config has unknown fields: {unknown}")

    config = SpeechGeneratorConfig(**raw)
    validate_generator_config(config)
    return config


def validate_generator_config(config: SpeechGeneratorConfig) -> None:
    positive_int_fields = (
        "num_codebooks",
        "num_phones_per_frame",
        "mimi_sr",
        "mimi_vocab_size",
        "mimi_frame_ms",
        "spk_enc_sr",
        "max_prompt_sec",
        "min_prompt_sec",
        "max_phone_tokens",
        "text_context_length",
        "audio_pad_token",
        "min_look_ahead_phones",
        "frame_repeat_counter",
    )
    for name in positive_int_fields:
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if config.max_prompt_sec < config.min_prompt_sec:
        raise ValueError("max_prompt_sec must be greater than or equal to min_prompt_sec")
    if config.max_audio_length_ms <= 0:
        raise ValueError("max_audio_length_ms must be positive")
    if config.audio_delay_frames < 0:
        raise ValueError("audio_delay_frames must be non-negative")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < config.top_p <= 1:
        raise ValueError("top_p must be in the range (0, 1]")
    if config.topk <= 0:
        raise ValueError("topk must be positive")
    if config.spk_rate_window_sec <= 0:
        raise ValueError("spk_rate_window_sec must be positive")
    if config.min_speech_seg_sec < 0:
        raise ValueError("min_speech_seg_sec must be non-negative")

    for mapping_name in ("phoneme_index_map", "punct_map"):
        mapping = getattr(config, mapping_name)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"{mapping_name} must be a non-empty object")

    for repo_field in ("model_repo", "mimi_repo", "spk_enc_repo"):
        value = getattr(config, repo_field)
        if not isinstance(value, str) or "/" not in value:
            raise ValueError(f"{repo_field} must be a Hugging Face or torch.hub repo id")


def load_speaking_rate_config(
    path: str | Path = "configs/speaking_rate.json",
) -> dict[str, dict[str, list[int] | float]]:
    raw = _json_object(
        load_json(path, "configs/speaking_rate.json"), "Speaking-rate config"
    )
    if not raw:
        raise ValueError("Speaking-rate config must be a non-empty JSON object")

    parsed: dict[str, dict[str, list[int] | float]] = {}
    for rate, params in raw.items():
        if not isinstance(params, dict):
            raise ValueError(f"Speaking-rate entry {rate!r} must be an object")
        duration_state = params.get("duration_state")
        weight = params.get("weight")
        cfg_gamma = params.get("cfg_gamma")
        if not isinstance(duration_state, list) or not duration_state:
            raise ValueError(f"duration_state for speaking rate {rate!r} must be a non-empty list")
        if not all(isinstance(value, int) and value > 0 for value in duration_state):
            raise ValueError(f"duration_state for speaking rate {rate!r} must contain positive integers")
        duration_state_values = [int(value) for value in duration_state]
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(f"weight for speaking rate {rate!r} must be positive")
        if not isinstance(cfg_gamma, (int, float)) or cfg_gamma <= 0:
            raise ValueError(f"cfg_gamma for speaking rate {rate!r} must be positive")
        parsed[str(rate)] = {
            "duration_state": duration_state_values,
            "weight": float(weight),
            "cfg_gamma": float(cfg_gamma),
        }
    return parsed
