from dataclasses import dataclass
from typing import Dict


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
    phoneme_index_map: Dict
    phoneme_dict_name: str
    max_prompt_sec: int
    min_prompt_sec: int
    max_phone_tokens: int
    cache_prompt: bool
    punct_map: Dict
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
