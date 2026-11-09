from dataclasses import dataclass
from typing import Dict


@dataclass
class SpeechGeneratorConfig:
    """
    Generator settings. Every field is set by the config file (`configs/generator*.json`);
    none has a default here.
    """

    # -- model files ------------------------------------------------------------------
    # Hugging Face repo of the model, or a local directory holding the same files
    model_repo: str
    model_name: str
    model_config_name: str
    # Phoneme vocabulary of the model: `phoneme_vocab.json` (factored vocabulary) or
    # `phoneme_to_token.json` (flat vocabulary of VoXtream2 and VoXtream2.1)
    phoneme_vocab_name: str
    # Local files in place of `model_name`, `model_config_name` and
    # `phoneme_vocab_name` of `model_repo` (a Lightning checkpoint is accepted for the
    # weights). The phoneme vocabulary must be the one the store was built with
    # (`emilia_train_arrays.py` writes it beside the arrays): the root ids are
    # positions in it, so a vocabulary that is not the checkpoint's own embeds the
    # wrong rows. The sizes are checked against the model config on load.
    model_weight_path: str
    model_config_path: str
    phoneme_vocab_path: str
    hf_token: str

    # -- audio codec and speaker encoder ------------------------------------------------
    num_codebooks: int
    audio_delay_frames: int
    audio_pad_token: int
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

    # -- acoustic prompt ----------------------------------------------------------------
    max_prompt_sec: int
    min_prompt_sec: int
    # Read the Mimi codes and the speaker embedding of a prompt from, else write them
    # to, `<stem>.prompt.npy` next to the audio
    cache_prompt: bool
    enhance_prompt: bool
    sidon_se_reload_model: bool
    apply_vad: bool
    min_speech_seg_sec: float
    normalize_prompt_loudness: bool
    prompt_loudness_lufs: float

    # -- text -----------------------------------------------------------------------------
    # "espeak" runs the espeak-ng command-line tool, one process per call: about 10 ms
    # for a word of a text stream, most of the text-side cost of a full-stream frame.
    # "espeak-lib" calls the espeak-ng shared library instead, well under a
    # millisecond for the same phonemes (`ESpeak(library="espeak-lib")`).
    phonemizer: str
    max_phone_tokens: int
    # Context appended to a chunk of a text stream for its phonemization, and the
    # number of phoneme characters it adds, cut off again
    text_context: str
    text_context_length: int
    # Phonemes a text stream must have past the pointer before a frame is generated
    min_look_ahead_phones: int
    # Phoneme transformer of the generator, factored vocabulary only. With the KV cache
    # a chunk of text encodes the new rows and the rows before them whose output can
    # still change (the look-ahead receptive field, `num_layers * max_look_ahead`
    # rows) instead of the whole sequence, under the largest look-ahead mask. False
    # re-encodes the whole sequence at every chunk with the mask matched to the text
    # available past the pointer.
    phone_kv_cache: bool
    # slots of that cache per layer; holds the window, the receptive field and a chunk,
    # and bounds the positions the model sees however long the text stream runs
    phone_cache_len: int

    # -- sampling -------------------------------------------------------------------------
    temperature: float
    topk: int
    top_p: float
    # Classifier-free guidance of the semantic token (None: no guidance), of the
    # acoustic codebooks of the AR head and of the MoG step (`dep_former_head: mog`)
    cfg_gamma: float
    cfg_ac_gamma: float
    mog_cfg_gamma: float
    # Phoneme states of the semantic head: {state: [pointer shift, phonemes per frame]}
    phoneme_index_map: Dict
    num_phones_per_frame: int
    # Seconds of speaking-rate history the duration guidance is measured over
    spk_rate_window_sec: float

    # -- generation length and pointer rules -------------------------------------------
    max_audio_length_ms: int
    # Frames the pointer may hold one phoneme before it is pushed to the next one
    frame_repeat_counter: int
    # Frames the pointer may hold the last phoneme, once the text has ended, before the
    # step onto `[EOS]` is taken on the model's behalf and generation closes.
    #
    # A crop that ends on a sentence supervises that step, so a model trained on such a
    # store takes it itself and this is only the net for when it does not. A model
    # trained before that never takes it: the step was unsupervised, holding the last
    # phoneme is what a pause before the next sentence looks like, and the model opens
    # a sentence the text never asked for, which this then clips mid-word. For a flat
    # vocabulary the rule applies from the second-to-last phoneme, as it was released.
    final_phone_hold_frames: int
    # Silent frames tolerated at the start before the draw is abandoned and taken again.
    # 0 turns the retry off and streams the first draw whatever it does.
    #
    # The model occasionally parks the pointer on the opening phoneme and renders
    # silence. It is a sampling failure rather than anything the data teaches - a crop
    # starts its content on a word onset - and the same utterance comes out clean on
    # another draw, so the cure is to notice it within the first second and draw again.
    # Measured on the seed-tts set: openings past 1 s fall from 63 to 4 and past 2 s
    # from 13 to 0, for no change in WER.
    start_retry_frames: int
    # How many draws to take in total before keeping one regardless.
    start_retry_attempts: int
    # Silent frames kept in front of the first frame that carries sound; the rest of the
    # opening silence is dropped, which is free because nothing audible is in it.
    start_lead_in_frames: int
    # Consecutive forced pointer steps (`frame_repeat_counter` kicks, with no step of the
    # model's own between them) after which a stalled utterance is restarted: the audio
    # context is rebuilt from the prompt alone and the pointer rewound to where the stall
    # began. 0 disables the rule.
    #
    # In long-form generation the model sometimes drops mid-utterance into a silent state
    # it never leaves on its own: the semantic token repeats in most frames, the audio is
    # a faint hum 20 dB below the speech, and the pointer only moves by the kick, silently
    # consuming the text at one state per kick. A rebuild of the sliding window sometimes
    # clears it and sometimes not, since the tail it keeps carries the silence into the
    # new context; a context of the prompt alone is the state generation starts from,
    # which speaks. Only one restart per stall position, so a spot the model cannot speak
    # is crawled past as before rather than looped on.
    stall_kicks: int

    # -- temporal transformer context ------------------------------------------------------
    # Frames the temporal transformer cache holds, the acoustic prompt included, in
    # place of the `audio_window_size` of the model config (None keeps it). Lim et al.
    # (arXiv:2610.19553) run VoXtream2 at 375 of its 625 to hold long-form WER: the
    # cache then never climbs to the positions a training crop reaches only at its
    # very end.
    audio_window_size: int
    # Constant added to the pre-softmax attention logit of every acoustic-prompt key, in
    # every layer and head of the temporal transformer, for every generated frame
    # (arXiv:2610.19553, b = 1.5). None leaves the attention as trained.
    prompt_attention_bias: float
    # Seconds of generated audio after which `prompt_attention_bias` applies; until then
    # the prompt keys are attended as trained. The bias holds long-form generation to the
    # prompt once the generated context fills the window, but on short utterances it
    # trades UTMOS for speaker similarity (seed-tts-en, bias 3.0: UTMOS 3.89 -> 3.72),
    # so a later start keeps the opening as trained. 0 applies it from the first frame.
    prompt_attention_bias_start_sec: float
    # Divisor of the window giving the recent frames kept at a cache rebuild
    # (`SinkAttentionConfig.tail_window_divisor`, 2 as trained: half the window). A
    # larger value re-anchors generation on the prompt with less of its own recent
    # output behind it, at the price of more frequent rebuilds.
    sink_tail_window_divisor: int

    # -- CPU backend ------------------------------------------------------------------------
    # Run the transformer stacks through llama.cpp when there is no GPU, from the q8_0
    # GGUF files written by `voxtream-convert-gguf`, read from `gguf_dir`, else from
    # `model_repo`. False runs them in PyTorch on the CPU, as does a missing
    # llama-cpp-python.
    gguf: bool
    gguf_dir: str
    # CPU threads of PyTorch and llama.cpp. None gives PyTorch up to 8 cores and
    # llama.cpp the performance cores.
    cpu_threads: int
