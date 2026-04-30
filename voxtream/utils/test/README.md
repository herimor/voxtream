# Emilia speaking-rate test

## Instructions

Download WavLM ECAPA-TDNN checkpoint follow [instructions](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/eval#objective-evaluation-on-generated-results) from F5-TTS repository. Place the checkpoint in `./wavlm-large/` directory in the repository root.

### Emilia speaking rate

Use the following command to run the test. The Emilia speaking-rate dataset, Whisper-ASR and UTMOS models will be downloaded automatically and stored in the `~/.cache` directory. Specify output directory to store generated audio files and target speaking rate from [1-7] range:

```bash
python run.py -o /path/to/output/directory --speaking-rate 4.0 --prompt-enhancement
```

### SEED TTS EN

Download dataset following [instructions](https://github.com/BytedanceSpeech/seed-tts-eval?tab=readme-ov-file#dataset) from the official repository. Specify dataset directory, dataset name, and output directory to store generated audio files:

```bash
python run.py -d seedtts-en -dd /path/to/seedtts_testset/en -o /path/to/output/directory --prompt-enhancement
```

* Note: Make sure the `$PYTHONPATH` variable contains a repository root directory.
