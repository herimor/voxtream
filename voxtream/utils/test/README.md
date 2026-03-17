# Emilia speaking-rate test

## Instructions

Download WavLM ECAPA-TDNN checkpoint follow [instructions](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/eval#objective-evaluation-on-generated-results) from F5-TTS repository. Place the checkpoint in `./wavlm-large/` directory in the repository root.

Use the following command to run the test. The Emilia speaking-rate dataset, Whisper-ASR and UTMOS models will be downloaded automatically and stored in the `~/.cache` directory. Specify output directory to store generated audio files:

```bash
python run.py -o /path/to/output/directory
```

* Note: Make sure the `$PYTHONPATH` variable contains a repository root directory.
