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

### Seamless (dynamic speaking rate)

The dataset will be downloaded automatically. The speaking rate is changed every 80 ms frame following the per-word rate of the reference recording (clipped to [2-6] syllables per second), so `--speaking-rate` is ignored. Generation only needs the alignments shipped with the dataset:

```bash
python run.py -d seamless -o /path/to/output/directory
```

To also compute speaking rate metrics (Correlation and MAE between the reference and generated speaking-rate curves), the generated audio has to be aligned with the Seamless UnitY2 forced aligner. Clone the aligner repository and install its extra dependencies:

```bash
git clone https://github.com/PalabraAI/forced-aligners-bench
pip install loguru overrides resampy sentencepiece matplotlib
```

Then pass the repository path to the test (aligner checkpoints will be downloaded on the first run):

```bash
python run.py -d seamless -o /path/to/output/directory --aligner-root /path/to/forced-aligners-bench
```

Correlation and MAE are computed between the reference and generated speaking-rate curves (80 ms frames, generated curve resampled to the reference length, both Gaussian-smoothed). Per-file scores are saved to `speaking_rate.csv` in the output directory.

* Note: Make sure the `$PYTHONPATH` variable contains a repository root directory.
