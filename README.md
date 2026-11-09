# VoXtream2: Full-stream TTS with dynamic speaking rate control

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2603.13518)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/herimor/voxtream2.5)

VoXtream2.5, a zero-shot full-stream TTS model with dynamic speaking-rate control that can be updated mid-utterance on the fly.

Check our [blogpost](https://herimor.github.io/voxtream2.5) about latest model updates. For audio examples, see our [demo page](https://herimor.github.io/voxtream2).

Try VoXtream2 in your browser on HuggingFace 🤗 [space](https://huggingface.co/spaces/herimor/voxtream2.5-q8).

## Key features

- **Dynamic speed control**: Duration distribution guidance allows for fine-grained speaking rate control, which can be adjusted as the model generates speech.
- **Hybrid depth transformer head**: 32 Mimi codebooks are generated in 4 steps (3 AR and a single Mixture of Gaussians step), combining high speech quality and low generation complexity.
- **Streaming performance**: Works **10x** times faster than real-time and achieves **30 ms** first packet latency in a full-stream on RTX3090 GPU. Real-time streaming support on a single CPU core via GGUF (int8) and llama.cpp.
- **Long-form generation**: Generation runs for up to 5 minutes with a WER below 2% measured on the LibriSpeech-long benchmark.

## Updates

- `2026/11`: Released new model [VoXtream2.5](https://huggingface.co/herimor/voxtream2.5) with significant improvement over generation speed and objective metrics.
- `2026/10`: Updated speaking rate control mechanism. Released new checkpoints [VoXtream2.1](https://huggingface.co/herimor/voxtream2.1) with improved objective metrics.
- `2026/04/30`:
    - Added a dynamic speaking rate control interactive demo. You can now adjust the speaking rate as the model produces speech in real-time. Run `voxtream-app` locally after installing the package or check our HuggingFace [space](https://huggingface.co/spaces/herimor/voxtream2).
    - Added cache reset in SynkAttention. Fixed the bug that caused the model to generate noise by relying on an invalid prompt cache.
- `2026/04/08`: Added a frame repeat counter. Reduces hallucinations caused by models getting stuck in the same frame. Controlled by the `frame_repeat_counter` parameter in `SpeechGeneratorConfig`. Recommended value (12-25), lower values for stricter control.
- `2026/03`: We released VoXtream2.
- `2026/01`: VoXtream is accepted for an oral presentation at ICASSP 2026.
- `2025/09`: We released VoXtream. Now available at [voxtream](https://github.com/herimor/voxtream/tree/voxtream) branch.

## Installation

### eSpeak NG phonemizer

```bash
# For Debian-like distribution (e.g. Ubuntu, Mint, etc.)
apt-get install espeak-ng
# For RedHat-like distribution (e.g. CentOS, Fedora, etc.) 
yum install espeak-ng
# For MacOS
brew install espeak-ng
```

### Pip package
```bash
pip install "voxtream>=0.2.6"
```

### Models

| Config | Model | Phoneme vocabulary | Depth transformer head |
| :- | :- | :- | :- |
| `configs/generator.json` | [VoXtream2.5](https://huggingface.co/herimor/voxtream2.5) | factored: phoneme root, stress, length, punctuation and end of word | 3 codebooks AR, then all the rest in one Mixture of Gaussians step |
| `configs/generator_voxtream2.1.json` | [VoXtream2.1](https://huggingface.co/herimor/voxtream2.1) | flat, one token per phoneme | 15 codebooks AR |

Every setting of a model is in its generator config (`voxtream/config.py` documents them); pass another config with `-c`/`--config` to any of the commands below. `model_repo` is a Hugging Face repo or a local directory with the same files.

## Usage

* Prompt audio: a file containing 3-10 seconds of the target voice. The maximum supported length is 20 seconds (longer audio will be trimmed).
* Text: What you want the model to say. Text longer than `max_phone_tokens` phonemes of the generator config is trimmed.
* Speaking rate (optional): target speaking rate in syllables per second.

**Notes**: 
* The model was tested on Ubuntu 22.04, CUDA 12.8 and PyTorch 2.8.
* The model requires 3Gb of VRAM (enabling prompt speech enhancement adds 2Gb).
* Generation length is limited by `max_audio_length_ms` of the generator config: 5 minutes for VoXtream2.5, 1 minute for VoXtream2.1.
* The initial run may take a bit longer to download model weights and warmup model graph.
* If you experience problems with CUDAGraphs please check this [issue](https://github.com/herimor/voxtream/issues/8).

### Command line

#### Output streaming
```bash
voxtream \
    --prompt-audio assets/audio/english_male.wav \
    --text "In general, some method is then needed to evaluate each approximation." \
    --output "output_stream.wav"
```

#### Full streaming (slow speech, 2 syllables per second)
```bash
voxtream \
    --prompt-audio assets/audio/english_female.wav \
    --text "Staff do not always do enough to prevent violence." \
    --output "full_stream_2sps.wav" \
    --full-stream \
    --spk-rate 2.0
```

#### Acoustic prompt enhancement
```bash
voxtream \
    --prompt-audio assets/test/english_male.wav \
    --text "In general, however, some method is then needed to evaluate each approximation." \
    --output "output_enhanced.wav" \
    --prompt-enhancement
```

### Python API

```python
import json
from itertools import repeat
from pathlib import Path

import numpy as np
import soundfile as sf

from voxtream.utils.generator import (
    set_seed,
    text_generator,
)
from voxtream.generator import SpeechGenerator, SpeechGeneratorConfig


set_seed()
with open('configs/generator.json') as f:
    config = SpeechGeneratorConfig(**json.load(f))

with open('configs/speaking_rate.json') as f:
    spk_rate_config = json.load(f)

speech_generator = SpeechGenerator(config, spk_rate_config)

# Output streaming, no speaking rate control
speech_stream = speech_generator.generate_stream(
    prompt_audio_path=Path('assets/audio/english_male.wav'),
    text="In general, however, some method is then needed to evaluate each approximation.",
)

audio_frames = [audio_frame for audio_frame, _ in speech_stream]
sf.write('output_stream.wav', np.concatenate(audio_frames), config.mimi_sr)

# Full streaming & fixed speaking rate control (2 syllables per second)
speech_stream = speech_generator.generate_stream(
    prompt_audio_path=Path('assets/audio/english_female.wav'),
    text=text_generator("Staff do not always do enough to prevent violence."),
    speaking_rate=repeat(2.0),
)

audio_frames = [audio_frame for audio_frame, _ in speech_stream]
sf.write('full_stream_2sps.wav', np.concatenate(audio_frames), config.mimi_sr)
```

### Gradio demo

To start a gradio web-demo run:
```bash
voxtream-app
```

### Websocket

To start a websocket server run:
```bash
voxtream-server [--config configs/generator.json] [--port 7860]
```

To send a request to the server run:
```bash
python voxtream/client.py [--prompt-audio assets/audio/english_male.wav] [--text "..."]
```

It sends a path to the audio prompt and the text word by word to the server and immediately plays audio from the output stream (`--output file.wav` saves it instead).

### Evaluation

To reproduce evaluation metrics from the paper check [evaluation](voxtream/utils/test/README.md) section.

Zero-shot TTS performance on SEED test-en set:

| Model   | WER (%)           | SPK-SIM | UTMOS   |
| :-:     | :-:                | :-:     | :-:   |
| [VoXtream](https://huggingface.co/herimor/voxtream)       | 3.24 | 0.524 | 3.92 |
| [VoXtream2](https://huggingface.co/herimor/voxtream2)     | 1.45 | 0.679 | 3.56 |
| [VoXtream2.1](https://huggingface.co/herimor/voxtream2.1) | 1.32 | 0.682 | 3.78 |
| [VoXtream2.5](https://huggingface.co/herimor/voxtream2.5) | 1.34 | 0.714 | 3.89 |

## Training

- Build the Docker container. If you have another version of Docker compose installed use `docker compose -f ...` instead.
```bash
docker-compose -f .devcontainer/docker-compose.yaml build voxtream
```

- Run training using the `train.py` script. You should specify GPU IDs that will be seen inside the container, ex. `GPU_IDS=0,1`. Specify the batch size according to your GPU. The default batch size is 64 (tested on H200), batch size 12 fits into RTX3090. Results will be stored at the `./experiments` directory.

There are two recipes:

- `configs/train.yaml`, VoXtream2.5: word-aligned crops of the Emilia and HiFiTTS-2 stores, the factored phoneme vocabulary and the hybrid AR + MoG depth transformer head, all trained from scratch. The stores are built locally (see [dataset](voxtream/utils/dataset/README.md)) and passed with `dataset_base_dir`.
```bash
GPU_IDS=0,1 docker-compose -f .devcontainer/docker-compose.yaml run voxtream python voxtream/train.py dataset_base_dir=/path/to/stores batch_size=12
```

- `configs/train_voxtream2.1.yaml`, VoXtream2.1: random crops of the flat-vocabulary arrays of [herimor/voxtream2-train](https://huggingface.co/datasets/herimor/voxtream2-train), downloaded automatically to the HF cache directory (80Gb, loaded to RAM during training, make sure you can allocate ~80Gb of RAM per GPU), and the AR depth transformer initialized from the pre-trained one of CSM. Its audio embeddings and AR head stay frozen (`freeze`), its layers are trained.
```bash
GPU_IDS=0,1 docker-compose -f .devcontainer/docker-compose.yaml run voxtream python voxtream/train.py --config-name train_voxtream2.1 batch_size=12
```

The dataset is picked by `dataset.name` from `DATASET_POOL` (`voxtream/dataset`), as the transformers are picked from `MODEL_POOL`: `aligned` for the stores, `flat` for the arrays.

### Inference with a trained model

`voxtream.utils.export_model` writes a checkpoint as a model directory laid out like the Hugging Face repo of a release (`model.safetensors`, `config.json`, `phoneme_vocab.json`); a generator config with `model_repo` set to that directory runs it.
```bash
python -m voxtream.utils.export_model \
    --checkpoint experiments/voxtream_train/lightning_logs/version_0/epoch=9.ckpt \
    --model-config experiments/voxtream_train/config.json \
    --phoneme-vocab $(hf download herimor/voxtream2.5 phoneme_vocab.json) \
    --out-dir models/my_model
```

### Custom dataset

To prepare a custom training dataset check [dataset](voxtream/utils/dataset/README.md) section.

## Benchmark

To evaluate model's real time factor (RTF) and First packet latency (FPL) run `voxtream-benchmark`. You can compile model for faster inference using `--compile` flag (note that initial compilation take some time).

| Model       | Device  | Compiled           | FPL, ms | RTF   |
| :-:         | :-:     | :-:                | :-:     | :-:   |
| VoXtream2.1 | RTX3090 |                    | 74      | 0.256 |
| VoXtream2.1 | RTX3090 | :heavy_check_mark: | 63      | 0.173 |
| VoXtream2.5 | RTX3090 |                    | 30      | 0.103 |

The depth transformer steps and the MoG step of a VoXtream2.5 frame run as one CUDA graph, which `--compile` gives up for compiled depth transformer steps, so it does not pay off there.

## License

The code in this repository is provided under the MIT License.

The Depth Transformer component from SesameAI-CSM is included under the Apache 2.0 License (see LICENSE-APACHE and NOTICE). This only relates to VoXtream < 2.5. VoXtream2.5 does not use the weights of the SesameAI-CSM model.

The model weights were trained on data licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0 for HiFiTTS2) and Creative Commons Attribution Non Commercial 4.0 (CC BY-NC 4.0 for Emilia). Redistribution of the weights must include proper attribution to the original dataset creators (see ATTRIBUTION.md).

## Acknowledgements

- [Mimi](https://huggingface.co/kyutai/mimi): Streaming audio codec from [Kyutai](https://kyutai.org)
- [CSM](https://github.com/SesameAILabs/csm): Conversation speech model from [Sesame](https://sesame.com)
- [ReDimNet](https://github.com/IDRnD/redimnet): Speaker recognition model from [IDR&D](https://idrnd.ai)
- [ReDimNet2](https://github.com/PalabraAI/redimnet2): Better speaker recognition model from [PalabraAI](https://palabra.ai)
- [Sidon](https://github.com/sarulab-speech/Sidon): Speech enhancemnet model from [SaruLab](https://sp.ipc.i.u-tokyo.ac.jp/index-en)
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng): IPA Phonemizer from eSpeak

## Citation
```
@inproceedings{torgashov2026voxtream,
  title={Vo{X}tream: Full-Stream Text-to-Speech with Extremely Low Latency},
  author={Torgashov, Nikita and Henter, Gustav Eje and Skantze, Gabriel},
  booktitle={Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  note={to appear},
  url={https://arxiv.org/abs/2509.15969}
}

@article{torgashov2026voxtream2,
  author    = {Torgashov, Nikita and Henter, Gustav Eje and Skantze, Gabriel},
  title     = {Vo{X}tream2: Full-stream TTS with dynamic speaking rate control},
  journal   = {arXiv:2603.13518},
  year      = {2026}
}
```

## Disclaimer
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
