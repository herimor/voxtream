# VoXtream: Full-Stream Text-to-Speech with Extremely Low Latency

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2509.15969)
[![demo](https://img.shields.io/badge/VoXtream-Demo-red)](https://herimor.github.io/voxtream)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/herimor/voxtream)
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3119)
[![pytorch](https://img.shields.io/badge/PyTorch_2.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally)

We present VoXtream, a fully autoregressive, zero-shot streaming text-to-speech system for real-time use that begins speaking from the first word.

### Key featues

- **Streaming**: Support a full-stream scenario, where the full sentence is not known in advance. The model takes the text stream coming word-by-word as input and outputs an audio stream in 80ms chunks.
- **Speed**: Works **5x** times faster than real-time and achieves **102 ms** first packet latency on GPU.
- **Quality and efficiency**: With only 9k hours of training data, it matches or surpasses the quality and intelligibility of larger models or models trained on large datasets.

Try [VoXtream ⚡](https://huggingface.co/spaces/herimor/voxtream) in your browser on HuggingFace 🤗 spaces.

## Installation

```bash
pip install voxtream
```

## Usage

* Prompt audio: a file containing 3-5 seconds of the target voice. The maximum supported length is 10 seconds (longer audio will be trimmed).
* Prompt transcript: text that matches the prompt audio. The maximum supported length is 250 characters (longer text will be trimmed).
* Text: What you want the model to say. The maximum supported length is 1000 characters (longer text will be trimmed).
* **Notes**: 
    * The VoXtream requires around 2GB of VRAM.
    * Maximum generation length is limited to 1 minute.
    * The initial run may take a bit longer to download model weights.

### Command line

#### Output streaming
```bash
voxtream \
    --prompt-audio assets/audio/male.wav \
    --prompt-text "The liquor was first created as 'Brandy Milk', produced with milk, brandy and vanilla." \
    --text "In general, however, some method is then needed to evaluate each approximation." \
    --output "output_stream.wav"
```

#### Full streaming
```bash
voxtream \
    --prompt-audio assets/audio/female.wav \
    --prompt-text "Betty Cooper helps Archie with cleaning a store room, when Reggie attacks her." \
    --text "Staff do not always do enough to prevent violence." \
    --output "full_stream.wav" \
    --full-stream # This flag enables input text streaming
```

### Python API

```python
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from voxtream.utils.generator import set_seed, text_generator
from voxtream.generator import SpeechGenerator, SpeechGeneratorConfig


set_seed()
with open('configs/generator.json') as f:
    config = SpeechGeneratorConfig(**json.load(f))

speech_generator = SpeechGenerator(config)

# Output streaming
speech_stream = speech_generator.generate_stream(
    prompt_text="The liquor was first created as 'Brandy Milk', produced with milk, brandy and vanilla.",
    prompt_audio_path=Path('assets/audio/male.wav'),
    text="In general, however, some method is then needed to evaluate each approximation."
)

audio_frames = [audio_frame for audio_frame, _ in speech_stream]
sf.write('output_stream.wav', np.concatenate(audio_frames), config.mimi_sr)

# Full streaming
speech_stream = speech_generator.generate_stream(
    prompt_text="Betty Cooper helps Archie with cleaning a store room, when Reggie attacks her.",
    prompt_audio_path=Path('assets/audio/female.wav'),
    text=text_generator("Staff do not always do enough to prevent violence.")
)

audio_frames = [audio_frame for audio_frame, _ in speech_stream]
sf.write('full_stream.wav', np.concatenate(audio_frames), config.mimi_sr)
```

### Gradio demo

```bash
voxtream-app
```

## Training

- Build the Docker container. If you have another version of Docker compose installed use `docker compose -f ...` instead.
```bash
docker-compose -f .devcontainer/docker-compose.yaml build voxtream
```

- Run training using the `train.py` script. You should specify GPU IDs that will be seen inside the container, ex. `GPU_IDS=0,1`. Specify the batch size according to your GPU. The default batch size is 32 (tested on RTX3090), 64 fits into A100-40Gb, and 128 fits into A100-80Gb. The dataset will be downloaded automatically to the HF cache directory. Dataset size is 20Gb. The data will be loaded to RAM during training, make sure you can allocate ~20Gb of RAM per GPU. Results will be stored at the `./experiments` directory.

Example of running the training using 2 GPUs with batch size 32:
```bash
GPU_IDS=0,1 docker-compose -f .devcontainer/docker-compose.yaml run voxtream python voxtream/train.py batch_size=32
```

## Benchmark

To evaluate model's real time factor (RTF) and First packet latency (FPL) run `voxtream-benchmark`. You can compile model for faster inference using `--compile` flag (note that initial compilation take some time).

| Device  | Compiled           | FPL, ms | RTF  |
| :-:     | :-:                | :-:     | :-:  |
| A100    |                    | 176     | 1.00 |
| A100    | :heavy_check_mark: | 102     | 0.17 |
| RTX3090 |                    | 205     | 1.19 |
| RTX3090 | :heavy_check_mark: | 123     | 0.19 |

## TODO

- [x] Add a neural phoneme aligner. Remove MFA dependency
- [x] Add PyPI package
- [x] Gradio demo
- [x] HuggingFace Spaces demo
- [ ] Evaluation scripts

## License

The code in this repository is provided under the MIT License.

The Depth Transformer component from SesameAI-CSM is included under the Apache 2.0 License (see LICENSE-APACHE and NOTICE).

The model weights were trained on data licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0). Redistribution of the weights must include proper attribution to the original dataset creators (see ATTRIBUTION.md).

## Acknowledgements

- [Mimi](https://huggingface.co/kyutai/mimi): Streaming audio codec from [Kyutai](https://kyutai.org)
- [CSM](https://github.com/SesameAILabs/csm): Conversation speech model from [Sesame](https://www.sesame.com)
- [ReDimNet](https://github.com/IDRnD/redimnet): Speaker recognition model from [IDR&D](https://www.idrnd.ai)

## Citation
```
@article{torgashov2025voxtream,
  author    = {Torgashov, Nikita and Henter, Gustav Eje and Skantze, Gabriel},
  title     = {Vo{X}tream: Full-Stream Text-to-Speech with Extremely Low Latency},
  journal   = {arXiv:2509.15969},
  year      = {2025}
}
```

## Disclaimer
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
