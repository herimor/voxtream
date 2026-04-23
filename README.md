# VoXtream2: Full-stream TTS with dynamic speaking rate control

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2603.13518)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/herimor/voxtream2)
<!-- [![Demo page](https://img.shields.io/badge/VoXtream2-Demo_page-red)](https://herimor.github.io/voxtream2) -->
<!-- [![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Live_demo-yellow)](https://huggingface.co/spaces/herimor/voxtream2) -->

We present VoXtream2, a zero-shot full-stream TTS model with dynamic speaking-rate control that can be updated mid-utterance on the fly.

For audio examples, see our [demo page](https://herimor.github.io/voxtream2).

Try VoXtream2 in your browser on HuggingFace 🤗 [space](https://huggingface.co/spaces/herimor/voxtream2).

## Key features

- **Dynamic speed control**: Distribution matching and Classifier-free guidance allow for a fine-grained speaking rate control, which can be adjusted as the model generates speech.
- **Streaming performance**: Works **4x** times faster than real-time and achieves **74 ms** first packet latency in a full-stream on a consumer GPU.
- **Translingual capability**: Prompt text masking enables support of acoustic prompts in any language.

## Updates

- `2026/04`: Added a frame repeat counter. Reduces hallucinations caused by models getting stuck in the same frame. Controlled by the `frame_repeat_counter` parameter in `SpeechGeneratorConfig`. Recommended value (12-25), lower values for stricter control.
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
pip install "voxtream>=0.2.1"
```

## Usage

* Prompt audio: a file containing 3-10 seconds of the target voice. The maximum supported length is 20 seconds (longer audio will be trimmed).
* Text: What you want the model to say. The maximum supported length is 1000 characters (longer text will be trimmed).
* Speaking rate (optional): target speaking rate in syllables per second.

**Notes**: 
* The model was tested on Ubuntu 22.04, CUDA 12 and PyTorch 2.4.
* The model requires 4.2Gb of VRAM (can be reduced to 2.2Gb by disabling speech enhancement module).
* Maximum generation length is limited to 1 minute.
* The initial run may take a bit longer to download model weights and warmup model graph.
* If you experience problems with CUDAGraphs please check this [issue](https://github.com/herimor/voxtream/issues/8).

### Command line

#### Output streaming
```bash
voxtream \
    --prompt-audio assets/audio/english_male.wav \
    --text "In general, however, some method is then needed to evaluate each approximation." \
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

### Python API

```python
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from voxtream.utils.generator import (
    set_seed,
    text_generator,
    interpolate_speaking_rate_params
)
from voxtream.generator import SpeechGenerator, SpeechGeneratorConfig


set_seed()
with open('configs/generator.json') as f:
    config = SpeechGeneratorConfig(**json.load(f))

with open('configs/speaking_rate.json') as f:
    speaking_rate_config = json.load(f)

speech_generator = SpeechGenerator(config)

# Output streaming, no speaking rate control
speech_stream = speech_generator.generate_stream(
    prompt_audio_path=Path('assets/audio/english_male.wav'),
    text="In general, however, some method is then needed to evaluate each approximation.",
)

audio_frames = [audio_frame for audio_frame, _ in speech_stream]
sf.write('output_stream.wav', np.concatenate(audio_frames), config.mimi_sr)

# Slow speech, 2 syllables per second
speaking_rate = 2.0
duration_state, weight, cfg_gamma = interpolate_speaking_rate_params(
    speaking_rate_config, speaking_rate, logger=speech_generator.logger
)

# Full streaming & speaking rate control
speech_stream = speech_generator.generate_stream(
    prompt_audio_path=Path('assets/audio/english_female.wav'),
    text=text_generator("Staff do not always do enough to prevent violence."),
    target_spk_rate_cnt=duration_state,
    spk_rate_weight=weight,
    cfg_gamma=cfg_gamma,
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
voxtream-server
```

To send a request to the server run:
```bash
python voxtream/client.py
```

It sends a path to the audio prompt and a text to the server and immediately plays audio from the output stream.

### Evaluation

To reproduce evaluation metrics from the paper check [evaluation](voxtream/utils/test/README.md) section.

## Training

- Build the Docker container. If you have another version of Docker compose installed use `docker compose -f ...` instead.
```bash
docker-compose -f .devcontainer/docker-compose.yaml build voxtream
```

- Run training using the `train.py` script. You should specify GPU IDs that will be seen inside the container, ex. `GPU_IDS=0,1`. Specify the batch size according to your GPU. The default batch size is 64 (tested on H200), batch size 12 fits into RTX3090. The dataset will be downloaded automatically to the HF cache directory. Dataset size is 80Gb. The data will be loaded to RAM during training, make sure you can allocate ~80Gb of RAM per GPU. Results will be stored at the `./experiments` directory.

Example of running the training using 2 GPUs with batch size 32:
```bash
GPU_IDS=0,1 docker-compose -f .devcontainer/docker-compose.yaml run voxtream python voxtream/train.py batch_size=12
```

### Custom dataset

To prepare a custom training dataset check [dataset](voxtream/utils/dataset/README.md) section.

## Benchmark

To evaluate model's real time factor (RTF) and First packet latency (FPL) run `voxtream-benchmark`. You can compile model for faster inference using `--compile` flag (note that initial compilation take some time).

| Device  | Compiled           | FPL, ms | RTF   |
| :-:     | :-:                | :-:     | :-:   |
| RTX3090 |                    | 74      | 0.256 |
| RTX3090 | :heavy_check_mark: | 63      | 0.173 |

## TODO

- [ ] Add finetuning instructions

## License

The code in this repository is provided under the MIT License.

The Depth Transformer component from SesameAI-CSM is included under the Apache 2.0 License (see LICENSE-APACHE and NOTICE).

The model weights were trained on data licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0). Redistribution of the weights must include proper attribution to the original dataset creators (see ATTRIBUTION.md).

## Acknowledgements

- [Mimi](https://huggingface.co/kyutai/mimi): Streaming audio codec from [Kyutai](https://kyutai.org)
- [CSM](https://github.com/SesameAILabs/csm): Conversation speech model from [Sesame](https://sesame.com)
- [ReDimNet](https://github.com/IDRnD/redimnet): Speaker recognition model from [IDR&D](https://idrnd.ai)
- [Sidon](https://github.com/sarulab-speech/Sidon): Speech enhancemnet model from [SaruLab](https://sp.ipc.i.u-tokyo.ac.jp/index-en)

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
