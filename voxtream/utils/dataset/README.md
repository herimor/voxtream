# VoXtream2.5 training data

VoXtream2.5 trains on word-aligned stores (`dataset.name: aligned`, `configs/train.yaml`)
built from two datasets:

| store | source | entries |
| --- | --- | --- |
| `emilia` | English part of [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) | 1,913,215 |
| `hifitts2` | [HiFiTTS-2](https://huggingface.co/datasets/nvidia/hifitts-2), 15.6k h subset | 693,420 |

Every entry is 80 s of one speaker: utterances of a speaker cluster laid end to end, ended
at a word boundary and padded with silence.

Two datasets are published:

- [herimor/voxtream2.5-train](https://huggingface.co/datasets/herimor/voxtream2.5-train):
  the stores, ready for training.
- [herimor/voxtream2.5-metadata](https://huggingface.co/datasets/herimor/voxtream2.5-metadata):
  everything needed to rebuild the stores from the original audio.

## Train on the published stores

```bash
hf download herimor/voxtream2.5-train --repo-type dataset --local-dir /data/voxtream2.5-train
python voxtream/train.py dataset_base_dir=/data/voxtream2.5-train \
    dataset.datasets.emilia=emilia dataset.datasets.hifitts2=hifitts2
```

Download into a writable directory rather than the Hugging Face cache: on its first run
the dataloader writes a `.crop_index_*.npy` cache next to each store. The phoneme
vocabulary of the stores is `phoneme_vocab.json` of the
[model](https://huggingface.co/herimor/voxtream2.5).

## Paths

Every stage reads one protocol per dataset, `metadata.parquet`: one row per utterance with
`path`, `duration` and `text`. The `path` is relative to the dataset root and has no
extension (`EN-B000000/EN_B00000_S00000_W000000`, `100/2315/100_2315_sea_fairies_..._17`);
the stages take the roots as arguments (`paths.py`):

| file | location | argument |
| --- | --- | --- |
| audio | `{audio_dir}/{path}.mp3` | `-a/--audio-dir` |
| phoneme alignment | `{alignment_dir}/{parent}/{name}.json` | `-d/--data-dir` |
| word alignment | `{alignment_dir}/{parent}/{name}.seamless.json` | `-d/--data-dir` |

`{parent}` is the last directory of the path: the batch for Emilia, the session for
HiFiTTS-2. All audio is 24 kHz mp3 in one directory per dataset. Every row index a stage
writes points into the protocol, so a protocol must never be reordered or filtered;
filters are boolean row masks.

## Rebuild the stores from the original audio

The metadata holds the protocols and the outputs of every stage that filters, clusters or
plans, so a rebuild runs only the last three stages:

```
valid_phoneme_vocab.json     phoneme roots of the vocabulary (-v)
{emilia,hifitts2}/
  metadata.parquet           protocol
  alignment.tar.gz           {parent}/{name}.json and {parent}/{name}.seamless.json
  spk_emb.npy                speaker embeddings, float16 [rows, 192]
  alignment_filter.npy       keep mask of the transcript and alignment filters, bool [rows]
  crop_plan/                 utterances of every entry, their order and the sample range
                             taken from each
```

Emilia (HiFiTTS-2 is the same with its own files and audio directory):

```bash
hf download herimor/voxtream2.5-metadata --repo-type dataset --local-dir /data/voxtream2.5-metadata
META=/data/voxtream2.5-metadata
mkdir -p /data/alignment && tar xzf $META/emilia/alignment.tar.gz -C /data/alignment

python -m voxtream.utils.dataset.emilia_train_arrays \
    -m $META/emilia/metadata.parquet -p $META/emilia/crop_plan \
    -e $META/emilia/spk_emb.npy -v $META/valid_phoneme_vocab.json \
    -d /data/alignment/emilia -o /data/stores/emilia -nj 32
python -m voxtream.utils.dataset.emilia_mimi_codes \
    -m $META/emilia/metadata.parquet -p $META/emilia/crop_plan -a /data/emilia \
    -o /data/stores/emilia -cb 32 -g 0,1,2,3
python -m voxtream.utils.dataset.silence_tokens -d /data/stores/emilia -g 0
```

The array builder is deterministic, so the rebuilt stores hold the published entries in the
published order. The published HiFiTTS-2 store encoded 40% of its utterances from the
22.05 kHz flac release, so its Mimi codes rebuilt from 24 kHz mp3 differ slightly on those
utterances; the phonemes, alignments and plan are the same.

## Build from scratch

For entries of another length, other clusters or another protocol, run every stage. Work
in a copy of the metadata layout (`META`), not in the downloaded metadata. Use
`valid_phoneme_vocab.json` of the metadata as the vocabulary: it is the one VoXtream2.5
was trained with. The transcript filter that `--mask sentence_filter.npy` refers to is
made outside this repository and is not published (`alignment_filter.npy` already
includes it); without it, leave `--mask` out.

### Emilia

```bash
E=$META/emilia
ALIGN=/data/alignment/emilia
STORE=/data/stores/emilia

# 1. word alignment with the Seamless aligner: {batch}/{name}.seamless.json
python -m voxtream.utils.dataset.emilia_seamless_align \
    -m $E/metadata.parquet -a /data/emilia -o $ALIGN \
    --aligner-root /path/to/forced-aligners-bench --gpus 0,1,2,3 --procs-per-gpu 2

# 2. phoneme alignment with Clap-IPA from the word alignments: {batch}/{name}.json
python -m voxtream.utils.dataset.phoneme_align \
    -m $E/metadata.parquet -a /data/emilia -d $ALIGN --gpus 0,1,2,3 -nw 8

# 3. speaker embeddings (ReDimNet2-B3, first 3 s of every file): $E/spk_emb.npy
python -m voxtream.utils.dataset.emilia_speaker_encoder \
    -m $E/metadata.parquet -a /data/emilia -o $E --gpus 0,1,2,3

# 4. phoneme duration model and alignment filter: $E/alignment_filter.npy
python -m voxtream.utils.dataset.parse_phoneme_alignment \
    -m $E/metadata.parquet --mask $E/sentence_filter.npy \
    -d $ALIGN -o $E/phoneme_stats.json -nj 32
python -m voxtream.utils.dataset.emilia_alignment_filter \
    -m $E/metadata.parquet --mask $E/sentence_filter.npy \
    -s $E/phoneme_stats.npz -d $ALIGN -o $E -nj 32

# 5. single-speaker clusters of --target-duration seconds
python -m voxtream.utils.dataset.emilia_speaker_clusters \
    -m $E/metadata.parquet -e $E/spk_emb.npy --mask $E/alignment_filter.npy \
    --target-duration 80 -o $E/clusters -j 40

# 6. crop plan: the 80 s entries, without touching audio
python -m voxtream.utils.dataset.emilia_crop_plan \
    -m $E/metadata.parquet -c $E/clusters -d $ALIGN -o $E/crop_plan -nj 32

# 7-9. training arrays, Mimi codes and silence tokens, as in the rebuild
python -m voxtream.utils.dataset.emilia_train_arrays \
    -m $E/metadata.parquet -p $E/crop_plan -e $E/spk_emb.npy \
    -v $META/valid_phoneme_vocab.json -d $ALIGN -o $STORE -nj 32
python -m voxtream.utils.dataset.emilia_mimi_codes \
    -m $E/metadata.parquet -p $E/crop_plan -a /data/emilia \
    -o $STORE -cb 32 -g 0,1,2,3 -bs 16 -w 10
python -m voxtream.utils.dataset.silence_tokens -d $STORE -g 0
```

Notes:

- The cluster target runs ahead of the entry length: an entry stops at the last word
  boundary that fits, so 80 s clusters give 80 s entries with ~74 s of speech.
- `emilia_alignment_filter.py --from-scores --drop <share>` rebuilds the mask for another
  drop share in seconds.
- Extract all the codebooks you need in one Mimi pass (`-cb`); `num_codebooks` in
  `train.yaml` then picks how many the dataloader reads. A later run with a larger `-cb`
  re-encodes all of them.
- Re-run `silence_tokens.py` whenever the Mimi codes are rebuilt; the dataloader refuses
  a store without it.
- `emilia_mimi_codes.py` fills the store in place and can be restarted.
- Copy the store's `phoneme_vocab.json` next to the weights of a model trained on it; the
  inference frontend loads it as `phoneme_vocab_name`.

### HiFiTTS-2

HiFiTTS-2 has real speaker labels (`{speaker}/{session}/{name}`), so
`hifitts2_speaker_groups.py` groups entries by `(speaker, session)` in recording order
instead of clustering embeddings. Stages 1-3 run as for Emilia, pointed at the HiFiTTS-2
protocol and audio; the duration model and the filter are fitted to its own alignments
and drop 1% instead of 2%, since its alignments are much cleaner:

```bash
H=$META/hifitts2
ALIGN=/data/alignment/hifitts2
STORE=/data/stores/hifitts2

python -m voxtream.utils.dataset.parse_phoneme_alignment \
    -m $H/metadata.parquet --mask $H/sentence_filter.npy \
    -d $ALIGN -o $H/phoneme_stats.json -nj 16
python -m voxtream.utils.dataset.emilia_alignment_filter \
    -m $H/metadata.parquet --mask $H/sentence_filter.npy \
    -s $H/phoneme_stats.npz -d $ALIGN -o $H --drop 0.01 -nj 16
python -m voxtream.utils.dataset.hifitts2_speaker_groups \
    -m $H/metadata.parquet --mask $H/alignment_filter.npy \
    -e $H/spk_emb.npy --target-duration 80 -o $H/clusters
python -m voxtream.utils.dataset.emilia_crop_plan \
    -m $H/metadata.parquet -c $H/clusters -d $ALIGN -o $H/crop_plan -nj 16
python -m voxtream.utils.dataset.emilia_train_arrays \
    -m $H/metadata.parquet -p $H/crop_plan -e $H/spk_emb.npy \
    -v $META/valid_phoneme_vocab.json -d $ALIGN -o $STORE -nj 16
python -m voxtream.utils.dataset.emilia_mimi_codes \
    -m $H/metadata.parquet -p $H/crop_plan -a /data/hifitts2 \
    -o $STORE -cb 32 -g 0 -bs 8 -w 4
python -m voxtream.utils.dataset.silence_tokens -d $STORE -g 0
```

The two stores share `num_frames`, the phoneme vocabulary and the speaker embedding size,
so training reads them side by side (`dataset.datasets` of `train.yaml`).
