"""
Training datasets, picked by `dataset.name` of the train config, as the transformers of
the model are picked from `MODEL_POOL`. A dataset takes the root of its stores, the model
config and the rest of the `dataset` section, and brings its own `collate_fn`, which
returns the batch as a dict of the arguments of `Model.forward` and the labels.
"""

from voxtream.dataset.aligned import AlignedDataset
from voxtream.dataset.flat import FlatDataset

DATASET_POOL = {
    # word-aligned crops of the speaker-cluster stores, factored phoneme vocabulary
    "aligned": AlignedDataset,
    # random crops of the flat-vocabulary arrays of VoXtream2 and VoXtream2.1
    "flat": FlatDataset,
}

__all__ = ["DATASET_POOL", "AlignedDataset", "FlatDataset"]
