from pathlib import Path

import pandas as pd
import pytest

from voxtream.utils.dataset.hifitts2_speaker_groups import path_labels
from voxtream.utils.dataset.paths import (
    SEAMLESS_SUFFIX,
    alignment_path,
    audio_path,
    check_protocol,
    file_names,
)

EMILIA = "EN-B000000/EN_B00000_S00000_W000000"
HIFITTS2 = "100/2315/100_2315_sea_fairies_0812_librivox-18_baum_sea_fairies_17"


def test_emilia_layout():
    assert audio_path(EMILIA, "/data/emilia") == f"/data/emilia/{EMILIA}.mp3"
    assert alignment_path(EMILIA, "/align") == Path(f"/align/{EMILIA}.json")
    assert alignment_path(EMILIA, "/align", SEAMLESS_SUFFIX) == Path(
        f"/align/{EMILIA}.seamless.json"
    )


def test_hifitts2_alignment_tree_is_keyed_by_session():
    name = HIFITTS2.rsplit("/", 1)[1]
    assert audio_path(HIFITTS2, "/data/hifitts2") == f"/data/hifitts2/{HIFITTS2}.mp3"
    assert alignment_path(HIFITTS2, "/align") == Path(f"/align/2315/{name}.json")


@pytest.mark.parametrize(
    "path",
    [
        f"/nfs/server/datasets/emilia/{EMILIA}.mp3",
        f"/data/{EMILIA}",
        f"{HIFITTS2}.flac",
    ],
)
def test_old_protocols_are_refused(path):
    with pytest.raises(SystemExit):
        check_protocol(path)


def test_relative_protocols_pass():
    check_protocol(EMILIA)
    check_protocol(HIFITTS2)


def test_file_names_strip_directories():
    args = {
        "meta_path": "/a/b/metadata.parquet",
        "mask": ["/a/sentence_filter.npy", "/b/alignment_filter.npy"],
        "embeddings_path": None,
        "target_duration": 80.0,
    }
    out = file_names(args, ("meta_path", "mask", "embeddings_path"))
    assert out == {
        "meta_path": "metadata.parquet",
        "mask": ["sentence_filter.npy", "alignment_filter.npy"],
        "embeddings_path": None,
        "target_duration": 80.0,
    }
    assert args["meta_path"] == "/a/b/metadata.parquet"


def test_hifitts2_labels_from_relative_paths():
    speakers, sessions, names = path_labels(pd.Series([HIFITTS2]))
    assert (speakers[0], sessions[0], names[0]) == (
        "100",
        "2315",
        HIFITTS2.rsplit("/", 1)[1],
    )
    with pytest.raises(SystemExit):
        path_labels(pd.Series([EMILIA]))
