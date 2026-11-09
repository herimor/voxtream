"""
Where the files of a protocol row live.

The protocol parquet (`metadata.parquet`) names every utterance by its path relative to
the root of the dataset, without an extension, so it holds no machine's directory layout:

    Emilia      EN-B000000/EN_B00000_S00000_W000000
    HiFiTTS-2   100/2315/100_2315_sea_fairies_0812_librivox-18_baum_sea_fairies_17

Every stage is given the roots instead and puts the pieces together here:

    audio                {audio_dir}/{path}.mp3
    phoneme alignment    {data_dir}/{parent}/{name}.json            (`phoneme_align.py`)
    word alignment       {data_dir}/{parent}/{name}.seamless.json   (`emilia_seamless_align.py`)

The alignment tree is keyed by the last directory of the path only: the batch for Emilia,
the session for HiFiTTS-2, whose file names start with the speaker and stay unique.

Every row index the stages write (`row` of the clusters and the crop plan, the masks, the
speaker embeddings) is a row of this protocol, so the protocol must never be reordered.
"""

import os
from pathlib import Path
from typing import Iterable, Union

import pyarrow.parquet as pq

AUDIO_SUFFIX = ".mp3"
ALIGNMENT_SUFFIX = ".json"
SEAMLESS_SUFFIX = ".seamless.json"

_AUDIO_SUFFIXES = (".mp3", ".flac", ".wav", ".opus", ".ogg")


def audio_path(path: str, audio_dir: Union[str, Path]) -> str:
    """The audio file of a protocol path."""
    return os.path.join(audio_dir, path + AUDIO_SUFFIX)


def alignment_path(
    path: str, data_dir: Union[str, Path], suffix: str = ALIGNMENT_SUFFIX
) -> Path:
    """The phoneme alignment (or, with `SEAMLESS_SUFFIX`, word alignment) of a protocol path."""
    parent, _, name = path.rpartition("/")
    return Path(data_dir) / parent.rpartition("/")[2] / f"{name}{suffix}"


def check_protocol(path: str) -> None:
    """
    Refuse a protocol written the old way, with absolute paths and extensions.

    Joined to a root, such a path names a file that does not exist, and every stage treats
    a missing file as one bad utterance, so the run would finish having skipped them all.
    """
    if os.path.isabs(path) or path.lower().endswith(_AUDIO_SUFFIXES):
        raise SystemExit(
            f"the protocol path {path!r} is absolute or carries an extension; the "
            "protocol must hold paths relative to the dataset root, without one"
        )


def check_protocol_file(meta_path: Union[str, Path]) -> None:
    """`check_protocol` on the first row of a protocol parquet, reading only that row."""
    table = pq.ParquetFile(meta_path).read_row_group(0, columns=["path"])
    check_protocol(table.column("path")[0].as_py())


def file_names(values: dict, keys: Iterable[str]) -> dict:
    """
    `values` with the paths under `keys` cut to their last component.

    The settings a stage writes next to its output name the inputs it read for the
    record, and only the file name is portable: the outputs are shared without the
    machine they were built on.
    """
    out = dict(values)
    for key in keys:
        value = out.get(key)
        if isinstance(value, (list, tuple)):
            out[key] = [Path(v).name for v in value]
        elif value:
            out[key] = Path(value).name
    return out
