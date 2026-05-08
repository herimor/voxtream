import numpy as np

from voxtream.dataset import TrainDataset


def test_train_dataset_loads_shards_without_pickle(monkeypatch, tmp_path):
    calls = []

    def fake_load(path, allow_pickle=False):
        calls.append((path, allow_pickle))
        return np.array([1])

    monkeypatch.setattr(np, "load", fake_load)
    monkeypatch.setattr(TrainDataset, "__len__", lambda self: 1)

    dataset = TrainDataset.__new__(TrainDataset)
    TrainDataset.__init__(
        dataset,
        base_dir=tmp_path,
        datasets={"one": {"audio_codes": "a.npy"}, "two": {"audio_codes": "b.npy"}},
        phone_vocab_size=10,
        audio_vocab_size=10,
        audio_pad_size=0,
        num_codebooks=2,
        audio_delay_frames=1,
        dtype="int64",
        audio_window_size=1,
        pad_len=1,
        semantic_label_pad=2,
        num_phones_per_frame=2,
        phoneme_index_map={"0": 1},
        cfg_prob=0.0,
        prompt_length_sec=[1, 2],
        mimi_tps=12.5,
    )

    assert all(allow_pickle is False for _, allow_pickle in calls)
    assert dataset.audio_codes.tolist() == [1, 1]
