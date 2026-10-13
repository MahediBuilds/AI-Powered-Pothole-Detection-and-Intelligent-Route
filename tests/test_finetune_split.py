from pathlib import Path

import pytest


def frames(n):
    return [Path(f"frame_{i * 5:06d}.jpg") for i in range(n)]


def test_split_holds_out_end_of_video(finetune):
    train, val = finetune.split_train_val(frames(10), 0.2)
    assert train == frames(10)[:8]
    assert val == frames(10)[8:]


@pytest.mark.parametrize("n,fraction,expected", [(2, 0.2, (1, 1)), (3, 0.9, (1, 2)), (100, 0.25, (75, 25))])
def test_split_always_leaves_both_sides_non_empty(finetune, n, fraction, expected):
    train, val = finetune.split_train_val(frames(n), fraction)
    assert (len(train), len(val)) == expected
    assert set(train).isdisjoint(val)


def test_split_needs_two_frames(finetune):
    with pytest.raises(ValueError):
        finetune.split_train_val(frames(1), 0.2)


def make_dataset(root, n, with_labels=True):
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(n):
        (images / f"{i:04d}.jpg").write_bytes(b"jpg")
        if with_labels:
            (labels / f"{i:04d}.txt").write_text(f"0 0.1 0.1 0.2 0.2 0.3 {i / 100}\n")
    return images


def test_sample_replay_images_is_reproducible_and_capped(tmp_path, finetune):
    images = make_dataset(tmp_path, 20)
    first = finetune.sample_replay_images(images, 5, seed=1)
    assert first == finetune.sample_replay_images(images, 5, seed=1)
    assert len(first) == 5 and len(set(first)) == 5
    assert len(finetune.sample_replay_images(images, 100)) == 20


def test_label_path_for(tmp_path, finetune):
    img = tmp_path / "images" / "train" / "0001_00003.jpg"
    assert finetune.label_path_for(img) == tmp_path / "labels" / "train" / "0001_00003.txt"


def test_add_replay_data_copies_images_and_labels(tmp_path, finetune):
    source = make_dataset(tmp_path / "orig", 10)
    out_images = tmp_path / "work" / "images" / "train"
    out_labels = tmp_path / "work" / "labels" / "train"
    out_images.mkdir(parents=True)
    out_labels.mkdir(parents=True)

    assert finetune.add_replay_data(source, 4, out_images, out_labels, seed=0) == 4

    copied = sorted(p.name for p in out_images.iterdir())
    assert len(copied) == 4 and all(name.startswith("replay_") for name in copied)
    for name in copied:
        original_label = tmp_path / "orig" / "labels" / "train" / f"{name[len('replay_'):-4]}.txt"
        assert (out_labels / f"{name[:-4]}.txt").read_text() == original_label.read_text()


def test_add_replay_data_without_label_writes_background_label(tmp_path, finetune):
    source = make_dataset(tmp_path / "orig", 1, with_labels=False)
    out_images, out_labels = tmp_path / "img", tmp_path / "lbl"
    out_images.mkdir()
    out_labels.mkdir()
    finetune.add_replay_data(source, 1, out_images, out_labels, seed=0)
    assert (out_labels / "replay_0000.txt").read_text() == ""


def test_add_replay_data_missing_source_or_zero_count(tmp_path, finetune):
    assert finetune.add_replay_data(tmp_path / "nope", 5, tmp_path, tmp_path, seed=0) == 0
    source = make_dataset(tmp_path / "orig", 3)
    assert finetune.add_replay_data(source, 0, tmp_path, tmp_path, seed=0) == 0
