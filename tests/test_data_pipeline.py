"""End-to-end test of the data pipeline on a tiny, fast dataset."""

import torch
from torch.utils.data import DataLoader

from mhd_fno.data.sweep import generate_sweep
from mhd_fno.data.generate import generate_dataset
from mhd_fno.data.dataset import MHDTrajectoryDataset


def _tiny_dataset(tmp_path):
    specs = generate_sweep(n_train=2, n_test=1, resolution=32, seed=0)
    generate_dataset(specs, tmp_path, t_end=2.0, n_snapshots=6,
                     perturbation_amp=1e-2, progress=False)
    return tmp_path


def test_generate_writes_files_and_manifest(tmp_path):
    root = _tiny_dataset(tmp_path)
    assert (root / "manifest.json").exists()
    assert len(list(root.glob("*.h5"))) == 3


def test_dataset_shapes_and_splits(tmp_path):
    root = _tiny_dataset(tmp_path)
    train = MHDTrajectoryDataset(root, split="train")
    test = MHDTrajectoryDataset(root, split="test")
    # 6 snapshots + initial frame = 7 frames -> 6 pairs per run.
    assert len(train) == 2 * 6
    assert len(test) == 1 * 6
    item = train[0]
    assert item["input"].shape == (2, 32, 32)
    assert item["target"].shape == (2, 32, 32)
    assert item["params"].shape == (2,)


def test_dataloader_batches(tmp_path):
    root = _tiny_dataset(tmp_path)
    ds = MHDTrajectoryDataset(root, split="train")
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))
    assert batch["input"].shape == (4, 2, 32, 32)
    assert batch["params"].shape == (4, 2)


def test_rollout_window_mode(tmp_path):
    root = _tiny_dataset(tmp_path)
    K = 3
    ds = MHDTrajectoryDataset(root, split="train", rollout_steps=K)
    item = ds[0]
    assert "window" in item and "input" not in item
    assert item["window"].shape == (K + 1, 2, 32, 32)   # K+1 consecutive frames
    assert item["params"].shape == (2,)
    # 7 frames per run, window of K+1=4 -> 7-3 = 4 start positions per run, x2 runs
    assert len(ds) == 2 * (7 - K)


def test_restartable_skips_existing(tmp_path):
    specs = generate_sweep(n_train=1, n_test=0, resolution=32, seed=0)
    m1 = generate_dataset(specs, tmp_path, t_end=1.0, n_snapshots=4, progress=False)
    m2 = generate_dataset(specs, tmp_path, t_end=1.0, n_snapshots=4, progress=False)
    assert m1[0]["skipped"] is False
    assert m2[0]["skipped"] is True   # second run reuses the existing file
