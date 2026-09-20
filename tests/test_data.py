from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from pcam_repro.config import DataConfig
from pcam_repro.data import (
    PCamH5Dataset,
    PerItemDataset,
    RandomD4,
    make_loader,
    stratified_indices,
)


def test_hdf5_reader_and_stratified_subset(tmp_path: Path) -> None:
    images = np.zeros((6, 96, 96, 3), dtype=np.uint8)
    images[3:] = 255
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8).reshape(-1, 1, 1, 1)
    for suffix, values in (("x", images), ("y", labels)):
        path = tmp_path / f"camelyonpatch_level_2_split_train_{suffix}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset(suffix, data=values)
    dataset = PCamH5Dataset(tmp_path, "train")
    image, target, sample_id = dataset[3]
    assert image.shape == (3, 96, 96)
    assert float(target) == 1.0
    assert sample_id == 3
    batch = dataset.__getitems__([4, 1, 4])
    assert [item[2] for item in batch] == [4, 1, 4]
    assert [float(item[1]) for item in batch] == [0.0, 1.0, 0.0]
    assert np.allclose(batch[0][0].numpy(), batch[2][0].numpy())
    subset = stratified_indices(dataset.labels, 0.5, seed=5)
    subset_labels = dataset.labels[subset]
    assert (subset_labels == 0).sum() == (subset_labels == 1).sum()


def test_loader_can_override_training_batch_size_for_evaluation() -> None:
    config = DataConfig(
        batch_size=2,
        eval_batch_size=5,
        workers=0,
        pin_memory=False,
    )
    dataset = TensorDataset(torch.arange(10))
    train_loader = make_loader(dataset, config, shuffle=False, seed=17)
    evaluation_loader = make_loader(
        dataset,
        config,
        shuffle=False,
        seed=17,
        batch_size=config.eval_batch_size,
    )
    assert train_loader.batch_size == 2
    assert evaluation_loader.batch_size == 5


def test_per_item_wrapper_hides_batched_dataset_access() -> None:
    class DatasetWithFailingBatchGetter(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int):
            return index

        def __getitems__(self, _indices: list[int]):
            raise AssertionError("Batched getter must remain hidden.")

    loader = torch.utils.data.DataLoader(
        PerItemDataset(DatasetWithFailingBatchGetter()),
        batch_size=4,
    )
    assert torch.equal(next(iter(loader)), torch.arange(4))


def test_random_d4_is_lossless_and_enumerates_eight_orientations() -> None:
    image = Image.fromarray(
        np.arange(3 * 3 * 3, dtype=np.uint8).reshape(3, 3, 3),
        mode="RGB",
    )
    payloads = {
        RandomD4.apply(image, group_index).tobytes()
        for group_index in range(8)
    }
    assert len(payloads) == 8
    assert all(len(payload) == len(image.tobytes()) for payload in payloads)
