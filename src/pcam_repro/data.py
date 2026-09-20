"""Official PCam HDF5 input pipeline and controlled augmentations."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Callable, Sequence

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .config import DataConfig
from .utils import seed_worker


PCAM_MEAN = (0.7008, 0.5384, 0.6916)
PCAM_STD = (0.2350, 0.2774, 0.2129)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomD4:
    """Uniform exact rotation/reflection of a square image.

    This is distribution-equivalent to independent horizontal/vertical flips
    followed by a uniformly selected multiple of 90 degrees, but performs a
    single lossless PIL transpose instead of a general affine rotation.
    """

    _TRANSPOSES = (
        None,
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
        Image.Transpose.FLIP_LEFT_RIGHT,
        Image.Transpose.TRANSVERSE,
        Image.Transpose.FLIP_TOP_BOTTOM,
        Image.Transpose.TRANSPOSE,
    )

    @classmethod
    def apply(cls, image: Image.Image, group_index: int) -> Image.Image:
        if group_index not in range(8):
            raise ValueError("D4 group index must lie in [0, 7].")
        operation = cls._TRANSPOSES[group_index]
        return image.copy() if operation is None else image.transpose(operation)

    def __call__(self, image: Image.Image) -> Image.Image:
        group_index = int(torch.randint(0, 8, ()).item())
        return self.apply(image, group_index)


def _first_dataset(handle: h5py.File) -> h5py.Dataset:
    datasets: list[h5py.Dataset] = []
    handle.visititems(lambda _name, item: datasets.append(item) if isinstance(item, h5py.Dataset) else None)
    if not datasets:
        raise ValueError(f"No dataset found in HDF5 file {handle.filename}.")
    return datasets[0]


class PCamH5Dataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Lazy, multi-worker-safe reader for the official uncompressed PCam files."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        target_transform: Callable[[float], float] | None = None,
    ) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be train, valid, or test.")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        stem = f"camelyonpatch_level_2_split_{split}"
        self.image_path = self.root / f"{stem}_x.h5"
        self.target_path = self.root / f"{stem}_y.h5"
        if not self.image_path.exists() or not self.target_path.exists():
            raise FileNotFoundError(
                f"Missing PCam files for split '{split}'. Expected {self.image_path.name} "
                f"and {self.target_path.name} in {self.root}."
            )
        self._image_file: h5py.File | None = None
        self._target_file: h5py.File | None = None
        self._images: h5py.Dataset | None = None
        self._targets: h5py.Dataset | None = None
        with h5py.File(self.target_path, "r") as handle:
            targets = np.asarray(_first_dataset(handle)).reshape(-1)
        self.labels = targets.astype(np.int64)

    def _ensure_open(self) -> None:
        if self._image_file is None:
            self._image_file = h5py.File(self.image_path, "r")
            self._target_file = h5py.File(self.target_path, "r")
            self._images = _first_dataset(self._image_file)
            self._targets = _first_dataset(self._target_file)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _read_images(self, indices: Sequence[int]) -> np.ndarray:
        self._ensure_open()
        assert self._images is not None
        index_array = np.asarray(indices, dtype=np.int64)
        # h5py requires monotonically increasing unique fancy indices. Reading
        # them once and applying the inverse map restores arbitrary sampler
        # order and also supports repeated indices.
        unique_indices, inverse = np.unique(index_array, return_inverse=True)
        unique_images = np.asarray(self._images[unique_indices], dtype=np.uint8)
        return unique_images[inverse]

    def _read_image(self, index: int) -> np.ndarray:
        """Read one sample without NumPy unique/fancy-index bookkeeping."""
        self._ensure_open()
        assert self._images is not None
        return np.asarray(self._images[index], dtype=np.uint8)

    def _make_item(
        self,
        image_array: np.ndarray,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        image = Image.fromarray(image_array, mode="RGB")
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)
        target = float(self.labels[index])
        if self.target_transform is not None:
            target = float(self.target_transform(target))
        return tensor, torch.tensor(target, dtype=torch.float32), index

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self._make_item(self._read_image(index), index)

    def __getitems__(
        self,
        indices: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        images = self._read_images(indices)
        return [
            self._make_item(image, int(index))
            for image, index in zip(images, indices, strict=True)
        ]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for key in ("_image_file", "_target_file", "_images", "_targets"):
            state[key] = None
        return state

    def close(self) -> None:
        if self._image_file is not None:
            self._image_file.close()
        if self._target_file is not None:
            self._target_file.close()
        self._image_file = self._target_file = None
        self._images = self._targets = None

    def __del__(self) -> None:
        self.close()


class TransformViewDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    """Produce weak/strong or two contrastive views from a base PCam dataset."""

    def __init__(
        self,
        base: PCamH5Dataset,
        view_one: Callable[[Image.Image], torch.Tensor],
        view_two: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.base = base
        self.view_one = view_one
        self.view_two = view_two

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        image = Image.fromarray(self.base._read_image(index), mode="RGB")
        target = torch.tensor(float(self.base.labels[index]), dtype=torch.float32)
        return self.view_one(image), self.view_two(image), target, index

    def __getitems__(
        self,
        indices: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        images = self.base._read_images(indices)
        items = []
        for image_array, index in zip(images, indices, strict=True):
            image = Image.fromarray(image_array, mode="RGB")
            target = torch.tensor(
                float(self.base.labels[index]),
                dtype=torch.float32,
            )
            items.append(
                (self.view_one(image), self.view_two(image), target, int(index))
            )
        return items


class PerItemDataset(Dataset):
    """Force independent reads for randomly sampled training examples.

    h5py fancy indexing sorts and materializes a complete shuffled batch.
    Independent reads distributed across persistent workers are faster for the
    random training sampler, while validation and test retain batched,
    sequential HDF5 access.
    """

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        return self.base[index]


class NoisyLabelDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Deterministic symmetric label-noise wrapper used only on the train split."""

    def __init__(self, base: Dataset, rate: float, seed: int) -> None:
        self.base = base
        generator = np.random.default_rng(seed)
        self.flip_mask = generator.random(len(base)) < rate

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target, sample_id = self.base[index]
        if self.flip_mask[index]:
            target = 1.0 - target
        return image, target, sample_id

    def __getitems__(self, indices: list[int]):
        batched_getter = getattr(self.base, "__getitems__", None)
        items = (
            batched_getter(indices)
            if callable(batched_getter)
            else [self.base[index] for index in indices]
        )
        return [
            (image, 1.0 - target if self.flip_mask[index] else target, sample_id)
            for index, (image, target, sample_id) in zip(
                indices, items, strict=True
            )
        ]


def normalization(name: str) -> transforms.Normalize:
    if name == "pcam":
        return transforms.Normalize(PCAM_MEAN, PCAM_STD)
    if name == "imagenet":
        return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if name == "half":
        return transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    raise ValueError(f"Unknown normalization: {name}")


def build_transform(
    image_size: int,
    policy: str,
    normalize: str,
) -> Callable[[Image.Image], torch.Tensor]:
    resize = [] if image_size == 96 else [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True)
    ]
    norm = normalization(normalize)
    if policy == "none":
        ops = resize + [transforms.ToTensor(), norm]
    elif policy == "minimal":
        ops = resize + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            norm,
        ]
    elif policy == "pathology":
        ops = resize + [
            RandomD4(),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            norm,
        ]
    elif policy == "weak":
        ops = resize + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            norm,
        ]
    elif policy == "strong":
        ops = resize + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.04),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
            transforms.ToTensor(),
            norm,
        ]
    elif policy == "simclr":
        # Crop scale is deliberately conservative: the central 32x32 region defines the label.
        ops = [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.25, 0.05)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
            norm,
        ]
    else:
        raise ValueError(f"Unknown augmentation policy: {policy}")
    return transforms.Compose(ops)


def stratified_indices(labels: Sequence[int], fraction: float, seed: int) -> list[int]:
    labels_array = np.asarray(labels, dtype=np.int64)
    if fraction >= 1:
        return list(range(len(labels_array)))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in (0, 1):
        class_indices = np.flatnonzero(labels_array == class_id)
        count = max(1, int(round(len(class_indices) * fraction)))
        selected.extend(rng.choice(class_indices, size=count, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def wsi_group_indices(
    metadata_csv: str | Path,
    fraction: float,
    seed: int,
    slide_column: str = "wsi",
) -> list[int]:
    """Select complete WSIs if the official metadata CSV is available."""
    with Path(metadata_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or slide_column not in rows[0]:
        raise ValueError(f"Column '{slide_column}' was not found in {metadata_csv}.")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row[slide_column], []).append(index)
    slides = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(slides)
    keep = set(slides[: max(1, round(len(slides) * fraction))])
    return [index for slide in keep for index in groups[slide]]


def make_loader(
    dataset: Dataset,
    config: DataConfig,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
    batch_size: int | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size if batch_size is None else batch_size,
        shuffle=shuffle,
        num_workers=config.workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.workers > 0,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_dataloaders(
    config: DataConfig,
    mode: str,
    seed: int,
    noise_rate: float = 0.0,
    noise_seed: int = 23,
) -> dict[str, DataLoader]:
    eval_transform = build_transform(config.image_size, "none", config.normalize)
    train_transform = build_transform(
        config.image_size,
        config.augmentation if config.augmentation not in {"weak_strong", "simclr"} else "pathology",
        config.normalize,
    )
    train_base = PCamH5Dataset(config.root, "train", transform=train_transform)
    validation = PCamH5Dataset(config.root, "valid", transform=eval_transform)
    test = PCamH5Dataset(config.root, "test", transform=eval_transform)
    labeled_indices = stratified_indices(train_base.labels, config.labeled_fraction, config.subset_seed)

    loaders: dict[str, DataLoader] = {
        "validation": make_loader(
            validation,
            config,
            False,
            seed,
            batch_size=config.eval_batch_size,
        ),
        "test": make_loader(
            test,
            config,
            False,
            seed,
            batch_size=config.eval_batch_size,
        ),
    }
    if mode == "supervised":
        train: Dataset = Subset(
            PerItemDataset(train_base),
            labeled_indices,
        )
        if noise_rate > 0:
            train = NoisyLabelDataset(train, noise_rate, noise_seed)
        loaders["train"] = make_loader(train, config, True, seed)
    elif mode == "mean_teacher":
        raw_base = PCamH5Dataset(config.root, "train", transform=None)
        weak = build_transform(config.image_size, "weak", config.normalize)
        strong = build_transform(config.image_size, "strong", config.normalize)
        views = PerItemDataset(
            TransformViewDataset(raw_base, weak, strong)
        )
        loaders["labeled"] = make_loader(Subset(views, labeled_indices), config, True, seed, drop_last=True)
        loaders["unlabeled"] = make_loader(views, config, True, seed + 1, drop_last=True)
    elif mode == "simclr":
        raw_base = PCamH5Dataset(config.root, "train", transform=None)
        contrastive = build_transform(config.image_size, "simclr", config.normalize)
        views = PerItemDataset(
            TransformViewDataset(raw_base, contrastive, contrastive)
        )
        loaders["train"] = make_loader(views, config, True, seed, drop_last=True)
    else:
        raise ValueError(f"Unknown training mode: {mode}")
    return loaders
