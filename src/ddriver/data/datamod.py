"""
Data module helpers that assemble DataLoaders with gentle augmentations.

Why "gentle"? Heavy jitter/flip hurt generalization to unseen drivers in prior
runs, so we only adjust scale slightly and tweak color a bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from torch.utils.data import DataLoader
from torchvision import transforms as T

from .dataset import AucDriverDataset


@dataclass
class DefaultCfg:
    """Tiny convenience container when the caller doesn't want a full config."""

    manifest_csv: str
    train_split_csv: str
    val_split_csv: str
    test_split_csv: str
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224


def _build_transforms(image_size: int, split: str) -> T.Compose:
    """
    Create gentle augmentations per split.

    - Train: resize slightly larger, random crop, small color jitter, no h-flip.
    - Val/Test: deterministic resize + center crop.
    """

    base_resize = image_size + 32  # small buffer for cropping

    if split == "train":
        return T.Compose([
            T.Resize(base_resize),
            T.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return T.Compose([
        T.Resize(base_resize),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _make_dataset(manifest_csv: str | Path, split_csv: str | Path, transforms: T.Compose) -> AucDriverDataset:
    """
    Helper that wires together the manifest with a split list.
    """

    return AucDriverDataset(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        transforms=transforms,
    )


def build_dataloaders(cfg: DefaultCfg | Dict) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders using the provided config.

    Expected cfg attributes/keys:
        - manifest_csv
        - train_split_csv / val_split_csv / test_split_csv
        - batch_size
        - num_workers
        - image_size
    """

    # Allow both dataclass-style access and dict-style configs.
    cfg_obj = cfg if isinstance(cfg, DefaultCfg) else DefaultCfg(**cfg)

    train_tfms = _build_transforms(cfg_obj.image_size, "train")
    eval_tfms = _build_transforms(cfg_obj.image_size, "eval")

    datasets = {
        "train": _make_dataset(cfg_obj.manifest_csv, cfg_obj.train_split_csv, train_tfms),
        "val": _make_dataset(cfg_obj.manifest_csv, cfg_obj.val_split_csv, eval_tfms),
        "test": _make_dataset(cfg_obj.manifest_csv, cfg_obj.test_split_csv, eval_tfms),
    }

    loaders = {}
    for split, dataset in datasets.items():
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg_obj.batch_size,
            shuffle=shuffle,
            num_workers=cfg_obj.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return loaders

