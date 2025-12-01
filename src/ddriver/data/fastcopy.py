"""Utilities that copy the Distracted Driver dataset into FAST_DATA with optional compression.

This module centralizes the logic that the Colab notebook uses when it wants
to materialize a smaller, IO-friendly copy of the dataset inside `/content/data`.
It keeps the folder structure identical to the source while optionally resizing
and recompressing JPEGs so they occupy less space without sacrificing the pose
information that the models rely on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import shutil

__all__ = [
    "CompressionSpec",
    "copy_splits_with_compression",
]

DATASET_MARKER = "auc.distracted.driver.dataset_v2"
JPEG_EXTENSIONS = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class CompressionSpec:
    """
    Parameters that describe how we shrink the dataset copy.

    Attributes
    ----------
    target_short_side : int
        Largest permissible length for the shorter image side. Only enforced
        when the image is larger; smaller images are left as-is to avoid
        up-scaling noise.
    jpeg_quality : int
        Pillow JPEG quality flag (0-100). Values in the 70-85 range are
        visually indistinguishable yet save significant space.
    progressive : bool
        Whether to emit progressive JPEGs (slightly faster to stream).
    optimize : bool
        Ask Pillow to spend extra CPU to find better Huffman tables.
    """

    target_short_side: int = 320
    jpeg_quality: int = 80
    progressive: bool = True
    optimize: bool = True

    def compute_size(self, width: int, height: int) -> tuple[int, int]:
        """
        Return the resized dimensions that keep aspect ratio intact.
        """
        if self.target_short_side is None:
            return width, height

        short_side = min(width, height)
        if short_side <= self.target_short_side:
            return width, height

        scale = self.target_short_side / short_side
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return new_width, new_height


def copy_splits_with_compression(
    split_csvs: Mapping[str, Path],
    src_root: Path,
    dst_root: Path,
    *,
    dataset_marker: str = DATASET_MARKER,
    compression: CompressionSpec | None = CompressionSpec(),
    skip_existing: bool = True,
) -> Dict[str, int]:
    """
    Copy every file referenced by the provided split CSVs into dst_root.

    If `compression` is provided, JPEG files are re-encoded at the supplied
    quality and optionally resized so the shorter side is <= target_short_side.
    Non-JPEG assets (if any) are copied byte-for-byte.

    Returns a tiny summary dict for logging.
    """

    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    rel_paths = _collect_relative_paths(split_csvs, dataset_marker)
    processed = 0
    skipped = 0

    for rel_path in tqdm(rel_paths, desc="Copying dataset to FAST_DATA"):
        src_file = src_root / rel_path
        dst_file = dst_root / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if not src_file.exists():
            raise FileNotFoundError(f"Missing source image on Drive: {src_file}")

        if dst_file.exists() and skip_existing:
            skipped += 1
            continue

        if compression and src_file.suffix.lower() in JPEG_EXTENSIONS:
            _compress_and_save(src_file, dst_file, compression)
        else:
            shutil.copy2(src_file, dst_file)

        processed += 1

    return {
        "total": len(rel_paths),
        "processed": processed,
        "skipped": skipped,
        "dst_root": str(dst_root),
    }


def _collect_relative_paths(split_csvs: Mapping[str, Path], dataset_marker: str) -> Sequence[Path]:
    """
    Read each split CSV and produce the relative (within dataset root) paths
    that must be materialized.
    """

    dataset_marker_lower = dataset_marker.lower()
    unique_paths: set[Path] = set()

    for split_name, csv_path in split_csvs.items():
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found for {split_name}: {csv_path}")
        df = pd.read_csv(csv_path)
        if "path" not in df.columns:
            raise ValueError(f"Split CSV {csv_path} is missing the 'path' column")

        for path_str in df["path"].astype(str):
            rel_path = _path_inside_dataset(path_str, dataset_marker_lower, dataset_marker)
            unique_paths.add(rel_path)

    return sorted(unique_paths)


def _path_inside_dataset(path_str: str, marker_lower: str, marker_original: str) -> Path:
    """
    Convert absolute or relative manifest paths into a path that is relative
    to the dataset root (e.g., c0/img.jpg).
    """

    candidate = Path(path_str)
    if candidate.is_absolute():
        lowered = path_str.lower()
        idx = lowered.find(marker_lower)
        if idx == -1:
            raise ValueError(
                f"Could not locate dataset marker '{marker_original}' inside absolute path: {path_str}"
            )
        relative = Path(path_str[idx:])  # keep original casing
    else:
        relative = candidate

    parts = list(relative.parts)
    if parts and parts[0].lower() == marker_lower:
        parts = parts[1:]

    if not parts:
        raise ValueError(f"Path '{path_str}' does not reference a file inside the dataset root.")

    return Path(*parts)


def _compress_and_save(src_file: Path, dst_file: Path, spec: CompressionSpec) -> None:
    """
    Resize (if needed) and re-encode src_file into dst_file with the given spec.
    """

    with Image.open(src_file) as img:
        img = img.convert("RGB")
        new_size = spec.compute_size(*img.size)
        if new_size != img.size:
            img = img.resize(new_size, Image.BILINEAR)

        save_kwargs = {
            "quality": spec.jpeg_quality,
            "optimize": spec.optimize,
        }
        if spec.progressive:
            save_kwargs["progressive"] = True

        img.save(dst_file, format="JPEG", **save_kwargs)

