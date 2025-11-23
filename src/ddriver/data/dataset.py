"""
Dataset utilities for the AUC Distracted Driver project.

Only the VAL split currently has explicit driver_id assignments. The dataset
handles missing IDs gracefully by returning None/"unknown" for train/test rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

try:
    from ddriver import config
except ImportError:
    raise ImportError("ddriver.config must be importable. Install the package with 'pip install -e .'")

# Mapping from c0..c9 to numeric labels (ensures 0-9 output for the model).
CLASS_TO_INDEX = {
    f"c{i}": i for i in range(10)
}


class AucDriverDataset(Dataset):
    """
    Dataset that joins a global manifest with a split CSV.

    Parameters
    ----------
    manifest_csv : str | Path
        Path to the master manifest containing metadata for every image.
    split_csv : str | Path
        CSV with the exact rows (image paths) to include in this dataset.
    transforms : callable, optional
        Torchvision transforms applied to each loaded image.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split_csv: str | Path,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.split_csv = Path(split_csv)
        self.transforms = transforms

        # Load manifest with metadata columns: path, class_id, driver_id, camera, ...
        manifest_df = pd.read_csv(self.manifest_csv)
        manifest_df["path"] = manifest_df["path"].astype(str)

        # Split CSV is simply a list of paths to include (plus optional columns).
        split_df = pd.read_csv(self.split_csv)
        if "path" not in split_df.columns:
            raise ValueError("split CSV must contain a 'path' column")
        split_df["path"] = split_df["path"].astype(str)

        # Join on path to keep metadata only for the selected rows.
        merged_df = split_df.merge(
            manifest_df,
            on="path",
            how="left",
            validate="one_to_one",
            suffixes=("", "_manifest"),
        )
        missing = merged_df["class_id"].isna().sum()
        if missing:
            raise ValueError(
                f"{missing} entries in {split_csv} were not found in {manifest_csv}"
            )

        # Convert to list of dicts for faster __getitem__ access.
        self.records: List[Dict[str, object]] = merged_df.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.records[idx]
        img_path_str = str(row["path"])
        img_path = Path(img_path_str)

        # Handle relative paths: CSVs should store relative paths for portability
        # Join with DATASET_ROOT if path is relative, or if absolute path doesn't exist
        if not img_path.is_absolute():
            # Relative path: join with DATASET_ROOT (contains auc.distracted.driver.dataset_v2)
            img_path = config.DATASET_ROOT / img_path_str
        elif not img_path.exists():
            # Absolute path doesn't exist (might be from another machine)
            # Try relative to DATASET_ROOT as fallback
            fallback = config.DATASET_ROOT / Path(img_path_str).name
            if fallback.exists():
                img_path = fallback
            # Otherwise use original path (will raise FileNotFoundError if truly missing)

        # Load image from disk (always RGB to keep models happy).
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        to_tensor = self.transforms or T.ToTensor()
        image = to_tensor(image)

        class_id = row["class_id"]
        label = CLASS_TO_INDEX.get(class_id)
        if label is None:
            raise ValueError(f"Unknown class_id '{class_id}' for {img_path}")

        driver_id = row.get("driver_id")
        camera = row.get("camera")

        # default_collate cannot handle None values; use string placeholders instead
        driver_id_clean = str(driver_id) if pd.notna(driver_id) else "unknown_driver"
        camera_clean = str(camera) if pd.notna(camera) else "unknown_camera"

        return {
            "image": image,
            "label": label,
            "driver_id": driver_id_clean,
            "camera": camera_clean,
            "path": str(img_path),
        }

