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
        img_path = Path(row["path"])

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

        return {
            "image": image,
            "label": label,
            "driver_id": driver_id if pd.notna(driver_id) else None,
            "camera": camera if pd.notna(camera) else None,
            "path": str(img_path),
        }

