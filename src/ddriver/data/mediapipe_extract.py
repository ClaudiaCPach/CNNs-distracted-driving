"""
MediaPipe-based ROI extraction for distracted driving images.

Generates cropped variants (face, hands, face+hands union) and writes new
manifest/split CSVs that mirror the originals but point to the cropped images.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from ddriver import config

Variant = Literal["face", "hands", "face_hands"]


@dataclasses.dataclass
class RoiBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def clamp(self, w: int, h: int) -> "RoiBox":
        return RoiBox(
            xmin=max(0, min(self.xmin, w - 1)),
            ymin=max(0, min(self.ymin, h - 1)),
            xmax=max(0, min(self.xmax, w - 1)),
            ymax=max(0, min(self.ymax, h - 1)),
        )

    def valid(self) -> bool:
        return self.xmax > self.xmin and self.ymax > self.ymin

    def union(self, other: "RoiBox") -> "RoiBox":
        return RoiBox(
            xmin=min(self.xmin, other.xmin),
            ymin=min(self.ymin, other.ymin),
            xmax=max(self.xmax, other.xmax),
            ymax=max(self.ymax, other.ymax),
        )


def _landmark_box(landmarks, image_w: int, image_h: int) -> Optional[RoiBox]:
    if not landmarks:
        return None
    xs = [lm.x * image_w for lm in landmarks]
    ys = [lm.y * image_h for lm in landmarks]
    xmin, xmax = int(np.min(xs)), int(np.max(xs))
    ymin, ymax = int(np.min(ys)), int(np.max(ys))
    box = RoiBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax).clamp(image_w, image_h)
    return box if box.valid() else None


def _select_box(
    results: mp.solutions.holistic.Holistic,
    w: int,
    h: int,
    variant: Variant,
) -> Optional[RoiBox]:
    face_box = _landmark_box(
        results.face_landmarks.landmark if results.face_landmarks else None, w, h
    )
    left_hand_box = _landmark_box(
        results.left_hand_landmarks.landmark if results.left_hand_landmarks else None, w, h
    )
    right_hand_box = _landmark_box(
        results.right_hand_landmarks.landmark if results.right_hand_landmarks else None, w, h
    )

    if variant == "face":
        return face_box
    if variant == "hands":
        if left_hand_box and right_hand_box:
            return left_hand_box.union(right_hand_box)
        return left_hand_box or right_hand_box
    if variant == "face_hands":
        boxes = [b for b in (face_box, left_hand_box, right_hand_box) if b]
        if not boxes:
            return None
        box = boxes[0]
        for b in boxes[1:]:
            box = box.union(b)
        return box
    raise ValueError(f"Unknown variant {variant}")


def _relative_path(path: Path, dataset_root: Path) -> Path:
    """
    Best-effort to preserve substructure relative to dataset root.
    """
    try:
        rel = path.relative_to(dataset_root)
        return rel
    except ValueError:
        return Path(path.name)


def extract_rois(
    manifest_csv: Path,
    split_csvs: dict[str, Path],
    output_root: Path,
    dataset_root: Path,
    variant: Variant,
    overwrite: bool = False,
    max_side: Optional[int] = 640,
    model_complexity: int = 1,
) -> dict:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_csv)
    manifest["path"] = manifest["path"].astype(str)

    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=model_complexity,
        refine_face_landmarks=True,
    )

    out_records = []
    total = len(manifest)

    for _, row in tqdm(manifest.iterrows(), total=total, desc=f"mediapipe-{variant}", dynamic_ncols=True):
        src_path = Path(row["path"])
        if not src_path.is_absolute():
            src_path = dataset_root / src_path
        if not src_path.exists():
            continue

        image_orig = cv2.imread(str(src_path))
        if image_orig is None:
            continue
        h_orig, w_orig = image_orig.shape[:2]

        # Optional downscale for speed
        scale = 1.0
        image_proc = image_orig
        if max_side and max(h_orig, w_orig) > max_side:
            scale = max_side / float(max(h_orig, w_orig))
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            image_proc = cv2.resize(image_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h_proc, w_proc = image_proc.shape[:2]
        image_rgb = cv2.cvtColor(image_proc, cv2.COLOR_BGR2RGB)
        results = mp_holistic.process(image_rgb)
        box_proc = _select_box(results, w_proc, h_proc, variant)
        if box_proc is None and scale != 1.0:
            # fallback: no detection, treat as full resized frame
            box_proc = RoiBox(0, 0, w_proc, h_proc)

        # Map box back to original resolution for cropping/saving
        if box_proc is None:
            box = RoiBox(0, 0, w_orig, h_orig)
        else:
            inv = 1.0 / scale
            box = RoiBox(
                xmin=int(box_proc.xmin * inv),
                ymin=int(box_proc.ymin * inv),
                xmax=int(box_proc.xmax * inv),
                ymax=int(box_proc.ymax * inv),
            ).clamp(w_orig, h_orig)
        if box is None:
            # fallback to full image
            box = RoiBox(0, 0, w_orig, h_orig)
        crop = image_orig[box.ymin : box.ymax, box.xmin : box.xmax]
        rel = _relative_path(src_path, dataset_root)
        dst_path = output_root / variant / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not dst_path.exists():
            cv2.imwrite(str(dst_path), crop)

        new_row = dict(row)
        new_row["original_path"] = new_row["path"]
        new_row["path"] = str(dst_path.resolve())
        out_records.append(new_row)

    out_manifest_path = output_root / f"manifest_{variant}.csv"
    pd.DataFrame(out_records).to_csv(out_manifest_path, index=False)

    # Write split CSVs that mirror originals but point to new paths.
    out_splits = {}
    mapping_df = pd.DataFrame(out_records)[["original_path", "path"]]
    for name, split_path in split_csvs.items():
        split_df = pd.read_csv(split_path)
        split_df["path"] = split_df["path"].astype(str)
        merged = split_df.merge(
            mapping_df,
            left_on="path",
            right_on="original_path",
            how="left",
            validate="one_to_one",
        )
        # Prefer new path; fall back to original if extraction failed.
        merged["path"] = merged["path_y"].fillna(merged["path_x"])
        merged = merged.drop(columns=["original_path", "path_y"])
        merged = merged.rename(columns={"path_x": "original_path"})
        out_split_path = output_root / f"{name}_{variant}.csv"
        merged.to_csv(out_split_path, index=False)
        out_splits[name] = out_split_path

    mp_holistic.close()
    return {
        "manifest": out_manifest_path,
        "splits": out_splits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MediaPipe ROI crops and write new manifest/splits.")
    parser.add_argument("--manifest", required=True, help="Path to original manifest CSV.")
    parser.add_argument("--splits-root", required=True, help="Directory containing train/val/test CSVs.")
    parser.add_argument("--dataset-root", default=None, help="Root of the original dataset (defaults to config.DATASET_ROOT).")
    parser.add_argument("--output-root", default=None, help="Where to write cropped images and new CSVs (defaults to config.OUT_ROOT/mediapipe).")
    parser.add_argument("--variant", choices=["face", "hands", "face_hands"], required=True, help="ROI variant to extract.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crops.")
    parser.add_argument("--max-side", type=int, default=None, help="Optional max size (long side) to downscale before inference for speed.")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2], help="MediaPipe Holistic model complexity (0=faster, 1=default, 2=highest).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root) if args.dataset_root else config.DATASET_ROOT
    output_root = Path(args.output_root) if args.output_root else (config.OUT_ROOT / "mediapipe")
    splits_root = Path(args.splits_root)
    split_csvs = {
        "train": splits_root / "train.csv",
        "val": splits_root / "val.csv",
        "test": splits_root / "test.csv",
    }
    for name, p in split_csvs.items():
        if not p.exists():
            raise FileNotFoundError(p)

    result = extract_rois(
        manifest_csv=Path(args.manifest),
        split_csvs=split_csvs,
        output_root=output_root,
        dataset_root=dataset_root,
        variant=args.variant,  # type: ignore
        overwrite=args.overwrite,
        max_side=args.max_side,
        model_complexity=args.model_complexity,
    )
    summary = {k: str(v) if not isinstance(v, dict) else {kk: str(vv) for kk, vv in v.items()} for k, v in result.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

