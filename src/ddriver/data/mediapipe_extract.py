"""
MediaPipe-based ROI extraction for distracted driving images.

Generates cropped variants (face, hands, face+hands union) and writes new
manifest/split CSVs that mirror the originals but point to the cropped images.

Also logs per-image detection metadata for quality auditing.
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


@dataclasses.dataclass
class DetectionMeta:
    """Per-image detection metadata for quality auditing."""
    original_path: str
    cropped_path: str
    face_detected: bool
    left_hand_detected: bool
    right_hand_detected: bool
    hands_count: int  # 0, 1, or 2
    detection_used: str  # "face", "hands", "face_hands", "face_only", "hands_only", "none"
    fallback_to_full: bool
    fallback_reason: str  # "", "no_detection", "area_too_small", "aspect_extreme"
    roi_area_frac: float  # ROI area as fraction of original image
    roi_aspect: float  # width / height
    crop_width: int
    crop_height: int
    original_width: int
    original_height: int
    pad_applied: int
    extra_down_applied: int
    class_id: Optional[int] = None
    camera: Optional[str] = None
    driver_id: Optional[str] = None
    split: Optional[str] = None


def _landmark_box(landmarks, image_w: int, image_h: int) -> Optional[RoiBox]:
    if not landmarks:
        return None
    xs = [lm.x * image_w for lm in landmarks]
    ys = [lm.y * image_h for lm in landmarks]
    xmin, xmax = int(np.min(xs)), int(np.max(xs))
    ymin, ymax = int(np.min(ys)), int(np.max(ys))
    box = RoiBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax).clamp(image_w, image_h)
    return box if box.valid() else None


@dataclasses.dataclass
class DetectionResult:
    """Result of landmark detection with metadata."""
    box: Optional[RoiBox]
    face_detected: bool
    left_hand_detected: bool
    right_hand_detected: bool
    detection_used: str  # what was actually used to form the box


def _select_box(
    results: mp.solutions.holistic.Holistic,
    w: int,
    h: int,
    variant: Variant,
) -> DetectionResult:
    face_box = _landmark_box(
        results.face_landmarks.landmark if results.face_landmarks else None, w, h
    )
    left_hand_box = _landmark_box(
        results.left_hand_landmarks.landmark if results.left_hand_landmarks else None, w, h
    )
    right_hand_box = _landmark_box(
        results.right_hand_landmarks.landmark if results.right_hand_landmarks else None, w, h
    )
    
    face_detected = face_box is not None
    left_hand_detected = left_hand_box is not None
    right_hand_detected = right_hand_box is not None

    if variant == "face":
        return DetectionResult(
            box=face_box,
            face_detected=face_detected,
            left_hand_detected=left_hand_detected,
            right_hand_detected=right_hand_detected,
            detection_used="face" if face_box else "none",
        )
    if variant == "hands":
        if left_hand_box and right_hand_box:
            box = left_hand_box.union(right_hand_box)
            detection_used = "hands"
        elif left_hand_box:
            box = left_hand_box
            detection_used = "left_hand_only"
        elif right_hand_box:
            box = right_hand_box
            detection_used = "right_hand_only"
        else:
            box = None
            detection_used = "none"
        return DetectionResult(
            box=box,
            face_detected=face_detected,
            left_hand_detected=left_hand_detected,
            right_hand_detected=right_hand_detected,
            detection_used=detection_used,
        )
    if variant == "face_hands":
        boxes = [b for b in (face_box, left_hand_box, right_hand_box) if b]
        if not boxes:
            return DetectionResult(
                box=None,
                face_detected=False,
                left_hand_detected=False,
                right_hand_detected=False,
                detection_used="none",
            )
        box = boxes[0]
        for b in boxes[1:]:
            box = box.union(b)
        
        # Determine what combination was used
        parts = []
        if face_detected:
            parts.append("face")
        if left_hand_detected or right_hand_detected:
            parts.append("hands")
        detection_used = "+".join(parts) if parts else "none"
        if detection_used == "face+hands":
            # Distinguish complete (both hands) from partial
            if left_hand_detected and right_hand_detected:
                detection_used = "face+2hands"
            else:
                detection_used = "face+1hand"
        elif detection_used == "hands":
            if left_hand_detected and right_hand_detected:
                detection_used = "hands_only_2"
            else:
                detection_used = "hands_only_1"
        elif detection_used == "face":
            detection_used = "face_only"
        
        return DetectionResult(
            box=box,
            face_detected=face_detected,
            left_hand_detected=left_hand_detected,
            right_hand_detected=right_hand_detected,
            detection_used=detection_used,
        )
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
    max_side: Optional[int] = 720,
    model_complexity: int = 2,
    min_area_frac: float = 0.10,
    min_aspect: float = 0.20,
    pad_frac: float = 0.20,
    face_extra_down_frac: float = 0.35,
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
    detection_metadata: list[DetectionMeta] = []
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
        detection = _select_box(results, w_proc, h_proc, variant)
        
        # Track fallback reasons
        fallback_to_full = False
        fallback_reason = ""
        
        if detection.box is None:
            # No detection at all
            fallback_to_full = True
            fallback_reason = "no_detection"
            box = RoiBox(0, 0, w_orig, h_orig)
        else:
            # Map box back to original resolution
            inv = 1.0 / scale
            box = RoiBox(
                xmin=int(detection.box.xmin * inv),
                ymin=int(detection.box.ymin * inv),
                xmax=int(detection.box.xmax * inv),
                ymax=int(detection.box.ymax * inv),
            ).clamp(w_orig, h_orig)

        # Expand box to reduce overly tight crops
        bw = box.xmax - box.xmin
        bh = box.ymax - box.ymin
        pad = int(max(bw, bh) * pad_frac)

        # If only face was found (hands missing), extend further downward
        if variant == "face_hands" and detection.face_detected and not (detection.left_hand_detected or detection.right_hand_detected):
            extra_down = int(bh * face_extra_down_frac)
        else:
            extra_down = 0

        box_padded = RoiBox(
            xmin=box.xmin - pad,
            ymin=box.ymin - pad,
            xmax=box.xmax + pad,
            ymax=box.ymax + pad + extra_down,
        ).clamp(w_orig, h_orig)

        # Safeguard: if ROI is too small or too skinny/wide, fall back to full frame
        area = (box_padded.xmax - box_padded.xmin) * (box_padded.ymax - box_padded.ymin)
        area_frac = area / float(w_orig * h_orig + 1e-6)
        aspect = (box_padded.xmax - box_padded.xmin) / float(box_padded.ymax - box_padded.ymin + 1e-6)
        
        if not fallback_to_full:
            if area_frac < min_area_frac:
                fallback_to_full = True
                fallback_reason = "area_too_small"
                box_padded = RoiBox(0, 0, w_orig, h_orig)
            elif aspect < min_aspect or aspect > (1.0 / min_aspect):
                fallback_to_full = True
                fallback_reason = "aspect_extreme"
                box_padded = RoiBox(0, 0, w_orig, h_orig)

        # Recalculate final area/aspect after potential fallback
        final_area = (box_padded.xmax - box_padded.xmin) * (box_padded.ymax - box_padded.ymin)
        final_area_frac = final_area / float(w_orig * h_orig + 1e-6)
        final_aspect = (box_padded.xmax - box_padded.xmin) / float(box_padded.ymax - box_padded.ymin + 1e-6)

        crop = image_orig[box_padded.ymin : box_padded.ymax, box_padded.xmin : box_padded.xmax]
        rel = _relative_path(src_path, dataset_root)
        dst_path = output_root / variant / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not dst_path.exists():
            cv2.imwrite(str(dst_path), crop)

        # Build manifest row
        new_row = dict(row)
        new_row["original_path"] = new_row["path"]
        new_row["path"] = str(dst_path.resolve())
        out_records.append(new_row)
        
        # Build detection metadata
        hands_count = int(detection.left_hand_detected) + int(detection.right_hand_detected)
        meta = DetectionMeta(
            original_path=str(src_path),
            cropped_path=str(dst_path.resolve()),
            face_detected=detection.face_detected,
            left_hand_detected=detection.left_hand_detected,
            right_hand_detected=detection.right_hand_detected,
            hands_count=hands_count,
            detection_used=detection.detection_used if not fallback_to_full else "fallback",
            fallback_to_full=fallback_to_full,
            fallback_reason=fallback_reason,
            roi_area_frac=round(final_area_frac, 4),
            roi_aspect=round(final_aspect, 4),
            crop_width=box_padded.xmax - box_padded.xmin,
            crop_height=box_padded.ymax - box_padded.ymin,
            original_width=w_orig,
            original_height=h_orig,
            pad_applied=pad,
            extra_down_applied=extra_down,
            class_id=row.get("class_id"),
            camera=row.get("camera"),
            driver_id=row.get("driver_id"),
            split=row.get("split"),
        )
        detection_metadata.append(meta)

    out_manifest_path = output_root / f"manifest_{variant}.csv"
    pd.DataFrame(out_records).to_csv(out_manifest_path, index=False)
    
    # Save detection metadata for auditing
    meta_df = pd.DataFrame([dataclasses.asdict(m) for m in detection_metadata])
    meta_path = output_root / f"detection_metadata_{variant}.csv"
    meta_df.to_csv(meta_path, index=False)
    
    # Print quick summary stats
    n_total = len(detection_metadata)
    n_fallback = sum(1 for m in detection_metadata if m.fallback_to_full)
    n_face_only = sum(1 for m in detection_metadata if m.detection_used == "face_only")
    n_no_hands = sum(1 for m in detection_metadata if m.hands_count == 0 and not m.fallback_to_full)
    n_one_hand = sum(1 for m in detection_metadata if m.hands_count == 1)
    n_two_hands = sum(1 for m in detection_metadata if m.hands_count == 2)
    
    print(f"\n📊 Detection Summary for variant={variant}:")
    print(f"   Total images: {n_total}")
    print(f"   Fallback to full frame: {n_fallback} ({100*n_fallback/n_total:.1f}%)")
    print(f"   Face-only (no hands): {n_face_only} ({100*n_face_only/n_total:.1f}%)")
    print(f"   0 hands detected: {n_no_hands} ({100*n_no_hands/n_total:.1f}%)")
    print(f"   1 hand detected: {n_one_hand} ({100*n_one_hand/n_total:.1f}%)")
    print(f"   2 hands detected: {n_two_hands} ({100*n_two_hands/n_total:.1f}%)")
    print(f"   Metadata saved to: {meta_path}")

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
        "detection_metadata": meta_path,
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
    parser.add_argument("--model-complexity", type=int, default=2, choices=[0, 1, 2], help="MediaPipe Holistic model complexity (0=faster, 1=default, 2=highest).")
    parser.add_argument("--min-area-frac", type=float, default=0.10, help="Minimum ROI area fraction; fallback to full frame if smaller.")
    parser.add_argument("--min-aspect", type=float, default=0.20, help="Minimum width/height aspect ratio; fallback if more extreme.")
    parser.add_argument("--pad-frac", type=float, default=0.20, help="Padding fraction applied to the detected box.")
    parser.add_argument("--face-extra-down-frac", type=float, default=0.35, help="Extra downward extension (as fraction of box height) when only face is present to include likely hand region.")
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
        min_area_frac=args.min_area_frac,
        min_aspect=args.min_aspect,
        pad_frac=args.pad_frac,
        face_extra_down_frac=args.face_extra_down_frac,
    )
    summary = {k: str(v) if not isinstance(v, dict) else {kk: str(vv) for kk, vv in v.items()} for k, v in result.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

