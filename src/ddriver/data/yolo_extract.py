"""
YOLO-World based ROI extraction for distracted driving images.

Uses open-vocabulary detection to find faces and hands, then generates
cropped variants (face, hands, face+hands union) and writes new
manifest/split CSVs that mirror the originals but point to the cropped images.

Also logs per-image detection metadata for quality auditing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from ddriver import config

Variant = Literal["face", "hands", "face_hands"]


@dataclasses.dataclass
class RoiBox:
    """Bounding box for a region of interest."""
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

    def area(self) -> int:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


@dataclasses.dataclass
class DetectionMeta:
    """Per-image detection metadata for quality auditing."""
    original_path: str
    cropped_path: str
    face_detected: bool
    face_count: int
    hand_count: int
    face_confidence: float  # max confidence among face detections (0 if none)
    hand_confidence: float  # max confidence among hand detections (0 if none)
    detection_used: str  # "face", "hands", "face_hands", "face_only", "hands_only", "none", "fallback"
    fallback_to_full: bool
    fallback_reason: str  # "", "no_detection", "detection_too_small", "area_too_small", "aspect_extreme"
    raw_detection_area_frac: float  # RAW detection area (before padding) as fraction of original image
    roi_area_frac: float  # FINAL ROI area (after padding) as fraction of original image
    roi_aspect: float  # width / height
    crop_width: int
    crop_height: int
    original_width: int
    original_height: int
    pad_applied: int
    class_id: Optional[int] = None
    camera: Optional[str] = None
    driver_id: Optional[str] = None
    split: Optional[str] = None


@dataclasses.dataclass
class YoloDetectionResult:
    """Result of YOLO detection with metadata."""
    box: Optional[RoiBox]
    face_detected: bool
    face_count: int
    hand_count: int
    face_confidence: float
    hand_confidence: float
    detection_used: str


def _get_class_boxes(
    results,
    class_keyword: str,
    w: int,
    h: int,
) -> list[tuple[RoiBox, float]]:
    """Extract bounding boxes and confidences for classes containing keyword from YOLO results.
    
    Args:
        results: YOLO prediction results
        class_keyword: Keyword to match in class names (e.g., "face" matches "human face", "person face")
        w: Image width
        h: Image height
    """
    boxes_with_conf = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                # Get class name from results
                cls_name = result.names.get(cls_id, "")
                # Match if keyword is in class name (e.g., "face" matches "human face")
                if class_keyword.lower() in cls_name.lower():
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    roi = RoiBox(
                        xmin=int(xyxy[0]),
                        ymin=int(xyxy[1]),
                        xmax=int(xyxy[2]),
                        ymax=int(xyxy[3]),
                    ).clamp(w, h)
                    if roi.valid():
                        boxes_with_conf.append((roi, conf))
    return boxes_with_conf


def _select_box_yolo(
    results,
    w: int,
    h: int,
    variant: Variant,
) -> YoloDetectionResult:
    """Select the appropriate bounding box based on variant from YOLO results."""
    # Get all face and hand detections
    face_boxes = _get_class_boxes(results, "face", w, h)
    hand_boxes = _get_class_boxes(results, "hand", w, h)
    
    face_detected = len(face_boxes) > 0
    face_count = len(face_boxes)
    hand_count = len(hand_boxes)
    face_confidence = max((conf for _, conf in face_boxes), default=0.0)
    hand_confidence = max((conf for _, conf in hand_boxes), default=0.0)
    
    if variant == "face":
        if face_boxes:
            # Union all face boxes (in case of multiple faces)
            box = face_boxes[0][0]
            for fb, _ in face_boxes[1:]:
                box = box.union(fb)
            detection_used = "face"
        else:
            box = None
            detection_used = "none"
        return YoloDetectionResult(
            box=box,
            face_detected=face_detected,
            face_count=face_count,
            hand_count=hand_count,
            face_confidence=face_confidence,
            hand_confidence=hand_confidence,
            detection_used=detection_used,
        )
    
    if variant == "hands":
        if hand_boxes:
            # Union all hand boxes
            box = hand_boxes[0][0]
            for hb, _ in hand_boxes[1:]:
                box = box.union(hb)
            detection_used = "hands"
        else:
            box = None
            detection_used = "none"
        return YoloDetectionResult(
            box=box,
            face_detected=face_detected,
            face_count=face_count,
            hand_count=hand_count,
            face_confidence=face_confidence,
            hand_confidence=hand_confidence,
            detection_used=detection_used,
        )
    
    # variant == "face_hands"
    all_boxes = [b for b, _ in face_boxes] + [b for b, _ in hand_boxes]
    if not all_boxes:
        return YoloDetectionResult(
            box=None,
            face_detected=False,
            face_count=0,
            hand_count=0,
            face_confidence=0.0,
            hand_confidence=0.0,
            detection_used="none",
        )
    
    # Union all detected boxes
    box = all_boxes[0]
    for b in all_boxes[1:]:
        box = box.union(b)
    
    # Determine detection type
    if face_detected and hand_count > 0:
        detection_used = f"face+{hand_count}hand" if hand_count == 1 else f"face+{hand_count}hands"
    elif face_detected:
        detection_used = "face_only"
    else:
        detection_used = f"{hand_count}hand" if hand_count == 1 else f"{hand_count}hands"
    
    return YoloDetectionResult(
        box=box,
        face_detected=face_detected,
        face_count=face_count,
        hand_count=hand_count,
        face_confidence=face_confidence,
        hand_confidence=hand_confidence,
        detection_used=detection_used,
    )


def _relative_path(path: Path, dataset_root: Path) -> Path:
    """
    Best-effort to preserve substructure relative to dataset root.
    
    CRITICAL: Always preserves class subfolder (c0-c9) to avoid filename collisions.
    Different classes have files with the same names (0.jpg, 1.jpg, etc.).
    """
    try:
        return path.relative_to(dataset_root)
    except ValueError:
        # Fallback: find class folder (c0-c9) in path and preserve from there
        parts = path.parts
        for i, part in enumerate(parts):
            if len(part) == 2 and part.startswith('c') and part[1].isdigit():
                # Found class folder - preserve from here (e.g., c0/0.jpg)
                return Path(*parts[i:])
        # Last resort: just filename (but this will cause collisions!)
        import warnings
        warnings.warn(f"Could not find class folder in path {path}, using filename only. This may cause collisions!")
        return Path(path.name)


def extract_rois_yolo(
    manifest_csv: Path,
    split_csvs: dict[str, Path],
    output_root: Path,
    dataset_root: Path,
    variant: Variant,
    overwrite: bool = False,
    model_size: str = "m",
    confidence: float = 0.15,
    min_detection_area_frac: float = 0.05,
    min_area_frac: float = 0.10,
    min_aspect: float = 0.20,
    pad_frac: float = 0.20,
    sample_csv: Optional[Path] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Extract ROI crops using YOLO-World open-vocabulary detection.
    
    Args:
        manifest_csv: Path to the original manifest CSV.
        split_csvs: Dict mapping split names to their CSV paths.
        output_root: Where to write cropped images and new CSVs.
        dataset_root: Root of the original dataset.
        variant: "face", "hands", or "face_hands".
        overwrite: Whether to overwrite existing crops.
        model_size: YOLO-World model size ("s", "m", "l", "x").
        confidence: Minimum confidence threshold for detections.
        min_detection_area_frac: Minimum raw detection area fraction before padding.
        min_area_frac: Minimum padded ROI area fraction.
        min_aspect: Minimum width/height aspect ratio.
        pad_frac: Padding fraction applied to the detected box.
        sample_csv: Optional CSV with subset of paths to process (e.g., train_small.csv for testing).
        limit: Optional max number of images to process (for quick testing).
    
    Returns:
        Dict with paths to output manifest, splits, and detection metadata.
    """
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_csv)
    manifest["path"] = manifest["path"].astype(str)
    
    # If sample_csv provided, filter manifest to only those paths
    if sample_csv is not None:
        sample_df = pd.read_csv(sample_csv)
        sample_paths = set(sample_df["path"].astype(str))
        original_len = len(manifest)
        manifest = manifest[manifest["path"].isin(sample_paths)]
        print(f"📋 Using sample CSV: {sample_csv}")
        print(f"   Filtered from {original_len} to {len(manifest)} images")
    
    # If limit provided, take only first N images
    if limit is not None and limit > 0:
        manifest = manifest.head(limit)
        print(f"⚡ Limited to first {limit} images (for quick testing)")

    # Load YOLO-World model
    model_name = f"yolov8{model_size}-worldv2.pt"
    print(f"Loading YOLO-World model: {model_name}")
    model = YOLO(model_name)
    
    # Set classes based on variant
    # Using more descriptive prompts for better YOLO-World detection
    if variant == "face":
        model.set_classes(["human face", "person face"])
    elif variant == "hands":
        model.set_classes(["human hand", "person hand"])
    else:  # face_hands
        model.set_classes(["human face", "person face", "human hand", "person hand"])
    
    print(f"Detecting classes: {model.names}")

    out_records = []
    detection_metadata: list[DetectionMeta] = []
    total = len(manifest)

    for _, row in tqdm(manifest.iterrows(), total=total, desc=f"yolo-{variant}", dynamic_ncols=True):
        src_path = Path(row["path"])
        if not src_path.is_absolute():
            src_path = dataset_root / src_path
        if not src_path.exists():
            continue

        image_orig = cv2.imread(str(src_path))
        if image_orig is None:
            continue
        h_orig, w_orig = image_orig.shape[:2]

        # Run YOLO inference
        results = model.predict(
            source=image_orig,
            conf=confidence,
            verbose=False,
        )
        
        detection = _select_box_yolo(results, w_orig, h_orig, variant)
        
        # Track fallback reasons and raw detection area
        fallback_to_full = False
        fallback_reason = ""
        raw_detection_area_frac = 0.0
        
        if detection.box is None:
            # No detection at all
            fallback_to_full = True
            fallback_reason = "no_detection"
            box = RoiBox(0, 0, w_orig, h_orig)
        else:
            box = detection.box
            
            # Calculate raw detection area BEFORE padding
            raw_detection_area_frac = box.area() / float(w_orig * h_orig + 1e-6)
            
            # Check raw detection box - catches tiny detections
            if raw_detection_area_frac < min_detection_area_frac:
                fallback_to_full = True
                fallback_reason = "detection_too_small"
                box = RoiBox(0, 0, w_orig, h_orig)

        # Expand box to reduce overly tight crops
        bw = box.xmax - box.xmin
        bh = box.ymax - box.ymin
        pad = int(max(bw, bh) * pad_frac)

        box_padded = RoiBox(
            xmin=box.xmin - pad,
            ymin=box.ymin - pad,
            xmax=box.xmax + pad,
            ymax=box.ymax + pad,
        ).clamp(w_orig, h_orig)

        # Safeguard: if ROI is too small or too skinny/wide, fall back to full frame
        area_frac = box_padded.area() / float(w_orig * h_orig + 1e-6)
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
        final_area_frac = box_padded.area() / float(w_orig * h_orig + 1e-6)
        final_aspect = (box_padded.xmax - box_padded.xmin) / float(box_padded.ymax - box_padded.ymin + 1e-6)

        crop = image_orig[box_padded.ymin : box_padded.ymax, box_padded.xmin : box_padded.xmax]
        rel = _relative_path(src_path, dataset_root)
        dst_path = output_root / variant / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not dst_path.exists():
            cv2.imwrite(str(dst_path), crop)

        # Build manifest row - store RELATIVE paths for portability
        crop_rel_path = str(Path(variant) / rel)
        orig_rel_path = str(rel)
        
        new_row = dict(row)
        new_row["original_path"] = orig_rel_path
        new_row["path"] = crop_rel_path
        out_records.append(new_row)
        
        # Build detection metadata
        meta = DetectionMeta(
            original_path=orig_rel_path,
            cropped_path=crop_rel_path,
            face_detected=detection.face_detected,
            face_count=detection.face_count,
            hand_count=detection.hand_count,
            face_confidence=round(detection.face_confidence, 4),
            hand_confidence=round(detection.hand_confidence, 4),
            detection_used=detection.detection_used if not fallback_to_full else "fallback",
            fallback_to_full=fallback_to_full,
            fallback_reason=fallback_reason,
            raw_detection_area_frac=round(raw_detection_area_frac, 4),
            roi_area_frac=round(final_area_frac, 4),
            roi_aspect=round(final_aspect, 4),
            crop_width=box_padded.xmax - box_padded.xmin,
            crop_height=box_padded.ymax - box_padded.ymin,
            original_width=w_orig,
            original_height=h_orig,
            pad_applied=pad,
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
    n_face_only = sum(1 for m in detection_metadata if "face_only" in m.detection_used)
    n_with_hands = sum(1 for m in detection_metadata if m.hand_count > 0 and not m.fallback_to_full)
    avg_face_conf = np.mean([m.face_confidence for m in detection_metadata if m.face_detected]) if any(m.face_detected for m in detection_metadata) else 0
    avg_hand_conf = np.mean([m.hand_confidence for m in detection_metadata if m.hand_count > 0]) if any(m.hand_count > 0 for m in detection_metadata) else 0
    
    # Fallback reason breakdown
    fallback_reasons = {}
    for m in detection_metadata:
        if m.fallback_to_full and m.fallback_reason:
            fallback_reasons[m.fallback_reason] = fallback_reasons.get(m.fallback_reason, 0) + 1
    
    print(f"\n📊 YOLO Detection Summary for variant={variant}:")
    print(f"   Total images: {n_total}")
    print(f"   Fallback to full frame: {n_fallback} ({100*n_fallback/n_total:.1f}%)")
    if fallback_reasons:
        print(f"   Fallback reasons:")
        for reason, count in sorted(fallback_reasons.items(), key=lambda x: -x[1]):
            print(f"      - {reason}: {count} ({100*count/n_total:.1f}%)")
    print(f"   Face-only (no hands): {n_face_only} ({100*n_face_only/n_total:.1f}%)")
    print(f"   With hands detected: {n_with_hands} ({100*n_with_hands/n_total:.1f}%)")
    print(f"   Avg face confidence: {avg_face_conf:.3f}")
    print(f"   Avg hand confidence: {avg_hand_conf:.3f}")
    print(f"   Metadata saved to: {meta_path}")

    # Write split CSVs that mirror originals but point to new paths
    out_splits = {}
    mapping_df = pd.DataFrame(out_records)[["original_path", "path"]]
    # Deduplicate in case manifest had duplicate paths
    mapping_df = mapping_df.drop_duplicates(subset="original_path")
    for name, split_path in split_csvs.items():
        split_df = pd.read_csv(split_path)
        split_df["path"] = split_df["path"].astype(str)
        merged = split_df.merge(
            mapping_df,
            left_on="path",
            right_on="original_path",
            how="left",
        )
        # Prefer new path; fall back to original if extraction failed
        merged["path"] = merged["path_y"].fillna(merged["path_x"])
        merged = merged.drop(columns=["original_path", "path_y"])
        merged = merged.rename(columns={"path_x": "original_path"})
        out_split_path = output_root / f"{name}_{variant}.csv"
        merged.to_csv(out_split_path, index=False)
        out_splits[name] = out_split_path

    return {
        "manifest": out_manifest_path,
        "splits": out_splits,
        "detection_metadata": meta_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract YOLO-World ROI crops and write new manifest/splits."
    )
    parser.add_argument("--manifest", required=True, help="Path to original manifest CSV.")
    parser.add_argument("--splits-root", required=True, help="Directory containing train/val/test CSVs.")
    parser.add_argument("--dataset-root", default=None, help="Root of the original dataset (defaults to config.DATASET_ROOT).")
    parser.add_argument("--output-root", default=None, help="Where to write cropped images and new CSVs (defaults to config.OUT_ROOT/yolo).")
    parser.add_argument("--variant", choices=["face", "hands", "face_hands"], required=True, help="ROI variant to extract.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crops.")
    parser.add_argument("--model-size", choices=["s", "m", "l", "x"], default="m", help="YOLO-World model size (s=fast, m=balanced, l/x=accurate). Default: m")
    parser.add_argument("--confidence", type=float, default=0.15, help="Minimum detection confidence threshold. Default: 0.15")
    parser.add_argument("--min-detection-area-frac", type=float, default=0.05, help="Minimum RAW detection area fraction (before padding); fallback if too small.")
    parser.add_argument("--min-area-frac", type=float, default=0.10, help="Minimum PADDED ROI area fraction; fallback to full frame if smaller.")
    parser.add_argument("--min-aspect", type=float, default=0.20, help="Minimum width/height aspect ratio; fallback if more extreme.")
    parser.add_argument("--pad-frac", type=float, default=0.20, help="Padding fraction applied to the detected box.")
    parser.add_argument("--sample-csv", default=None, help="Optional CSV with subset of paths to process (e.g., train_small.csv for testing).")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images to process (for quick testing).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root) if args.dataset_root else config.DATASET_ROOT
    output_root = Path(args.output_root) if args.output_root else (config.OUT_ROOT / "yolo")
    splits_root = Path(args.splits_root)
    split_csvs = {
        "train": splits_root / "train.csv",
        "val": splits_root / "val.csv",
        "test": splits_root / "test.csv",
    }
    for name, p in split_csvs.items():
        if not p.exists():
            raise FileNotFoundError(p)

    result = extract_rois_yolo(
        manifest_csv=Path(args.manifest),
        split_csvs=split_csvs,
        output_root=output_root,
        dataset_root=dataset_root,
        variant=args.variant,  # type: ignore
        overwrite=args.overwrite,
        model_size=args.model_size,
        confidence=args.confidence,
        min_detection_area_frac=args.min_detection_area_frac,
        min_area_frac=args.min_area_frac,
        min_aspect=args.min_aspect,
        pad_frac=args.pad_frac,
        sample_csv=Path(args.sample_csv) if args.sample_csv else None,
        limit=args.limit,
    )
    summary = {
        k: str(v) if not isinstance(v, dict) else {kk: str(vv) for kk, vv in v.items()}
        for k, v in result.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

