"""
Hybrid ROI extraction using InsightFace (face) + MediaPipe Hands (hands).

This combines:
- InsightFace: RetinaFace via ONNX (no TensorFlow!), handles occlusion/angles well
- MediaPipe Hands: Google's dedicated hand model, much better than Holistic

Generates cropped variants (face, hands, face+hands union) and writes new
manifest/split CSVs that mirror the originals but point to the cropped images.

Also logs per-image detection metadata for quality auditing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from tqdm import tqdm

# MediaPipe compatibility - handle both old and new API
try:
    import mediapipe as mp
    # Try old API first (mediapipe < 0.10.14)
    _mp_hands_module = mp.solutions.hands
    _MediaPipeHands = _mp_hands_module.Hands
except AttributeError:
    # New API (mediapipe >= 0.10.14) - try alternative import
    try:
        from mediapipe.python.solutions import hands as _mp_hands_module
        _MediaPipeHands = _mp_hands_module.Hands
    except ImportError:
        # Fallback: install older version hint
        raise ImportError(
            "MediaPipe hands module not found. Try: pip install mediapipe==0.10.9"
        )

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
    left_hand_detected: bool
    right_hand_detected: bool
    hand_count: int
    face_confidence: float  # max confidence among face detections (0 if none)
    left_hand_confidence: float
    right_hand_confidence: float
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
    skipped: bool = False  # True if image was skipped (for hands/face-only variants with fallback)
    class_id: Optional[int] = None
    camera: Optional[str] = None
    driver_id: Optional[str] = None
    split: Optional[str] = None


def _detect_faces_insightface(
    image_bgr: np.ndarray,
    face_app: FaceAnalysis,
) -> list[tuple[RoiBox, float]]:
    """Detect faces using InsightFace (RetinaFace via ONNX).
    
    Returns list of (box, confidence) tuples.
    """
    h, w = image_bgr.shape[:2]
    boxes_with_conf = []
    
    try:
        faces = face_app.get(image_bgr)
        
        for face in faces:
            bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
            confidence = float(face.det_score)
            box = RoiBox(
                xmin=int(bbox[0]),
                ymin=int(bbox[1]),
                xmax=int(bbox[2]),
                ymax=int(bbox[3]),
            ).clamp(w, h)
            if box.valid():
                boxes_with_conf.append((box, confidence))
    except Exception:
        # InsightFace can fail on some images
        pass
    
    return boxes_with_conf


def _detect_hands_mediapipe(
    image_rgb: np.ndarray,
    hands_detector,  # MediaPipe Hands detector (type varies by version)
) -> tuple[Optional[tuple[RoiBox, float]], Optional[tuple[RoiBox, float]]]:
    """Detect hands using MediaPipe Hands.
    
    Returns (left_hand, right_hand) where each is (box, confidence) or None.
    MediaPipe Hands is much more accurate than Holistic for hand detection.
    """
    h, w = image_rgb.shape[:2]
    
    results = hands_detector.process(image_rgb)
    
    left_hand = None
    right_hand = None
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get hand label (Left/Right) and confidence
            label = handedness.classification[0].label
            confidence = handedness.classification[0].score
            
            # Get bounding box from landmarks
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            
            box = RoiBox(
                xmin=int(min(xs)),
                ymin=int(min(ys)),
                xmax=int(max(xs)),
                ymax=int(max(ys)),
            ).clamp(w, h)
            
            if box.valid():
                if label == "Left":
                    # Note: MediaPipe mirrors, so "Left" in image is actually right hand
                    right_hand = (box, confidence)
                else:
                    left_hand = (box, confidence)
    
    return left_hand, right_hand


@dataclasses.dataclass 
class HybridDetectionResult:
    """Result of hybrid detection with metadata."""
    box: Optional[RoiBox]
    face_detected: bool
    face_count: int
    left_hand_detected: bool
    right_hand_detected: bool
    hand_count: int
    face_confidence: float
    left_hand_confidence: float
    right_hand_confidence: float
    detection_used: str


def _select_box_hybrid(
    face_boxes: list[tuple[RoiBox, float]],
    left_hand: Optional[tuple[RoiBox, float]],
    right_hand: Optional[tuple[RoiBox, float]],
    variant: Variant,
) -> HybridDetectionResult:
    """Select the appropriate bounding box based on variant from hybrid detections."""
    
    face_detected = len(face_boxes) > 0
    face_count = len(face_boxes)
    left_hand_detected = left_hand is not None
    right_hand_detected = right_hand is not None
    hand_count = int(left_hand_detected) + int(right_hand_detected)
    
    face_confidence = max((conf for _, conf in face_boxes), default=0.0)
    left_hand_confidence = left_hand[1] if left_hand else 0.0
    right_hand_confidence = right_hand[1] if right_hand else 0.0
    
    if variant == "face":
        if face_boxes:
            # Union all face boxes
            box = face_boxes[0][0]
            for fb, _ in face_boxes[1:]:
                box = box.union(fb)
            detection_used = "face"
        else:
            box = None
            detection_used = "none"
        return HybridDetectionResult(
            box=box,
            face_detected=face_detected,
            face_count=face_count,
            left_hand_detected=left_hand_detected,
            right_hand_detected=right_hand_detected,
            hand_count=hand_count,
            face_confidence=face_confidence,
            left_hand_confidence=left_hand_confidence,
            right_hand_confidence=right_hand_confidence,
            detection_used=detection_used,
        )
    
    if variant == "hands":
        hand_boxes = []
        if left_hand:
            hand_boxes.append(left_hand[0])
        if right_hand:
            hand_boxes.append(right_hand[0])
        
        if hand_boxes:
            box = hand_boxes[0]
            for hb in hand_boxes[1:]:
                box = box.union(hb)
            detection_used = "hands"
        else:
            box = None
            detection_used = "none"
        return HybridDetectionResult(
            box=box,
            face_detected=face_detected,
            face_count=face_count,
            left_hand_detected=left_hand_detected,
            right_hand_detected=right_hand_detected,
            hand_count=hand_count,
            face_confidence=face_confidence,
            left_hand_confidence=left_hand_confidence,
            right_hand_confidence=right_hand_confidence,
            detection_used=detection_used,
        )
    
    # variant == "face_hands"
    # REQUIRE both face AND at least one hand for a valid face_hands crop
    if not face_detected or hand_count == 0:
        # Missing face or hands - return None to trigger skip
        if not face_detected and hand_count == 0:
            detection_used = "none"
        elif not face_detected:
            detection_used = "hands_only_rejected"
        else:
            detection_used = "face_only_rejected"
        return HybridDetectionResult(
            box=None,
            face_detected=face_detected,
            face_count=face_count,
            left_hand_detected=left_hand_detected,
            right_hand_detected=right_hand_detected,
            hand_count=hand_count,
            face_confidence=face_confidence,
            left_hand_confidence=left_hand_confidence,
            right_hand_confidence=right_hand_confidence,
            detection_used=detection_used,
        )
    
    # Both face and hands detected - union all boxes
    all_boxes = [fb for fb, _ in face_boxes]
    if left_hand:
        all_boxes.append(left_hand[0])
    if right_hand:
        all_boxes.append(right_hand[0])
    
    box = all_boxes[0]
    for b in all_boxes[1:]:
        box = box.union(b)
    
    detection_used = f"face+{hand_count}hand" if hand_count == 1 else f"face+{hand_count}hands"
    
    return HybridDetectionResult(
        box=box,
        face_detected=face_detected,
        face_count=face_count,
        left_hand_detected=left_hand_detected,
        right_hand_detected=right_hand_detected,
        hand_count=hand_count,
        face_confidence=face_confidence,
        left_hand_confidence=left_hand_confidence,
        right_hand_confidence=right_hand_confidence,
        detection_used=detection_used,
    )


def _relative_path(path: Path, dataset_root: Path) -> Path:
    """
    Best-effort to preserve substructure relative to dataset root.
    
    CRITICAL: Always preserves Camera/split/class structure to avoid filename collisions.
    The dataset has identical filenames across Camera 1/Camera 2 and train/test.
    """
    parts = path.parts
    
    # Strategy 1: Find "Camera" folder and preserve from there
    # This handles: .../Camera 1/train/c0/551.jpg -> Camera 1/train/c0/551.jpg
    for i, part in enumerate(parts):
        if part.lower().startswith("camera"):
            return Path(*parts[i:])
    
    # Strategy 2: Find train/test folder and preserve from there  
    # This handles edge cases where Camera folder is missing
    for i, part in enumerate(parts):
        if part.lower() in ("train", "test"):
            return Path(*parts[i:])
    
    # Strategy 3: Find class folder (c0-c9) and preserve from there
    # Last resort - still better than just filename
    for i, part in enumerate(parts):
        if len(part) == 2 and part.startswith('c') and part[1].isdigit():
            return Path(*parts[i:])
    
    # Absolute last resort: just filename (will cause collisions!)
    import warnings
    warnings.warn(f"Could not find Camera/train/class folder in path {path}, using filename only. This may cause collisions!")
    return Path(path.name)


def _extract_class(path_str: str) -> Optional[str]:
    for part in Path(path_str).parts:
        if re.fullmatch(r"c[0-9]", part):
            return part
    return None


def _extract_camera(path_str: str) -> Optional[str]:
    for part in Path(path_str).parts:
        if part.lower().startswith("camera"):
            return part
    return None


def _coerce_class_id(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    value_str = str(value)
    if re.fullmatch(r"c[0-9]", value_str):
        return value_str
    if value_str.isdigit():
        return f"c{int(value_str)}"
    return None


def extract_rois_hybrid(
    manifest_csv: Path,
    split_csvs: dict[str, Path],
    output_root: Path,
    dataset_root: Path,
    variant: Variant,
    overwrite: bool = False,
    min_detection_area_frac: float = 0.05,
    min_area_frac: float = 0.10,
    min_aspect: float = 0.20,
    pad_frac: float = 0.20,
    max_area_frac: Optional[float] = None,
    min_face_conf: float = 0.4,
    sample_csv: Optional[Path] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Extract ROI crops using hybrid RetinaFace + MediaPipe Hands detection.
    
    Args:
        manifest_csv: Path to the original manifest CSV.
        split_csvs: Dict mapping split names to their CSV paths.
        output_root: Where to write cropped images and new CSVs.
        dataset_root: Root of the original dataset.
        variant: "face", "hands", or "face_hands".
        overwrite: Whether to overwrite existing crops.
        min_detection_area_frac: Minimum raw detection area fraction before padding.
        min_area_frac: Minimum padded ROI area fraction.
        min_aspect: Minimum width/height aspect ratio.
        pad_frac: Padding fraction applied to the detected box.
        max_area_frac: Maximum padded ROI area fraction; fallback to full frame if larger.
                       Useful for ablations (hands-only/face-only) to avoid including other modalities.
        min_face_conf: Minimum face detection confidence threshold; faces below this are filtered out.
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

    # Initialize InsightFace (RetinaFace via ONNX - no TensorFlow!)
    # Using 'buffalo_sc' which is lightweight and fast
    face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    
    # Initialize MediaPipe Hands (NOT Holistic - much better accuracy)
    mp_hands = _MediaPipeHands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    
    print(f"🔧 Using InsightFace (RetinaFace/ONNX) for faces + MediaPipe Hands for hands")
    print(f"   Variant: {variant}")
    print(f"   Min face confidence: {min_face_conf}")

    out_records = []
    detection_metadata: list[DetectionMeta] = []
    total = len(manifest)

    for _, row in tqdm(manifest.iterrows(), total=total, desc=f"hybrid-{variant}", dynamic_ncols=True):
        src_path = Path(row["path"])
        if not src_path.is_absolute():
            src_path = dataset_root / src_path
        if not src_path.exists():
            continue

        image_bgr = cv2.imread(str(src_path))
        if image_bgr is None:
            continue
        h_orig, w_orig = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Detect faces with InsightFace (RetinaFace via ONNX)
        face_boxes = _detect_faces_insightface(image_bgr, face_app)
        
        # Filter out low-confidence faces
        face_boxes = [(box, conf) for box, conf in face_boxes if conf >= min_face_conf]
        
        # Detect hands with MediaPipe Hands
        left_hand, right_hand = _detect_hands_mediapipe(image_rgb, mp_hands)
        
        # Select appropriate box based on variant
        detection = _select_box_hybrid(face_boxes, left_hand, right_hand, variant)
        
        # Track fallback reasons and raw detection area
        fallback_to_full = False
        fallback_reason = ""
        raw_detection_area_frac = 0.0
        
        if detection.box is None:
            # No detection at all
            fallback_to_full = True
            fallback_reason = "no_detection"
            box = RoiBox(0, 0, w_orig, h_orig)
        elif detection.face_count > 1:
            # Skip multi-face images to avoid giant union crops from false positives
            fallback_to_full = True
            fallback_reason = "multi_face"
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
            elif max_area_frac is not None and area_frac > max_area_frac:
                # For hands-only or face-only, reject crops that are too large
                # (likely include other modality or too much background)
                fallback_to_full = True
                fallback_reason = "area_too_large"
                box_padded = RoiBox(0, 0, w_orig, h_orig)

        # Recalculate final area/aspect after potential fallback
        final_area_frac = box_padded.area() / float(w_orig * h_orig + 1e-6)
        final_aspect = (box_padded.xmax - box_padded.xmin) / float(box_padded.ymax - box_padded.ymin + 1e-6)

        # Compute relative paths early (needed for metadata and camera/class extraction)
        rel = _relative_path(src_path, dataset_root)
        crop_rel_path = str(Path(variant) / rel)
        orig_rel_path = str(rel)

        camera_value = _extract_camera(str(src_path)) or _extract_camera(orig_rel_path)
        class_value = _extract_class(str(src_path)) or _extract_class(orig_rel_path)
        if class_value is None:
            class_value = _coerce_class_id(row.get("class_id"))

        # Skip saving ANY fallback case for ALL variants
        # This ensures every saved image is a valid ROI crop (no full-frame fallbacks).
        # Metadata CSV still logs all attempts including failed ones.
        skip_save = False
        if fallback_to_full:
            skip_save = True

        crop = image_bgr[box_padded.ymin : box_padded.ymax, box_padded.xmin : box_padded.xmax]
        dst_path = output_root / variant / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not skip_save and (overwrite or not dst_path.exists()):
            cv2.imwrite(str(dst_path), crop)

        # Build manifest row - store RELATIVE paths for portability
        # Only add to manifest if we actually saved the image (or it already exists)
        
        if not skip_save:
            new_row = dict(row)
            new_row["original_path"] = orig_rel_path
            new_row["path"] = crop_rel_path
            new_row["fallback_to_full"] = fallback_to_full
            if not new_row.get("class_id") and class_value is not None:
                new_row["class_id"] = class_value
            if not new_row.get("camera") and camera_value is not None:
                new_row["camera"] = camera_value
            out_records.append(new_row)
        
        # Build detection metadata (tracks ALL images, including skipped ones)
        meta = DetectionMeta(
            original_path=orig_rel_path,
            cropped_path=crop_rel_path if not skip_save else "",  # Empty if skipped
            face_detected=detection.face_detected,
            face_count=detection.face_count,
            left_hand_detected=detection.left_hand_detected,
            right_hand_detected=detection.right_hand_detected,
            hand_count=detection.hand_count,
            face_confidence=round(detection.face_confidence, 4),
            left_hand_confidence=round(detection.left_hand_confidence, 4),
            right_hand_confidence=round(detection.right_hand_confidence, 4),
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
            skipped=skip_save,
            class_id=class_value,
            camera=camera_value,
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
    n_saved = len(out_records)
    n_skipped = sum(1 for m in detection_metadata if m.skipped)
    n_fallback = sum(1 for m in detection_metadata if m.fallback_to_full)
    n_face_only = sum(1 for m in detection_metadata if "face_only" in m.detection_used)
    n_with_hands = sum(1 for m in detection_metadata if m.hand_count > 0 and not m.fallback_to_full)
    n_both_hands = sum(1 for m in detection_metadata if m.hand_count == 2 and not m.fallback_to_full)
    avg_face_conf = np.mean([m.face_confidence for m in detection_metadata if m.face_detected]) if any(m.face_detected for m in detection_metadata) else 0
    avg_hand_conf = np.mean([max(m.left_hand_confidence, m.right_hand_confidence) for m in detection_metadata if m.hand_count > 0]) if any(m.hand_count > 0 for m in detection_metadata) else 0
    
    # Fallback reason breakdown (only for non-skipped fallbacks)
    fallback_reasons = {}
    skip_reasons = {}
    for m in detection_metadata:
        if m.fallback_to_full and m.fallback_reason:
            if m.skipped:
                skip_reasons[m.fallback_reason] = skip_reasons.get(m.fallback_reason, 0) + 1
            else:
                fallback_reasons[m.fallback_reason] = fallback_reasons.get(m.fallback_reason, 0) + 1
    
    print(f"\n📊 Hybrid Detection Summary for variant={variant}:")
    print(f"   Total images processed: {n_total}")
    print(f"   Images SAVED: {n_saved} ({100*n_saved/n_total:.1f}%)")
    if n_skipped > 0:
        print(f"   Images SKIPPED: {n_skipped} ({100*n_skipped/n_total:.1f}%)")
        if skip_reasons:
            print(f"   Skip reasons (no valid {variant} ROI):")
            for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
                print(f"      - {reason}: {count} ({100*count/n_total:.1f}%)")
    if n_fallback > n_skipped:
        # Note: with current policy all fallbacks are skipped, so this should not trigger
        print(f"   Fallback to full frame (saved): {n_fallback - n_skipped} ({100*(n_fallback - n_skipped)/n_total:.1f}%)")
        if fallback_reasons:
            print(f"   Fallback reasons:")
            for reason, count in sorted(fallback_reasons.items(), key=lambda x: -x[1]):
                print(f"      - {reason}: {count} ({100*count/n_total:.1f}%)")
    print(f"   Face-only (no hands): {n_face_only} ({100*n_face_only/n_total:.1f}%)")
    print(f"   With 1+ hands detected: {n_with_hands} ({100*n_with_hands/n_total:.1f}%)")
    print(f"   With both hands: {n_both_hands} ({100*n_both_hands/n_total:.1f}%)")
    print(f"   Avg face confidence: {avg_face_conf:.3f}")
    print(f"   Avg hand confidence: {avg_hand_conf:.3f}")
    print(f"   Metadata saved to: {meta_path}")

    # Write split CSVs that mirror originals but point to new paths
    out_splits = {}
    
    # Handle empty out_records (all images skipped)
    if len(out_records) == 0:
        print(f"\n⚠️  WARNING: All images were skipped for variant={variant}!")
        print(f"   No crops saved, no split CSVs generated.")
        print(f"   This is expected if no {variant} detections were found.")
        for name, split_path in split_csvs.items():
            out_split_path = output_root / f"{name}_{variant}.csv"
            # Create empty CSV with same columns
            split_df = pd.read_csv(split_path)
            empty_df = split_df.head(0).copy()
            empty_df["original_path"] = []
            empty_df.to_csv(out_split_path, index=False)
            out_splits[name] = out_split_path
    else:
        mapping_df = pd.DataFrame(out_records)[["original_path", "path", "fallback_to_full", "camera", "class_id"]]
        mapping_df["original_path"] = mapping_df["original_path"].astype(str)
        mapping_df["_filename"] = mapping_df["original_path"].map(lambda p: Path(p).name)
        mapping_df["_class"] = mapping_df["original_path"].map(_extract_class)
        mapping_df["_class"] = mapping_df["_class"].fillna(mapping_df["class_id"].map(_coerce_class_id))
        mapping_df["_camera"] = mapping_df["original_path"].map(_extract_camera)
        mapping_df["_camera"] = mapping_df["_camera"].fillna(mapping_df["camera"])
        # Deduplicate in case manifest had duplicate paths
        mapping_df = mapping_df.drop_duplicates(subset=["_filename", "_class", "_camera"])
        # Note: fallback_to_full is always False in out_records now (fallbacks are skipped)
        
        for name, split_path in split_csvs.items():
            split_df = pd.read_csv(split_path)
            split_df["path"] = split_df["path"].astype(str)

            split_df["_filename"] = split_df["path"].map(lambda p: Path(p).name)
            split_df["_class"] = split_df["path"].map(_extract_class)
            if "class_id" in split_df.columns:
                split_df["_class"] = split_df["_class"].fillna(split_df["class_id"].map(_coerce_class_id))
            split_df["_camera"] = split_df["path"].map(_extract_camera)

            merged = split_df.merge(
                mapping_df,
                on=["_filename", "_class", "_camera"],
                how="inner",
            )

            merged["path"] = merged["path_y"]
            merged = merged.drop(columns=["path_y", "_filename", "_class", "_camera", "fallback_to_full"])
            merged = merged.rename(columns={"path_x": "original_path"})
            out_split_path = output_root / f"{name}_{variant}.csv"
            merged.to_csv(out_split_path, index=False)
            out_splits[name] = out_split_path
            
            # Report split sizes
            original_count = len(pd.read_csv(split_path))
            new_count = len(merged)
            if new_count < original_count:
                print(f"   {name}: {new_count}/{original_count} images ({100*new_count/original_count:.1f}%)")

    mp_hands.close()
    
    return {
        "manifest": out_manifest_path,
        "splits": out_splits,
        "detection_metadata": meta_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ROI crops using InsightFace (face) + MediaPipe Hands (hands)."
    )
    parser.add_argument("--manifest", required=True, help="Path to original manifest CSV.")
    parser.add_argument("--splits-root", required=True, help="Directory containing train/val/test CSVs.")
    parser.add_argument("--dataset-root", default=None, help="Root of the original dataset (defaults to config.DATASET_ROOT).")
    parser.add_argument("--output-root", default=None, help="Where to write cropped images and new CSVs (defaults to config.OUT_ROOT/hybrid).")
    parser.add_argument("--variant", choices=["face", "hands", "face_hands"], required=True, help="ROI variant to extract.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crops.")
    parser.add_argument("--min-detection-area-frac", type=float, default=0.05, help="Minimum RAW detection area fraction (before padding); fallback if too small.")
    parser.add_argument("--min-area-frac", type=float, default=0.10, help="Minimum PADDED ROI area fraction; fallback to full frame if smaller.")
    parser.add_argument("--min-aspect", type=float, default=0.20, help="Minimum width/height aspect ratio; fallback if more extreme.")
    parser.add_argument("--pad-frac", type=float, default=0.20, help="Padding fraction applied to the detected box.")
    parser.add_argument("--max-area-frac", type=float, default=None, help="Maximum PADDED ROI area fraction; fallback to full frame if larger (useful for hands-only/face-only to avoid including other modalities).")
    parser.add_argument("--min-face-conf", type=float, default=0.4, help="Minimum face detection confidence; faces below this are filtered out.")
    parser.add_argument("--sample-csv", default=None, help="Optional CSV with subset of paths to process (e.g., train_small.csv for testing).")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images to process (for quick testing).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root) if args.dataset_root else config.DATASET_ROOT
    output_root = Path(args.output_root) if args.output_root else (config.OUT_ROOT / "hybrid")
    splits_root = Path(args.splits_root)
    split_csvs = {
        "train": splits_root / "train.csv",
        "val": splits_root / "val.csv",
        "test": splits_root / "test.csv",
    }
    for name, p in split_csvs.items():
        if not p.exists():
            raise FileNotFoundError(p)

    result = extract_rois_hybrid(
        manifest_csv=Path(args.manifest),
        split_csvs=split_csvs,
        output_root=output_root,
        dataset_root=dataset_root,
        variant=args.variant,  # type: ignore
        overwrite=args.overwrite,
        min_detection_area_frac=args.min_detection_area_frac,
        min_area_frac=args.min_area_frac,
        min_aspect=args.min_aspect,
        pad_frac=args.pad_frac,
        max_area_frac=args.max_area_frac,
        min_face_conf=args.min_face_conf,
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
