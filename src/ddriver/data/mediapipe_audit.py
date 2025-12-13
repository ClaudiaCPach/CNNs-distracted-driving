"""
MediaPipe crop quality auditing tools.

Generates:
1. Numeric summary statistics (fallback rates, detection rates by class/camera)
2. Visual inspection grids ("worst suspects", per-class samples, etc.)
3. Exportable reports for thesis documentation

Supports two modes:
- Full audit: Uses detection_metadata CSV from new extraction (has face/hand detection info)
- Lite audit: Works from existing manifest CSVs by inferring fallback from crop dimensions

Path conventions:
- CSVs store RELATIVE paths (e.g., "face_hands/v2_cam1.../c0/img.jpg")
- At runtime, paths are resolved using config.OUT_ROOT or config.FAST_DATA
- This allows portability between local Mac and Colab environments
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ddriver import config


def resolve_crop_path(relative_path: str, crop_root: Path) -> Path:
    """
    Resolve a relative crop path to an absolute path.
    
    Args:
        relative_path: Path stored in CSV (e.g., "face_hands/v2_cam1.../c0/img.jpg")
        crop_root: Base directory (config.OUT_ROOT / "mediapipe" or config.FAST_DATA / "mediapipe")
    
    Returns:
        Absolute path to the crop file
    """
    # Handle legacy absolute paths (for backwards compatibility)
    if Path(relative_path).is_absolute():
        return Path(relative_path)
    return crop_root / relative_path


def get_crop_root(prefer_fast: bool = True) -> Path:
    """
    Get the root directory for MediaPipe crops.
    
    Args:
        prefer_fast: If True and FAST_DATA exists with crops, use it (faster I/O)
    
    Returns:
        Path to mediapipe crops root (either FAST_DATA/mediapipe or OUT_ROOT/mediapipe)
    """
    if prefer_fast and config.FAST_DATA:
        fast_mp = config.FAST_DATA / "mediapipe"
        if fast_mp.exists():
            return fast_mp
    return config.OUT_ROOT / "mediapipe"


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    """Load detection metadata CSV."""
    df = pd.read_csv(metadata_csv)
    # Ensure proper types
    df["fallback_to_full"] = df["fallback_to_full"].astype(bool)
    df["face_detected"] = df["face_detected"].astype(bool)
    df["left_hand_detected"] = df["left_hand_detected"].astype(bool)
    df["right_hand_detected"] = df["right_hand_detected"].astype(bool)
    return df


def build_lite_metadata_from_manifest(
    manifest_csv: Path,
    crop_root: Path,
    dataset_root: Optional[Path] = None,
    sample_originals: int = 100,
) -> pd.DataFrame:
    """
    Build a 'lite' metadata DataFrame from an existing manifest CSV.
    
    This doesn't have face/hand detection info, but can infer:
    - Crop dimensions (by reading cropped images)
    - Whether crop is likely full-frame (by comparing to sampled original dimensions)
    - ROI area fraction and aspect ratio
    
    Args:
        manifest_csv: Path to manifest_{variant}.csv
        crop_root: Base directory for resolving crop paths (e.g., OUT_ROOT/mediapipe)
        dataset_root: Root of original dataset (to read original image dimensions)
        sample_originals: How many original images to sample to estimate typical dimensions
    
    Returns:
        DataFrame with inferred metadata
    """
    df = pd.read_csv(manifest_csv)
    
    # Use config dataset root if not provided
    if dataset_root is None:
        dataset_root = config.DATASET_ROOT
    
    # Determine original dimensions by sampling a few original images
    orig_dims = []
    if "original_path" in df.columns:
        sample_paths = df["original_path"].dropna().head(sample_originals)
        for p in sample_paths:
            try:
                orig_path = Path(p)
                if not orig_path.is_absolute():
                    # Try dataset root first, then crop root parent
                    orig_path = dataset_root / p
                if orig_path.exists():
                    img = cv2.imread(str(orig_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        orig_dims.append((w, h))
            except Exception:
                pass
    
    # Estimate typical original dimensions
    if orig_dims:
        typical_orig_w = int(np.median([d[0] for d in orig_dims]))
        typical_orig_h = int(np.median([d[1] for d in orig_dims]))
        typical_orig_area = typical_orig_w * typical_orig_h
    else:
        # Fallback to common camera resolution
        typical_orig_w, typical_orig_h = 1920, 1080
        typical_orig_area = typical_orig_w * typical_orig_h
    
    print(f"   Estimated original dimensions: {typical_orig_w}x{typical_orig_h}")
    print(f"   Reading crop dimensions from: {crop_root}")
    
    # Read crop dimensions and compute metrics
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading crops"):
        rel_path = row["path"]
        crop_path = resolve_crop_path(rel_path, crop_root)
        
        # Try to read crop dimensions
        crop_w, crop_h = None, None
        try:
            if crop_path.exists():
                img = cv2.imread(str(crop_path))
                if img is not None:
                    crop_h, crop_w = img.shape[:2]
        except Exception:
            pass
        
        if crop_w is None or crop_h is None:
            continue
        
        crop_area = crop_w * crop_h
        area_frac = crop_area / (typical_orig_area + 1e-6)
        aspect = crop_w / (crop_h + 1e-6)
        
        # Infer if this is likely a fallback (crop ≈ full frame)
        is_full_frame = area_frac > 0.95
        
        records.append({
            "cropped_path": rel_path,  # Store relative path
            "original_path": row.get("original_path", ""),
            "crop_width": crop_w,
            "crop_height": crop_h,
            "original_width": typical_orig_w,
            "original_height": typical_orig_h,
            "roi_area_frac": round(area_frac, 4),
            "roi_aspect": round(aspect, 4),
            "fallback_to_full": is_full_frame,
            "fallback_reason": "inferred_full_frame" if is_full_frame else "",
            # These are unknown without re-running detection
            "face_detected": None,
            "left_hand_detected": None,
            "right_hand_detected": None,
            "hands_count": None,
            "detection_used": "unknown",
            "class_id": row.get("class_id"),
            "camera": row.get("camera"),
            "driver_id": row.get("driver_id"),
            "split": row.get("split"),
        })
    
    return pd.DataFrame(records)


def compute_summary_stats(df: pd.DataFrame, lite_mode: bool = False) -> dict:
    """
    Compute overall detection statistics.
    
    Args:
        df: Metadata DataFrame
        lite_mode: If True, skip face/hand detection stats (not available in lite mode)
    """
    n_total = len(df)
    if n_total == 0:
        return {"error": "No data"}
    
    stats = {
        "total_images": n_total,
        "lite_mode": lite_mode,
        "fallback_count": int(df["fallback_to_full"].sum()),
        "fallback_pct": round(100 * df["fallback_to_full"].mean(), 2),
        "fallback_reasons": df[df["fallback_to_full"]]["fallback_reason"].value_counts().to_dict(),
        "roi_area_frac_mean": round(df["roi_area_frac"].mean(), 4),
        "roi_area_frac_std": round(df["roi_area_frac"].std(), 4),
        "roi_area_frac_min": round(df["roi_area_frac"].min(), 4),
        "roi_area_frac_p5": round(df["roi_area_frac"].quantile(0.05), 4),
        "roi_area_frac_p25": round(df["roi_area_frac"].quantile(0.25), 4),
        "roi_area_frac_median": round(df["roi_area_frac"].quantile(0.50), 4),
        "roi_aspect_mean": round(df["roi_aspect"].mean(), 4),
        "roi_aspect_std": round(df["roi_aspect"].std(), 4),
        "roi_aspect_min": round(df["roi_aspect"].min(), 4),
        "roi_aspect_max": round(df["roi_aspect"].max(), 4),
    }
    
    # Add face/hand stats only if available (full mode)
    if not lite_mode and "face_detected" in df.columns and df["face_detected"].notna().any():
        stats.update({
            "face_detected_pct": round(100 * df["face_detected"].mean(), 2),
            "hands_0_pct": round(100 * (df["hands_count"] == 0).mean(), 2),
            "hands_1_pct": round(100 * (df["hands_count"] == 1).mean(), 2),
            "hands_2_pct": round(100 * (df["hands_count"] == 2).mean(), 2),
            "detection_used_distribution": df["detection_used"].value_counts().to_dict(),
        })
    
    return stats


def compute_breakdown_by_column(df: pd.DataFrame, column: str, lite_mode: bool = False) -> pd.DataFrame:
    """Compute detection stats grouped by a column (class_id, camera, split, driver_id)."""
    if column not in df.columns or df[column].isna().all():
        return pd.DataFrame()
    
    # Base aggregations (always available)
    agg_dict = {
        "count": ("fallback_to_full", "count"),
        "fallback_pct": ("fallback_to_full", lambda x: round(100 * x.mean(), 2)),
        "roi_area_mean": ("roi_area_frac", lambda x: round(x.mean(), 4)),
        "roi_aspect_mean": ("roi_aspect", lambda x: round(x.mean(), 4)),
    }
    
    # Add face/hand stats only in full mode
    if not lite_mode and "face_detected" in df.columns and df["face_detected"].notna().any():
        agg_dict.update({
            "face_detected_pct": ("face_detected", lambda x: round(100 * x.mean(), 2)),
            "hands_0_pct": ("hands_count", lambda x: round(100 * (x == 0).mean(), 2)),
            "hands_1_pct": ("hands_count", lambda x: round(100 * (x == 1).mean(), 2)),
            "hands_2_pct": ("hands_count", lambda x: round(100 * (x == 2).mean(), 2)),
        })
    
    grouped = df.groupby(column).agg(**agg_dict).reset_index()
    return grouped


def find_worst_suspects(
    df: pd.DataFrame,
    n: int = 50,
    criteria: str = "area_small",
    lite_mode: bool = False,
) -> pd.DataFrame:
    """
    Find the 'worst' crops according to different criteria.
    
    criteria options:
    - area_small: smallest ROI area fraction (tiny crops)
    - area_large: largest ROI area fraction (near full-frame, maybe fallbacks)
    - aspect_extreme: most extreme aspect ratios
    - fallback: all fallbacks
    - no_hands: face detected but no hands (full mode only)
    - one_hand: only one hand detected (full mode only)
    """
    if criteria == "area_small":
        # Exclude full-frame fallbacks to find actual small crops
        subset = df[~df["fallback_to_full"]]
        if subset.empty:
            subset = df
        return subset.nsmallest(n, "roi_area_frac")
    elif criteria == "area_large":
        # Exclude true full-frame (area_frac ≈ 1.0) to find "almost full"
        subset = df[df["roi_area_frac"] < 0.95]
        if subset.empty:
            return df.head(0)  # Return empty if all are full-frame
        return subset.nlargest(n, "roi_area_frac")
    elif criteria == "aspect_extreme":
        df_copy = df.copy()
        df_copy["aspect_deviation"] = np.abs(np.log(df_copy["roi_aspect"] + 1e-6))
        return df_copy.nlargest(n, "aspect_deviation")
    elif criteria == "fallback":
        return df[df["fallback_to_full"]].head(n)
    elif criteria == "no_hands":
        if lite_mode:
            return df.head(0)  # Not available in lite mode
        return df[(df["face_detected"] == True) & (df["hands_count"] == 0)].head(n)
    elif criteria == "one_hand":
        if lite_mode:
            return df.head(0)  # Not available in lite mode
        return df[df["hands_count"] == 1].head(n)
    else:
        raise ValueError(f"Unknown criteria: {criteria}")


def create_image_grid(
    image_paths: list[str],
    crop_root: Path,
    titles: Optional[list[str]] = None,
    grid_cols: int = 5,
    figsize_per_img: float = 2.5,
    suptitle: Optional[str] = None,
    max_images: int = 25,
) -> plt.Figure:
    """
    Create a grid of images for visual inspection.
    
    Args:
        image_paths: List of relative paths to crop images
        crop_root: Base directory for resolving paths
        titles: Optional titles for each image
        grid_cols: Number of columns in the grid
        figsize_per_img: Size per image in inches
        suptitle: Overall title for the figure
        max_images: Maximum number of images to show
    """
    paths = image_paths[:max_images]
    n = len(paths)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.text(0.5, 0.5, "No images to display", ha="center", va="center")
        ax.axis("off")
        return fig
    
    grid_rows = (n + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(grid_cols * figsize_per_img, grid_rows * figsize_per_img),
    )
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1:
        axes = axes.reshape(1, -1)
    elif grid_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, rel_path in enumerate(paths):
        row, col = divmod(idx, grid_cols)
        ax = axes[row, col]
        
        try:
            abs_path = resolve_crop_path(rel_path, crop_root)
            img = cv2.imread(str(abs_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            else:
                ax.text(0.5, 0.5, "Load failed", ha="center", va="center", fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", fontsize=6)
        
        ax.axis("off")
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=8)
    
    # Hide unused axes
    for idx in range(n, grid_rows * grid_cols):
        row, col = divmod(idx, grid_cols)
        axes[row, col].axis("off")
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    return fig


def generate_audit_report(
    metadata_csv: Optional[Path] = None,
    manifest_csv: Optional[Path] = None,
    crop_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    variant: str = "face_hands",
    n_samples: int = 25,
    save_figures: bool = True,
    show_figures: bool = False,
    prefer_fast: bool = True,
) -> dict:
    """
    Generate a complete audit report with stats and visualizations.
    
    Supports two modes:
    - Full mode: Pass metadata_csv (from new extraction with detection info)
    - Lite mode: Pass manifest_csv (works with existing crops, infers fallback)
    
    Args:
        metadata_csv: Path to detection_metadata_{variant}.csv (full mode)
        manifest_csv: Path to manifest_{variant}.csv (lite mode)
        crop_root: Base directory for crop images. If None, auto-detects using config.
                   Set prefer_fast=True to prefer FAST_DATA if available.
        output_dir: Where to save outputs (optional if only showing figures)
        variant: Variant name for labeling
        n_samples: Number of samples per grid
        save_figures: Whether to save figures to disk
        show_figures: Whether to display figures inline (for Colab)
        prefer_fast: If True and crop_root is None, prefer FAST_DATA over OUT_ROOT
    
    Returns:
        Dict with stats and optionally figures
    """
    # Auto-detect crop root if not provided
    if crop_root is None:
        crop_root = get_crop_root(prefer_fast=prefer_fast)
    crop_root = Path(crop_root)
    
    # Determine mode
    if metadata_csv and Path(metadata_csv).exists():
        print(f"📊 Running FULL audit (detection metadata available)")
        print(f"   Crop root: {crop_root}")
        df = load_metadata(Path(metadata_csv))
        lite_mode = False
    elif manifest_csv and Path(manifest_csv).exists():
        print(f"📊 Running LITE audit (inferring from crop dimensions)")
        print("   Note: Face/hand detection info not available in lite mode.")
        df = build_lite_metadata_from_manifest(
            Path(manifest_csv),
            crop_root=crop_root,
            dataset_root=config.DATASET_ROOT,
        )
        lite_mode = True
    else:
        raise ValueError("Must provide either metadata_csv or manifest_csv")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {"lite_mode": lite_mode, "crop_root": crop_root, "figures": {}}
    
    # 1. Overall summary stats
    stats = compute_summary_stats(df, lite_mode=lite_mode)
    result["stats"] = stats
    
    if output_dir and save_figures:
        stats_path = output_dir / "summary_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"📊 Saved summary stats to {stats_path}")
    
    # 2. Breakdowns by class, camera, split
    result["breakdowns"] = {}
    for col in ["class_id", "camera", "split", "driver_id"]:
        breakdown = compute_breakdown_by_column(df, col, lite_mode=lite_mode)
        if not breakdown.empty:
            result["breakdowns"][col] = breakdown
            if output_dir and save_figures:
                out_path = output_dir / f"breakdown_by_{col}.csv"
                breakdown.to_csv(out_path, index=False)
                print(f"📋 Saved {col} breakdown to {out_path}")
    
    # 3. Visual grids for "worst suspects"
    criteria_list = [
        ("area_small", "Smallest ROI crops (potential micro-crops)"),
        ("aspect_extreme", "Most extreme aspect ratios"),
        ("fallback", "Fallback to full frame"),
    ]
    # Add face/hand criteria only in full mode
    if not lite_mode:
        criteria_list.extend([
            ("no_hands", "Face detected but no hands"),
            ("one_hand", "Only one hand detected"),
        ])
    
    for criteria, title in criteria_list:
        suspects = find_worst_suspects(df, n=n_samples, criteria=criteria, lite_mode=lite_mode)
        if suspects.empty:
            print(f"   ⚠️ No samples for {criteria}")
            continue
        
        # Build titles with metadata
        titles = []
        for _, row in suspects.iterrows():
            title_parts = [f"c{row.get('class_id', '?')}"]
            if pd.notna(row.get("camera")):
                title_parts.append(str(row["camera"]))
            title_parts.append(f"area={row['roi_area_frac']:.2f}")
            title_parts.append(f"asp={row['roi_aspect']:.2f}")
            titles.append(" | ".join(title_parts))
        
        fig = create_image_grid(
            suspects["cropped_path"].tolist(),
            crop_root=crop_root,
            titles=titles,
            suptitle=f"{title} (n={len(suspects)})",
            max_images=n_samples,
        )
        result["figures"][criteria] = fig
        
        if output_dir and save_figures:
            fig_path = output_dir / f"grid_{criteria}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"🖼️  Saved {criteria} grid to {fig_path}")
        
        if show_figures:
            plt.show()
        else:
            plt.close(fig)
    
    # 4. Per-class sample grids (random samples from each class)
    if "class_id" in df.columns and df["class_id"].notna().any():
        for class_id in sorted(df["class_id"].dropna().unique()):
            class_df = df[df["class_id"] == class_id]
            sample = class_df.sample(n=min(n_samples, len(class_df)), random_state=42)
            
            titles = []
            for _, row in sample.iterrows():
                title_parts = [str(row.get("camera", "?"))]
                title_parts.append(f"area={row['roi_area_frac']:.2f}")
                if not lite_mode:
                    title_parts.append(str(row["detection_used"]))
                titles.append(" | ".join(title_parts))
            
            fig = create_image_grid(
                sample["cropped_path"].tolist(),
                crop_root=crop_root,
                titles=titles,
                suptitle=f"Class {int(class_id)} samples (n={len(sample)})",
                max_images=n_samples,
            )
            result["figures"][f"class_{int(class_id)}"] = fig
            
            if output_dir and save_figures:
                fig_path = output_dir / f"grid_class_{int(class_id)}.png"
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            
            if show_figures:
                plt.show()
            else:
                plt.close(fig)
        
        if output_dir and save_figures:
            print(f"🖼️  Saved per-class grids to {output_dir}/grid_class_*.png")
    
    # 5. Print thesis-ready summary sentence
    print("\n" + "=" * 60)
    print("📝 THESIS-READY SUMMARY:")
    print("=" * 60)
    if lite_mode:
        print(f"[LITE MODE - face/hand detection info not available]")
    print(f"MediaPipe extraction produced valid {variant} crops for")
    print(f"{100 - stats['fallback_pct']:.1f}% of images ({stats['total_images'] - stats['fallback_count']}/{stats['total_images']}).")
    print(f"Fallback to full frame occurred in {stats['fallback_pct']:.1f}% of cases.")
    if not lite_mode and "face_detected_pct" in stats:
        print(f"Face detection rate: {stats['face_detected_pct']:.1f}%")
        print(f"Hand detection: 0 hands={stats['hands_0_pct']:.1f}%, 1 hand={stats['hands_1_pct']:.1f}%, 2 hands={stats['hands_2_pct']:.1f}%")
    print(f"ROI area fraction: mean={stats['roi_area_frac_mean']:.2f}, std={stats['roi_area_frac_std']:.2f}")
    print("=" * 60 + "\n")
    
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MediaPipe crop quality.")
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Path to detection_metadata_<variant>.csv from extraction (full mode).",
    )
    parser.add_argument(
        "--manifest-csv",
        default=None,
        help="Path to manifest_<variant>.csv (lite mode, for existing crops).",
    )
    parser.add_argument(
        "--crop-root",
        default=None,
        help="Root directory for crop images. If not provided, auto-detects using config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save audit outputs (default: same dir as input CSV).",
    )
    parser.add_argument(
        "--variant",
        default="face_hands",
        help="Variant name for labeling (face/hands/face_hands).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=25,
        help="Number of samples per grid.",
    )
    parser.add_argument(
        "--prefer-fast",
        action="store_true",
        help="Prefer FAST_DATA location if available (for faster I/O).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.metadata_csv and not args.manifest_csv:
        raise ValueError("Must provide either --metadata-csv or --manifest-csv")
    
    # Determine output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.metadata_csv:
        output_dir = Path(args.metadata_csv).parent / "audit"
    else:
        output_dir = Path(args.manifest_csv).parent / "audit"
    
    generate_audit_report(
        metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
        manifest_csv=Path(args.manifest_csv) if args.manifest_csv else None,
        crop_root=Path(args.crop_root) if args.crop_root else None,
        output_dir=output_dir,
        variant=args.variant,
        n_samples=args.n_samples,
        save_figures=True,
        show_figures=False,
        prefer_fast=args.prefer_fast,
    )


if __name__ == "__main__":
    main()

