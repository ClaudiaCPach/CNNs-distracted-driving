"""
ID set operations for experimental filtering.

This module provides utilities for:
1. Extracting ID sets from manifests (which images have which crops)
2. Filtering split CSVs by ID sets (for control experiments)
3. Generating control splits for the 5-run experimental plan

The "ID" for an image is its full original path from the manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

try:
    from ddriver import config
except ImportError:
    config = None


def extract_id_sets(
    manifest_full: Path,
    manifest_face: Optional[Path] = None,
    manifest_fh: Optional[Path] = None,
) -> Dict[str, Set[str]]:
    """
    Extract ID sets from manifests.
    
    The "ID" is the full original path. For crop manifests, we use the
    'original_path' column which maps back to the full-frame path.
    
    Args:
        manifest_full: Path to full-frame manifest CSV (has 'path' column)
        manifest_face: Path to face crop manifest CSV (has 'original_path' column)
        manifest_fh: Path to face+hands crop manifest CSV (has 'original_path' column)
    
    Returns:
        Dict with keys: "full", "face", "fh", "both"
        Each value is a set of path strings.
    """
    # Read full-frame manifest
    df_full = pd.read_csv(manifest_full)
    ids_full = set(df_full["path"].astype(str).tolist())
    
    # Read face manifest (if provided)
    ids_face: Set[str] = set()
    if manifest_face and manifest_face.exists():
        df_face = pd.read_csv(manifest_face)
        # Crop manifests store the original full-frame path in 'original_path'
        if "original_path" in df_face.columns:
            ids_face = set(df_face["original_path"].astype(str).tolist())
        else:
            # Fallback: use 'path' if original_path doesn't exist
            ids_face = set(df_face["path"].astype(str).tolist())
    
    # Read face+hands manifest (if provided)
    ids_fh: Set[str] = set()
    if manifest_fh and manifest_fh.exists():
        df_fh = pd.read_csv(manifest_fh)
        if "original_path" in df_fh.columns:
            ids_fh = set(df_fh["original_path"].astype(str).tolist())
        else:
            ids_fh = set(df_fh["path"].astype(str).tolist())
    
    # Compute intersection (both face AND face+hands available)
    ids_both = ids_face & ids_fh if (ids_face and ids_fh) else set()
    
    return {
        "full": ids_full,
        "face": ids_face,
        "fh": ids_fh,
        "both": ids_both,
    }


def filter_split_by_ids(
    split_csv: Path,
    id_set: Set[str],
    output_csv: Path,
) -> int:
    """
    Filter a split CSV to only include paths in the ID set.
    
    Preserves all columns and driver-disjointness (filtering is per-row).
    
    Args:
        split_csv: Path to input split CSV (train.csv, val.csv, or test.csv)
        id_set: Set of paths to keep
        output_csv: Where to write the filtered CSV
    
    Returns:
        Number of rows in filtered output
    """
    df = pd.read_csv(split_csv)
    df["path"] = df["path"].astype(str)
    
    # Filter to only rows where path is in the ID set
    mask = df["path"].isin(id_set)
    filtered = df[mask].copy()
    
    # Write output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    
    return len(filtered)


def generate_control_splits(
    splits_root: Path,
    id_sets: Dict[str, Set[str]],
    output_root: Path,
    generate_both: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    Generate all control split CSVs for the 5-run experimental plan.
    
    Creates:
    - {train,val,test}_facesubset.csv: Full-frame paths filtered to face-available IDs
    - {train,val,test}_fhsubset.csv: Full-frame paths filtered to FH-available IDs
    - (Optional) {train,val,test}_both.csv: Full-frame paths filtered to both-available IDs
    
    Args:
        splits_root: Directory containing original train.csv, val.csv, test.csv
        id_sets: Dict from extract_id_sets()
        output_root: Where to write control split CSVs
        generate_both: Whether to also generate the _both.csv files (for S_both approach)
    
    Returns:
        Dict mapping subset name to dict of split name to output path
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    split_names = ["train", "val", "test"]
    results: Dict[str, Dict[str, Path]] = {}
    
    # Generate face-subset control splits
    if id_sets.get("face"):
        results["facesubset"] = {}
        print(f"\n📁 Generating face-subset control splits...")
        for split_name in split_names:
            input_csv = splits_root / f"{split_name}.csv"
            if not input_csv.exists():
                print(f"   ⚠️  {input_csv} not found, skipping")
                continue
            output_csv = output_root / f"{split_name}_facesubset.csv"
            count = filter_split_by_ids(input_csv, id_sets["face"], output_csv)
            original_count = len(pd.read_csv(input_csv))
            print(f"   {split_name}: {count}/{original_count} ({100*count/original_count:.1f}%)")
            results["facesubset"][split_name] = output_csv
    
    # Generate FH-subset control splits
    if id_sets.get("fh"):
        results["fhsubset"] = {}
        print(f"\n📁 Generating face+hands-subset control splits...")
        for split_name in split_names:
            input_csv = splits_root / f"{split_name}.csv"
            if not input_csv.exists():
                print(f"   ⚠️  {input_csv} not found, skipping")
                continue
            output_csv = output_root / f"{split_name}_fhsubset.csv"
            count = filter_split_by_ids(input_csv, id_sets["fh"], output_csv)
            original_count = len(pd.read_csv(input_csv))
            print(f"   {split_name}: {count}/{original_count} ({100*count/original_count:.1f}%)")
            results["fhsubset"][split_name] = output_csv
    
    # Generate both-subset control splits (optional, for S_both approach)
    if generate_both and id_sets.get("both"):
        results["both"] = {}
        print(f"\n📁 Generating both-subset control splits (S_both approach)...")
        for split_name in split_names:
            input_csv = splits_root / f"{split_name}.csv"
            if not input_csv.exists():
                print(f"   ⚠️  {input_csv} not found, skipping")
                continue
            output_csv = output_root / f"{split_name}_both.csv"
            count = filter_split_by_ids(input_csv, id_sets["both"], output_csv)
            original_count = len(pd.read_csv(input_csv))
            print(f"   {split_name}: {count}/{original_count} ({100*count/original_count:.1f}%)")
            results["both"][split_name] = output_csv
    
    return results


def save_id_sets(
    id_sets: Dict[str, Set[str]],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Save ID sets to text files for reference and auditing.
    
    Args:
        id_sets: Dict from extract_id_sets()
        output_dir: Where to write the ID set files
    
    Returns:
        Dict mapping set name to output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    for name, ids in id_sets.items():
        if ids:
            out_path = output_dir / f"ids_{name}.txt"
            with open(out_path, "w") as f:
                for id_str in sorted(ids):
                    f.write(f"{id_str}\n")
            paths[name] = out_path
            print(f"   ids_{name}.txt: {len(ids)} IDs")
    
    return paths


def print_id_set_summary(id_sets: Dict[str, Set[str]]) -> None:
    """Print a summary of ID set sizes and overlaps."""
    print("\n📊 ID Set Summary:")
    print(f"   Full-frame:   {len(id_sets.get('full', set())):,} images")
    print(f"   Face:         {len(id_sets.get('face', set())):,} images")
    print(f"   Face+Hands:   {len(id_sets.get('fh', set())):,} images")
    print(f"   Both:         {len(id_sets.get('both', set())):,} images")
    
    # Compute coverage percentages
    n_full = len(id_sets.get("full", set()))
    if n_full > 0:
        n_face = len(id_sets.get("face", set()))
        n_fh = len(id_sets.get("fh", set()))
        n_both = len(id_sets.get("both", set()))
        print(f"\n   Coverage vs full-frame:")
        print(f"   Face:         {100*n_face/n_full:.1f}%")
        print(f"   Face+Hands:   {100*n_fh/n_full:.1f}%")
        print(f"   Both:         {100*n_both/n_full:.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate control splits for the 5-run experimental plan."
    )
    parser.add_argument(
        "--manifest-full",
        required=True,
        help="Path to full-frame manifest CSV."
    )
    parser.add_argument(
        "--manifest-face",
        default=None,
        help="Path to face crop manifest CSV (with original_path column)."
    )
    parser.add_argument(
        "--manifest-fh",
        default=None,
        help="Path to face+hands crop manifest CSV (with original_path column)."
    )
    parser.add_argument(
        "--splits-root",
        required=True,
        help="Directory containing train.csv, val.csv, test.csv."
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Where to write control split CSVs (defaults to splits-root/control)."
    )
    parser.add_argument(
        "--save-id-sets",
        action="store_true",
        help="Also save ID sets as text files for auditing."
    )
    parser.add_argument(
        "--no-both",
        action="store_true",
        help="Skip generating the _both.csv files (S_both approach)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    manifest_full = Path(args.manifest_full)
    manifest_face = Path(args.manifest_face) if args.manifest_face else None
    manifest_fh = Path(args.manifest_fh) if args.manifest_fh else None
    splits_root = Path(args.splits_root)
    output_root = Path(args.output_root) if args.output_root else (splits_root / "control")
    
    print("🔍 Extracting ID sets from manifests...")
    id_sets = extract_id_sets(manifest_full, manifest_face, manifest_fh)
    print_id_set_summary(id_sets)
    
    if args.save_id_sets:
        id_sets_dir = output_root / "id_sets"
        print(f"\n💾 Saving ID sets to {id_sets_dir}...")
        save_id_sets(id_sets, id_sets_dir)
    
    print("\n🔧 Generating control splits...")
    results = generate_control_splits(
        splits_root=splits_root,
        id_sets=id_sets,
        output_root=output_root,
        generate_both=not args.no_both,
    )
    
    print(f"\n✅ Control splits saved to: {output_root}")


if __name__ == "__main__":
    main()

