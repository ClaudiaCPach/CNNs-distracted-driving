"""Utilities that copy the Distracted Driver dataset into FAST_DATA with optional compression.

This module centralizes the logic that the Colab notebook uses when it wants
to materialize a smaller, IO-friendly copy of the dataset inside `/content/data`.
It keeps the folder structure identical to the source while optionally resizing
and recompressing JPEGs so they occupy less space without sacrificing the pose
information that the models rely on.

TAR ARCHIVE SUPPORT
-------------------
Copying ~13,000 small files one-by-one over a FUSE-mounted Google Drive is very
slow (often 2+ hours). The tar-based approach bundles files into a single archive,
which can be copied as one large file (much faster) and then extracted locally.

Workflow:
1. ONE-TIME: Create compressed images + tar archive on Drive
2. EACH SESSION: Copy tar from Drive → /content (fast), extract locally (instant)

This typically turns 2-hour copy operations into 5-10 minutes.
"""

from __future__ import annotations

import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import shutil

__all__ = [
    "CompressionSpec",
    "copy_splits_with_compression",
    "create_tar_archive",
    "extract_tar_archive",
    "fast_copy_from_tar",
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


# =============================================================================
# TAR ARCHIVE UTILITIES
# =============================================================================

def create_tar_archive(
    source_dir: Path,
    tar_path: Path,
    *,
    use_gzip: bool = False,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Create a tar archive from a directory.
    
    This is used ONE TIME to bundle images into a single archive for faster
    copying from Google Drive to /content on subsequent sessions.
    
    Parameters
    ----------
    source_dir : Path
        Directory containing files to archive (e.g., /content/data/hybrid/face)
    tar_path : Path
        Output tar file path (e.g., /content/hybrid_face.tar)
    use_gzip : bool
        If True, create .tar.gz (smaller but slower to extract). Default False
        because extraction speed matters more than archive size for our use case.
    verbose : bool
        Print progress messages
        
    Returns
    -------
    dict with keys: n_files, size_bytes, tar_path
    """
    source_dir = Path(source_dir)
    tar_path = Path(tar_path)
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Count files for progress
    all_files = list(source_dir.rglob("*"))
    file_count = sum(1 for f in all_files if f.is_file())
    
    if verbose:
        print(f"📦 Creating tar archive from {source_dir}")
        print(f"   Files to archive: {file_count}")
    
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = "w:gz" if use_gzip else "w"
    with tarfile.open(tar_path, mode) as tar:
        # Add all files with relative paths
        for file_path in tqdm(all_files, desc="Archiving", disable=not verbose):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir)
                tar.add(file_path, arcname=arcname)
    
    size_bytes = tar_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    if verbose:
        print(f"✅ Created: {tar_path}")
        print(f"   Size: {size_mb:.1f} MB")
    
    return {
        "n_files": file_count,
        "size_bytes": size_bytes,
        "tar_path": str(tar_path),
    }


def extract_tar_archive(
    tar_path: Path,
    dest_dir: Path,
    *,
    remove_tar_after: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Extract a tar archive to a directory.
    
    Parameters
    ----------
    tar_path : Path
        Path to tar file (can be .tar or .tar.gz)
    dest_dir : Path
        Destination directory for extracted files
    remove_tar_after : bool
        Delete the tar file after successful extraction to save space
    verbose : bool
        Print progress messages
        
    Returns
    -------
    dict with keys: n_files, dest_dir
    """
    tar_path = Path(tar_path)
    dest_dir = Path(dest_dir)
    
    if not tar_path.exists():
        raise FileNotFoundError(f"Tar file not found: {tar_path}")
    
    if verbose:
        size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(f"📂 Extracting {tar_path.name} ({size_mb:.1f} MB)")
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect compression
    mode = "r:gz" if str(tar_path).endswith(".gz") else "r"
    
    with tarfile.open(tar_path, mode) as tar:
        members = tar.getmembers()
        if verbose:
            print(f"   Extracting {len(members)} files...")
        tar.extractall(dest_dir)
    
    if remove_tar_after:
        tar_path.unlink()
        if verbose:
            print(f"   🗑️  Removed tar file to save space")
    
    if verbose:
        print(f"✅ Extracted to {dest_dir}")
    
    return {
        "n_files": len(members),
        "dest_dir": str(dest_dir),
    }


def fast_copy_from_tar(
    tar_path_on_drive: Path,
    dest_dir: Path,
    *,
    local_tar_path: Optional[Path] = None,
    check_marker_file: Optional[str] = None,
    remove_tar_after: bool = True,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Copy a tar archive from Drive to local storage and extract it.
    
    This is the main entry point for fast data loading in Colab notebooks.
    It replaces the slow file-by-file copy with a single tar copy + extract.
    
    Parameters
    ----------
    tar_path_on_drive : Path
        Path to tar file on Google Drive (e.g., /content/drive/MyDrive/TFM/data/full_compressed.tar)
    dest_dir : Path
        Destination directory for extracted files (e.g., /content/data/hybrid/face)
    local_tar_path : Path, optional
        Where to copy tar before extracting. Defaults to /content/{tar_name}
    check_marker_file : str, optional
        If provided, check for this file in dest_dir to skip if already extracted
        (e.g., "c0/img_001.jpg" or just check for any .jpg)
    remove_tar_after : bool
        Delete the tar file after extraction
    verbose : bool
        Print progress messages
        
    Returns
    -------
    dict with status info
    """
    tar_path_on_drive = Path(tar_path_on_drive)
    dest_dir = Path(dest_dir)
    
    # Check if already extracted
    if check_marker_file:
        marker = dest_dir / check_marker_file
        if marker.exists():
            if verbose:
                print(f"✅ Data already extracted in {dest_dir}")
            return {"status": "skipped", "reason": "already_extracted"}
    else:
        # Default check: look for any jpg files
        if dest_dir.exists() and any(dest_dir.rglob("*.jpg")):
            if verbose:
                jpg_count = sum(1 for _ in dest_dir.rglob("*.jpg"))
                print(f"✅ Data already in {dest_dir} ({jpg_count} jpgs)")
            return {"status": "skipped", "reason": "already_extracted"}
    
    # Verify tar exists on Drive
    if not tar_path_on_drive.exists():
        raise FileNotFoundError(
            f"Tar archive not found on Drive: {tar_path_on_drive}\n"
            "Run the one-time archive creation step first (see 01_data_preparation.ipynb)"
        )
    
    # Determine local tar path
    if local_tar_path is None:
        local_tar_path = Path("/content") / tar_path_on_drive.name
    local_tar_path = Path(local_tar_path)
    
    # Copy tar from Drive to local (the fast part!)
    if verbose:
        size_mb = tar_path_on_drive.stat().st_size / (1024 * 1024)
        print(f"📦 Copying tar from Drive ({size_mb:.1f} MB)...")
    
    shutil.copy2(tar_path_on_drive, local_tar_path)
    
    if verbose:
        print(f"   ✅ Copied to {local_tar_path}")
    
    # Extract (very fast on local SSD)
    result = extract_tar_archive(
        local_tar_path, 
        dest_dir,
        remove_tar_after=remove_tar_after,
        verbose=verbose,
    )
    
    return {
        "status": "extracted",
        "n_files": result["n_files"],
        "dest_dir": str(dest_dir),
    }

