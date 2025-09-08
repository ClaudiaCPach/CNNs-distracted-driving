# src/ddriver/config.py
from __future__ import annotations
import os, sys
from pathlib import Path

# 1) Optional: load .env if you have one (local machines)
try:
    from dotenv import load_dotenv  # make sure python-dotenv is in pyproject.toml
    load_dotenv()
except Exception:
    pass  # fine on Colab if not installed yet

def _in_colab() -> bool:
    """True if running inside Google Colab."""
    return "google.colab" in sys.modules

# 2) Decide bases depending on where we are
if _in_colab():
    # On Colab: default to Drive (persistent) + optional fast local copy
    DRIVE_BASE = Path(os.environ.get("DRIVE_BASE", "/content/drive/MyDrive/TFM"))
    DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", str(DRIVE_BASE / "data")))
    OUT_ROOT     = Path(os.environ.get("OUT_ROOT",     str(DRIVE_BASE / "outputs")))
    CKPT_ROOT    = Path(os.environ.get("CKPT_ROOT",    str(DRIVE_BASE / "checkpoints")))
    # Where you rsync to for speed (optional). You decide to use it or not at runtime.
    FAST_DATA    = Path(os.environ.get("FAST_DATA", "/content/data"))
else:
    # On your PC/cluster: default to ~/TFM with env-var overrides
    TFM_HOME     = Path(os.environ.get("TFM_HOME", str(Path.home() / "TFM")))
    DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", str(TFM_HOME / "data")))
    OUT_ROOT     = Path(os.environ.get("OUT_ROOT",     str(TFM_HOME / "outputs")))
    CKPT_ROOT    = Path(os.environ.get("CKPT_ROOT",    str(TFM_HOME / "checkpoints")))
    FAST_DATA    = None  # only meaningful on Colab

# 3) Create persistent output/checkpoint dirs if missing (safe, idempotent)
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_ROOT.mkdir(parents=True, exist_ok=True)

def dataset_dir(prefer_fast: bool = False) -> Path:
    """
    Return the directory to read training images from.
    - prefer_fast=True → use FAST_DATA if it exists (good on Colab after rsync)
    - prefer_fast=False → use DATASET_ROOT (Drive/local persistent)
    """
    if prefer_fast and FAST_DATA and FAST_DATA.exists():
        return FAST_DATA
    return DATASET_ROOT

def split_csv(name: str, base: str | Path | None = None) -> Path:
    """
    Build a path to a split CSV by name.
    Example: split_csv('cam2_seed42/train.csv') → OUT_ROOT/splits/cam2_seed42/train.csv
    """
    base_path = Path(base) if base else (OUT_ROOT / "splits")
    return base_path / name

# Nice for logging
__all__ = ["DATASET_ROOT", "OUT_ROOT", "CKPT_ROOT", "FAST_DATA", "dataset_dir", "split_csv"]
