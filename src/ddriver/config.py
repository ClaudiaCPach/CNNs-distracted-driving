# src/ddriver/config.py

### Imports
from __future__ import annotations # treat type hints like lazy strings
import os # os allows us to read env variables (defined in .env)
''' needed to see if google.colab is in python's list of loaded modules.
# sys can be thought of like a dictionary of the python modules that have 
# been imported so far, so sys can be checked to see if we're in colab. '''
import sys 
'''
pathlib Path gives path objects instead of messy text strings as paths,
and using Path prevents paths from breaking across Windows, macOS, Linux
'''
from pathlib import Path
'''
works as long as pyproject.toml includes python-dotenv.
'''
try:
    from dotenv import load_dotenv  # make sure python-dotenv is in pyproject.toml
    load_dotenv()
except Exception:
    pass  # if in colab, we won't be using .env because paths are hardcoded
###

def _in_colab() -> bool:
    # returns true if running inside Google Colab.
    return "google.colab" in sys.modules

# set distinct paths for local vs colab drive
# to remain flexible to run both on colab and locally. 
if _in_colab():
    # DRIVE_PATH is persistent
    DRIVE_PATH = Path(os.environ.get("DRIVE_PATH", "/content/drive/MyDrive/TFM"))
    DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", str(DRIVE_PATH / "data")))
    OUT_ROOT     = Path(os.environ.get("OUT_ROOT",     str(DRIVE_PATH / "outputs")))
    CKPT_ROOT    = Path(os.environ.get("CKPT_ROOT",    str(DRIVE_PATH / "checkpoints")))
    # Where you rsync to for speed (optional). You decide to use it or not at runtime.
    FAST_DATA    = Path(os.environ.get("FAST_DATA", "/content/data"))
else:
    # On your PC/cluster: default to ~/TFM with env-var overrides
    LOCAL_PATH     = Path(os.environ.get("LOCAL_PATH", str(Path.home() / "TFM")))
    DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", str(LOCAL_PATH / "data")))
    OUT_ROOT     = Path(os.environ.get("OUT_ROOT",     str(LOCAL_PATH / "outputs")))
    CKPT_ROOT    = Path(os.environ.get("CKPT_ROOT",    str(LOCAL_PATH / "checkpoints")))
    FAST_DATA    = None  # not relevant here

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
