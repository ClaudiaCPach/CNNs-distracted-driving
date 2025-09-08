import argparse, re
from pathlib import Path
import pandas as pd


# Tools to: 
# Read/write the dataset manifest (one row per image)
# Filter by camera, class, driver. 


# Walk through the folder structure in the AUC v1 plus v2 dataset folder

# map c0 to c9 to friendly names in a dictionary so it can be used everywhere
CLASS_MAP = {
  'c0':'safe_driving',
  'c1':'text_right',
  'c2':'phone_right',
  'c3':'text_left',
  'c4':'phone_left',
  'c5':'adjusting_radio',
  'c6':'drinking',
  'c7':'reaching_behind',
  'c8':'hair_makeup',
  'c9':'talking_to_passenger'
}

def parse_img_num(filename: str):
    """
    Tries to pull a number from the end of a file name, like 803.jpg -> 803.
    If there isn't one, it returns none. 
    This number will help define the driver range (the range of numbers
    in which the same driver appears) for each driver. 
    """

    m = re.search(r'(\d+)(?=\.(jpg)$)', filename, re.I)
    return int(m.group(1)) if m else None

def build_manifest(root: Path, source = 'v2_cam1_cam2_ split_by_driver'):
    rows = []
    # the dataset is structured like camera1/2 contains train/test contains
    # c0 ... c9 contains images
    
    for cam_dir in ['Camera 1', 'Camera 2']:
        cam = 'cam1' if cam_dir.endswith('1') else 'cam2'
        for split in ['train', 'test']:

    
