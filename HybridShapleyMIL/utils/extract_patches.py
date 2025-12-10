
import openslide
import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm
import pandas as pd

# PATHS
NORMAL_ROOT = r"E:\FYP\HybridShapleyMIL\dataset\normal"
TUMOR_ROOT  = r"E:\FYP\HybridShapleyMIL\dataset\tumor"
COORD_DIR   = r"E:\FYP\HybridShapleyMIL\patches_coord"
CSV_PATH    = r"E:\FYP\HybridShapleyMIL\data_splits\DATA_SPLIT.csv"

PATCH_SIZE = 256
STEP_SIZE  = 256
LEVEL      = 0

os.makedirs(COORD_DIR, exist_ok=True)

def extract_coords_ultra_fast(slide_path):
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions

    # Get tiny thumbnail → Otsu once → resize to full size
    thumb = slide.get_thumbnail((1024, 1024))
    thumb_gray = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(thumb_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)

    # Keep only foreground regions (black in Otsu = tissue)
    coords = []
    for x in range(0, width - PATCH_SIZE + 1, STEP_SIZE):
        for y in range(0, height - PATCH_SIZE + 1, STEP_SIZE):
            # Just check one pixel in the center of the future patch
            if mask[y + 128, x + 128] == 0:  # 0 = tissue in binary Otsu mask
                coords.append((x, y))

    slide.close()
    return coords

df = pd.read_csv(CSV_PATH)
total_patches = 0

print("ULTRA-FAST COORDINATES EXTRACTION STARTED (PMIL/CLAM  method)\n")

for i, slide_name in enumerate(df["slide_id"]):
    found = False
    for root in [NORMAL_ROOT, TUMOR_ROOT]:
        for sub in os.listdir(root):
            path = os.path.join(root, sub, slide_name)
            if os.path.exists(path):
                print(f"[{i+1:3d}/215] {slide_name.ljust(20)} → extracting...", end=" ")
                coords = extract_coords_ultra_fast(path)
                out_path = os.path.join(COORD_DIR, slide_name.replace(".tif", ".pkl"))
                with open(out_path, "wb") as f:
                    pickle.dump(coords, f)
                print(f"{len(coords):,} patches ({len(coords)*256*256//1024//1024} MB saved)")
                total_patches += len(coords)
                found = True
                break
        if found: break

print("\n" + "═"*80)
print(f"Total patches extracted: {total_patches:,}")
print(f"Estimated final feature size: ~{total_patches*4*1024//1024//1024:.1f} GB")
print(f"Coordinates saved in: {COORD_DIR}")
print("═"*80)