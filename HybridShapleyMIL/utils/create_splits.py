
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

NORMAL_ROOT = r"E:\FYP\HybridShapleyMIL\dataset\normal"
TUMOR_ROOT  = r"E:\FYP\HybridShapleyMIL\dataset\tumor"
OUTPUT_CSV  = r"E:\FYP\HybridShapleyMIL\data_splits\DATA_SPLIT.csv"

print("Scanning all subfolders for .tif files...\n")

# Collect ALL .tif files from ALL subfolders
slides = []
labels = []

# Normal slides (label = 0)
for root, dirs, files in os.walk(NORMAL_ROOT):
    for f in files:
        if f.lower().endswith(".tif"):
            full_path = os.path.join(root, f)
            slides.append(f)                    
            labels.append(0)
            print(f"Normal: {f}")

# Tumor slides (label = 1)
for root, dirs, files in os.walk(TUMOR_ROOT):
    for f in files:
        if f.lower().endswith(".tif"):
            full_path = os.path.join(root, f)
            slides.append(f)
            labels.append(1)
            print(f"Tumor : {f}")

print(f"\nTotal found: {len(slides)} slides")
print(f"Normal: {labels.count(0)} | Tumor: {labels.count(1)}\n")

# Create DataFrame
df = pd.DataFrame({"slide_id": slides, "label": labels})

# 5-fold stratified split 
print("Creating 5-fold cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
    df.loc[val_idx, "fold"] = fold

# Save
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print("DATA_SPLIT.csv created SUCCESSFULLY!")
print(f"Saved → {OUTPUT_CSV}")
print("\nFirst 10 rows:")
print(df.head(10))
print("\nFold distribution:")
print(df["fold"].value_counts().sort_index())