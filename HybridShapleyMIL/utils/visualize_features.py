# utils/inspect_features.py   ← save this file
import torch
import os

# CHANGE THIS to any .pt file you want to check
FEATURE_PT = r"E:\FYP\HybridShapleyMIL\features\normal_003.pt"   # ← example

if not os.path.exists(FEATURE_PT):
    print("File not found! Check the path.")
else:
    data = torch.load(FEATURE_PT)
    print(f"Slide: {os.path.basename(FEATURE_PT)}")
    print(f"Shape: {data.shape}   →  (num_patches, 1280)")
    print(f"Total patches: {data.shape[0]:,}")
    print(f"Feature dim: {data.shape[1]}")
    print(f"Memory: {data.element_size() * data.nelement() / 1024 / 1024:.1f} MB")
    print(f"Min value: {data.min():.4f} | Max value: {data.max():.4f}")
    print(f"Mean: {data.mean():.4f} | Std: {data.std():.4f}")
    
    # Optional: show first 5 patch features
    print("\nFirst 5 feature vectors (first 10 values each):")
    print(data[:5, :10])