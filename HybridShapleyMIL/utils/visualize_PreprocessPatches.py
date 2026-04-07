import openslide
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os

# CHANGE ONLY THESE TWO LINES
PKL_FILE = r"E:\FYP\HybridShapleyMIL\patches_coord\normal_011.pkl"   
WSI_ROOTS = [r"E:\FYP\HybridShapleyMIL\dataset\normal", 
             r"E:\FYP\HybridShapleyMIL\dataset\tumor"]

# Find the .tif
slide_name = os.path.basename(PKL_FILE).replace(".pkl", ".tif")
wsi_path = None
for root in WSI_ROOTS:
    for sub in os.listdir(root):
        path = os.path.join(root, sub, slide_name)
        if os.path.exists(path):
            wsi_path = path
            break
    if wsi_path: break

if not wsi_path:
    print("WSI not found!")
    exit()

# Load your coordinates
with open(PKL_FILE, "rb") as f:
    coords = pickle.load(f)

print(f"Found {len(coords):,} tissue patches in {slide_name}")

# Pick 5 random coordinates
random.seed(42)
sample_coords = random.sample(coords, 5)

# EfficientNet-B0 preprocessing
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

slide = openslide.OpenSlide(wsi_path)

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle(f"Raw vs Preprocessed Patches — {slide_name}\n"
             f"Total patches: {len(coords):,}", fontsize=20, y=0.98)

for i, (x, y) in enumerate(sample_coords):
    # Raw patch
    raw = slide.read_region((x, y), 0, (256, 256)).convert("RGB")
    raw_np = np.array(raw)
    axes[0, i].imshow(raw_np)
    axes[0, i].set_title(f"Raw {i+1}\n({x}, {y})", fontsize=14)
    axes[0, i].axis('off')
    
    # Preprocessed
    tensor = transform(raw_np)
    display = tensor.permute(1, 2, 0).numpy()
    display = display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    display = np.clip(display, 0, 1)
    axes[1, i].imshow(display)
    axes[1, i].set_title(f"Preprocessed {i+1}\n(EfficientNet Input)", fontsize=14)
    axes[1, i].axis('off')


os.makedirs("results", exist_ok=True)
plt.tight_layout()
plt.savefig("results/raw_vs_preprocessed_patches.png", dpi=400, bbox_inches='tight')
plt.show()

print("SAVED: results/raw_vs_preprocessed_patches.png")
