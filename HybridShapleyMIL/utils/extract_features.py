
import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
import openslide
import os
import numpy as np
from tqdm import tqdm
import pickle

# PATHS
COORD_DIR    = r"E:\FYP\HybridShapleyMIL\patches_coord"
FEATURES_DIR = r"E:\FYP\HybridShapleyMIL\features"  
NORMAL_ROOT  = r"E:\FYP\HybridShapleyMIL\dataset\normal"
TUMOR_ROOT   = r"E:\FYP\HybridShapleyMIL\dataset\tumor"

os.makedirs(FEATURES_DIR, exist_ok=True)

# Device + EfficientNet-B0 (1280-D)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Backbone: EfficientNet-B0")

model = efficientnet_b0(pretrained=True)
model.classifier = torch.nn.Identity()  # Remove final layer
model = model.to(device)
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def extract_features(slide_path, coords):
    slide = openslide.OpenSlide(slide_path)
    feats = []
    for x, y in tqdm(coords, desc=os.path.basename(slide_path)[:15], leave=False):
        patch = slide.read_region((x, y), 0, (256, 256))
        patch = np.array(patch.convert("RGB"))
        tensor = transform(patch).unsqueeze(0).to(device)
        feat = model(tensor)
        feats.append(feat.cpu().numpy())
    slide.close()
    return np.squeeze(np.array(feats))

# MAIN LOOP
pkl_files = [f for f in os.listdir(COORD_DIR) if f.endswith(".pkl")]
total = len(pkl_files)

print(f"Starting EfficientNet-B0 feature extraction for {total} slides\n")

for i, pkl_file in enumerate(pkl_files):
    slide_name = pkl_file.replace(".pkl", "")
    slide_tif = slide_name + ".tif"
    coord_path = os.path.join(COORD_DIR, pkl_file)
    
    # Find the .tif
    wsi_path = None
    for root in [NORMAL_ROOT, TUMOR_ROOT]:
        for sub in os.listdir(root):
            path = os.path.join(root, sub, slide_tif)
            if os.path.exists(path):
                wsi_path = path
                break
        if wsi_path: break
    
    if wsi_path is None:
        print(f"   Warning: {slide_tif} not found!")
        continue
    
    print(f"[{i+1:3d}/{total}] {slide_name}")
    with open(coord_path, "rb") as f:
        coords = pickle.load(f)
    
    features = extract_features(wsi_path, coords)
    save_path = os.path.join(FEATURES_DIR, slide_name + ".pt")
    torch.save(torch.FloatTensor(features), save_path)
    print(f"   → {features.shape} (1280-D) features saved\n")

print("EFFICIENTNET-B0 FEATURE EXTRACTION COMPLETE!")
print(f"Features saved in: {FEATURES_DIR}")
