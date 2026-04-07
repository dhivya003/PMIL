# ============================================================
# PMIL SINGLE WSI TESTING + PREDICTED CANCER HEATMAP
# ============================================================

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import openslide
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# PATHS (EDIT)
# ============================================================

WSI_PATH   = r"E:\FYP\HybridShapleyMIL\dataset\tumor\tumor4\tumor_085.tif"
MODEL_PATH = r"E:\FYP\HybridShapleyMIL\models\pmil_combined_5fold.pth"
#MODEL_PATH = r"E:\FYP\HybridShapleyMIL\models\pmil_fold2.pth"
LOG_ROOT   = r"E:\FYP\HybridShapleyMIL\test\test_logs"

PATCH_SIZE = 256
STEP_SIZE  = 256
LEVEL      = 0

VIS_SCALE   = 32     # downsample for visualization
TOP_PERCENT = 5      # top % patches used for heatmap

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

slide_id = os.path.splitext(os.path.basename(WSI_PATH))[0]
SLIDE_LOG_DIR = os.path.join(LOG_ROOT, slide_id)
os.makedirs(SLIDE_LOG_DIR, exist_ok=True)

# ============================================================
# PMIL MODEL (IDENTICAL TO TRAINING)
# ============================================================

class PMILHybrid(nn.Module):
    def __init__(self, feat_dim=1280):
        super().__init__()
        self.attention_V = nn.Linear(feat_dim, 128)
        self.attention_U = nn.Linear(128, 1)     # ✅ FIXED
        self.classifier = nn.Linear(feat_dim, 2)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        V = torch.tanh(self.attention_V(x))
        A = self.attention_U(V).squeeze(-1)
        A = torch.softmax(A, dim=0)

        M = torch.mm(A.unsqueeze(0), x)
        logits = self.classifier(M)

        return logits, A

# ============================================================
# STEP 1: PATCH COORDINATE EXTRACTION (TISSUE ONLY)
# ============================================================

def extract_coords_ultra_fast(slide_path):
    slide = openslide.OpenSlide(slide_path)
    w, h = slide.dimensions

    thumb = slide.get_thumbnail((1024, 1024))
    gray = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)

    coords = []
    for x in range(0, w - PATCH_SIZE + 1, STEP_SIZE):
        for y in range(0, h - PATCH_SIZE + 1, STEP_SIZE):
            if mask[y + PATCH_SIZE // 2, x + PATCH_SIZE // 2] == 0:
                coords.append((x, y))

    slide.close()
    return coords

# ============================================================
# STEP 2: FEATURE EXTRACTION (EFFICIENTNET-B0)
# ============================================================

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# clean, warning-free loading
backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
backbone.classifier = nn.Identity()
backbone = backbone.to(device)
backbone.eval()

@torch.no_grad()
def extract_features(slide_path, coords):
    slide = openslide.OpenSlide(slide_path)
    feats = []

    for x, y in tqdm(coords, desc="Extracting features"):
        patch = slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE))
        patch = np.array(patch.convert("RGB"))
        tensor = transform(patch).unsqueeze(0).to(device)
        feat = backbone(tensor)
        feats.append(feat.cpu())

    slide.close()
    return torch.cat(feats, dim=0)

# ============================================================
# STEP 3: PMIL INFERENCE
# ============================================================

def predict_slide(features):
    model = PMILHybrid().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        logits, attention = model(features.to(device))
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    return {
        "prediction": "CANCER" if pred == 1 else "NON-CANCER",
        "confidence": probs[pred].item(),
        "prob_cancer": probs[1].item(),
        "prob_non_cancer": probs[0].item(),
        "attention": attention.cpu().numpy()
    }

# ============================================================
# STEP 4: VISUALIZE PREDICTED CANCER REGIONS
# ============================================================

def visualize_predicted_heatmap(wsi_path, coords, attention):
    slide = openslide.OpenSlide(wsi_path)
    w, h = slide.dimensions

    thumb = slide.get_thumbnail((w // VIS_SCALE, h // VIS_SCALE))
    canvas = np.array(thumb.convert("RGB"))
    slide.close()

    attn = attention.astype(np.float32)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    K = max(1, int(len(attn) * TOP_PERCENT / 100))
    top_idx = np.argsort(attn)[-K:]

    for i in top_idx:
        x, y = coords[i]
        vx, vy = x // VIS_SCALE, y // VIS_SCALE
        vps = PATCH_SIZE // VIS_SCALE

        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (vx, vy),
            (vx + vps, vy + vps),
            (255, 0, 0),   # 🔴 predicted cancer
            -1
        )
        canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

    out_path = os.path.join(SLIDE_LOG_DIR, "predicted_attention-heatmap-tumor085.png")

    plt.figure(figsize=(12, 12))
    plt.imshow(canvas)
    plt.title("PMIL Predicted Cancer Regions")
    plt.axis("off")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✓ Predicted heatmap saved to:\n{out_path}")

# ============================================================
# SAVE LOG
# ============================================================

def save_prediction_log(result, num_patches):
    log_path = os.path.join(SLIDE_LOG_DIR, "prediction.txt")
    with open(log_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("PMIL SINGLE WSI TEST RESULT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Slide: {slide_id}.tif\n")
        f.write(f"Prediction: {result['prediction']}\n")
        f.write(f"Confidence: {result['confidence']*100:.2f}%\n")
        f.write(f"Prob Cancer: {result['prob_cancer']*100:.2f}%\n")
        f.write(f"Prob Non-Cancer: {result['prob_non_cancer']*100:.2f}%\n")
        f.write(f"Total patches: {num_patches}\n")

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print("PMIL SINGLE WSI TESTING + HEATMAP")
    print("="*70)

    print("\n[1] Extracting patch coordinates...")
    coords = extract_coords_ultra_fast(WSI_PATH)
    print(f"✓ {len(coords)} tissue patches")

    print("\n[2] Extracting features...")
    features = extract_features(WSI_PATH, coords)

    print("\n[3] Running PMIL inference...")
    result = predict_slide(features)

    save_prediction_log(result, len(coords))

    print("\nFINAL PREDICTION")
    print(f"Slide: {slide_id}.tif")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")

    if result["prediction"] == "CANCER":
        print("\n[4] Visualizing predicted cancer regions...")
        visualize_predicted_heatmap(WSI_PATH, coords, result["attention"])
    else:
        print("\nSlide predicted NON-CANCER — heatmap skipped")

    print("\n✓ Testing completed successfully")
    print("="*70)

if __name__ == "__main__":
    main()
