# ============================================================
# test_pmil.py — Modified for Flask integration with heatmap
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = 256
STEP_SIZE  = 256
LEVEL      = 0
VIS_SCALE  = 32
TOP_PERCENT = 5

# ============================================================
# PMIL MODEL
# ============================================================

class PMILHybrid(nn.Module):
    def __init__(self, feat_dim=1280):
        super().__init__()
        self.attention_V = nn.Linear(feat_dim, 128)
        self.attention_U = nn.Linear(128, 1)
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
# HELPER FUNCTIONS
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

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def extract_features(slide_path, coords, device):
    backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    backbone.classifier = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()
    
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

def predict_slide(features, model_path, device):
    model = PMILHybrid().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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

def visualize_predicted_heatmap(wsi_path, coords, attention, output_path):
    """Generate heatmap and save to output_path"""
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
            (255, 0, 0),
            -1
        )
        canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

    plt.figure(figsize=(12, 12))
    plt.imshow(canvas)
    plt.title("PMIL Predicted Cancer Regions", fontsize=16, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================
# MAIN FUNCTION FOR FLASK
# ============================================================

def run_single_wsi_test(wsi_path, model_path=None, output_dir="outputs"):
    """
    Main function called by Flask app
    
    Args:
        wsi_path: Path to the WSI file
        model_path: Path to the trained model (optional, uses default if None)
        output_dir: Directory to save heatmap
        
    Returns:
        dict with prediction, confidence, probabilities, and heatmap path
    """
    # Default model path
    if model_path is None:
        model_path = r"E:\FYP\HybridShapleyMIL\models\pmil_fold0.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract slide ID
    slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
    
    # Extract coordinates
    print(f"Extracting patch coordinates from {slide_id}...")
    coords = extract_coords_ultra_fast(wsi_path)
    print(f"Found {len(coords)} tissue patches")
    
    # Extract features
    print("Extracting features...")
    features = extract_features(wsi_path, coords, device)
    
    # Run prediction
    print("Running PMIL inference...")
    result = predict_slide(features, model_path, device)
    result["num_patches"] = len(coords)
    
    # Generate heatmap if cancer detected
    heatmap_path = None
    if result["prediction"] == "CANCER":
        print("Generating heatmap...")
        os.makedirs(output_dir, exist_ok=True)
        heatmap_path = os.path.join(output_dir, f"{slide_id}_heatmap.png")
        visualize_predicted_heatmap(wsi_path, coords, result["attention"], heatmap_path)
        result["heatmap_path"] = heatmap_path
        print(f"Heatmap saved to {heatmap_path}")
    
    return result