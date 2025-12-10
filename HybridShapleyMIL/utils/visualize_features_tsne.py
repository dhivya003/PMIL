# utils/visualize_features_tsne.py  ← FINAL 100% WORKING VERSION
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from glob import glob

FEATURE_DIR = "features"   # ← YOUR REAL FOLDER

all_feats = []
all_labels = []
all_names = []

print("Loading features from both normal and tumor slides...")

for pt_file in glob(os.path.join(FEATURE_DIR, "*.pt")):
    name = os.path.basename(pt_file).replace(".pt", "")
    label = 0 if "normal" in name.lower() else 1
    feats = torch.load(pt_file, map_location='cpu').numpy()
    
    # Subsample max 1200 patches per slide
    if feats.shape[0] > 1200:
        idx = np.random.choice(feats.shape[0], 1200, replace=False)
        feats = feats[idx]
    
    all_feats.append(feats)
    all_labels.extend([label] * len(feats))
    all_names.append(name)

X = np.vstack(all_feats)
y = np.array(all_labels)

print(f"Running t-SNE on {X.shape[0]:,} patches from {len(all_names)} slides...")

tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000, learning_rate='auto', init='pca')
X_2d = tsne.fit_transform(X)

# ← FIXED: Create folder first
os.makedirs("results/Feature_visualization", exist_ok=True)

plt.figure(figsize=(15, 11))
colors = ['#1f77b4', '#d62728']  # blue, red
labels = ['Normal (benign)', 'Tumor (malignant)']

for i in [0, 1]:
    mask = y == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                c=colors[i], label=f"{labels[i]} ({mask.sum():,} patches)", 
                alpha=0.7, s=20, edgecolors='none')

plt.legend(markerscale=2, fontsize=16, loc='upper right')
plt.title("t-SNE of EfficientNet-B0 1280-D Features (CAMELYON-16)\n"
          "Blue = Normal patches | Red = Tumor patches", fontsize=18, pad=20)
plt.xlabel("t-SNE dimension 1", fontsize=14)
plt.ylabel("t-SNE dimension 2", fontsize=14)
plt.tight_layout()

# ← FIXED: Correct path
plt.savefig("results/Feature_visualization/tsne_feature.png", dpi=400, bbox_inches='tight')
plt.show()

print("SAVED: results/Feature_visualization/tsne_feature.png")
