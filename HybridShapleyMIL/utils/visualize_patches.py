
import openslide
import os
import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np

PKL_FILE   = r"E:\FYP\HybridShapleyMIL\patches_coord\normal_040.pkl"   
WSI_ROOTS  = [r"E:\FYP\HybridShapleyMIL\dataset\normal", 
              r"E:\FYP\HybridShapleyMIL\dataset\tumor"]

PATCH_SIZE = 256

slide_name = os.path.basename(PKL_FILE).replace(".pkl", ".tif")
wsi_path = None
for root in WSI_ROOTS:
    for sub in os.listdir(root):
        path = os.path.join(root, sub, slide_name)
        if os.path.exists(path):
            wsi_path = path
            break
    if wsi_path: break

if wsi_path is None:
    print("WSI not found!")
    exit()

# Load coordinates
with open(PKL_FILE, "rb") as f:
    coords = pickle.load(f)

print(f"Loaded {len(coords)} patch coordinates from {PKL_FILE}")
print(f"WSI: {wsi_path}")

# Open slide and get thumbnail
slide = openslide.OpenSlide(wsi_path)
thumb = slide.get_thumbnail((2048, 2048))  # bigger for clear view
thumb_np = np.array(thumb)

# Draw red rectangles
overlay = thumb_np.copy()
for x, y in coords[:5000]:  # limit for speed, remove [:5000] to draw all
    x_thumb = int(x * 2048 / slide.dimensions[0])
    y_thumb = int(y * 2048 / slide.dimensions[1])
    cv2.rectangle(overlay, 
                  (x_thumb, y_thumb), 
                  (x_thumb + int(256 * 2048 / slide.dimensions[0]), 
                   y_thumb + int(256 * 2048 / slide.dimensions[1])),
                  (255, 0, 0), 2)  # red border

# Blend
result = cv2.addWeighted(overlay, 0.5, thumb_np, 0.5, 0)

plt.figure(figsize=(16, 12))
plt.imshow(result)
plt.title(f"{slide_name}\n{len(coords)} patches extracted (red boxes)\n"
          f"Blue = tissue kept | White/gray = background removed", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()

print("Visualization complete!")
