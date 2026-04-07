

import os
import cv2
import numpy as np
import openslide
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# ============================================================
# PATHS (EDIT)
# ============================================================

WSI_PATH = r"E:\FYP\HybridShapleyMIL\dataset\tumor\tumor4\tumor_085.tif"
XML_PATH = r"E:\FYP\HybridShapleyMIL\annotations\tumor_085.xml"

PATCH_SIZE = 256
STEP_SIZE = 256
VIS_SCALE = 32   # higher = more downsampling

OUT_DIR = os.path.join(os.path.dirname(WSI_PATH), "ground_truth")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD WSI SAFELY (THUMBNAIL ONLY)
# ============================================================

slide = openslide.OpenSlide(WSI_PATH)
w, h = slide.dimensions

thumb = slide.get_thumbnail((w // VIS_SCALE, h // VIS_SCALE))
canvas = np.array(thumb.convert("RGB"))

print(f"WSI size: {w} x {h}")
print(f"Thumbnail size: {canvas.shape[1]} x {canvas.shape[0]}")

# ============================================================
# STEP 2: GENERATE PATCH COORDS (ON THE FLY)
# ============================================================

coords = []
for x in range(0, w - PATCH_SIZE + 1, STEP_SIZE):
    for y in range(0, h - PATCH_SIZE + 1, STEP_SIZE):
        coords.append((x, y))

print(f"✓ Generated {len(coords)} patch coordinates")

# ============================================================
# STEP 3: PARSE ASAP XML (CORRECT)
# ============================================================

def parse_asap_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    cancer, normal = [], []

    for ann in root.iter("Annotation"):
        group = ann.attrib.get("PartOfGroup", "")

        pts = []
        for c in ann.iter("Coordinate"):
            pts.append([float(c.attrib["X"]), float(c.attrib["Y"])])

        if len(pts) < 3:
            continue

        poly = np.array(pts, dtype=np.int32)

        # 🔴 DEFINE GROUP MAPPING HERE
        if group == "_0":        # cancer
            cancer.append(poly)
        elif group == "_1":      # normal
            normal.append(poly)

    return cancer, normal

cancer_polys, normal_polys = parse_asap_xml(XML_PATH)

print(f"Cancer regions: {len(cancer_polys)}")
print(f"Normal regions: {len(normal_polys)}")

# ============================================================
# STEP 4: PATCH-LEVEL LABELING
# ============================================================

labels = []
for x, y in coords:
    cx, cy = x + PATCH_SIZE // 2, y + PATCH_SIZE // 2
    lab = -1

    for poly in cancer_polys:
        if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
            lab = 1
            break

    if lab == -1:
        for poly in normal_polys:
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                lab = 0
                break

    labels.append(lab)

# ============================================================
# STEP 5: OVERLAY PATCHES ON THUMBNAIL
# ============================================================

for (x, y), lab in zip(coords, labels):
    if lab == -1:
        continue

    vx, vy = x // VIS_SCALE, y // VIS_SCALE
    vps = PATCH_SIZE // VIS_SCALE

    color = (255, 0, 0) if lab == 1 else (0, 255, 0)

    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (vx, vy),
        (vx + vps, vy + vps),
        color,
        -1
    )
    canvas = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)

# ============================================================
# SAVE & SHOW
# ============================================================

out_path = os.path.join(OUT_DIR, "patch_ground_truth_085.png")

plt.figure(figsize=(12, 12))
plt.imshow(canvas)
plt.title("Patch-level Ground Truth (ASAP XML)")
plt.axis("off")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

slide.close()

print(f"\n✓ Ground truth visualization saved to:\n{out_path}")
