import openslide
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PATCH_SIZE = 256
STEP_SIZE  = 256

def visualize_tissue_detection(slide_path, max_patches_shown=500, thumb_size=1024):
    """
    Visualizes Otsu-based tissue detection on a WSI thumbnail.
    Shows: raw thumbnail | Otsu binary mask | tissue overlay + patch grid
    """
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions
    slide_name = slide_path.split("\\")[-1]

    # ── Step 1: Thumbnail ──────────────────────────────────────────────────────
    thumb = slide.get_thumbnail((thumb_size, thumb_size))
    thumb_np = np.array(thumb)
    thumb_gray = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2GRAY)

    # ── Step 2: Otsu threshold ─────────────────────────────────────────────────
    otsu_thresh, binary = cv2.threshold(
        thumb_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ── Step 3: Resize mask to full-slide dimensions ───────────────────────────
    mask_full = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)

    # ── Step 4: Collect tissue coords (same logic as your pipeline) ────────────
    coords = []
    for x in range(0, width - PATCH_SIZE + 1, STEP_SIZE):
        for y in range(0, height - PATCH_SIZE + 1, STEP_SIZE):
            if mask_full[y + 128, x + 128] == 0:   # 0 = tissue
                coords.append((x, y))

    slide.close()

    # ── Step 5: Scale coords → thumbnail space ─────────────────────────────────
    thumb_w, thumb_h = thumb.size
    scale_x = thumb_w / width
    scale_y = thumb_h / height

    # ── Step 6: Build tissue overlay ──────────────────────────────────────────
    tissue_mask_thumb = cv2.resize(
        (mask_full == 0).astype(np.uint8) * 255,
        (thumb_w, thumb_h),
        interpolation=cv2.INTER_NEAREST
    )
    overlay = thumb_np.copy()
    green_layer = np.zeros_like(thumb_np)
    green_layer[tissue_mask_thumb == 255] = [0, 200, 80]
    overlay = cv2.addWeighted(overlay, 0.6, green_layer, 0.4, 0)

    # ── Step 7: Draw patch grid (subsample if many) ───────────────────────────
    step = max(1, len(coords) // max_patches_shown)
    patch_overlay = overlay.copy()
    for (x, y) in coords[::step]:
        rx = int(x * scale_x)
        ry = int(y * scale_y)
        rw = max(1, int(PATCH_SIZE * scale_x))
        rh = max(1, int(PATCH_SIZE * scale_y))
        cv2.rectangle(patch_overlay, (rx, ry), (rx + rw, ry + rh), (255, 220, 0), 1)

    # ── Step 8: Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#0e0e0e")
    titles = ["Raw Thumbnail", f"Otsu Mask  (thresh={otsu_thresh:.0f})", "Tissue + Patch Grid"]
    images = [thumb_np, cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB), patch_overlay]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=13, pad=10, fontfamily="monospace")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#00c850", label="Tissue (foreground)"),
        mpatches.Patch(color="#ffdc00", label=f"Patch centres sampled ({len(coords):,} total)"),
        mpatches.Patch(color="white",   label=f"Background (excluded)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=3,
        frameon=True, facecolor="#1a1a1a",
        edgecolor="#555", labelcolor="white",
        fontsize=10, bbox_to_anchor=(0.5, 0.01)
    )

    fig.suptitle(
        f"Tissue Detection — {slide_name}\n"
        f"Slide: {width}×{height} px  |  Patches: {len(coords):,}  |  "
        f"Patch size: {PATCH_SIZE}px  |  Step: {STEP_SIZE}px",
        color="white", fontsize=12, fontfamily="monospace",
        y=1.01
    )

    plt.tight_layout()
    plt.savefig(
        slide_name.replace(".tif", "_tissue_vis.png"),
        dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    plt.show()
    print(f"Saved → {slide_name.replace('.tif', '_tissue_vis.png')}")
    print(f"Tissue patches: {len(coords):,}  |  Otsu threshold: {otsu_thresh:.1f}")


# ── Usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Single slide
    visualize_tissue_detection(
        r"C:\Users\TEAM1\Downloads\valid\valid\61_HE_val.tif"
    )

    # Or loop over several slides for a quick batch check
    import pandas as pd, os
    CSV_PATH   = r"E:\FYP\HybridShapleyMIL\data_splits\DATA_SPLIT.csv"
    NORMAL_ROOT = r"E:\FYP\HybridShapleyMIL\dataset\normal"
    TUMOR_ROOT  = r"E:\FYP\HybridShapleyMIL\dataset\tumor"

    df = pd.read_csv(CSV_PATH)
    for slide_name in df["slide_id"][:5]:          # preview first 5
        for root in [NORMAL_ROOT, TUMOR_ROOT]:
            for sub in os.listdir(root):
                path = os.path.join(root, sub, slide_name)
                if os.path.exists(path):
                    visualize_tissue_detection(path)
                    break