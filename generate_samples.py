import os
import cv2
import numpy as np

def generate_soil_texture(h: int, w: int) -> np.ndarray:
    """Generates a realistic soil/table background with grain noise."""
    base = np.full((h, w, 3), [45, 55, 65], dtype=np.uint8)  # Earth brown BGR
    noise = np.random.normal(0, 18, (h, w, 3)).astype(np.float32)
    soil = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    soil = cv2.GaussianBlur(soil, (3, 3), 0)
    # Add some random pebble spots
    for _ in range(35):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        r = np.random.randint(3, 9)
        color = int(np.random.randint(90, 140))
        cv2.circle(soil, (cx, cy), r, (color, color, color), -1)
    return soil

def generate_healthy_leaf(h: int = 512, w: int = 512) -> np.ndarray:
    """Generates a vibrant green leaf with visible venation on soil."""
    img = generate_soil_texture(h, w)
    leaf_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Elliptical leaf contour
    center = (w // 2, h // 2)
    axes = (130, 220)
    angle = 20
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    
    # Base green color with gradient
    for y in range(h):
        ratio = y / h
        leaf_canvas[y, :] = [30 + int(ratio * 20), 160 + int(ratio * 30), 40 + int(ratio * 20)]

    # Main vein
    pt1 = (int(w * 0.40), int(h * 0.15))
    pt2 = (int(w * 0.60), int(h * 0.85))
    cv2.line(leaf_canvas, pt1, pt2, (45, 215, 90), 4)

    # Lateral veins
    for i in range(1, 9):
        t = i / 10.0
        vx = int(pt1[0] + t * (pt2[0] - pt1[0]))
        vy = int(pt1[1] + t * (pt2[1] - pt1[1]))
        cv2.line(leaf_canvas, (vx, vy), (vx - 65, vy - 25), (40, 195, 80), 2)
        cv2.line(leaf_canvas, (vx, vy), (vx + 65, vy + 20), (40, 195, 80), 2)

    # Merge leaf onto background
    mask_3d = (mask > 0)[:, :, np.newaxis]
    img = np.where(mask_3d, leaf_canvas, img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def generate_diseased_leaf(h: int = 512, w: int = 512) -> np.ndarray:
    """
    Generates a diseased leaf with yellow chlorosis margins, 
    and brown/black necrotic fungal lesions on soil.
    This demonstrates why classical color thresholding fails.
    """
    img = generate_soil_texture(h, w)
    leaf_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (w // 2, h // 2)
    axes = (140, 210)
    angle = -15
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

    # Chlorotic (yellowing) base green
    for y in range(h):
        ratio = y / h
        leaf_canvas[y, :] = [25, 140 + int(ratio * 25), 90 + int(ratio * 40)]  # More yellow/olive

    # Main vein
    pt1 = (int(w * 0.60), int(h * 0.15))
    pt2 = (int(w * 0.40), int(h * 0.85))
    cv2.line(leaf_canvas, pt1, pt2, (40, 175, 120), 4)

    # Draw large brown/black necrotic lesions
    lesion_spots = [
        ((int(w * 0.45), int(h * 0.35)), 35, (20, 45, 95)),    # Dark brown
        ((int(w * 0.58), int(h * 0.55)), 45, (15, 35, 75)),    # Severe necrosis
        ((int(w * 0.38), int(h * 0.65)), 25, (30, 80, 140)),   # Brown lesion
        ((int(w * 0.52), int(h * 0.72)), 20, (20, 50, 100)),
    ]

    for (lx, ly), rad, col in lesion_spots:
        # Yellow chlorotic halo around lesion
        cv2.circle(leaf_canvas, (lx, ly), rad + 12, (20, 190, 220), -1)
        # Brown necrotic center
        cv2.circle(leaf_canvas, (lx, ly), rad, col, -1)
        # Inner black necrotic core
        cv2.circle(leaf_canvas, (lx, ly), rad // 2, (10, 20, 40), -1)

    # Merge leaf onto background
    mask_3d = (mask > 0)[:, :, np.newaxis]
    img = np.where(mask_3d, leaf_canvas, img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    os.makedirs("samples", exist_ok=True)
    healthy = generate_healthy_leaf()
    diseased = generate_diseased_leaf()

    cv2.imwrite("samples/healthy_leaf_sample.png", cv2.cvtColor(healthy, cv2.COLOR_RGB2BGR))
    cv2.imwrite("samples/diseased_leaf_sample.png", cv2.cvtColor(diseased, cv2.COLOR_RGB2BGR))
    print("Sample plant images generated successfully in samples/")

if __name__ == "__main__":
    main()
