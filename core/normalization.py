import cv2
import numpy as np
from typing import Tuple, Dict, Any

def check_blur(img_bgr: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
    """
    Computes the variance of the Laplacian to evaluate image sharpness.
    Returns (is_sharp, variance_score).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return score >= threshold, round(score, 2)

def apply_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Applies the Gray-World white balance algorithm to normalize illumination
    and remove environmental/sunlight color temperature casts.
    """
    result = img_bgr.copy().astype(np.float32)
    b_mean, g_mean, r_mean = np.mean(result[:, :, 0]), np.mean(result[:, :, 1]), np.mean(result[:, :, 2])
    k = (b_mean + g_mean + r_mean) / 3.0
    
    if b_mean > 0:
        result[:, :, 0] = np.clip(result[:, :, 0] * (k / b_mean), 0, 255)
    if g_mean > 0:
        result[:, :, 1] = np.clip(result[:, :, 1] * (k / g_mean), 0, 255)
    if r_mean > 0:
        result[:, :, 2] = np.clip(result[:, :, 2] * (k / r_mean), 0, 255)
        
    return result.astype(np.uint8)

def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    on the Lightness channel (L in LAB color space) to enhance lesion contrast
    without distorting chromatic hues.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def clean_mask_components(mask: np.ndarray, min_area_ratio: float = 0.01) -> np.ndarray:
    """
    Removes small disconnected noise specks and keeps the primary plant structures.
    Uses Connected Component Analysis.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask

    total_pixels = mask.shape[0] * mask.shape[1]
    min_area = total_pixels * min_area_ratio

    # Label 0 is background; find components larger than threshold
    clean_mask = np.zeros_like(mask)
    # Find largest component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask[labels == largest_label] = 255

    # Also keep any other significant components (e.g. secondary leaves of same plant)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255

    return clean_mask

def composite_background(
    img_rgb: np.ndarray, 
    alpha_mask: np.ndarray, 
    bg_style: str = "Neutral Gray (128)",
    custom_color: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """
    Composites the segmented plant onto the chosen background.
    Options: 'Neutral Gray (128)', 'Pure White', 'Pure Black', 'Blurred Background', 'Transparent (RGBA)'
    """
    h, w = img_rgb.shape[:2]
    norm_alpha = (alpha_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    if bg_style == "Transparent (RGBA)":
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = img_rgb
        rgba[:, :, 3] = alpha_mask
        return rgba

    if bg_style == "Blurred Background":
        bg = cv2.GaussianBlur(img_rgb, (51, 51), 0)
    elif bg_style == "Pure White":
        bg = np.full_like(img_rgb, 255)
    elif bg_style == "Pure Black":
        bg = np.zeros_like(img_rgb)
    else:  # Neutral Gray
        bg = np.full_like(img_rgb, custom_color)

    composite = (img_rgb.astype(np.float32) * norm_alpha + bg.astype(np.float32) * (1.0 - norm_alpha))
    return np.clip(composite, 0, 255).astype(np.uint8)

def crop_to_plant_bbox(img: np.ndarray, mask: np.ndarray, padding_ratio: float = 0.08) -> np.ndarray:
    """
    Crops the image and mask to the bounding box of the plant with padding.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return img

    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return img

    h, w = mask.shape[:2]
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)

    pad_y = int((y_max - y_min) * padding_ratio)
    pad_x = int((x_max - x_min) * padding_ratio)

    y1 = max(0, y_min - pad_y)
    y2 = min(h, y_max + pad_y)
    x1 = max(0, x_min - pad_x)
    x2 = min(w, x_max + pad_x)

    return img[y1:y2, x1:x2]
