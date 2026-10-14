import cv2
import numpy as np
from typing import Dict, Any

class MaskEvaluator:
    """
    Objective Quality Assessment Engine for Plant Segmentation Masks.
    Evaluates boundary alignment, anatomical continuity, hole penalties (lesion preservation),
    and noise fragmentation to compute a holistic confidence score.
    """

    @staticmethod
    def evaluate(img_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Calculates an objective confidence score and diagnostic sub-metrics.
        img_rgb: original RGB image
        mask: uint8 binary mask (0 or 255)
        """
        h, w = mask.shape[:2]
        total_pixels = h * w
        fg_pixels = int(np.count_nonzero(mask))

        if fg_pixels == 0:
            return {
                "confidence_score": 0.0,
                "confidence_pct": 0.0,
                "area_ratio": 0.0,
                "edge_coherence": 0.0,
                "solidity": 0.0,
                "hole_penalty": 1.0,
                "fragmentation_score": 0.0,
                "diagnostic": "Empty mask: No foreground detected."
            }

        # 1. Area Plausibility (Leaves usually occupy 5% to 92% of the frame)
        area_ratio = fg_pixels / total_pixels
        if area_ratio < 0.03:
            area_score = 0.2
        elif area_ratio > 0.95:
            area_score = 0.2
        else:
            area_score = 1.0

        # 2. Edge Gradient Coherence
        # Extract 1-pixel boundary of the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {
                "confidence_score": 0.0,
                "confidence_pct": 0.0,
                "area_ratio": area_ratio,
                "edge_coherence": 0.0,
                "solidity": 0.0,
                "hole_penalty": 1.0,
                "fragmentation_score": 0.0,
                "diagnostic": "Failed contour extraction."
            }

        mask_edges = np.zeros_like(mask)
        cv2.drawContours(mask_edges, contours, -1, 255, 1)

        # Image Sobel gradients on luminance
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobelx, sobely)
        max_grad = np.max(grad_mag)
        if max_grad > 0:
            grad_mag = grad_mag / max_grad
        else:
            grad_mag = np.zeros_like(grad_mag)

        edge_indices = mask_edges > 0
        if np.any(edge_indices):
            edge_coherence = float(np.mean(grad_mag[edge_indices]))
            # Scale non-linearly to reward true edges
            edge_coherence = min(1.0, edge_coherence * 2.2)
        else:
            edge_coherence = 0.0

        # 3. Anatomical Solidity & Compactness
        largest_cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(largest_cnt)
        hull = cv2.convexHull(largest_cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(cnt_area / hull_area) if hull_area > 0 else 0.0

        # 4. Internal Holes & Lesion Integrity
        # Naive color thresholding punches holes in diseased/brown spots
        # Count total contour area vs total external area
        total_external_area = 0
        total_internal_holes_area = 0
        if hierarchy is not None:
            # hierarchy: [Next, Previous, First_Child, Parent]
            for i, h_info in enumerate(hierarchy[0]):
                c_area = cv2.contourArea(contours[i])
                if h_info[3] == -1:  # External contour
                    total_external_area += c_area
                else:  # Internal hole
                    total_internal_holes_area += c_area

        if total_external_area > 0:
            hole_ratio = total_internal_holes_area / total_external_area
        else:
            hole_ratio = 0.0

        # Small holes from natural leaf fenestration are okay (<3%), but >15% usually indicates diseased tissue was hollowed out
        if hole_ratio <= 0.03:
            hole_score = 1.0
        elif hole_ratio <= 0.15:
            hole_score = 1.0 - (hole_ratio - 0.03) * 3.0
        else:
            hole_score = max(0.1, 1.0 - hole_ratio * 1.5)

        # 5. Fragmentation & Noise Analysis
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        # Discard tiny speckle components
        sig_components = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > (fg_pixels * 0.005):
                sig_components += 1

        if sig_components <= 1:
            frag_score = 1.0
        elif sig_components <= 3:
            frag_score = 0.85
        elif sig_components <= 6:
            frag_score = 0.65
        else:
            frag_score = max(0.1, 1.0 - (sig_components * 0.08))

        # Weighted Final Confidence Score (0.0 to 1.0)
        # Edge alignment (30%), Solidity (25%), Hole Integrity (25%), Cleanliness/Noise (20%) * Area Score
        raw_confidence = (
            0.30 * edge_coherence +
            0.25 * solidity +
            0.25 * hole_score +
            0.20 * frag_score
        ) * area_score

        confidence_score = round(float(np.clip(raw_confidence, 0.0, 1.0)), 4)
        confidence_pct = round(confidence_score * 100.0, 1)

        # Diagnostic message
        if confidence_pct >= 85:
            diagnostic = "High confidence: Crisp boundaries, intact lamina, minimal background noise."
        elif confidence_pct >= 70:
            diagnostic = "Moderate confidence: Usable segmentation, minor boundary roughness or small fragments."
        elif confidence_pct >= 50:
            diagnostic = "Fair confidence: Potential edge bleed or partial internal holes."
        else:
            diagnostic = "Low confidence: Severe holes, excessive background clutter, or misclassification."

        return {
            "confidence_score": confidence_score,
            "confidence_pct": confidence_pct,
            "area_ratio": round(area_ratio, 3),
            "edge_coherence": round(edge_coherence, 3),
            "solidity": round(solidity, 3),
            "hole_score": round(hole_score, 3),
            "fragmentation_score": round(frag_score, 3),
            "diagnostic": diagnostic
        }
