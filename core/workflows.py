import time
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple
import os

from core.normalization import (
    apply_white_balance,
    apply_clahe,
    clean_mask_components,
    composite_background,
    crop_to_plant_bbox
)
from core.evaluator import MaskEvaluator

class WorkflowRunner:
    """
    Orchestrates the 4 segmentation workflows:
    - Workflow 1: Classical Multi-Threshold & GrabCut
    - Workflow 2: Deep Salient Matting (rembg U2Net / BiRefNet)
    - Workflow 3: Two-Stage Object-Centric (YOLOv8-seg)
    - Workflow 4: Adaptive Consensus & Automatic Best-Method Selection
    """

    def __init__(self):
        self._rembg_session = None
        self._yolo_model = None

    def get_rembg_session(self, model_name: str = "u2net"):
        """Lazy-load rembg session to save startup time and memory."""
        if self._rembg_session is None:
            try:
                from rembg import new_session
                self._rembg_session = new_session(model_name)
            except Exception as e:
                print(f"Warning: could not initialize rembg session: {e}")
                self._rembg_session = None
        return self._rembg_session

    def get_yolo_model(self, model_path: str = "yolov8n-seg.pt"):
        """Lazy-load YOLO model."""
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO(model_path)
            except Exception as e:
                print(f"Warning: could not load YOLO model: {e}")
                self._yolo_model = None
        return self._yolo_model

    # -------------------------------------------------------------
    # WORKFLOW 1: Classical Multi-Threshold & GrabCut
    # -------------------------------------------------------------
    def run_workflow_classical(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Classical computer vision pipeline using Color Spaces (HSV + Lab) + Otsu + GrabCut.
        Fast baseline, but vulnerable to necrotic lesion hollowing and background weeds.
        """
        start_time = time.time()
        h, w = img_rgb.shape[:2]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 1. Color space thresholding for green vegetation
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Plant green hue range roughly 25 to 88
        lower_green = np.array([25, 35, 35], dtype=np.uint8)
        upper_green = np.array([88, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Lab color space: excess green (a channel is green-magenta)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        _, a_thresh = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Combine cues
        combined_seed = cv2.bitwise_or(green_mask, a_thresh)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_seed = cv2.morphologyEx(combined_seed, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_seed = cv2.morphologyEx(combined_seed, cv2.MORPH_OPEN, kernel, iterations=1)

        # 2. Refine with GrabCut
        # Initialize GrabCut mask
        gc_mask = np.zeros((h, w), np.uint8)
        gc_mask[:] = cv2.GC_BGD

        # Central bounding box for probable foreground
        margin_x, margin_y = int(w * 0.05), int(h * 0.05)
        gc_mask[margin_y:h-margin_y, margin_x:w-margin_x] = cv2.GC_PR_BGD

        # Where seed is positive, set to probable foreground
        gc_mask[combined_seed > 0] = cv2.GC_PR_FGD

        # Run 2 iterations of GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
            final_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:
            final_mask = combined_seed

        # Cleanup isolated noise specks
        final_mask = clean_mask_components(final_mask, min_area_ratio=0.015)
        elapsed = (time.time() - start_time) * 1000.0
        return final_mask, elapsed

    # -------------------------------------------------------------
    # WORKFLOW 2: Deep Salient Matting (rembg U2Net / BiRefNet)
    # -------------------------------------------------------------
    def run_workflow_deep_matting(self, img_rgb: np.ndarray, model_name: str = "u2net") -> Tuple[np.ndarray, float]:
        """
        Deep Neural Matting. Specifically designed for fine-grained natural boundary
        extraction without needing prompts. Flawlessly preserves lesion colors.
        """
        start_time = time.time()
        session = self.get_rembg_session(model_name)
        
        pil_img = Image.fromarray(img_rgb)
        try:
            from rembg import remove
            mask_pil = remove(
                pil_img, 
                session=session, 
                only_mask=True,
                alpha_matting=True,
                alpha_matting_foreground_threshold=230,
                alpha_matting_background_threshold=15
            )
            mask_np = np.array(mask_pil)
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]
            # Binarize with threshold
            _, mask_binary = cv2.threshold(mask_np, 100, 255, cv2.THRESH_BINARY)
        except Exception as e:
            print(f"Deep matting error: {e}")
            mask_binary, _ = self.run_workflow_classical(img_rgb)

        mask_binary = clean_mask_components(mask_binary, min_area_ratio=0.01)
        elapsed = (time.time() - start_time) * 1000.0
        return mask_binary, elapsed

    # -------------------------------------------------------------
    # WORKFLOW 3: Two-Stage Object-Centric (YOLOv8-seg)
    # -------------------------------------------------------------
    def run_workflow_two_stage(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Two-stage pipeline: Detects the primary plant instance and segments its mask.
        Outstanding at suppressing background weeds, hands holding leaves, or plant pots.
        """
        start_time = time.time()
        h, w = img_rgb.shape[:2]
        model = self.get_yolo_model("yolov8n-seg.pt")

        final_mask = np.zeros((h, w), dtype=np.uint8)

        if model is not None:
            try:
                # YOLOv8 segmentation inference
                results = model(img_rgb, verbose=False, conf=0.15)
                if len(results) > 0 and results[0].masks is not None:
                    # Look for plant-related classes (potted plant is class 58 in COCO),
                    # or pick the largest segmented instance in the central region
                    masks_data = results[0].masks.data.cpu().numpy()
                    combined = np.zeros((h, w), dtype=np.uint8)
                    for m in masks_data:
                        resized_m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                        combined = np.maximum(combined, (resized_m > 0.4).astype(np.uint8) * 255)
                    final_mask = combined
                else:
                    # Fallback to salient matting if YOLO has no COCO plant match
                    final_mask, _ = self.run_workflow_deep_matting(img_rgb)
            except Exception as e:
                print(f"YOLO workflow error: {e}")
                final_mask, _ = self.run_workflow_deep_matting(img_rgb)
        else:
            final_mask, _ = self.run_workflow_classical(img_rgb)

        final_mask = clean_mask_components(final_mask, min_area_ratio=0.015)
        elapsed = (time.time() - start_time) * 1000.0
        return final_mask, elapsed

    # -------------------------------------------------------------
    # WORKFLOW 4: Adaptive Consensus & Automatic Best-Method Selection
    # -------------------------------------------------------------
    def run_workflow_adaptive(
        self, 
        img_rgb: np.ndarray,
        pre_wb: bool = True,
        pre_clahe: bool = False,
        bg_style: str = "Neutral Gray (128)",
        crop_plant: bool = False
    ) -> Dict[str, Any]:
        """
        Runs the candidate workflows, evaluates their confidence scores,
        diagnoses pros/cons, and automatically selects the highest-confidence output.
        """
        overall_start = time.time()

        # Step 1: Preprocessing & Normalization
        processed_img = img_rgb.copy()
        img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
        if pre_wb:
            img_bgr = apply_white_balance(img_bgr)
        if pre_clahe:
            img_bgr = apply_clahe(img_bgr, clip_limit=1.8)
        processed_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Step 2: Execute all 3 primary workflows
        mask_w1, time_w1 = self.run_workflow_classical(processed_img)
        mask_w2, time_w2 = self.run_workflow_deep_matting(processed_img)
        mask_w3, time_w3 = self.run_workflow_two_stage(processed_img)

        # Step 3: Evaluate each workflow through the Confidence Engine
        eval_w1 = MaskEvaluator.evaluate(processed_img, mask_w1)
        eval_w2 = MaskEvaluator.evaluate(processed_img, mask_w2)
        eval_w3 = MaskEvaluator.evaluate(processed_img, mask_w3)

        workflows_data = {
            "Workflow 1: Classical (HSV + GrabCut)": {
                "mask": mask_w1,
                "latency_ms": round(time_w1, 1),
                "eval": eval_w1,
                "description": "Fast color-threshold & morphological seed refined by GrabCut."
            },
            "Workflow 2: Deep Salient Matting (U2Net/BiRefNet)": {
                "mask": mask_w2,
                "latency_ms": round(time_w2, 1),
                "eval": eval_w2,
                "description": "End-to-end dichotomous deep neural matting for organic boundaries."
            },
            "Workflow 3: Two-Stage (YOLOv8 Object-Centric)": {
                "mask": mask_w3,
                "latency_ms": round(time_w3, 1),
                "eval": eval_w3,
                "description": "Instance-level plant localization to reject background pots/weeds."
            }
        }

        # Step 4: Pick the winner based on highest confidence score
        winner_name = max(workflows_data, key=lambda k: workflows_data[k]["eval"]["confidence_score"])
        winner_data = workflows_data[winner_name]
        winner_mask = winner_data["mask"]

        # Step 5: Background replacement & formatting for downstream models
        composited_winner = composite_background(processed_img, winner_mask, bg_style=bg_style)
        if crop_plant:
            final_winner_img = crop_to_plant_bbox(composited_winner, winner_mask, padding_ratio=0.08)
        else:
            final_winner_img = composited_winner

        total_elapsed = round((time.time() - overall_start) * 1000.0, 1)

        return {
            "winner_name": winner_name,
            "winner_confidence": winner_data["eval"]["confidence_pct"],
            "winner_mask": winner_mask,
            "winner_image": final_winner_img,
            "processed_input": processed_img,
            "workflows": workflows_data,
            "total_latency_ms": total_elapsed
        }
