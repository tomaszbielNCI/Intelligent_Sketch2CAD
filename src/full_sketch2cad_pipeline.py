#!/usr/bin/env python3
"""
Full Sketch2CAD Pipeline - Complete automation from raw image to technical drawing.
Faithfully reproduces the logic from notebook v6:
SAM2 preprocessing → adaptive threshold → component filtering → thinning → DeepLSD → classification → rectangle detection → circles → calibration → technical drawing PDF.
"""

import logging
import sys
import os
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import torch
from loguru import logger

# Add DeepLSD to path
sys.path.append(str(Path(__file__).parent.parent / "DeepLSD"))
from deeplsd.models.deeplsd_inference import DeepLSD

# Add SAM2 to path (if available)
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    logger.warning("SAM2 not available - preprocessing will use basic methods")
    SAM2_AVAILABLE = False


class FullSketch2CADPipeline:
    """
    Complete pipeline from raw sketch to technical drawing.
    Replicates notebook logic: SAM2 → adaptive threshold → component filtering → thinning → DeepLSD → classification → rectangle → circles → calibration → PDF.
    """

    def __init__(self, project_dir: Optional[Path] = None):
        """Initialize full pipeline with all components."""
        if project_dir is None:
            project_dir = Path(r"C:\python\Intelligent_Sketch2CAD")

        self.project_dir = project_dir
        self.input_dir = project_dir / "input_data" / "raw_sketches"
        self.intermediate_dir = project_dir / "intermediate_data"
        self.output_dir = project_dir / "output_data"
        self.deeplsd_dir = project_dir / "DeepLSD"
        self.weights_path = self.deeplsd_dir / "weights" / "deeplsd_md.tar"

        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.sam2_model = None
        self.deeplsd_model = None

        # DeepLSD configuration (same as notebook)
        self.device = torch.device('cpu')
        self.deeplsd_conf = {
            'detect_lines': True,
            'line_detection_params': {
                'merge': True, 'filtering': True,
                'grad_thresh': 2, 'grad_nfa': True,
            }
        }

        # Classification parameters (from notebook v6)
        self.ANGLE_TOL_STRAIGHT = 20
        self.ANGLE_TOL_TICK = 18
        self.MIN_MAIN = 80
        self.MIN_DIM = 40
        self.MIN_TICK = 20
        self.MAX_TICK = 90

        # Rectangle filtering
        self.MAX_H_LENGTH = 400  # filter out wall lines > 400px
        self.MIN_H_LENGTH = 80   # minimum length for horizontal lines

        # Real dimensions (from the original sketch) - used for calibration
        self.real_width_mm = 857
        self.real_height_mm = 660

        logger.info("Full Sketch2CAD Pipeline initialized")
        logger.info(f"Input dir: {self.input_dir}")
        logger.info(f"Output dir: {self.output_dir}")

    def load_sam2_model(self) -> bool:
        """Load SAM2 model for preprocessing."""
        if not SAM2_AVAILABLE:
            logger.warning("SAM2 not available - skipping SAM2 preprocessing")
            return False

        try:
            checkpoint_path = "sam2_hiera_small.pt"
            config_path = "sam2_hiera_s.yaml"

            logger.info("Loading SAM2 model...")
            self.sam2_model = build_sam2(config_path, checkpoint_path, device=self.device)
            logger.info("SAM2 model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load SAM2: {e}")
            return False

    def load_deeplsd_model(self) -> bool:
        """Load DeepLSD model for line detection."""
        try:
            logger.info("Loading DeepLSD model...")
            ckpt = torch.load(str(self.weights_path), map_location=self.device, weights_only=False)
            self.deeplsd_model = DeepLSD(self.deeplsd_conf)
            self.deeplsd_model.load_state_dict(ckpt['model'])
            self.deeplsd_model.eval()
            logger.info("DeepLSD model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load DeepLSD: {e}")
            return False

    def load_raw_image(self, image_path: Optional[str] = None) -> Tuple[np.ndarray, str]:
        """Load raw image from input directory."""
        if image_path is None:
            # Use specific default file
            default_image = self.input_dir / "WhatsApp Image 2026-04-24 at 21.41.48.jpeg"
            if default_image.exists():
                image_path = str(default_image)
                logger.info(f"Using default image: {image_path}")
            else:
                # Fallback: most recent file
                patterns = ["*.jpeg", "*.jpg", "*.png", "*.bmp", "*.tiff"]
                files = []
                for pattern in patterns:
                    files.extend(glob.glob(str(self.input_dir / pattern)))

                if not files:
                    raise FileNotFoundError(f"No images found in {self.input_dir}")

                image_path = max(files, key=os.path.getctime)
                logger.info(f"Using latest raw sketch: {image_path}")
        else:
            logger.info(f"Using user-specified image: {image_path}")

        logger.info(f"Loading raw image: {image_path}")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        return image_bgr, image_path

    def preprocess_with_sam2(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image using SAM2 to extract drawing from background (identical to notebook)."""
        if not SAM2_AVAILABLE or self.sam2_model is None:
            logger.warning("SAM2 not available - using basic preprocessing")
            return self.basic_preprocessing(image_bgr)

        try:
            logger.info("Applying SAM2 preprocessing...")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
            masks = mask_generator.generate(image_rgb)
            logger.info(f"Generated {len(masks)} masks")

            # Find largest mask (the paper/drawing)
            masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
            paper_mask = masks_sorted[0]
            mask = paper_mask['segmentation']
            logger.info(f"Using mask with area: {paper_mask['area']} pixels")

            # Create white background with only the masked object
            white_background = np.ones_like(image_rgb) * 255
            white_background[mask] = image_rgb[mask]

            # Crop to object with padding
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                padding = 20
                x_min = max(0, x_min - padding)
                x_max = min(white_background.shape[1], x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(white_background.shape[0], y_max + padding)
                cropped = white_background[y_min:y_max, x_min:x_max]
            else:
                cropped = white_background

            # Save intermediate
            processed_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            logger.info("SAM2 preprocessing completed")
            return processed_bgr, mask

        except Exception as e:
            logger.error(f"SAM2 preprocessing failed: {e}")
            return self.basic_preprocessing(image_bgr)

    def basic_preprocessing(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback preprocessing without SAM2."""
        logger.info("Applying basic preprocessing...")
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        processed = np.ones_like(image_bgr) * 255
        processed[mask > 0] = image_bgr[mask > 0]
        return processed, mask

    def adaptive_threshold_and_mask_removal(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply adaptive threshold and remove border artifacts using mask.
        Replicates notebook logic exactly.
        """
        logger.info("Applying adaptive threshold and mask removal...")

        # Convert to grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold (same parameters as notebook: window_size=25, C=20)
        window_size = 25
        C = 20
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, window_size, C)

        # Crop mask to same size as binary
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            padding = 20
            x_min = max(0, x_min - padding)
            x_max = min(mask.shape[1], x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(mask.shape[0], y_max + padding)
            mask_cropped = mask[y_min:y_max, x_min:x_max]

            # Shrink mask to remove border artifacts
            mask_uint8 = (mask_cropped * 255).astype(np.uint8)
            shrink = 15
            kernel_shrink = np.ones((shrink, shrink), np.uint8)
            mask_shrinked = cv2.erode(mask_uint8, kernel_shrink, iterations=1) > 0

            # Apply mask to binary image
            final = binary.copy()
            final[~mask_shrinked] = 255
        else:
            final = binary

        logger.info("Adaptive threshold completed")
        return final

    def component_filtering(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Component filtering - removes noise and keeps only lines connected to the main drawing.
        This is a KEY step from the notebook that was missing in the original pipeline.
        """
        logger.info("Applying component filtering (connected components)...")

        # Prepare image (white background, black lines)
        contour_img = cv2.bitwise_not(gray_image)
        contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No contours found - skipping component filtering")
            return gray_image

        # Find main contour (largest area)
        main_contour = max(contours, key=cv2.contourArea)
        logger.info(f"Main contour area: {cv2.contourArea(main_contour):.0f} px")

        # Mask inside the main contour
        mask_inside = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.drawContours(mask_inside, [main_contour], -1, 255, thickness=cv2.FILLED)

        # Mask near edge (dilated edge)
        mask_edge = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.drawContours(mask_edge, [main_contour], -1, 255, thickness=2)
        kernel_large = np.ones((120, 120), np.uint8)
        mask_near = cv2.dilate(mask_edge, kernel_large, iterations=1)

        # Connected components analysis
        cc_img = cv2.bitwise_not(gray_image)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(cc_img, connectivity=8)

        mask_final = np.zeros_like(gray_image, dtype=np.uint8)
        MAX_AREA_SMALL = 1500  # Keep small components (ticks, digits) near edge

        kept_inside = 0
        kept_near = 0

        for label in range(1, len(stats)):
            comp_mask = (labels == label).astype(np.uint8) * 255
            area = stats[label, cv2.CC_STAT_AREA]

            # Keep if inside main contour
            if cv2.bitwise_and(comp_mask, mask_inside).any():
                mask_final = cv2.bitwise_or(mask_final, comp_mask)
                kept_inside += 1
                continue

            # Keep if near edge and small (ticks, dimensions, digits)
            if cv2.bitwise_and(comp_mask, mask_near).any() and area < MAX_AREA_SMALL:
                mask_final = cv2.bitwise_or(mask_final, comp_mask)
                kept_near += 1
                continue

        logger.info(f"Component filtering: kept {kept_inside} inside, {kept_near} near edge")

        # Apply mask
        filtered = cv2.bitwise_and(gray_image, mask_final)
        filtered[mask_final == 0] = 255

        return filtered

    def apply_thinning(self, image: np.ndarray) -> np.ndarray:
        """Apply Zhang-Suen thinning using ximgproc (same as notebook)."""
        logger.info("Applying thinning...")

        # Invert so lines are white on black background
        binary_thin = cv2.bitwise_not(image)

        try:
            import cv2.ximgproc as xip
            skeleton = xip.thinning(binary_thin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except ImportError:
            logger.warning("ximgproc not available - using skeletonization fallback")
            # Simple skeletonization fallback
            kernel = np.ones((2, 2), np.uint8)
            skeleton = cv2.erode(binary_thin, kernel, iterations=1)

        logger.info("Thinning completed")
        return skeleton

    def run_deeplsd(self, image: np.ndarray) -> List[Tuple]:
        """Run DeepLSD line detection on preprocessed image."""
        if self.deeplsd_model is None:
            raise RuntimeError("DeepLSD model not loaded")

        logger.info("Running DeepLSD...")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Fix polarity if needed (white background, black lines)
        if np.mean(gray) < 128:
            gray = cv2.bitwise_not(gray)
            logger.info("Polarity fixed")

        # Prepare input tensor
        img_tensor = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        # Run inference
        with torch.no_grad():
            out = self.deeplsd_model({'image': img_tensor})

        lines_raw = out['lines'][0]
        logger.info(f"DeepLSD detected {len(lines_raw)} raw segments")

        # Convert to format (x1, y1, x2, y2, length, angle)
        all_segments = []
        for line in lines_raw:
            x1, y1 = float(line[0][0]), float(line[0][1])
            x2, y2 = float(line[1][0]), float(line[1][1])
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            all_segments.append((int(x1), int(y1), int(x2), int(y2), length, angle))

        return all_segments

    def classify_line(self, x1: int, y1: int, x2: int, y2: int, length: float, angle: float) -> str:
        """Classify a single line segment based on notebook v6 logic."""
        is_h = angle < self.ANGLE_TOL_STRAIGHT or angle > (180 - self.ANGLE_TOL_STRAIGHT)
        is_v = abs(angle - 90) < self.ANGLE_TOL_STRAIGHT
        is_tick = (abs(angle - 45) < self.ANGLE_TOL_TICK or abs(angle - 135) < self.ANGLE_TOL_TICK)

        if (is_h or is_v) and length >= self.MIN_MAIN:
            return "main"
        elif (is_h or is_v) and length >= self.MIN_DIM:
            return "dimension_line"
        elif is_tick and self.MIN_TICK <= length <= self.MAX_TICK:
            return "tick"
        elif length >= self.MIN_DIM:
            return "other"
        return "noise"

    def classify_lines(self, all_segments: List[Tuple]) -> Dict[str, List]:
        """Classify all line segments into categories."""
        logger.info("Classifying line segments...")

        lines = {
            "main": [],
            "dimension_line": [],
            "tick": [],
            "other": [],
            "noise": []
        }

        for seg in all_segments:
            x1, y1, x2, y2, length, angle = seg
            category = self.classify_line(x1, y1, x2, y2, length, angle)

            entry = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "length_px": round(length, 1), "angle_deg": round(angle, 1)
            }
            lines[category].append(entry)

        # Move long "other" lines to main (same as notebook)
        for l in lines["other"]:
            if l["length_px"] > 150:
                lines["main"].append(l)
        lines["other"] = [l for l in lines["other"] if l["length_px"] <= 150]

        logger.info(f"Classification: main={len(lines['main'])}, dimension={len(lines['dimension_line'])}, ticks={len(lines['tick'])}")
        return lines

    def detect_rectangle(self, lines_main: List[Dict]) -> Optional[Dict]:
        """Detect outer rectangle from main horizontal and vertical lines (notebook v6 logic)."""
        logger.info("Detecting rectangle from main lines...")

        if len(lines_main) < 4:
            logger.warning("Not enough main lines for rectangle detection")
            return None

        # Split into horizontal and vertical
        main_h = [l for l in lines_main
                  if l["angle_deg"] < self.ANGLE_TOL_STRAIGHT or
                  l["angle_deg"] > (180 - self.ANGLE_TOL_STRAIGHT)]
        main_v = [l for l in lines_main
                  if abs(l["angle_deg"] - 90) < self.ANGLE_TOL_STRAIGHT]

        # Filter horizontal lines by length (remove wall lines > MAX_H_LENGTH)
        main_h_filtered = [l for l in main_h
                           if self.MIN_H_LENGTH <= l["length_px"] <= self.MAX_H_LENGTH]
        main_v_filtered = [l for l in main_v
                           if l["length_px"] >= self.MIN_H_LENGTH]

        logger.info(f"Filtered: H={len(main_h_filtered)}, V={len(main_v_filtered)}")

        if len(main_h_filtered) < 2 or len(main_v_filtered) < 2:
            logger.warning("Not enough filtered lines for rectangle")
            return None

        # Find top and bottom (min/max Y)
        def mid_y(l):
            return (l["y1"] + l["y2"]) / 2

        def mid_x(l):
            return (l["x1"] + l["x2"]) / 2

        top_line = min(main_h_filtered, key=mid_y)
        bottom_line = max(main_h_filtered, key=mid_y)
        left_line = min(main_v_filtered, key=mid_x)
        right_line = max(main_v_filtered, key=mid_x)

        x1 = int(mid_x(left_line))
        x2 = int(mid_x(right_line))
        y1 = int(mid_y(top_line))
        y2 = int(mid_y(bottom_line))

        rectangle = {
            "label": "outer",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width_px": x2 - x1, "height_px": y2 - y1
        }

        logger.info(f"Rectangle detected: ({x1},{y1})→({x2},{y2}) {x2 - x1}x{y2 - y1}px")
        return rectangle

    def detect_circles(self, skeleton: np.ndarray) -> List[Dict]:
        """Detect circles from skeletonized image using contour analysis (notebook v6 logic)."""
        logger.info("Detecting circles...")

        # Close small gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skeleton_closed = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel_close)

        # Find contours
        contours, _ = cv2.findContours(skeleton_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        circles_all = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300 or area > 12000:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.60:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                circles_all.append({
                    "cx": int(cx), "cy": int(cy),
                    "radius_px": int(radius),
                    "circularity": round(circularity, 3)
                })

        # Remove duplicates
        circles_clean = []
        for c in sorted(circles_all, key=lambda x: -x["radius_px"]):
            if not any(np.hypot(c["cx"] - e["cx"], c["cy"] - e["cy"]) < 25 for e in circles_clean):
                circles_clean.append(c)

        # Take up to 4 largest (mounting holes)
        circles_mounting = circles_clean[:4]
        for i, c in enumerate(circles_mounting):
            c["label"] = f"mounting_{i + 1}"
            c["radius_mm"] = None

        logger.info(f"Detected {len(circles_mounting)} circles")
        return circles_mounting

    def create_visualization(self, gray_image: np.ndarray, lines: Dict,
                             rectangle: Optional[Dict], circles: List[Dict]) -> np.ndarray:
        """Create color-coded visualization (same as notebook)."""
        logger.info("Creating visualization...")

        vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # Draw dimension lines (cyan)
        for l in lines["dimension_line"]:
            cv2.line(vis, (l["x1"], l["y1"]), (l["x2"], l["y2"]), (0, 200, 200), 1)

        # Draw ticks (green)
        for l in lines["tick"]:
            cv2.line(vis, (l["x1"], l["y1"]), (l["x2"], l["y2"]), (0, 255, 80), 2)

        # Draw main lines (red)
        for l in lines["main"]:
            cv2.line(vis, (l["x1"], l["y1"]), (l["x2"], l["y2"]), (180, 60, 60), 1)

        # Draw rectangle (yellow)
        if rectangle:
            cv2.rectangle(vis, (rectangle["x1"], rectangle["y1"]),
                          (rectangle["x2"], rectangle["y2"]), (60, 60, 255), 3)
            cv2.putText(vis, f"{rectangle['width_px']}x{rectangle['height_px']}px",
                        (rectangle["x1"] + 5, rectangle["y1"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)

        # Draw circles (orange)
        for c in circles:
            cv2.circle(vis, (c["cx"], c["cy"]), c["radius_px"], (60, 140, 255), 2)
            cv2.circle(vis, (c["cx"], c["cy"]), 3, (60, 140, 255), -1)
            cv2.putText(vis, c["label"], (c["cx"] + c["radius_px"] + 3, c["cy"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 140, 255), 1)

        return vis

    def save_results(self, data: Dict, vis_final: np.ndarray) -> Tuple[str, str, str]:
        """Save JSON and PNG visualization to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_path = self.output_dir / f"full_pipeline_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON saved: {json_path}")

        # Save PNG visualization
        png_path = self.output_dir / f"technical_drawing_{timestamp}.png"
        cv2.imwrite(str(png_path), vis_final)
        logger.info(f"PNG saved: {png_path}")

        return str(json_path), str(png_path)

    def export_dxf(self, rectangle: Optional[Dict], circles: List[Dict], timestamp: str) -> Optional[str]:
        """Export geometry to DXF file."""
        try:
            import ezdxf
            doc = ezdxf.new("R2010")
            msp = doc.modelspace()

            # Draw rectangle
            if rectangle:
                msp.add_lwpolyline([
                    (rectangle["x1"], -rectangle["y1"]), (rectangle["x2"], -rectangle["y1"]),
                    (rectangle["x2"], -rectangle["y2"]), (rectangle["x1"], -rectangle["y2"]),
                    (rectangle["x1"], -rectangle["y1"])
                ])

            # Draw circles
            for c in circles:
                msp.add_circle(center=(c["cx"], -c["cy"]), radius=c["radius_px"])

            dxf_path = self.output_dir / f"sketch_{timestamp}.dxf"
            doc.saveas(str(dxf_path))
            logger.info(f"DXF saved: {dxf_path}")
            return str(dxf_path)

        except ImportError:
            logger.warning("ezdxf not available - skipping DXF export")
            return None

    def create_technical_pdf(self, rectangle: Dict, circles: List[Dict],
                              lines_main: List[Dict], timestamp: str) -> Optional[str]:
        """
        Create professional technical drawing PDF with dimensions, title block, and mm calibration.
        This fully replicates the notebook's final output.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            # Real dimensions in mm (from sketch)
            width_mm = self.real_width_mm
            height_mm = self.real_height_mm

            # Calibration px → mm
            width_px = rectangle["width_px"]
            height_px = rectangle["height_px"]
            scale_x = width_mm / width_px
            scale_y = height_mm / height_px

            logger.info(f"Scale: {scale_x:.3f} mm/px (X)  {scale_y:.3f} mm/px (Y)")

            # Convert circles to mm
            circles_mm = []
            for c in circles:
                cx_mm = (c["cx"] - rectangle["x1"]) * scale_x
                cy_mm = (c["cy"] - rectangle["y1"]) * scale_y
                circles_mm.append({
                    "cx": cx_mm, "cy": cy_mm,
                    "r": 18.0,  # Half of 36mm diameter
                    "label": c["label"]
                })

            # Manual dimensions (from the original sketch)
            dims = {
                "width_mm": width_mm, "height_mm": height_mm,
                "offset_top_mm": 95, "offset_bottom_mm": 95,
                "offset_left_mm": 102, "offset_right_mm": 96,
                "hole_spacing_h_mm": 665, "hole_spacing_v_mm": 514,
                "hole_diameter_mm": 36,
                "glass_type": "Float", "glass_thickness_mm": 6,
                "note": "Bathroom mirror — technical sketch v1"
            }

            # Plot technical drawing
            MARGIN = 180
            fig, ax = plt.subplots(figsize=(16, 14))
            ax.set_aspect('equal')
            ax.set_facecolor('white')
            ax.axis('off')

            # Main rectangle
            ax.add_patch(patches.Rectangle((0, 0), width_mm, height_mm,
                                           linewidth=2.5, edgecolor='black', facecolor='#f8f8ff'))

            # Circles with crosshairs
            for c in circles_mm:
                ax.add_patch(patches.Circle((c["cx"], height_mm - c["cy"]), c["r"],
                                            linewidth=1.8, edgecolor='#cc6600', facecolor='white'))
                ax.plot(c["cx"], height_mm - c["cy"], '+', color='#cc6600', markersize=10, markeredgewidth=1.5)

            # Dimension line helper
            def dim_line(ax, x1, y1, x2, y2, text, offset=0, orient='h', fontsize=9):
                color = '#444444'
                if orient == 'h':
                    y = y1 + offset
                    ax.plot([x1, x1], [min(y1, y), max(y1, y)], color=color, lw=0.7, ls='--')
                    ax.plot([x2, x2], [min(y2, y), max(y2, y)], color=color, lw=0.7, ls='--')
                    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                                arrowprops=dict(arrowstyle='<->', color=color, lw=1.2, shrinkA=0, shrinkB=0))
                    text_y = y - 8 if offset > 0 else y + 8
                    va = 'top' if offset > 0 else 'bottom'
                    ax.text((x1 + x2) / 2, text_y, text, ha='center', va=va,
                            fontsize=fontsize, fontweight='bold', color='black',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='lightgray', alpha=0.8))
                else:  # vertical
                    x = x1 + offset
                    ax.plot([min(x1, x), max(x1, x)], [y1, y1], color=color, lw=0.7, ls='--')
                    ax.plot([min(x2, x), max(x2, x)], [y2, y2], color=color, lw=0.7, ls='--')
                    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                                arrowprops=dict(arrowstyle='<->', color=color, lw=1.2, shrinkA=0, shrinkB=0))
                    text_x = x - 10 if offset > 0 else x + 10
                    ha = 'right' if offset > 0 else 'left'
                    ax.text(text_x, (y1 + y2) / 2, text, ha=ha, va='center',
                            fontsize=fontsize, fontweight='bold', color='black',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='lightgray', alpha=0.8))

            # Main dimensions
            dim_line(ax, 0, 0, width_mm, 0, f'{width_mm} mm', offset=-55, orient='h', fontsize=10)
            dim_line(ax, 0, 0, 0, height_mm, f'{height_mm} mm', offset=-65, orient='v', fontsize=10)

            # Hole dimensions from edges
            if len(circles_mm) >= 1:
                c = circles_mm[0]
                dim_line(ax, 0, height_mm - c["cy"], c["cx"], height_mm - c["cy"],
                         f'{dims["offset_left_mm"]} mm', offset=25, orient='h', fontsize=8)
                dim_line(ax, c["cx"], height_mm, c["cx"], height_mm - c["cy"],
                         f'{dims["offset_top_mm"]} mm', offset=30, orient='v', fontsize=8)

            # Hole spacing
            if len(circles_mm) >= 2:
                c_left = min(circles_mm, key=lambda c: c["cx"])
                c_right = max(circles_mm, key=lambda c: c["cx"])
                dim_line(ax, c_left["cx"], height_mm - c_left["cy"] - 50,
                         c_right["cx"], height_mm - c_right["cy"] - 50,
                         f'{dims["hole_spacing_h_mm"]} mm', offset=-35, orient='h', fontsize=8)

                c_top = min(circles_mm, key=lambda c: c["cy"])
                c_bottom = max(circles_mm, key=lambda c: c["cy"])
                dim_line(ax, width_mm + 35, height_mm - c_bottom["cy"],
                         width_mm + 35, height_mm - c_top["cy"],
                         f'{dims["hole_spacing_v_mm"]} mm', offset=45, orient='v', fontsize=8)

            # Hole diameter annotation
            if circles_mm:
                c = circles_mm[0]
                ax.annotate(f'⌀ {dims["hole_diameter_mm"]} mm',
                            xy=(c["cx"], height_mm - c["cy"]),
                            xytext=(c["cx"] + 75, height_mm - c["cy"] + 60),
                            fontsize=9, fontweight='bold', color='#cc6600',
                            arrowprops=dict(arrowstyle='->', color='#cc6600', lw=1.2),
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cc6600', alpha=0.9))

            # Title block
            title_y = -110
            title_block_height = 70
            ax.add_patch(patches.Rectangle((-MARGIN/2, title_y - title_block_height),
                                           width_mm + MARGIN, title_block_height,
                                           linewidth=1.5, edgecolor='black', facecolor='#e8e8ff'))

            ax.text(width_mm / 2, title_y - 15, dims["note"],
                    ha='center', va='top', fontsize=12, fontweight='bold')
            ax.text(width_mm / 2, title_y - 33,
                    f'Glass: {dims["glass_type"]}  |  Thickness: {dims["glass_thickness_mm"]} mm',
                    ha='center', va='top', fontsize=9, color='#333333')
            ax.text(width_mm / 2, title_y - 50,
                    f'All dimensions in mm  |  Mounting holes: 4 × ⌀{dims["hole_diameter_mm"]} mm',
                    ha='center', va='top', fontsize=9, color='#333333')
            ax.text(width_mm / 2, title_y - 65,
                    f'Project: Intelligent Sketch2CAD  |  Date: {datetime.now().strftime("%Y-%m-%d")}',
                    ha='center', va='top', fontsize=8, color='#555555')

            # Set limits
            ax.set_xlim(-MARGIN, width_mm + MARGIN)
            ax.set_ylim(title_y - title_block_height - 15, height_mm + MARGIN)
            plt.tight_layout()

            # Save PDF
            pdf_path = self.output_dir / f"technical_drawing_{timestamp}.pdf"
            plt.savefig(str(pdf_path), bbox_inches='tight', facecolor='white')
            plt.close(fig)

            logger.info(f"Technical PDF saved: {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"Failed to create technical PDF: {e}")
            return None

    def run_full_pipeline(self, image_path: Optional[str] = None,
                          save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete pipeline from raw image to technical drawing.
        Follows notebook v6 logic exactly with component filtering and PDF export.
        """
        logger.info("Starting Full Sketch2CAD Pipeline (notebook v6 logic)...")

        # Load models
        sam2_loaded = self.load_sam2_model()
        deeplsd_loaded = self.load_deeplsd_model()
        if not deeplsd_loaded:
            raise RuntimeError("Failed to load DeepLSD model")

        # Step 1: Load raw image
        logger.info("Step 1: Loading raw image...")
        raw_image, source_path = self.load_raw_image(image_path)
        logger.info(f"Loaded: {Path(source_path).name}, shape: {raw_image.shape}")

        # Step 2: SAM2 preprocessing
        logger.info("Step 2: SAM2 preprocessing...")
        preprocessed_image, mask = self.preprocess_with_sam2(raw_image)

        # Step 3: Adaptive threshold + mask removal
        logger.info("Step 3: Adaptive threshold and mask removal...")
        binary_image = self.adaptive_threshold_and_mask_removal(preprocessed_image, mask)

        # Step 4: Component filtering (KEY STEP that was missing)
        logger.info("Step 4: Component filtering...")
        filtered_image = self.component_filtering(binary_image)

        # Step 5: Thinning
        logger.info("Step 5: Thinning...")
        skeleton = self.apply_thinning(filtered_image)

        # Step 6: DeepLSD line detection
        logger.info("Step 6: DeepLSD line detection...")
        all_segments = self.run_deeplsd(filtered_image)

        # Step 7: Line classification
        logger.info("Step 7: Line classification...")
        lines = self.classify_lines(all_segments)

        # Step 8: Rectangle detection
        logger.info("Step 8: Rectangle detection...")
        rectangle = self.detect_rectangle(lines["main"])

        # Step 9: Circle detection
        logger.info("Step 9: Circle detection...")
        circles = self.detect_circles(skeleton)

        # Step 10: Create visualization
        logger.info("Step 10: Creating visualization...")
        vis_final = self.create_visualization(filtered_image, lines, rectangle, circles)

        # Step 11: Prepare results
        h, w = filtered_image.shape
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "timestamp": timestamp,
            "source_image": source_path,
            "image_size": {"height": h, "width": w},
            "method": "Full Pipeline (notebook v6 logic) - SAM2 + component filtering + DeepLSD + PDF export",
            "preprocessing": {"sam2_available": sam2_loaded},
            "rectangle": rectangle,
            "circles": circles,
            "lines": lines,
            "statistics": {
                "deeplsd_raw": len(all_segments),
                "main": len(lines["main"]),
                "dimension": len(lines["dimension_line"]),
                "ticks": len(lines["tick"]),
                "circles_mounting": len(circles)
            }
        }

        # Step 12: Save results
        if save_results:
            logger.info("Step 12: Saving results...")
            json_path, png_path = self.save_results(data, vis_final)
            dxf_path = self.export_dxf(rectangle, circles, timestamp)
            pdf_path = self.create_technical_pdf(rectangle, circles, lines["main"], timestamp) if rectangle else None

            data["output_files"] = {
                "json": json_path,
                "png": png_path,
                "dxf": dxf_path,
                "pdf": pdf_path
            }

        logger.info("Full pipeline completed successfully!")
        return data


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Full Sketch2CAD Pipeline - Raw to Technical Drawing")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--project-dir", type=str, help="Project directory path")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")

    args = parser.parse_args()

    pipeline = FullSketch2CADPipeline(
        project_dir=Path(args.project_dir) if args.project_dir else None
    )

    try:
        results = pipeline.run_full_pipeline(
            image_path=args.image,
            save_results=not args.no_save
        )

        print("\n" + "=" * 50)
        print("FULL PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Processed: {results['source_image']}")
        print(f"Lines detected: {results['statistics']['deeplsd_raw']}")
        if results['rectangle']:
            print(f"Rectangle: {results['rectangle']['width_px']}x{results['rectangle']['height_px']}px")
        else:
            print("No rectangle detected")
        print(f"Circles: {len(results['circles'])}")

        if 'output_files' in results:
            print(f"\nOutput files:")
            print(f"  JSON: {results['output_files']['json']}")
            print(f"  PNG: {results['output_files']['png']}")
            print(f"  DXF: {results['output_files']['dxf']}")
            print(f"  PDF: {results['output_files']['pdf']}")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())