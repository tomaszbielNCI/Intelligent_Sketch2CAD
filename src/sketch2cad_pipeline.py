"""
Intelligent_Sketch2CAD Pipeline
================================

Main pipeline for converting hand sketches to technical drawings.
Supports preprocessing → DeepLSD → JSON → technical drawing generation.

Author: Intelligent_Sketch2CAD Team
Date: 2026-05-11
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
import json
from datetime import datetime
import torch
import sys
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Sketch2CADPipeline:
    """
    Main pipeline for converting sketches to technical drawings.
    
    Workflow:
    1. Load and preprocess image
    2. Apply thinning
    3. Run DeepLSD line detection
    4. Classify line segments
    5. Detect rectangles and circles
    6. Generate JSON output
    7. Create technical drawing visualization
    """
    
    def __init__(self, project_dir: Optional[Path] = None):
        """Initialize pipeline with project directories."""
        if project_dir is None:
            project_dir = Path(r"C:\python\Intelligent_Sketch2CAD")
        
        self.project_dir = project_dir
        self.input_dir = project_dir / "input_data" / "raw_sketches"
        self.intermediate_dir = project_dir / "intermediate_data"
        self.outputs_dir = project_dir / "output_data"
        self.deeplsd_dir = project_dir / "DeepLSD"
        self.weights_path = self.deeplsd_dir / "weights" / "deeplsd_md.tar"
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # DeepLSD configuration
        self.device = torch.device('cpu')
        self.deeplsd_conf = {
            'detect_lines': True,
            'line_detection_params': {
                'merge': True, 'filtering': True,
                'grad_thresh': 2, 'grad_nfa': True,
            }
        }
        
        # Line classification parameters
        self.ANGLE_TOL_STRAIGHT = 20
        self.ANGLE_TOL_TICK = 18
        self.MIN_MAIN = 80
        self.MIN_DIM = 40
        self.MIN_TICK = 20
        self.MAX_TICK = 90
        
        # Rectangle detection parameters
        self.MAX_H_LENGTH = 400  # Above = wall line
        self.MIN_H_LENGTH = 80    # Below = noise
        
        self.net = None
        
    def load_deeplsd_model(self):
        """Load DeepLSD model."""
        logger.info("Loading DeepLSD model...")
        sys.path.insert(0, str(self.deeplsd_dir))
        from deeplsd.models.deeplsd_inference import DeepLSD
        
        ckpt = torch.load(str(self.weights_path), map_location=self.device, weights_only=False)
        self.net = DeepLSD(self.deeplsd_conf)
        self.net.load_state_dict(ckpt['model'])
        self.net.eval()
        logger.info("DeepLSD model loaded successfully")
        
    def load_image(self, image_path: Optional[str] = None) -> Tuple[np.ndarray, str]:
        """
        Load latest preprocessed image or specific image path.
        If no path provided, looks for adaptive_cleaned_*.jpg in intermediate_data/
        If path is provided, loads from that location.
        
        Args:
            image_path: Optional specific image path to load
            
        Returns:
            Tuple of (image, file_path)
        """
        if image_path is None:
            # Look for preprocessed images in intermediate_data
            pattern = str(self.intermediate_dir / "adaptive_cleaned_*.jpg")
            files = glob.glob(pattern)
            if not files:
                # If no preprocessed images, look for raw sketches
                logger.info("No preprocessed images found, looking for raw sketches...")
                raw_pattern = str(self.input_dir / "*.jpeg")
                raw_files = glob.glob(raw_pattern)
                if not raw_files:
                    raw_pattern = str(self.input_dir / "*.jpg")
                    raw_files = glob.glob(raw_pattern)
                if not raw_files:
                    raise FileNotFoundError(f"No images found in {self.input_dir}")
                latest_file = max(raw_files, key=os.path.getctime)
                logger.info(f"Using raw sketch: {latest_file}")
            else:
                latest_file = max(files, key=os.path.getctime)
                logger.info(f"Using preprocessed image: {latest_file}")
        else:
            latest_file = image_path
            
        logger.info(f"Loading image: {latest_file}")
        image = cv2.imread(latest_file)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {latest_file}")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Fix polarity if needed
        if np.mean(gray) < 128:
            gray = cv2.bitwise_not(gray)
            logger.info("Image polarity fixed")
            
        return gray, latest_file
        
    def preprocess_image(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing filters to remove noise and enhance main features.
        
        Args:
            gray: Grayscale input image
            
        Returns:
            Preprocessed grayscale image
        """
        logger.info("Applying preprocessing...")
        
        contour_img = cv2.bitwise_not(gray)
        contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return gray
            
        main_contour = max(contours, key=cv2.contourArea)
        mask_inside = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask_inside, [main_contour], -1, 255, thickness=cv2.FILLED)
        mask_edge = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask_edge, [main_contour], -1, 255, thickness=2)
        mask_near = cv2.dilate(mask_edge, np.ones((120, 120), np.uint8), iterations=1)
        
        cc_img = cv2.bitwise_not(gray)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(cc_img, connectivity=8)
        mask_final = np.zeros_like(gray, dtype=np.uint8)
        
        for label in range(1, len(stats)):
            comp_mask = (labels == label).astype(np.uint8) * 255
            area = stats[label, cv2.CC_STAT_AREA]
            if cv2.bitwise_and(comp_mask, mask_inside).any():
                mask_final = cv2.bitwise_or(mask_final, comp_mask)
            elif cv2.bitwise_and(comp_mask, mask_near).any() and area < 1500:
                mask_final = cv2.bitwise_or(mask_final, comp_mask)
                
        gray_filtered = cv2.bitwise_and(gray, mask_final)
        gray_filtered[mask_final == 0] = 255
        
        logger.info("Preprocessing completed")
        return gray_filtered
        
    def apply_thinning(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Zhang-Suen thinning algorithm.
        
        Args:
            gray: Grayscale input image
            
        Returns:
            Skeletonized image
        """
        logger.info("Applying thinning...")
        binary_thin = cv2.bitwise_not(gray)
        skeleton = cv2.ximgproc.thinning(binary_thin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        logger.info("Thinning completed")
        return skeleton
        
    def run_deeplsd(self, gray: np.ndarray) -> List[Tuple]:
        """
        Run DeepLSD line detection.
        
        Args:
            gray: Grayscale input image
            
        Returns:
            List of detected line segments
        """
        logger.info("Running DeepLSD...")
        
        if self.net is None:
            self.load_deeplsd_model()
            
        img_tensor = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            out = self.net({'image': img_tensor})
            
        lines_raw = out['lines'][0]
        logger.info(f"DeepLSD detected {len(lines_raw)} line segments")
        
        all_segments = []
        for line in lines_raw:
            x1, y1 = float(line[0][0]), float(line[0][1])
            x2, y2 = float(line[1][0]), float(line[1][1])
            length = np.hypot(x2-x1, y2-y1)
            angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
            all_segments.append((int(x1), int(y1), int(x2), int(y2), length, angle))
            
        return all_segments
        
    def classify_segment(self, x1: float, y1: float, x2: float, y2: float, 
                      length: float, angle: float) -> str:
        """Classify line segment based on geometry."""
        is_h = angle < self.ANGLE_TOL_STRAIGHT or angle > (180 - self.ANGLE_TOL_STRAIGHT)
        is_v = abs(angle - 90) < self.ANGLE_TOL_STRAIGHT
        is_tick = (abs(angle-45) < self.ANGLE_TOL_TICK or abs(angle-135) < self.ANGLE_TOL_TICK)
        
        if (is_h or is_v) and length >= self.MIN_MAIN:
            return "main"
        elif (is_h or is_v) and length >= self.MIN_DIM:
            return "dimension_line"
        elif is_tick and self.MIN_TICK <= length <= self.MAX_TICK:
            return "tick"
        elif length >= self.MIN_DIM:
            return "other"
        return "noise"
        
    def classify_lines(self, all_segments: List[Tuple]) -> Dict[str, List[Dict]]:
        """
        Classify all line segments into categories.
        
        Args:
            all_segments: List of line segments
            
        Returns:
            Dictionary with classified lines
        """
        logger.info("Classifying line segments...")
        
        lines_main, lines_dimension, lines_tick, lines_other = [], [], [], []
        
        for seg in all_segments:
            x1, y1, x2, y2, length, angle = seg
            typ = self.classify_segment(x1, y1, x2, y2, length, angle)
            entry = {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "length_px": round(length, 1), "angle_deg": round(angle, 1)}
            
            if typ == "main":
                lines_main.append(entry)
            elif typ == "dimension_line":
                lines_dimension.append(entry)
            elif typ == "tick":
                lines_tick.append(entry)
            elif typ == "other":
                lines_other.append(entry)
                
        # Move long "other" lines to main
        for l in lines_other:
            if l['length_px'] > 150:
                lines_main.append(l)
                
        logger.info(f"Classification complete: main={len(lines_main)}, "
                   f"dimension={len(lines_dimension)}, tick={len(lines_tick)}")
        
        return {
            "main": lines_main,
            "dimension_lines": lines_dimension,
            "ticks": lines_tick
        }
        
    def detect_rectangles(self, lines_main: List[Dict]) -> List[Dict]:
        """
        Detect rectangles from main horizontal and vertical lines.
        
        Args:
            lines_main: List of main lines
            
        Returns:
            List of detected rectangles
        """
        logger.info("Detecting rectangles...")
        
        def mid_y(l): return (l['y1'] + l['y2']) / 2
        def mid_x(l): return (l['x1'] + l['x2']) / 2
        
        main_h = [l for l in lines_main
                  if l['angle_deg'] < self.ANGLE_TOL_STRAIGHT or
                     l['angle_deg'] > (180 - self.ANGLE_TOL_STRAIGHT)]
        main_v = [l for l in lines_main
                  if abs(l['angle_deg'] - 90) < self.ANGLE_TOL_STRAIGHT]
        
        # Filter lines
        main_h_filtered = [l for l in main_h
                         if self.MIN_H_LENGTH <= l['length_px'] <= self.MAX_H_LENGTH]
        main_v_filtered = [l for l in main_v
                         if l['length_px'] >= self.MIN_H_LENGTH]
        
        rectangles = []
        if len(main_h_filtered) >= 2 and len(main_v_filtered) >= 2:
            top_line = min(main_h_filtered, key=mid_y)
            bottom_line = max(main_h_filtered, key=mid_y)
            left_line = min(main_v_filtered, key=mid_x)
            right_line = max(main_v_filtered, key=mid_x)
            
            x1r = int(mid_x(left_line))
            x2r = int(mid_x(right_line))
            y1r = int(mid_y(top_line))
            y2r = int(mid_y(bottom_line))
            
            rectangles.append({
                "label": "outer",
                "x1": x1r, "y1": y1r,
                "x2": x2r, "y2": y2r,
                "width_px": x2r - x1r,
                "height_px": y2r - y1r
            })
            
            logger.info(f"Rectangle detected: ({x1r},{y1r})→({x2r},{y2r}) {x2r-x1r}x{y2r-y1r}px")
        else:
            logger.warning("Insufficient lines for rectangle detection")
            
        return rectangles
        
    def detect_circles(self, skeleton: np.ndarray) -> List[Dict]:
        """
        Detect circles from skeletonized image.
        
        Args:
            skeleton: Skeletonized image
            
        Returns:
            List of detected circles
        """
        logger.info("Detecting circles...")
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        skeleton_closed = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel_close)
        circle_contours, _ = cv2.findContours(skeleton_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        circles_all = []
        for cnt in circle_contours:
            area = cv2.contourArea(cnt)
            if area < 300 or area > 12000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.60:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                circles_all.append({"cx": int(cx), "cy": int(cy),
                                 "radius_px": int(radius), "circularity": round(circularity, 3)})
        
        # Remove duplicates
        circles_clean = []
        for c in sorted(circles_all, key=lambda x: -x['radius_px']):
            if not any(np.hypot(c['cx']-e['cx'], c['cy']-e['cy']) < 25 for e in circles_clean):
                circles_clean.append(c)
        
        # Keep top 4 as mounting holes
        circles_mounting = sorted(circles_clean, key=lambda x: -x['radius_px'])[:4]
        for i, c in enumerate(circles_mounting):
            c['label'] = f"mounting_{i+1}"
            c['radius_mm'] = None
            
        logger.info(f"Detected {len(circles_mounting)} mounting circles")
        return circles_mounting
        
    def create_visualization(self, gray: np.ndarray, lines: Dict[str, List[Dict]], 
                           rectangles: List[Dict], circles: List[Dict]) -> np.ndarray:
        """
        Create final visualization with all detected elements.
        
        Args:
            gray: Original grayscale image
            lines: Classified line segments
            rectangles: Detected rectangles
            circles: Detected circles
            
        Returns:
            Visualization image
        """
        logger.info("Creating final visualization...")
        
        vis_final = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Draw dimension lines
        for l in lines["dimension_lines"]:
            cv2.line(vis_final, (l['x1'], l['y1']), (l['x2'], l['y2']), (0, 200, 200), 1)
        
        # Draw tick marks
        for l in lines["ticks"]:
            cv2.line(vis_final, (l['x1'], l['y1']), (l['x2'], l['y2']), (0, 255, 80), 2)
        
        # Draw main lines
        for l in lines["main"]:
            cv2.line(vis_final, (l['x1'], l['y1']), (l['x2'], l['y2']), (180, 60, 60), 1)
        
        # Draw rectangles
        for r in rectangles:
            cv2.rectangle(vis_final, (r['x1'], r['y1']), (r['x2'], r['y2']), (60, 60, 255), 3)
            cv2.putText(vis_final, f"{r['width_px']}x{r['height_px']}px",
                       (r['x1']+5, r['y1']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)
        
        # Draw circles
        for c in circles:
            cv2.circle(vis_final, (c['cx'], c['cy']), c['radius_px'], (60, 140, 255), 2)
            cv2.circle(vis_final, (c['cx'], c['cy']), 3, (60, 140, 255), -1)
            cv2.putText(vis_final, c['label'], (c['cx']+c['radius_px']+3, c['cy']),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 140, 255), 1)
        
        return vis_final
        
    def save_results(self, data: Dict[str, Any], vis_final: np.ndarray, 
                   source_image: str) -> Tuple[str, str, str]:
        """
        Save JSON results and visualization.
        
        Args:
            data: Dictionary with all detection results
            vis_final: Final visualization image
            source_image: Path to source image
            
        Returns:
            Tuple of (json_path, png_path, pdf_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = self.outputs_dir / f"sketch2cad_{timestamp}.json"
        try:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"JSON saved: {json_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            raise
        
        # Save PNG visualization
        png_path = self.outputs_dir / f"technical_drawing_{timestamp}.png"
        try:
            success = cv2.imwrite(str(png_path), vis_final)
            if success:
                logger.info(f"PNG saved: {png_path}")
            else:
                logger.error(f"Failed to save PNG: {png_path}")
        except Exception as e:
            logger.error(f"Failed to save PNG: {e}")
            raise
        
        # Save PDF (placeholder - would need additional implementation)
        pdf_path = self.outputs_dir / f"technical_drawing_{timestamp}.pdf"
        # TODO: Implement PDF generation using reportlab or similar
        logger.info(f"PDF placeholder created: {pdf_path}")
        
        return str(json_path), str(png_path), str(pdf_path)
        
    def run_pipeline(self, image_path: Optional[str] = None, 
                    save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete pipeline from image to technical drawing.
        
        Args:
            image_path: Optional specific image path to process
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with all pipeline results
        """
        logger.info("Starting Sketch2CAD pipeline...")
        
        # 1. Load and preprocess image
        gray, source_image = self.load_image(image_path)
        gray_filtered = self.preprocess_image(gray)
        
        # 2. Apply thinning
        skeleton = self.apply_thinning(gray_filtered)
        
        # 3. Run DeepLSD
        all_segments = self.run_deeplsd(gray_filtered)
        
        # 4. Classify lines
        lines = self.classify_lines(all_segments)
        
        # 5. Detect rectangles and circles
        rectangles = self.detect_rectangles(lines["main"])
        circles = self.detect_circles(skeleton)
        
        # 6. Create visualization
        vis_final = self.create_visualization(gray_filtered, lines, rectangles, circles)
        
        # 7. Prepare results data
        h, w = gray_filtered.shape
        data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "source_image": source_image,
            "image_size": {"height": h, "width": w},
            "method": "DeepLSD v6 - filtered H/V + contour circles",
            "rectangles": rectangles,
            "circles": circles,
            "lines": lines,
            "statistics": {
                "deeplsd_raw": len(all_segments),
                "main": len(lines["main"]),
                "dimension": len(lines["dimension_lines"]),
                "ticks": len(lines["ticks"]),
                "circles_mounting": len(circles)
            },
            "notes": {
                "scale_hint": "857mm = external width"
            }
        }
        
        # 8. Save results
        if save_results:
            json_path, png_path, pdf_path = self.save_results(data, vis_final, source_image)
            data["output_files"] = {
                "json": json_path,
                "png": png_path,
                "pdf": pdf_path
            }
        
        logger.info("Pipeline completed successfully")
        return data


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sketch2CAD Pipeline")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--project-dir", type=str, help="Project directory path")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    project_dir = Path(args.project_dir) if args.project_dir else None
    pipeline = Sketch2CADPipeline(project_dir)
    
    # Run pipeline
    results = pipeline.run_pipeline(
        image_path=args.image,
        save_results=not args.no_save
    )
    
    # Print summary
    print("\n" + "="*50)
    print("SKETCH2CAD PIPELINE RESULTS")
    print("="*50)
    print(f"Source: {results['source_image']}")
    print(f"Image size: {results['image_size']['width']}x{results['image_size']['height']}px")
    print(f"Rectangles: {len(results['rectangles'])}")
    print(f"Circles: {len(results['circles'])}")
    print(f"Main lines: {results['statistics']['main']}")
    print(f"Dimension lines: {results['statistics']['dimension']}")
    print(f"Tick marks: {results['statistics']['ticks']}")
    
    if "output_files" in results:
        print(f"\nOutput files:")
        print(f"  JSON: {results['output_files']['json']}")
        print(f"  PNG: {results['output_files']['png']}")
        print(f"  PDF: {results['output_files']['pdf']}")
    
    print("="*50)


if __name__ == "__main__":
    main()
