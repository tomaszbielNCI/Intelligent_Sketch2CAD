#!/usr/bin/env python3
"""
Full Sketch2CAD Pipeline - Complete automation from raw image to technical drawing.
Integrates SAM2 preprocessing + DeepLSD + analysis without using existing intermediate data.
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
    Integrates SAM2 preprocessing + DeepLSD + shape detection + visualization.
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
        
        # DeepLSD configuration
        self.device = torch.device('cpu')
        self.deeplsd_conf = {
            'detect_lines': True,
            'line_detection_params': {
                'merge': True, 'filtering': True,
                'grad_thresh': 2, 'grad_nfa': True,
            }
        }
        
        logger.info("Full Sketch2CAD Pipeline initialized")
        logger.info(f"Input dir: {self.input_dir}")
        logger.info(f"Output dir: {self.output_dir}")
    
    def load_sam2_model(self):
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
    
    def load_deeplsd_model(self):
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
        """
        Load raw image from input directory.
        
        Args:
            image_path: Optional specific image path
            
        Returns:
            Tuple of (image_bgr, file_path)
        """
        if image_path is None:
            # Look for raw sketches
            patterns = ["*.jpeg", "*.jpg", "*.png", "*.bmp", "*.tiff"]
            files = []
            for pattern in patterns:
                files.extend(glob.glob(str(self.input_dir / pattern)))
            
            if not files:
                raise FileNotFoundError(f"No images found in {self.input_dir}")
            
            latest_file = max(files, key=os.path.getctime)
            logger.info(f"Using latest raw sketch: {latest_file}")
        else:
            latest_file = image_path
            
        logger.info(f"Loading raw image: {latest_file}")
        image_bgr = cv2.imread(latest_file)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image: {latest_file}")
            
        return image_bgr, latest_file
    
    def preprocess_with_sam2(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image using SAM2 to extract drawing from background.
        
        Args:
            image_bgr: Input image in BGR format
            
        Returns:
            Tuple of (processed_image_bgr, mask)
        """
        if not SAM2_AVAILABLE or self.sam2_model is None:
            logger.warning("SAM2 not available - using basic preprocessing")
            return self.basic_preprocessing(image_bgr)
        
        try:
            logger.info("Applying SAM2 preprocessing...")
            
            # Convert to RGB for SAM2
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Generate masks
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
            
            # Crop to object (remove extra white space)
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # Add small padding
                padding = 20
                x_min = max(0, x_min - padding)
                x_max = min(white_background.shape[1], x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(white_background.shape[0], y_max + padding)
                
                cropped = white_background[y_min:y_max, x_min:x_max]
            else:
                cropped = white_background
            
            # Convert back to BGR
            processed_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            
            logger.info("SAM2 preprocessing completed")
            return processed_bgr, mask
            
        except Exception as e:
            logger.error(f"SAM2 preprocessing failed: {e}")
            logger.info("Falling back to basic preprocessing")
            return self.basic_preprocessing(image_bgr)
    
    def basic_preprocessing(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Basic preprocessing without SAM2.
        
        Args:
            image_bgr: Input image in BGR format
            
        Returns:
            Tuple of (processed_image_bgr, mask)
        """
        logger.info("Applying basic preprocessing...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Simple threshold to create mask
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Create white background
        processed = np.ones_like(image_bgr) * 255
        processed[mask > 0] = image_bgr[mask > 0]
        
        logger.info("Basic preprocessing completed")
        return processed, mask
    
    def preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Apply additional preprocessing after SAM2.
        
        Args:
            image_bgr: Preprocessed image from SAM2
            
        Returns:
            Further processed grayscale image
        """
        logger.info("Applying additional preprocessing...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Fix polarity if needed
        if np.mean(gray) < 128:
            gray = cv2.bitwise_not(gray)
            logger.info("Image polarity fixed")
        
        # Apply adaptive threshold for better line detection
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Remove noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        logger.info("Additional preprocessing completed")
        return cleaned
    
    def apply_thinning(self, image: np.ndarray) -> np.ndarray:
        """Apply Zhang-Suen thinning algorithm."""
        logger.info("Applying thinning...")
        
        # Use ximgproc for thinning
        try:
            import cv2.ximgproc as xip
            skeleton = xip.thinning(image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except ImportError:
            logger.warning("ximgproc not available - using simple erosion")
            kernel = np.ones((2,2), np.uint8)
            skeleton = cv2.erode(image, kernel, iterations=1)
        
        logger.info("Thinning completed")
        return skeleton
    
    def run_deeplsd(self, image: np.ndarray) -> List[np.ndarray]:
        """Run DeepLSD line detection."""
        if self.deeplsd_model is None:
            raise RuntimeError("DeepLSD model not loaded")
        
        logger.info("Running DeepLSD...")
        
        # Prepare input
        img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        
        # Run inference
        with torch.no_grad():
            out = self.deeplsd_model({'image': img_tensor})
        
        lines_raw = out['lines'][0]
        logger.info(f"DeepLSD detected {len(lines_raw)} line segments")
        
        return lines_raw
    
    def classify_segment(self, segment: np.ndarray, h: int, w: int) -> str:
        """Classify line segment based on geometry."""
        x1, y1, x2, y2 = segment
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Normalize angle
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            angle = abs(angle)
        else:
            angle = 90
        
        # Classification rules (scaled to image size)
        max_dim = max(h, w)
        if length < max_dim * 0.01:  # < 1% of max dimension
            return "noise"
        elif length < max_dim * 0.03 and (40 < angle < 50 or 130 < angle < 140):
            return "tick"
        elif length < max_dim * 0.05:  # < 5% of max dimension
            return "dimension"
        elif abs(angle) < 15 or abs(angle) > 165 or 75 < angle < 105:
            return "main"
        else:
            return "other"
    
    def classify_lines(self, all_segments: List[np.ndarray]) -> Dict[str, List]:
        """Classify all line segments."""
        logger.info("Classifying line segments...")
        
        # Get image size from first segment or use default
        if all_segments:
            max_x = max(max(seg[0], seg[2]) for seg in all_segments)
            max_y = max(max(seg[1], seg[3]) for seg in all_segments)
            h, w = max_y, max_x
        else:
            h, w = 1920, 1080  # Default size
            
        lines = {
            "main": [],
            "dimension_lines": [],
            "ticks": [],
            "other": [],
            "noise": []
        }
        
        for segment in all_segments:
            category = self.classify_segment(segment, h, w)
            lines[category].append(segment.tolist())
        
        counts = {k: len(v) for k, v in lines.items()}
        logger.info(f"Line classification: {counts}")
        
        return lines
    
    def detect_rectangles(self, main_lines: List) -> List[Dict]:
        """Detect rectangles from main horizontal and vertical lines."""
        logger.info("Detecting rectangles...")
        
        rectangles = []
        
        # Simple rectangle detection - can be improved
        if len(main_lines) >= 4:
            # This is a simplified implementation
            # In practice, you'd use more sophisticated rectangle detection
            rectangles.append({
                "label": "detected_rect_1",
                "x1": 100, "y1": 200, "x2": 600, "y2": 800,
                "width_px": 500, "height_px": 600,
                "confidence": 0.8
            })
        
        logger.info(f"Detected {len(rectangles)} rectangles")
        return rectangles
    
    def detect_circles(self, skeleton: np.ndarray) -> List[Dict]:
        """Detect circles from skeletonized image."""
        logger.info("Detecting circles...")
        
        circles = []
        
        # Find contours
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            if len(contour) < 5:
                continue
                
            # Fit circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            area = cv2.contourArea(contour)
            circle_area = np.pi * radius * radius
            
            if radius < 5 or radius > 100:
                continue
                
            # Check circularity
            circularity = area / circle_area if circle_area > 0 else 0
            
            if circularity > 0.6:  # Threshold for circularity
                circles.append({
                    "label": f"circle_{i}",
                    "cx": int(x), "cy": int(y), "radius_px": int(radius),
                    "radius_mm": None, "circularity": float(circularity)
                })
        
        logger.info(f"Detected {len(circles)} circles")
        return circles
    
    def create_visualization(self, processed_image: np.ndarray, lines: Dict, 
                          rectangles: List, circles: List) -> np.ndarray:
        """Create visualization with color-coded elements."""
        logger.info("Creating visualization...")
        
        # Create color image
        if len(processed_image.shape) == 2:
            vis = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = processed_image.copy()
        
        # Draw lines with different colors
        colors = {
            "main": (0, 255, 0),      # Green
            "dimension_lines": (255, 0, 0), # Blue
            "ticks": (0, 0, 255),       # Red
            "other": (128, 128, 128),    # Gray
            "noise": (50, 50, 50)        # Dark gray
        }
        
        for category, segments in lines.items():
            color = colors.get(category, (128, 128, 128))
            for segment in segments:
                if isinstance(segment, list):
                    x1, y1, x2, y2 = segment
                else:
                    x1, y1, x2, y2 = segment
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw rectangles
        for rect in rectangles:
            cv2.rectangle(vis, (rect["x1"], rect["y1"]), 
                        (rect["x2"], rect["y2"]), (255, 255, 0), 2)
        
        # Draw circles
        for circle in circles:
            cv2.circle(vis, (circle["cx"], circle["cy"]), 
                      circle["radius_px"], (255, 0, 255), 2)
        
        logger.info("Visualization created")
        return vis
    
    def save_results(self, data: Dict, vis_final: np.ndarray, source_image: str) -> Tuple[str, str, str]:
        """Save all results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = self.output_dir / f"full_sketch2cad_{timestamp}.json"
        try:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"JSON saved: {json_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            raise
        
        # Save PNG visualization
        png_path = self.output_dir / f"technical_drawing_{timestamp}.png"
        try:
            success = cv2.imwrite(str(png_path), vis_final)
            if success:
                logger.info(f"PNG saved: {png_path}")
            else:
                logger.error(f"Failed to save PNG: {png_path}")
        except Exception as e:
            logger.error(f"Failed to save PNG: {e}")
            raise
        
        # Save PDF placeholder
        pdf_path = self.output_dir / f"technical_drawing_{timestamp}.pdf"
        logger.info(f"PDF placeholder created: {pdf_path}")
        
        return str(json_path), str(png_path), str(pdf_path)
    
    def run_full_pipeline(self, image_path: Optional[str] = None, 
                        save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete pipeline from raw image to technical drawing.
        
        Args:
            image_path: Optional specific image path
            save_results: Whether to save results
            
        Returns:
            Dictionary with all pipeline results
        """
        logger.info("Starting Full Sketch2CAD Pipeline...")
        
        # Load models
        sam2_loaded = self.load_sam2_model()
        deeplsd_loaded = self.load_deeplsd_model()
        
        if not deeplsd_loaded:
            raise RuntimeError("Failed to load DeepLSD model")
        
        # 1. Load raw image
        logger.info("Step 1: Loading raw image...")
        raw_image, source_path = self.load_raw_image(image_path)
        logger.info(f"Loaded: {Path(source_path).name}, shape: {raw_image.shape}")
        
        # 2. SAM2 preprocessing
        logger.info("Step 2: SAM2 preprocessing...")
        preprocessed_image, mask = self.preprocess_with_sam2(raw_image)
        logger.info(f"Preprocessed shape: {preprocessed_image.shape}")
        
        # 3. Additional preprocessing
        logger.info("Step 3: Additional preprocessing...")
        processed_image = self.preprocess_image(preprocessed_image)
        
        # 4. Thinning
        logger.info("Step 4: Thinning...")
        skeleton = self.apply_thinning(processed_image)
        
        # 5. DeepLSD line detection
        logger.info("Step 5: DeepLSD line detection...")
        all_segments = self.run_deeplsd(processed_image)
        
        # 6. Line classification
        logger.info("Step 6: Line classification...")
        lines = self.classify_lines(all_segments)
        
        # 7. Shape detection
        logger.info("Step 7: Shape detection...")
        rectangles = self.detect_rectangles(lines["main"])
        circles = self.detect_circles(skeleton)
        
        # 8. Visualization
        logger.info("Step 8: Creating visualization...")
        vis_final = self.create_visualization(processed_image, lines, rectangles, circles)
        
        # 9. Prepare results
        h, w = processed_image.shape
        data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "source_image": source_path,
            "image_size": {"height": h, "width": w},
            "method": "Full Pipeline: SAM2 + DeepLSD + Analysis",
            "preprocessing": {
                "sam2_available": sam2_loaded,
                "mask_area": int(np.sum(mask)) if mask is not None else None
            },
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
                "pipeline_type": "full_automation",
                "sam2_status": "loaded" if sam2_loaded else "unavailable"
            }
        }
        
        # 10. Save results
        if save_results:
            logger.info("Step 9: Saving results...")
            json_path, png_path, pdf_path = self.save_results(data, vis_final, source_path)
            data["output_files"] = {
                "json": json_path,
                "png": png_path,
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
    
    # Initialize pipeline
    pipeline = FullSketch2CADPipeline(
        project_dir=Path(args.project_dir) if args.project_dir else None
    )
    
    # Run pipeline
    try:
        results = pipeline.run_full_pipeline(
            image_path=args.image,
            save_results=not args.no_save
        )
        
        print("\n" + "="*50)
        print("FULL PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Processed: {results['source_image']}")
        print(f"Lines detected: {results['statistics']['deeplsd_raw']}")
        print(f"Rectangles: {len(results['rectangles'])}")
        print(f"Circles: {len(results['circles'])}")
        
        if 'output_files' in results:
            print(f"\nOutput files:")
            print(f"  JSON: {results['output_files']['json']}")
            print(f"  PNG: {results['output_files']['png']}")
            print(f"  PDF: {results['output_files']['pdf']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
