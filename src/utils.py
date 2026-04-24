"""Utility functions for Intelligent Sketch to CAD."""

import logging
import os
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np


def setup_logging(log_dir: str = "output/logs", log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path / "sketch_to_cad.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def preprocess_image(image_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Preprocess image for analysis."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Resize if needed
    h, w = img.shape[:2]
    max_h, max_w = max_size
    
    if h > max_h or w > max_w:
        scale = min(max_h / h, max_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return img, gray, blurred


def enhance_edges(image: np.ndarray, canny_threshold1: int = 50, canny_threshold2: int = 150) -> np.ndarray:
    """Enhance edges using Canny edge detection."""
    return cv2.Canny(image, canny_threshold1, canny_threshold2)


def find_contours(edges: np.ndarray, min_area: int = 100, max_area: int = 50000) -> List[np.ndarray]:
    """Find significant contours in the image."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    significant_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            significant_contours.append(contour)
    
    return significant_contours


def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_intermediate_result(data: any, output_path: str) -> None:
    """Save intermediate processing results."""
    ensure_directory_exists(str(Path(output_path).parent))
    
    if isinstance(data, np.ndarray):
        cv2.imwrite(output_path, data)
    else:
        with open(output_path, 'w') as f:
            f.write(str(data))
