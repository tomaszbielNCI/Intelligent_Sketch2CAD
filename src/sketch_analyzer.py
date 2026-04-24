"""Sketch analysis module for extracting shapes and dimensions."""

import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .utils import preprocess_image, enhance_edges, find_contours


class SketchAnalyzer:
    """Analyze hand-drawn sketches to extract shapes and dimensions."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.image_config = config.get('image', {})
        self.ocr_config = config.get('ocr', {})
        self.shape_config = config.get('shape_detection', {})
    
    def analyze_sketch(self, image_path: str) -> Dict:
        """Analyze a sketch image and extract shapes and dimensions."""
        # Preprocess image
        original, gray, blurred = preprocess_image(
            image_path, 
            max_size=tuple(self.image_config.get('max_size', [1024, 1024]))
        )
        
        # Enhance edges
        edges = enhance_edges(
            blurred,
            self.image_config.get('canny_threshold1', 50),
            self.image_config.get('canny_threshold2', 150)
        )
        
        # Find contours
        contours = find_contours(
            edges,
            self.shape_config.get('min_contour_area', 100),
            self.shape_config.get('max_contour_area', 50000)
        )
        
        # Analyze shapes
        shapes = self._analyze_shapes(contours)
        
        # Extract text and dimensions
        dimensions = self._extract_dimensions(gray)
        
        return {
            'shapes': shapes,
            'dimensions': dimensions,
            'image_info': {
                'original_size': original.shape[:2],
                'processed_size': gray.shape[:2],
                'num_contours': len(contours)
            }
        }
    
    def _analyze_shapes(self, contours: List[np.ndarray]) -> List[Dict]:
        """Analyze contours to identify basic shapes."""
        shapes = []
        approximation = self.shape_config.get('contour_approximation', 0.02)
        angle_tolerance = self.shape_config.get('angle_tolerance', 5.0)
        
        for i, contour in enumerate(contours):
            # Approximate contour
            epsilon = approximation * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Determine shape type
            shape_type = self._classify_shape(approx, w, h, angle_tolerance)
            
            shapes.append({
                'id': i,
                'type': shape_type,
                'contour': contour.tolist(),
                'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h},
                'area': area,
                'center': {'x': x + w // 2, 'y': y + h // 2}
            })
        
        return shapes
    
    def _classify_shape(self, approx: np.ndarray, width: int, height: int, angle_tolerance: float) -> str:
        """Classify shape based on contour approximation."""
        vertices = len(approx)
        
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            # Check if rectangle or square
            aspect_ratio = width / float(height)
            if 0.9 <= aspect_ratio <= 1.1:
                return "square"
            else:
                return "rectangle"
        elif vertices > 4:
            # Check if circle
            (x, y), radius = cv2.minEnclosingCircle(approx)
            circle_area = np.pi * radius * radius
            contour_area = cv2.contourArea(approx)
            
            if abs(circle_area - contour_area) / circle_area < 0.2:
                return "circle"
            else:
                return "polygon"
        else:
            return "unknown"
    
    def _extract_dimensions(self, gray_image: np.ndarray) -> List[Dict]:
        """Extract text and numerical dimensions from the image."""
        # Configure Tesseract
        custom_config = self.ocr_config.get('tesseract_config', '--psm 6 --oem 3')
        
        # Extract text
        text = pytesseract.image_to_string(gray_image, config=custom_config)
        
        # Extract dimensions using regex patterns
        dimensions = []
        patterns = self.ocr_config.get('dimension_patterns', [
            r'\d+\.?\d*\s*mm',
            r'\d+\.?\d*\s*cm', 
            r'\d+\.?\d*\s*m'
        ])
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value_str = re.findall(r'\d+\.?\d*', match.group())[0]
                unit = re.findall(r'[a-zA-Z]+', match.group())[0].lower()
                
                dimensions.append({
                    'value': float(value_str),
                    'unit': unit,
                    'text': match.group(),
                    'position': match.start()
                })
        
        return dimensions
    
    def save_analysis(self, analysis: Dict, output_path: str) -> None:
        """Save analysis results to file."""
        import json
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
