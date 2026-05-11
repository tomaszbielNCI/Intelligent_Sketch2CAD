#!/usr/bin/env python3
"""
Simple test to verify full pipeline works step by step.
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    print("Testing Full Pipeline Step by Step...")
    print("=" * 50)
    
    try:
        from full_sketch2cad_pipeline import FullSketch2CADPipeline
        
        # Initialize
        pipeline = FullSketch2CADPipeline()
        
        # Test each step
        print("1. Loading models...")
        sam2_ok = pipeline.load_sam2_model()
        deeplsd_ok = pipeline.load_deeplsd_model()
        print(f"   SAM2: {sam2_ok}, DeepLSD: {deeplsd_ok}")
        
        print("2. Loading raw image...")
        raw_image, source = pipeline.load_raw_image()
        print(f"   Source: {Path(source).name}")
        print(f"   Shape: {raw_image.shape}")
        
        print("3. Preprocessing...")
        preprocessed, mask = pipeline.preprocess_with_sam2(raw_image)
        print(f"   Preprocessed: {preprocessed.shape}")
        
        print("4. Additional preprocessing...")
        processed = pipeline.preprocess_image(preprocessed)
        print(f"   Processed: {processed.shape}")
        
        print("5. Thinning...")
        skeleton = pipeline.apply_thinning(processed)
        print(f"   Skeleton: {skeleton.shape}")
        
        print("6. DeepLSD line detection...")
        segments = pipeline.run_deeplsd(processed)
        print(f"   Segments: {len(segments)}")
        
        print("7. Line classification...")
        lines = pipeline.classify_lines(segments)
        print(f"   Main: {len(lines['main'])}")
        print(f"   Dimension: {len(lines['dimension_lines'])}")
        print(f"   Ticks: {len(lines['ticks'])}")
        
        print("8. Shape detection...")
        rectangles = pipeline.detect_rectangles(lines['main'])
        circles = pipeline.detect_circles(skeleton)
        print(f"   Rectangles: {len(rectangles)}")
        print(f"   Circles: {len(circles)}")
        
        print("9. Visualization...")
        vis = pipeline.create_visualization(processed, lines, rectangles, circles)
        print(f"   Visualization: {vis.shape}")
        
        print("10. Preparing results...")
        h, w = processed.shape
        data = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'source_image': source,
            'image_size': {'height': h, 'width': w},
            'method': 'Full Pipeline - Step by Step Test',
            'preprocessing': {
                'sam2_available': sam2_ok,
                'mask_area': int(np.sum(mask)) if mask is not None else None
            },
            'rectangles': rectangles,
            'circles': circles,
            'lines': lines,
            'statistics': {
                'deeplsd_raw': len(segments),
                'main': len(lines['main']),
                'dimension': len(lines['dimension_lines']),
                'ticks': len(lines['ticks']),
                'circles_mounting': len(circles)
            }
        }
        
        print("11. Saving results...")
        json_path, png_path, pdf_path = pipeline.save_results(data, vis, source)
        print(f"   JSON: {json_path}")
        print(f"   PNG: {png_path}")
        print(f"   PDF: {pdf_path}")
        
        print("\n" + "=" * 50)
        print("FULL PIPELINE TEST: SUCCESS!")
        print("All steps completed successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
