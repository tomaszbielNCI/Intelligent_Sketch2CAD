#!/usr/bin/env python3
"""
Test script to verify corrected folder structure works with pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    print("Testing pipeline with corrected folder structure...")
    print("=" * 50)
    
    try:
        from sketch2cad_pipeline import Sketch2CADPipeline
        from datetime import datetime
        import json
        
        # Initialize pipeline
        pipeline = Sketch2CADPipeline()
        
        print("1. Directory structure:")
        print(f"   Input: {pipeline.input_dir}")
        print(f"   Intermediate: {pipeline.intermediate_dir}")
        print(f"   Output: {pipeline.outputs_dir}")
        
        # Test loading
        print("\n2. Testing image loading...")
        gray, source = pipeline.load_image()
        print("   SUCCESS: Image loaded")
        print(f"   Source: {Path(source).name}")
        
        # Test preprocessing
        print("\n3. Testing preprocessing...")
        filtered = pipeline.preprocess_image(gray)
        print("   SUCCESS: Preprocessing completed")
        
        # Test thinning
        print("\n4. Testing thinning...")
        skeleton = pipeline.apply_thinning(filtered)
        print("   SUCCESS: Thinning completed")
        
        # Test DeepLSD
        print("\n5. Testing DeepLSD...")
        all_segments = pipeline.run_deeplsd(filtered)
        print(f"   SUCCESS: {len(all_segments)} segments detected")
        
        # Test classification
        print("\n6. Testing classification...")
        lines = pipeline.classify_lines(all_segments)
        print(f"   SUCCESS: Main={len(lines['main'])}, Dim={len(lines['dimension_lines'])}, Ticks={len(lines['ticks'])}")
        
        # Test shape detection
        print("\n7. Testing shape detection...")
        rectangles = pipeline.detect_rectangles(lines['main'])
        circles = pipeline.detect_circles(skeleton)
        print(f"   SUCCESS: Rectangles={len(rectangles)}, Circles={len(circles)}")
        
        # Test save
        print("\n8. Testing save to output_data...")
        h, w = filtered.shape
        data = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'source_image': source,
            'image_size': {'height': h, 'width': w},
            'test': True,
            'rectangles': rectangles,
            'circles': circles,
            'lines': lines
        }
        
        json_path = pipeline.outputs_dir / 'structure_test.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"   SUCCESS: Saved to {json_path}")
        
        print("\n" + "=" * 50)
        print("STRUCTURE TEST: COMPLETE SUCCESS!")
        print("All pipeline components work with corrected folders.")
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
