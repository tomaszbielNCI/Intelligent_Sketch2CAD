#!/usr/bin/env python3
"""
Test script to verify the Sketch2CAD pipeline functionality.
This ensures the existing working code remains functional after refactoring.
"""

import sys
from pathlib import Path
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_pipeline():
    """Test the complete pipeline functionality."""
    print("=" * 60)
    print("TESTING SKETCH2CAD PIPELINE")
    print("=" * 60)
    
    try:
        # Import pipeline
        print("1. Importing pipeline...")
        from sketch2cad_pipeline import Sketch2CADPipeline
        print("   [OK] Import successful")
        
        # Initialize pipeline
        print("2. Initializing pipeline...")
        pipeline = Sketch2CADPipeline()
        print("   ✓ Pipeline initialized")
        
        # Test individual components
        print("3. Testing image loading...")
        gray, source = pipeline.load_image()
        print(f"   ✓ Image loaded: {Path(source).name}")
        print(f"   ✓ Shape: {gray.shape}")
        
        print("4. Testing preprocessing...")
        filtered = pipeline.preprocess_image(gray)
        print("   ✓ Preprocessing completed")
        print(f"   ✓ Shape: {filtered.shape}")
        
        print("5. Testing thinning...")
        skeleton = pipeline.apply_thinning(filtered)
        print("   ✓ Thinning completed")
        print(f"   ✓ Shape: {skeleton.shape}")
        
        print("6. Testing DeepLSD...")
        all_segments = pipeline.run_deeplsd(filtered)
        print(f"   ✓ DeepLSD detected {len(all_segments)} segments")
        
        print("7. Testing line classification...")
        lines = pipeline.classify_lines(all_segments)
        print(f"   ✓ Main lines: {len(lines['main'])}")
        print(f"   ✓ Dimension lines: {len(lines['dimension_lines'])}")
        print(f"   ✓ Tick marks: {len(lines['ticks'])}")
        
        print("8. Testing rectangle detection...")
        rectangles = pipeline.detect_rectangles(lines['main'])
        print(f"   ✓ Rectangles detected: {len(rectangles)}")
        
        print("9. Testing circle detection...")
        circles = pipeline.detect_circles(skeleton)
        print(f"   ✓ Circles detected: {len(circles)}")
        
        print("10. Testing visualization...")
        vis_final = pipeline.create_visualization(filtered, lines, rectangles, circles)
        print(f"   ✓ Visualization created: {vis_final.shape}")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("The pipeline is working correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("TESTS FAILED! ✗")
        print("=" * 60)
        return False

def test_notebook_compatibility():
    """Test that notebooks can still access the original functionality."""
    print("\n" + "=" * 60)
    print("TESTING NOTEBOOK COMPATIBILITY")
    print("=" * 60)
    
    try:
        # Test that original pipeline.py still exists and can be imported
        print("1. Checking original pipeline...")
        original_pipeline = Path(__file__).parent / "src" / "pipeline.py"
        if original_pipeline.exists():
            print("   ✓ Original pipeline.py exists")
        else:
            print("   ✗ Original pipeline.py missing")
            return False
        
        # Test that notebooks can still run (basic import test)
        print("2. Testing notebook imports...")
        sys.path.insert(0, str(Path(__file__).parent / "DeepLSD"))
        try:
            from deeplsd.models.deeplsd_inference import DeepLSD
            print("   ✓ DeepLSD import successful")
        except ImportError as e:
            print(f"   ✗ DeepLSD import failed: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("NOTEBOOK COMPATIBILITY: PASSED ✓")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n" + "=" * 60)
        print("NOTEBOOK COMPATIBILITY: FAILED ✗")
        print("=" * 60)
        return False

if __name__ == "__main__":
    print("Testing Sketch2CAD refactoring...")
    
    # Run tests
    pipeline_ok = test_pipeline()
    notebook_ok = test_notebook_compatibility()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pipeline functionality: {'✓ PASSED' if pipeline_ok else '✗ FAILED'}")
    print(f"Notebook compatibility: {'✓ PASSED' if notebook_ok else '✗ FAILED'}")
    
    if pipeline_ok and notebook_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("The refactoring was successful and existing code remains functional.")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the errors above.")
        sys.exit(1)
