#!/usr/bin/env python3
"""
Test script to verify full pipeline works end-to-end.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_full_pipeline():
    """Test the complete full pipeline."""
    print("Testing Full Sketch2CAD Pipeline...")
    print("=" * 50)
    
    try:
        from full_sketch2cad_pipeline import FullSketch2CADPipeline
        
        # Initialize pipeline
        pipeline = FullSketch2CADPipeline()
        
        # Run pipeline with debug
        print("Starting pipeline execution...")
        results = pipeline.run_full_pipeline(save_results=True)
        
        # Check results
        print("\nPipeline completed successfully!")
        print("Results summary:")
        print(f"  Source: {Path(results['source_image']).name}")
        print(f"  Method: {results.get('method', 'unknown')}")
        print(f"  Lines detected: {results['statistics']['deeplsd_raw']}")
        print(f"  Main lines: {results['statistics']['main']}")
        print(f"  Dimension lines: {results['statistics']['dimension']}")
        print(f"  Ticks: {results['statistics']['ticks']}")
        print(f"  Circles: {results['statistics']['circles_mounting']}")
        
        # Check output files
        if 'output_files' in results:
            print("\nOutput files created:")
            for file_type, path in results['output_files'].items():
                file_path = Path(path)
                if file_path.exists():
                    print(f"  {file_type}: {file_path.name} ({file_path.stat().st_size} bytes)")
                else:
                    print(f"  {file_type}: {path} (NOT FOUND)")
        
        return True
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_output_files():
    """Check what files were created."""
    print("\nChecking output files...")
    output_dir = Path("/output_data")
    
    if not output_dir.exists():
        print("Output directory does not exist!")
        return
    
    files = list(output_dir.glob("*"))
    if not files:
        print("No files in output directory")
        return
    
    print(f"Files in {output_dir}:")
    for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        print(f"  {file.name} - {file.stat().st_size} bytes")

if __name__ == "__main__":
    # Test pipeline
    success = test_full_pipeline()
    
    # Check files
    check_output_files()
    
    print("\n" + "=" * 50)
    if success:
        print("FULL PIPELINE TEST: SUCCESS")
    else:
        print("FULL PIPELINE TEST: FAILED")
    print("=" * 50)
    
    sys.exit(0 if success else 1)
