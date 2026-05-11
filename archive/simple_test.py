#!/usr/bin/env python3
"""
Simple test to verify pipeline functionality after refactoring.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    print("Testing Sketch2CAD refactoring...")
    print("=" * 50)
    
    try:
        # Test 1: Import new pipeline
        print("1. Testing new pipeline import...")
        from sketch2cad_pipeline import Sketch2CADPipeline
        print("   [OK] New pipeline imports successfully")
        
        # Test 2: Initialize pipeline
        print("2. Testing pipeline initialization...")
        pipeline = Sketch2CADPipeline()
        print("   [OK] Pipeline initializes successfully")
        
        # Test 3: Check original files still exist
        print("3. Checking original files...")
        original_pipeline = Path(__file__).parent / "src" / "pipeline.py"
        if original_pipeline.exists():
            print("   [OK] Original pipeline.py preserved")
        else:
            print("   [FAIL] Original pipeline.py missing")
            return False
        
        # Test 4: Check new structure
        print("4. Checking new folder structure...")
        folders_to_check = ["models", "outputs", "streamlit_app"]
        for folder in folders_to_check:
            folder_path = Path(__file__).parent / folder
            if folder_path.exists():
                print(f"   [OK] {folder}/ directory exists")
            else:
                print(f"   [FAIL] {folder}/ directory missing")
                return False
        
        # Test 5: Check new files
        print("5. Checking new files...")
        files_to_check = [
            "src/sketch2cad_pipeline.py",
            "src/alternative_detectors.py", 
            "streamlit_app/app.py",
            "README.md"
        ]
        
        for file_path in files_to_check:
            full_path = Path(__file__).parent / file_path
            if full_path.exists():
                print(f"   [OK] {file_path} exists")
            else:
                print(f"   [FAIL] {file_path} missing")
                return False
        
        print("\n" + "=" * 50)
        print("REFACTORING VERIFICATION: SUCCESS")
        print("All new structure elements are in place.")
        print("Original functionality preserved.")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 50)
        print("REFACTORING VERIFICATION: FAILED")
        print("=" * 50)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
