#!/usr/bin/env python3
"""
Setup script to create proper project structure and organize files.
This creates the correct folder structure and moves files to proper locations.
"""

import os
import shutil
from pathlib import Path

def setup_project_structure():
    """Create and organize project structure properly."""
    
    # Define project root
    project_root = Path(r"/")
    
    print("🔧 Setting up Intelligent_Sketch2CAD project structure...")
    print(f"Project root: {project_root}")
    
    # Define folder structure
    folders = {
        "input_data": project_root / "input_data",
        "raw_sketches": project_root / "input_data" / "raw_sketches", 
        "intermediate_data": project_root / "intermediate_data",
        "output_data": project_root / "output_data",
        "src": project_root / "src",
        "models": project_root / "models",
        "streamlit_app": project_root / "streamlit_app",
        "archive": project_root / "archive",
        "docs": project_root / "docs",
        "config": project_root / "config",
        "tests": project_root / "tests"
    }
    
    # Create folders
    print("\n📁 Creating folders...")
    for name, path in folders.items():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {name}")
        else:
            print(f"  ✓ Exists: {name}")
    
    # Check for SAM2 files
    print("\n🔍 Checking for SAM2 model files...")
    sam2_files = ["sam2_hiera_small.pt", "sam2_hiera_s.yaml"]
    sam2_found = []
    
    for file in sam2_files:
        file_path = project_root / file
        if file_path.exists():
            sam2_found.append(file_path)
            print(f"  ✓ Found: {file}")
        else:
            print(f"  ✗ Missing: {file}")
    
    # Check DeepLSD model
    deeplsd_path = project_root / "DeepLSD" / "weights" / "deeplsd_md.tar"
    if deeplsd_path.exists():
        print(f"  ✓ Found: DeepLSD model")
    else:
        print(f"  ✗ Missing: DeepLSD model at {deeplsd_path}")
    
    # Check input images
    print("\n📸 Checking input images...")
    raw_sketches_dir = folders["raw_sketches"]
    if raw_sketches_dir.exists():
        images = list(raw_sketches_dir.glob("*.jpeg")) + list(raw_sketches_dir.glob("*.jpg"))
        if images:
            print(f"  ✓ Found {len(images)} raw sketch images")
            for img in images[:3]:  # Show first 3
                print(f"    - {img.name}")
        else:
            print("  ✗ No raw sketch images found")
    else:
        print("  ✗ Raw sketches directory not found")
    
    # Clean up old output files
    print("\n🧹 Cleaning up old test files...")
    output_dir = folders["output_data"]
    if output_dir.exists():
        test_files = list(output_dir.glob("full_sketch2cad_*.json"))
        test_files.extend(list(output_dir.glob("technical_drawing_*.png")))
        test_files.extend(list(output_dir.glob("sketch2cad_*.json")))
        
        for file in test_files:
            try:
                file.unlink()
                print(f"  🗑️  Removed: {file.name}")
            except Exception as e:
                print(f"  ❌ Could not remove {file.name}: {e}")
    
    # Create __init__.py files for Python packages
    print("\n🐍 Creating Python package files...")
    python_dirs = ["src", "config", "tests"]
    for dir_name in python_dirs:
        init_file = folders[dir_name] / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Package initialization file\n")
            print(f"  ✓ Created: {dir_name}/__init__.py")
    
    # Create requirements.txt if not exists
    requirements_file = project_root / "requirements.txt"
    if not requirements_file.exists():
        requirements_content = """# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.4.0
loguru>=0.6.0

# Computer vision
opencv-contrib-python>=4.5.0

# SAM2 (if available)
# git+https://github.com/facebookresearch/segment-anything-2.git

# Streamlit (for web interface)
streamlit>=1.28.0

# Development
pytest>=6.2.0
black>=21.0.0
flake8>=3.9.0

# Optional: FreeCAD integration
# FreeCAD (conda install -c conda-forge freecad)
"""
        requirements_file.write_text(requirements_content)
        print("  ✓ Created: requirements.txt")
    else:
        print("  ✓ Exists: requirements.txt")
    
    # Create .gitignore if not exists
    gitignore_file = project_root / ".gitignore"
    if not gitignore_file.exists():
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
intermediate_data/*
output_data/*
!output_data/.gitkeep
models/*
!models/.gitkeep
archive/*
!archive/.gitkeep

# SAM2 models (if large)
*.pt
*.pth

# Temporary files
*.tmp
*.temp
"""
        gitignore_file.write_text(gitignore_content)
        print("  ✓ Created: .gitignore")
    else:
        print("  ✓ Exists: .gitignore")
    
    # Create placeholder files
    print("\n📄 Creating placeholder files...")
    placeholders = [
        (folders["models"] / ".gitkeep", "Keep this directory in git"),
        (folders["output_data"] / ".gitkeep", "Keep this directory in git"),
        (folders["archive"] / ".gitkeep", "Keep this directory in git")
    ]
    
    for file_path, content in placeholders:
        if not file_path.exists():
            file_path.write_text(content)
            print(f"  ✓ Created: {file_path.name}")
    
    print("\n" + "="*60)
    print("✅ PROJECT SETUP COMPLETED!")
    print("="*60)
    
    print("\n📋 Next steps:")
    print("1. Add your sketch images to: input_data/raw_sketches/")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run full pipeline: python src/full_sketch2cad_pipeline.py")
    print("4. Check results in: output_data/")
    
    if not sam2_found:
        print("\n⚠️  SAM2 Setup Required:")
        print("1. Clone SAM2: git clone https://github.com/facebookresearch/segment-anything-2.git")
        print("2. Download SAM2 model from the SAM2 repository")
        print("3. Place sam2_hiera_small.pt and sam2_hiera_s.yaml in project root")
    
    print("\n🎯 Project is ready for development!")

if __name__ == "__main__":
    setup_project_structure()
