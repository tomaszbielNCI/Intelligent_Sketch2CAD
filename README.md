# Intelligent Sketch2CAD

**Vertical Slice** – Converts hand-drawn technical sketches into professional technical drawings (PDF, DXF) compatible with **FreeCAD** for glass construction and fitting (windows, doors, mirrors).

## 🎯 Real-World Application

This project solves a real business problem for a glass construction and fitting company (windows, doors, mirrors). Instead of outsourcing technical drawings or spending hours in AutoCAD, the owner can now:
1. Take a photo of a hand-drawn sketch
2. Run the pipeline
3. Get a ready-to-use technical drawing (PDF/DXF) with dimensions, mounting holes, and title block
4. Open directly in **FreeCAD** for further editing or CAM export

**Tested use case**: Bathroom mirror – 857×660 mm with 4 mounting holes (⌀36 mm)

## 🔧 Pipeline Workflow

Raw Sketch → SAM2 (background removal) → Adaptive Threshold → Component Filtering → Thinning → DeepLSD (line detection) → Classification → Rectangle & Circle Detection → Calibration (px → mm) → Technical Drawing (PDF/DXF) → FreeCAD

## 🏗️ Project Architecture

This project provides a complete pipeline for converting hand sketches to technical drawings, with support for multiple line detection methods and extensible architecture.

### 📁 Project Structure

```
Intelligent_Sketch2CAD/
├── 📓 notebooks/                    # Jupyter notebooks for development and testing
│   ├── 0_preprocessing_load_image.ipynb
│   ├── 1_extract_contours_*.ipynb
│   ├── 2_extract_deepLSD_*.ipynb
│   └── vertical_slice_0.ipynb
├── 📂 src/                          # Core pipeline modules
│   ├── full_sketch2cad_pipeline.py    # Complete pipeline with SAM2 + DeepLSD (recommended)
│   ├── alternative_detectors.py        # Alternative line detection methods (TODO)
│   └── __init__.py                    # Package initialization
├── 📂 models/                       # Model storage
├── 📂 streamlit_app/               # Web interface (TODO)
│   └── app.py                        # Streamlit UI stub
├── 📂 input_data/                   # Input sketches
│   └── raw_sketches/               # Raw sketch images
├── 📂 intermediate_data/             # Preprocessed images and intermediate results
├── 📂 output_data/                  # Generated technical drawings and JSON results
├── 📂 archive/                      # Archived files and test scripts
├── 📂 DeepLSD/                      # DeepLSD model and code
├── 📂 docs/                         # Documentation
├── 📂 config/                       # Configuration files
├── 📂 tests/                        # Unit tests
├── app.py                            # Main application entry
├── main.py                           # Alternative entry point
└── requirements.txt                   # Python dependencies
```

### 🔧 Core Pipeline Workflow

The project provides two main pipelines:

#### **Full Pipeline** (`src/full_sketch2cad_pipeline.py`) - Recommended
Complete automation from raw image to technical drawing with SAM2 preprocessing:

```mermaid
graph TD
    A[Raw Image Input] --> B[SAM2 Preprocessing]
    B --> C[Background Removal]
    C --> D[Image Enhancement]
    D --> E[Thinning]
    E --> F[DeepLSD Line Detection]
    F --> G[Line Classification]
    G --> H[Shape Detection]
    H --> I[Rectangle Detection]
    H --> J[Circle Detection]
    I --> K[Visualization]
    J --> K
    K --> L[JSON Export]
    K --> M[PNG Export]
    K --> N[PDF Export]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
    style N fill:#9f9,stroke:#333,stroke-width:2px
```

#### 1. **Image Preprocessing**
- Load and convert to grayscale
- Fix image polarity (dark/light adjustment)
- Contour-based filtering to remove noise
- Connected components analysis

#### 2. **Thinning**
- Apply Zhang-Suen thinning algorithm
- Convert lines to 1-pixel width
- Preserve connectivity

#### 3. **Line Detection** (DeepLSD)
- Load pre-trained DeepLSD model
- Detect line segments with confidence scoring
- Extract geometric properties (length, angle)

#### 4. **Line Classification**
- Classify lines into categories:
  - **Main lines**: Long horizontal/vertical structural lines
  - **Dimension lines**: Medium-length measurement lines
  - **Tick marks**: Short diagonal marks at 45°/135°
  - **Other**: Unclassified geometric elements
  - **Noise**: Very short segments

#### 5. **Shape Detection**
- **Rectangles**: Filter and combine main H/V lines
- **Circles**: Detect from skeletonized contours using circularity

#### 6. **Output Generation**
- JSON with all detected elements and metadata
- PNG visualization with color-coded elements
- PDF technical drawing (placeholder implementation)

### 🤖 Alternative Line Detection Methods

The project includes TODO stubs for alternative detection methods:

#### **PaddleOCR** (`src/alternative_detectors.py`)
- Extract text and geometric structures
- Useful for dimension annotations
- Multi-language support

#### **YOLOv8-seg**
- Semantic segmentation for geometric primitives
- Deep learning-based object detection
- Customizable for specific sketch types

#### **ScanLSD**
- Alternative line segment detector
- Different algorithmic approach
- Complementary to DeepLSD

#### **Hybrid Detector**
- Combine multiple methods
- Voting mechanism for robustness
- Confidence-weighted fusion

### 🌐 Web Interface (TODO)

Streamlit-based web interface (`streamlit_app/app.py`):

#### Features to Implement:
- **File Upload**: Drag-and-drop sketch upload
- **Configuration Panel**: Adjust detection parameters
- **Real-time Preview**: Live parameter adjustment
- **Results Display**: Interactive visualization
- **Download Options**: JSON, PNG, PDF exports
- **Batch Processing**: Multiple image support
- **Detector Comparison**: Side-by-side results

### 📊 Current Capabilities

#### ✅ **Working Features**
- DeepLSD line detection with pre-trained model
- Image preprocessing and thinning
- Line classification by geometry
- Rectangle and circle detection
- JSON and PNG output generation
- Mirror sketch processing (tested use case)

#### 🚧 **TODO Features**
- PDF technical drawing generation
- PaddleOCR integration
- YOLOv8-seg implementation
- ScanLSD alternative
- Streamlit web interface
- Hybrid detection methods
- Batch processing
- Parameter optimization
- Additional shape detection (arcs, ellipses)
- Dimension text extraction
- CAD model export (DXF/DWG)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/tomaszbielNCI/Intelligent_Sketch2CAD.git
cd Intelligent_Sketch2CAD

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download SAM2 model (place in project root)
# - sam2_hiera_small.pt
# - sam2_hiera_s.yaml

# DeepLSD model should be at: DeepLSD/weights/deeplsd_md.tar
```

### Run Pipeline

```bash
# Process default image (input_data/raw_sketches/WhatsApp Image 2026-04-24 at 21.41.48.jpeg)
python src/full_sketch2cad_pipeline.py

# Process specific image
python src/full_sketch2cad_pipeline.py --image "path/to/your/sketch.jpg"

# Test without saving files
python src/full_sketch2cad_pipeline.py --no-save
```

## 📁 Input / Output

| Type | Location | Format |
|------|----------|--------|
| Input sketches | `input_data/raw_sketches/` | JPG, PNG, JPEG |
| Output JSON | `output_data/full_pipeline_*.json` | Detection data |
| Output PNG | `output_data/technical_drawing_*.png` | Color-coded visualization |
| Output PDF | `output_data/technical_drawing_*.pdf` | Professional technical drawing |
| Output DXF | `output_data/sketch_*.dxf` | CAD format (FreeCAD compatible) |

## 🖼️ Technical Drawing Output

The pipeline generates a professional technical drawing fully compatible with **FreeCAD**:
- Main rectangle in real dimensions (calibrated from sketch)
- Mounting holes with crosshairs
- Dimension lines (width, height, hole spacing, edge offsets)
- Hole diameter annotation
- Title block (glass type, thickness, project info, date)

**FreeCAD compatibility:**
- **DXF export** – can be opened directly in FreeCAD's Draft Workbench
- **PDF output** – ready for printing or sharing with clients
- **JSON data** – can be used to generate parametric FreeCAD models via Python script

**Calibration**: Based on known dimensions (default: 857×660 mm) – can be modified in code.

## 🏗️ Project Structure (Active Files Only)

```
Intelligent_Sketch2CAD/
├── src/
│   └── full_sketch2cad_pipeline.py   # MAIN PIPELINE (working)
├── input_data/raw_sketches/          # Place sketches here
├── output_data/                      # Results (JSON, PNG, PDF, DXF)
├── intermediate_data/                # Temporary files
├── DeepLSD/                          # Line detection model
├── notebooks/                        # Development experiments
└── requirements.txt                  # Dependencies
```

## 🧪 Tested On

Hand-drawn mirror sketch for glass construction (bathroom mirror)
- Dimensions: 857×660 mm
- 4 mounting holes (⌀36 mm)
- White background, black lines
- Successfully imported to FreeCAD for validation

## 🛠️ Use with FreeCAD

After running the pipeline, you can:

**Open DXF directly in FreeCAD:**
1. Launch FreeCAD
2. File → Open → Select `sketch_*.dxf` from `output_data/`
3. Switch to Draft Workbench for editing

**Generate parametric model from JSON:**
- Use included JSON data to create a scripted FreeCAD model
- Modify dimensions programmatically

**Print PDF for client approval:**
- The PDF contains all dimensions and specifications
- Ready for glass order submission

## ⚠️ Current Limitations

- Requires good contrast sketch (white background, dark lines)
- SAM2 model files must be downloaded manually
- Real dimensions are hardcoded (857×660 mm) – can be changed in `full_sketch2cad_pipeline.py`
- Works best for rectangular glass elements (mirrors, window panes, door glass) with circular mounting holes

## 🔬 Technologies

| Component | Technology |
|-----------|------------|
| Background removal | SAM2 (Meta) |
| Line detection | DeepLSD |
| Image processing | OpenCV |
| Thinning | Zhang-Suen algorithm |
| Technical drawing | Matplotlib |
| DXF export | ezdxf |
| CAD compatibility | FreeCAD (DXF import) |

## 📚 Academic Context

This project was developed for **Intelligent Agents and Process Automation** module at National College of Ireland.

Automation types demonstrated:
- **AI/Agentic automation**: DeepLSD + SAM2 for intelligent line detection
- **RPA automation**: Watchdog script for folder monitoring (separate file)

## 📄 License

MIT License – see LICENSE file for details.

## 👤 Author

Tomasz Biel – MSc in Artificial Intelligence

**Status**: ✅ Vertical slice complete – from raw sketch to professional technical drawing (PDF/DXF), fully compatible with **FreeCAD** – ready for glass fitting orders
