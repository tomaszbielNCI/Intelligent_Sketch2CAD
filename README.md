# Intelligent Sketch to CAD

AI-assisted pipeline that converts hand-drawn sketches with dimensions into parametric CAD models using FreeCAD.

## Overview

This project helps small construction/fit-out companies automate the process of creating technical drawings from hand sketches. It analyzes sketches, extracts shapes and dimensions, and generates parametric CAD templates that can be quickly reviewed and adjusted by humans.

## Features

- **Sketch Analysis**: Computer vision and OCR to extract shapes and dimensions from hand-drawn sketches
- **Template Matching**: Automatically selects appropriate CAD templates based on detected shapes
- **Parametric Generation**: Creates parametric FreeCAD models with extracted dimensions
- **Batch Processing**: Process multiple sketches automatically
- **Configurable Templates**: Easy to extend with new CAD templates

## Installation

### Prerequisites

- Python 3.8+
- FreeCAD (optional - for CAD generation)
- Tesseract OCR

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tomaszbielNCI/Intelligent_Sketch2CAD.git
cd Intelligent_Sketch2CAD
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
- **Windows**: Download from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

4. Install FreeCAD (optional):
- **Windows**: Download from [FreeCAD website](https://www.freecadweb.org/downloads.php)
- **macOS**: `brew install freecad`
- **Linux**: `sudo apt-get install freecad`

## Usage

### Basic Usage

Process a single sketch:
```bash
python main.py --input data/raw/sketches/sketch1.jpg
```

### Advanced Usage

Batch process multiple sketches:
```bash
python main.py --input data/raw/sketches/ --batch
```

Specify output directory:
```bash
python main.py --input sketch.jpg --output my_output/
```

Use custom configuration:
```bash
python main.py --input sketch.jpg --config my_config.yaml
```

Enable verbose logging:
```bash
python main.py --input sketch.jpg --verbose
```

## Project Structure

```
sketch_to_cad/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── config/                   # Configuration files
│   ├── __init__.py
│   ├── config.yaml           # Main configuration
│   └── templates.yaml        # CAD templates definition
├── src/                      # Source code
│   ├── __init__.py
│   ├── pipeline.py           # Main processing pipeline
│   ├── sketch_analyzer.py    # Sketch analysis (CV/OCR)
│   ├── cad_generator.py      # CAD generation (FreeCAD)
│   └── utils.py              # Utility functions
├── data/                     # Data directories
│   ├── raw/sketches/         # Input sketches
│   ├── processed/            # Processed images
│   └── labels/               # Manual labels
├── models/                   # ML models
│   └── checkpoints/          # Pre-trained models
├── templates/                # CAD templates
│   ├── cad_templates.fcstd   # FreeCAD templates
│   └── dxf_examples/         # DXF examples
├── output/                   # Generated files
│   ├── cad_files/            # CAD exports
│   └── logs/                 # Log files
├── tests/                    # Test suite
│   └── test_pipeline.py      # Pipeline tests
├── docs/                     # Documentation
│   └── diagram.mmd           # Architecture diagram
└── main.py                   # Entry point
```

## Configuration

### Main Configuration (config/config.yaml)

Key settings:
- Image processing parameters
- OCR configuration
- Shape detection thresholds
- CAD generation settings
- Logging configuration

### Templates Configuration (config/templates.yaml)

Define:
- CAD templates with default parameters
- Shape-to-template mapping rules
- Dimension extraction patterns

## Supported Templates

- **Mirror**: Rectangular mirrors with frames
- **Frame**: Picture frames with inner/outer dimensions
- **Panel**: Wall panels with customizable dimensions

## Adding New Templates

1. Define template in `config/templates.yaml`
2. Add generation logic in `src/cad_generator.py`
3. Update shape mapping if needed

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

Follow PEP 8 guidelines. Use type hints where possible.

## Troubleshooting

### FreeCAD Issues

If FreeCAD is not available, the system will generate mock data for testing purposes.

### OCR Issues

- Ensure Tesseract is properly installed
- Check image quality and resolution
- Adjust OCR configuration in config.yaml

### Performance Issues

- Reduce image size in config.yaml
- Adjust contour detection parameters
- Use batch processing for multiple files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue on GitHub.
