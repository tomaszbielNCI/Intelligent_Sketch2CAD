"""Test suite for the Sketch to CAD pipeline."""

import pytest
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np

from src.pipeline import SketchToCADPipeline


class TestSketchToCADPipeline:
    """Test cases for the main pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample test image."""
        # Create a simple rectangle image
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        img[:] = (255, 255, 255)  # White background
        
        # Draw a rectangle
        cv2.rectangle(img, (100, 100), (500, 300), (0, 0, 0), 3)
        
        # Add some text
        cv2.putText(img, "600mm", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "200mm", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        image_path = Path(temp_dir) / "test_sketch.jpg"
        cv2.imwrite(str(image_path), img)
        
        return str(image_path)
    
    @pytest.fixture
    def pipeline(self, temp_dir):
        """Create pipeline instance for testing."""
        # Create minimal config for testing
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        
        config_content = """
project:
  name: "Test"
paths:
  input_dir: "data/raw/sketches"
  output_dir: "output/cad_files"
  log_dir: "output/logs"
image:
  max_size: [1024, 1024]
ocr:
  tesseract_config: "--psm 6 --oem 3"
shape_detection:
  min_contour_area: 100
cad:
  default_units: "mm"
logging:
  level: "INFO"
"""
        
        templates_content = """
templates:
  mirror:
    default_parameters:
      width: 600.0
      height: 200.0
shape_mapping:
  rectangle:
    - mirror
"""
        
        config_file = config_dir / "config.yaml"
        templates_file = config_dir / "templates.yaml"
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        with open(templates_file, 'w') as f:
            f.write(templates_content)
        
        return SketchToCADPipeline(str(config_file), str(templates_file))
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config is not None
        assert pipeline.templates is not None
        assert pipeline.sketch_analyzer is not None
        assert pipeline.cad_generator is not None
    
    def test_process_sketch(self, pipeline, sample_image, temp_dir):
        """Test processing a single sketch."""
        output_dir = Path(temp_dir) / "output"
        
        result = pipeline.process_sketch(sample_image, str(output_dir))
        
        assert 'template_used' in result
        assert 'parameters' in result
        assert 'processing_info' in result
        assert Path(result['analysis_file']).exists()
        
        # Check that analysis contains expected data
        assert result['processing_info']['shapes_found'] >= 0
        assert result['processing_info']['dimensions_found'] >= 0
    
    def test_invalid_input_file(self, pipeline):
        """Test handling of invalid input file."""
        with pytest.raises(FileNotFoundError):
            pipeline.process_sketch("nonexistent.jpg")
    
    def test_invalid_image_format(self, pipeline, temp_dir):
        """Test handling of invalid image format."""
        # Create a text file instead of image
        text_file = Path(temp_dir) / "test.txt"
        text_file.write_text("not an image")
        
        with pytest.raises(ValueError):
            pipeline.process_sketch(str(text_file))
    
    def test_batch_processing(self, pipeline, temp_dir):
        """Test batch processing of multiple images."""
        # Create multiple test images
        input_dir = Path(temp_dir) / "input"
        input_dir.mkdir()
        
        for i in range(3):
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            img[:] = (255, 255, 255)
            cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 2)
            
            image_path = input_dir / f"test_{i}.jpg"
            cv2.imwrite(str(image_path), img)
        
        output_dir = Path(temp_dir) / "batch_output"
        results = pipeline.batch_process(str(input_dir), str(output_dir))
        
        assert 'processed' in results
        assert 'failed' in results
        assert len(results['processed']) == 3
        assert len(results['failed']) == 0
    
    def test_batch_processing_empty_dir(self, pipeline, temp_dir):
        """Test batch processing with empty directory."""
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()
        
        results = pipeline.batch_process(str(empty_dir))
        
        assert len(results['processed']) == 0
        assert len(results['failed']) == 0


class TestSketchAnalyzer:
    """Test cases for sketch analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        config = {
            'image': {'max_size': [1024, 1024]},
            'ocr': {'tesseract_config': '--psm 6 --oem 3'},
            'shape_detection': {'min_contour_area': 100}
        }
        from src.sketch_analyzer import SketchAnalyzer
        return SketchAnalyzer(config)
    
    def test_analyze_simple_rectangle(self, analyzer, temp_dir):
        """Test analyzing a simple rectangle."""
        # Create test image
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:] = (255, 255, 255)
        cv2.rectangle(img, (50, 50), (350, 250), (0, 0, 0), 3)
        
        image_path = Path(temp_dir) / "rectangle.jpg"
        cv2.imwrite(str(image_path), img)
        
        analysis = analyzer.analyze_sketch(str(image_path))
        
        assert 'shapes' in analysis
        assert 'dimensions' in analysis
        assert 'image_info' in analysis
        assert len(analysis['shapes']) >= 1
        assert analysis['shapes'][0]['type'] in ['rectangle', 'square']


if __name__ == '__main__':
    pytest.main([__file__])
