"""Main pipeline module for Intelligent Sketch to CAD processing."""

from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from .sketch_analyzer import SketchAnalyzer
from .cad_generator import CADGenerator
from .utils import setup_logging, save_intermediate_result


class SketchToCADPipeline:
    """Main pipeline for converting sketches to CAD models."""
    
    def __init__(self, config_path: str = "config/config.yaml", templates_path: str = "config/templates.yaml"):
        """Initialize the pipeline with configuration."""
        # Load configuration
        from config import load_config, load_templates
        self.config = load_config(config_path)
        self.templates = load_templates(templates_path)
        
        # Setup logging
        self.logger = setup_logging(
            log_dir=self.config['paths']['log_dir'],
            log_level=self.config['logging']['level']
        )
        
        # Initialize components
        self.sketch_analyzer = SketchAnalyzer(self.config)
        self.cad_generator = CADGenerator(self.config, self.templates)
        
        self.logger.info("Sketch to CAD Pipeline initialized")
    
    def process_sketch(self, input_path: str, output_dir: Optional[str] = None) -> Dict:
        """Process a sketch image and generate CAD model."""
        self.logger.info(f"Processing sketch: {input_path}")
        
        # Validate input
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if not input_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            raise ValueError(f"Unsupported image format: {input_file.suffix}")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.config['paths']['output_dir']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Analyze sketch
        self.logger.info("Step 1: Analyzing sketch...")
        analysis = self.sketch_analyzer.analyze_sketch(str(input_file))
        
        # Save analysis results
        analysis_path = output_path / f"{input_file.stem}_analysis.json"
        self.sketch_analyzer.save_analysis(analysis, str(analysis_path))
        
        # Step 2: Generate CAD model
        self.logger.info("Step 2: Generating CAD model...")
        cad_result = self.cad_generator.generate_cad_model(analysis)
        
        # Step 3: Export CAD file
        self.logger.info("Step 3: Exporting CAD file...")
        cad_filename = f"{input_file.stem}_model.fcstd"
        cad_path = output_path / cad_filename
        export_success = self.cad_generator.export_model(str(cad_path))
        
        # Step 4: Save pipeline results
        pipeline_result = {
            'input_file': str(input_file),
            'analysis_file': str(analysis_path),
            'cad_file': str(cad_path) if export_success else None,
            'template_used': cad_result['template_used'],
            'parameters': cad_result['parameters'],
            'processing_info': {
                'shapes_found': len(analysis['shapes']),
                'dimensions_found': len(analysis['dimensions']),
                'export_success': export_success
            }
        }
        
        # Save pipeline summary
        summary_path = output_path / f"{input_file.stem}_pipeline.json"
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Processing completed. Results saved to: {output_path}")
        
        return pipeline_result
    
    def batch_process(self, input_dir: str, output_dir: Optional[str] = None) -> Dict:
        """Process multiple sketches in a directory."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            self.logger.warning(f"No image files found in: {input_dir}")
            return {'processed': [], 'failed': []}
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        results = {'processed': [], 'failed': []}
        
        for i, image_file in enumerate(image_files, 1):
            self.logger.info(f"Processing file {i}/{len(image_files)}: {image_file.name}")
            
            try:
                result = self.process_sketch(str(image_file), output_dir)
                results['processed'].append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {image_file.name}: {e}")
                results['failed'].append({
                    'file': str(image_file),
                    'error': str(e)
                })
        
        self.logger.info(f"Batch processing completed. Processed: {len(results['processed'])}, Failed: {len(results['failed'])}")
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.cad_generator, 'close_document'):
            self.cad_generator.close_document()
        self.logger.info("Pipeline cleanup completed")
