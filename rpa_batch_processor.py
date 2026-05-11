#!/usr/bin/env python3
"""
RPA-based Batch Processor for Intelligent Sketch2CAD
====================================================

This script implements RPA-style automation by processing multiple sketch images
through the Sketch2CAD pipeline automatically. It demonstrates process automation
without human intervention, suitable for the NCI H9IAPA assignment requirements.

Author: Intelligent_Sketch2CAD Team
Date: 2026-05-11
"""

import os
import sys
import glob
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.full_sketch2cad_pipeline import FullSketch2CADPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rpa_batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RPABatchProcessor:
    """
    RPA-style batch processor for Sketch2CAD pipeline.
    Automates processing of multiple images without human intervention.
    """
    
    def __init__(self, input_dir: str = None, project_dir: str = None):
        """
        Initialize RPA batch processor.
        
        Args:
            input_dir: Directory containing sketch images
            project_dir: Project directory path
        """
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.input_dir = Path(input_dir) if input_dir else self.project_dir / "input_data" / "raw_sketches"
        
        # Initialize pipeline
        self.pipeline = FullSketch2CADPipeline(project_dir=self.project_dir)
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None,
            'failed_files': []
        }
        
        logger.info(f"RPA Batch Processor initialized")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Project directory: {self.project_dir}")
    
    def discover_images(self) -> List[str]:
        """
        Discover all image files in input directory.
        
        Returns:
            List of image file paths
        """
        logger.info("Discovering image files...")
        
        # Supported image formats
        patterns = ["*.jpeg", "*.jpg", "*.png", "*.bmp", "*.tiff", "*.tif"]
        image_files = []
        
        for pattern in patterns:
            files = glob.glob(str(self.input_dir / pattern))
            image_files.extend(files)
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        self.stats['total_files'] = len(image_files)
        logger.info(f"Found {len(image_files)} image files")
        
        return image_files
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image through the pipeline.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Processing results dictionary
        """
        logger.info(f"Processing: {Path(image_path).name}")
        
        try:
            # Run pipeline
            results = self.pipeline.run_full_pipeline(
                image_path=image_path,
                save_results=True
            )
            
            # Extract key metrics
            metrics = {
                'file': Path(image_path).name,
                'status': 'success',
                'lines_detected': results['statistics']['deeplsd_raw'],
                'rectangles': len(results['rectangles']),
                'circles': len(results['circles']),
                'processing_time': results.get('processing_time', 'N/A'),
                'output_files': results.get('output_files', {})
            }
            
            logger.info(f"✓ Success: {metrics['lines_detected']} lines, {metrics['rectangles']} rectangles")
            return metrics
            
        except Exception as e:
            error_info = {
                'file': Path(image_path).name,
                'status': 'failed',
                'error': str(e),
                'lines_detected': 0,
                'rectangles': 0,
                'circles': 0
            }
            
            logger.error(f"✗ Failed: {e}")
            self.stats['failed_files'].append({'file': image_path, 'error': str(e)})
            return error_info
    
    def run_batch_processing(self) -> Dict[str, Any]:
        """
        Run automated batch processing of all discovered images.
        
        Returns:
            Complete processing results and statistics
        """
        logger.info("="*60)
        logger.info("STARTING RPA BATCH PROCESSING")
        logger.info("="*60)
        
        self.stats['start_time'] = datetime.now()
        
        # Discover images
        image_files = self.discover_images()
        
        if not image_files:
            logger.warning("No image files found. Nothing to process.")
            return self.stats
        
        # Process each image
        all_results = []
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing file {i}/{len(image_files)}")
            
            # Process image
            result = self.process_single_image(image_path)
            all_results.append(result)
            
            # Update statistics
            if result['status'] == 'success':
                self.stats['processed'] += 1
            else:
                self.stats['failed'] += 1
            
            # Small delay to prevent overwhelming
            time.sleep(0.5)
        
        self.stats['end_time'] = datetime.now()
        self.stats['duration'] = self.stats['end_time'] - self.stats['start_time']
        self.stats['all_results'] = all_results
        
        # Print summary
        self.print_summary()
        
        # Save batch report
        self.save_batch_report()
        
        return self.stats
    
    def print_summary(self):
        """Print processing summary."""
        logger.info("="*60)
        logger.info("RPA BATCH PROCESSING COMPLETED")
        logger.info("="*60)
        logger.info(f"Total files: {self.stats['total_files']}")
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Duration: {self.stats['duration']}")
        
        if self.stats['failed_files']:
            logger.warning("Failed files:")
            for failed in self.stats['failed_files']:
                logger.warning(f"  - {Path(failed['file']).name}: {failed['error']}")
    
    def save_batch_report(self):
        """Save detailed batch processing report."""
        report_path = self.project_dir / "output_data" / f"rpa_batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        
        # Prepare report data
        report_data = {
            'batch_info': {
                'start_time': self.stats['start_time'].isoformat(),
                'end_time': self.stats['end_time'].isoformat(),
                'duration': str(self.stats['duration']),
                'total_files': self.stats['total_files'],
                'processed': self.stats['processed'],
                'failed': self.stats['failed'],
                'success_rate': f"{(self.stats['processed'] / self.stats['total_files'] * 100):.1f}%" if self.stats['total_files'] > 0 else "0%"
            },
            'detailed_results': self.stats['all_results'],
            'failed_files': self.stats['failed_files']
        }
        
        # Save report
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"Batch report saved: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save batch report: {e}")


def main():
    """Main function for RPA batch processing."""
    parser = argparse.ArgumentParser(description="RPA Batch Processor for Sketch2CAD")
    parser.add_argument("--input-dir", type=str, help="Input directory with sketch images")
    parser.add_argument("--project-dir", type=str, help="Project directory path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize RPA processor
    processor = RPABatchProcessor(
        input_dir=args.input_dir,
        project_dir=args.project_dir
    )
    
    # Run batch processing
    try:
        results = processor.run_batch_processing()
        
        print("\n" + "="*60)
        print("RPA BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"✓ Processed: {results['processed']}/{results['total_files']} files")
        print(f"✗ Failed: {results['failed']} files")
        print(f"⏱️  Duration: {results['duration']}")
        print(f"📊 Success Rate: {(results['processed'] / results['total_files'] * 100):.1f}%" if results['total_files'] > 0 else "0%")
        
        return 0 if results['failed'] == 0 else 1
        
    except Exception as e:
        logger.error(f"RPA batch processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
