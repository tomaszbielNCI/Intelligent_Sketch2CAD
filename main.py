#!/usr/bin/env python3
"""
Intelligent Sketch to CAD - Main Entry Point

This script converts hand-drawn sketches with dimensions into parametric CAD models.
Usage: python main.py --input data/raw/sketches/sketch1.jpg
"""

import click
import sys
from pathlib import Path

from src.pipeline import SketchToCADPipeline


@click.command()
@click.option('--input', '-i', required=True, help='Input sketch file or directory')
@click.option('--output', '-o', help='Output directory for CAD files')
@click.option('--batch', '-b', is_flag=True, help='Process all images in directory')
@click.option('--config', '-c', default='config/config.yaml', help='Configuration file path')
@click.option('--templates', '-t', default='config/templates.yaml', help='Templates file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input, output, batch, config, templates, verbose):
    """
    Intelligent Sketch to CAD Converter
    
    Convert hand-drawn sketches with dimensions into parametric CAD models.
    """
    try:
        # Initialize pipeline
        pipeline = SketchToCADPipeline(config, templates)
        
        if verbose:
            click.echo(f"Input: {input}")
            click.echo(f"Output: {output or 'default'}")
            click.echo(f"Batch mode: {batch}")
        
        # Process input
        if batch:
            # Batch processing
            results = pipeline.batch_process(input, output)
            
            click.echo(f"\nProcessing completed:")
            click.echo(f"✓ Processed: {len(results['processed'])} files")
            click.echo(f"✗ Failed: {len(results['failed'])} files")
            
            if results['failed']:
                click.echo("\nFailed files:")
                for failed in results['failed']:
                    click.echo(f"  - {failed['file']}: {failed['error']}")
        else:
            # Single file processing
            result = pipeline.process_sketch(input, output)
            
            click.echo(f"\nProcessing completed successfully!")
            click.echo(f"Template used: {result['template_used']}")
            click.echo(f"Parameters: {result['parameters']}")
            click.echo(f"CAD file: {result['cad_file']}")
            click.echo(f"Analysis file: {result['analysis_file']}")
        
        # Cleanup
        pipeline.cleanup()
        
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
