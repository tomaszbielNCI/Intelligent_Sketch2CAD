#!/usr/bin/env python3
"""
Streamlit UI for Intelligent Sketch to CAD
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import SketchToCADPipeline


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Intelligent Sketch to CAD",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ Intelligent Sketch to CAD")
    st.markdown("Convert hand-drawn sketches with dimensions into parametric CAD models")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Configuration files
        config_path = st.text_input(
            "Config file",
            value="config/config.yaml",
            help="Path to configuration file"
        )
        
        templates_path = st.text_input(
            "Templates file", 
            value="config/templates.yaml",
            help="Path to templates file"
        )
        
        # Processing options
        st.subheader("Processing Options")
        output_format = st.selectbox(
            "Output format",
            ["fcstd", "dxf"],
            help="CAD file format"
        )
        
        verbose_logging = st.checkbox("Verbose logging", value=False)
        
        # Initialize pipeline
        try:
            pipeline = None
            if st.button("Initialize Pipeline"):
                with st.spinner("Initializing pipeline..."):
                    pipeline = SketchToCADPipeline(config_path, templates_path)
                st.success("Pipeline initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            pipeline = None
    
    # Main content area
    if pipeline:
        # File upload
        st.header("Upload Sketch")
        uploaded_file = st.file_uploader(
            "Choose a sketch image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a hand-drawn sketch with dimensions"
        )
        
        if uploaded_file:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Original Sketch")
                st.image(uploaded_file, use_column_width=True)
            
            # Process button
            if st.button("🚀 Process Sketch", type="primary"):
                with st.spinner("Processing sketch..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            shutil.copyfileobj(uploaded_file, tmp_file)
                            tmp_path = tmp_file.name
                        
                        # Process sketch
                        result = pipeline.process_sketch(tmp_path)
                        
                        # Display results
                        with col2:
                            st.subheader("Analysis Results")
                            
                            # Template info
                            st.info(f"📋 Template used: `{result['template_used']}`")
                            
                            # Parameters
                            st.subheader("Extracted Parameters")
                            params = result['parameters']
                            for key, value in params.items():
                                st.metric(key.replace('_', ' ').title(), f"{value} mm")
                            
                            # Processing info
                            st.subheader("Processing Info")
                            info = result['processing_info']
                            col2_1, col2_2, col2_3 = st.columns(3)
                            with col2_1:
                                st.metric("Shapes Found", info['shapes_found'])
                            with col2_2:
                                st.metric("Dimensions Found", info['dimensions_found'])
                            with col2_3:
                                st.metric("Export Success", "✅" if info['export_success'] else "❌")
                        
                        # Download section
                        st.header("Download Results")
                        
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            if result['cad_file'] and Path(result['cad_file']).exists():
                                with open(result['cad_file'], 'rb') as f:
                                    st.download_button(
                                        label="📥 Download CAD File",
                                        data=f.read(),
                                        file_name=Path(result['cad_file']).name,
                                        help="Download the generated CAD file"
                                    )
                        
                        with col4:
                            if result['analysis_file'] and Path(result['analysis_file']).exists():
                                with open(result['analysis_file'], 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Analysis",
                                        data=f.read(),
                                        file_name=Path(result['analysis_file']).name,
                                        help="Download the analysis results (JSON)"
                                    )
                        
                        # Cleanup
                        Path(tmp_path).unlink(missing_ok=True)
                        
                    except Exception as e:
                        st.error(f"Error processing sketch: {e}")
        
        # Batch processing section
        st.header("Batch Processing")
        st.info("Upload multiple sketch files for batch processing")
        
        uploaded_files = st.file_uploader(
            "Choose sketch images",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple sketches for batch processing"
        )
        
        if uploaded_files and st.button("🔄 Process Batch"):
            if len(uploaded_files) > 5:
                st.warning("Processing more than 5 files at once may take a while...")
            
            progress_bar = st.progress(0)
            results_container = st.container()
            
            batch_results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Save temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            shutil.copyfileobj(uploaded_file, tmp_file)
                            tmp_path = tmp_file.name
                        
                        # Process
                        result = pipeline.process_sketch(tmp_path)
                        batch_results.append({
                            'file': uploaded_file.name,
                            'success': True,
                            'result': result
                        })
                        
                        # Cleanup
                        Path(tmp_path).unlink(missing_ok=True)
                        
                    except Exception as e:
                        batch_results.append({
                            'file': uploaded_file.name,
                            'success': False,
                            'error': str(e)
                        })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Display batch results
            with results_container:
                st.subheader("Batch Processing Results")
                
                successful = [r for r in batch_results if r['success']]
                failed = [r for r in batch_results if not r['success']]
                
                col_success, col_failed = st.columns(2)
                
                with col_success:
                    st.success(f"✅ Successfully processed: {len(successful)} files")
                    if successful:
                        for result in successful[:5]:  # Show first 5
                            st.text(f"• {result['file']} → {result['result']['template_used']}")
                
                with col_failed:
                    st.error(f"❌ Failed: {len(failed)} files")
                    if failed:
                        for result in failed:
                            st.text(f"• {result['file']}: {result['error']}")
    
    else:
        st.warning("⚠️ Please initialize the pipeline first from the sidebar")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **About**: This application converts hand-drawn sketches with dimensions into parametric CAD models 
        using computer vision, OCR, and CAD generation technologies.
        
        **GitHub**: [Intelligent_Sketch2CAD](https://github.com/tomaszbielNCI/Intelligent_Sketch2CAD)
        """
    )


if __name__ == "__main__":
    main()
