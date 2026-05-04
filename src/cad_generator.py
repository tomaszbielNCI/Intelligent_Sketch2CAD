"""CAD generation module for creating parametric models from sketch analysis."""

from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger

try:
    import FreeCAD
    import Part
    import Draft
    import Mesh
except ImportError:
    logger.warning("FreeCAD not available. CAD generation will be limited.")

try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    logger.warning("CadQuery not available. Using FreeCAD only.")
    CADQUERY_AVAILABLE = False


class CADGenerator:
    """Generate CAD models from sketch analysis results."""
    
    def __init__(self, config: Dict, templates: Dict):
        self.config = config
        self.templates = templates
        self.cad_config = config.get('cad', {})
        self.logger = logger
        
        # Initialize FreeCAD document
        try:
            self.doc = FreeCAD.newDocument("Sketch2CAD")
        except:
            self.doc = None
            self.logger.warning("FreeCAD not available, using mock generation")
    
    def generate_cad_model(self, analysis: Dict, template_name: Optional[str] = None) -> Dict:
        """Generate CAD model from sketch analysis."""
        shapes = analysis.get('shapes', [])
        dimensions = analysis.get('dimensions', [])
        
        # Select appropriate template
        if not template_name:
            template_name = self._select_template(shapes, dimensions)
        
        # Get template configuration
        template_config = self.templates.get('templates', {}).get(template_name, {})
        default_params = template_config.get('default_parameters', {})
        
        # Extract parameters from analysis
        extracted_params = self._extract_parameters(shapes, dimensions)
        
        # Merge default and extracted parameters
        parameters = {**default_params, **extracted_params}
        
        # Generate CAD model
        if self.doc:
            model_data = self._generate_freecad_model(template_name, parameters)
        else:
            model_data = self._generate_mock_model(template_name, parameters)
        
        return {
            'template_used': template_name,
            'parameters': parameters,
            'model_data': model_data,
            'output_format': 'fcstd'
        }
    
    def _select_template(self, shapes: List[Dict], dimensions: List[Dict]) -> str:
        """Select appropriate template based on shapes and dimensions."""
        shape_mapping = self.templates.get('shape_mapping', {})
        
        # Get shape types from analysis
        shape_types = [shape['type'] for shape in shapes]
        
        # Find matching templates
        possible_templates = []
        for shape_type in shape_types:
            if shape_type in shape_mapping:
                possible_templates.extend(shape_mapping[shape_type])
        
        # Remove duplicates and return first match
        if possible_templates:
            return list(set(possible_templates))[0]
        
        # Default to mirror template
        return 'mirror'
    
    def _extract_parameters(self, shapes: List[Dict], dimensions: List[Dict]) -> Dict:
        """Extract CAD parameters from shapes and dimensions."""
        params = {}
        
        # Extract dimensions from text
        for dim in dimensions:
            value = dim['value']
            unit = dim['unit']
            
            # Convert to mm if needed
            if unit == 'cm':
                value_mm = value * 10
            elif unit == 'm':
                value_mm = value * 1000
            else:  # mm
                value_mm = value
            
            # Try to map to parameter names
            dim_rules = self.templates.get('dimension_rules', {})
            
            # Simple heuristic: largest dimension is width/height
            if 'width' not in params:
                params['width'] = value_mm
            elif 'height' not in params:
                params['height'] = value_mm
            elif 'thickness' not in params:
                params['thickness'] = value_mm
        
        # Extract parameters from shapes
        if shapes:
            main_shape = max(shapes, key=lambda s: s['area'])
            bbox = main_shape['bounding_box']
            
            # Use bounding box if no text dimensions found
            if 'width' not in params:
                params['width'] = float(bbox['width'])
            if 'height' not in params:
                params['height'] = float(bbox['height'])
        
        return params
    
    def _generate_freecad_model(self, template_name: str, parameters: Dict) -> Dict:
        """Generate actual FreeCAD model."""
        if not self.doc:
            return {}
        
        self.logger.info(f"Generating FreeCAD model with template: {template_name}")
        
        # Clear document
        for obj in self.doc.Objects:
            self.doc.removeObject(obj.Name)
        
        # Generate model based on template
        if template_name == 'mirror':
            return self._create_mirror_template(parameters)
        elif template_name == 'frame':
            return self._create_frame_template(parameters)
        elif template_name == 'panel':
            return self._create_panel_template(parameters)
        else:
            return self._create_generic_rectangle(parameters)
    
    def _create_mirror_template(self, params: Dict) -> Dict:
        """Create mirror template with frame."""
        width = params.get('width', 600.0)
        height = params.get('height', 400.0)
        frame_width = params.get('frame_width', 20.0)
        thickness = params.get('thickness', 5.0)
        
        # Create mirror glass
        glass = Part.makeBox(width, height, thickness)
        glass_obj = self.doc.addObject("Part::Feature", "Glass")
        glass_obj.Shape = glass
        
        # Create frame (outer box)
        outer = Part.makeBox(width + 2*frame_width, height + 2*frame_width, thickness + 10.0)
        outer_obj = self.doc.addObject("Part::Feature", "OuterFrame")
        outer_obj.Shape = outer
        
        # Create inner cutout
        inner = Part.makeBox(width, height, thickness + 10.0)
        inner_obj = self.doc.addObject("Part::Feature", "InnerCutout")
        inner_obj.Shape = inner
        
        # Create frame by cutting
        frame = outer.cut(inner)
        frame_obj = self.doc.addObject("Part::Feature", "Frame")
        frame_obj.Shape = frame
        
        # Position glass in frame
        glass_obj.Placement.Base.x = frame_width
        glass_obj.Placement.Base.y = frame_width
        glass_obj.Placement.Base.z = 5.0
        
        return {
            'objects': ['Glass', 'Frame'],
            'dimensions': {'width': width, 'height': height, 'frame_width': frame_width}
        }
    
    def _create_frame_template(self, params: Dict) -> Dict:
        """Create picture frame template."""
        outer_width = params.get('outer_width', 500.0)
        outer_height = params.get('outer_height', 700.0)
        inner_width = params.get('inner_width', 450.0)
        inner_height = params.get('inner_height', 650.0)
        depth = params.get('depth', 25.0)
        
        # Create outer frame
        outer = Part.makeBox(outer_width, outer_height, depth)
        outer_obj = self.doc.addObject("Part::Feature", "OuterFrame")
        outer_obj.Shape = outer
        
        # Create inner cutout
        inner_offset_x = (outer_width - inner_width) / 2
        inner_offset_y = (outer_height - inner_height) / 2
        inner = Part.makeBox(inner_width, inner_height, depth)
        inner_obj = self.doc.addObject("Part::Feature", "InnerCutout")
        inner_obj.Shape = inner
        inner_obj.Placement.Base.x = inner_offset_x
        inner_obj.Placement.Base.y = inner_offset_y
        
        # Create frame by cutting
        frame = outer.cut(inner)
        frame_obj = self.doc.addObject("Part::Feature", "Frame")
        frame_obj.Shape = frame
        
        return {
            'objects': ['Frame'],
            'dimensions': {
                'outer_width': outer_width, 
                'outer_height': outer_height,
                'inner_width': inner_width,
                'inner_height': inner_height
            }
        }
    
    def _create_panel_template(self, params: Dict) -> Dict:
        """Create wall panel template."""
        width = params.get('width', 1200.0)
        height = params.get('height', 2400.0)
        thickness = params.get('thickness', 18.0)
        
        # Create simple panel
        panel = Part.makeBox(width, height, thickness)
        panel_obj = self.doc.addObject("Part::Feature", "Panel")
        panel_obj.Shape = panel
        
        return {
            'objects': ['Panel'],
            'dimensions': {'width': width, 'height': height, 'thickness': thickness}
        }
    
    def _create_generic_rectangle(self, params: Dict) -> Dict:
        """Create generic rectangle when no template matches."""
        width = params.get('width', 100.0)
        height = params.get('height', 100.0)
        thickness = params.get('thickness', 10.0)
        
        # Create simple box
        box = Part.makeBox(width, height, thickness)
        box_obj = self.doc.addObject("Part::Feature", "Rectangle")
        box_obj.Shape = box
        
        return {
            'objects': ['Rectangle'],
            'dimensions': {'width': width, 'height': height, 'thickness': thickness}
        }
    
    def _generate_mock_model(self, template_name: str, parameters: Dict) -> Dict:
        """Generate mock model data when FreeCAD is not available."""
        self.logger.info(f"Generating mock model with template: {template_name}")
        
        return {
            'template': template_name,
            'parameters': parameters,
            'objects': [f"Mock_{template_name.title()}"],
            'note': "FreeCAD not available - mock data generated"
        }
    
    def export_model(self, output_path: str, format_type: str = 'fcstd') -> bool:
        """Export CAD model to file."""
        if not self.doc:
            self.logger.warning("No FreeCAD document to export")
            return False
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type.lower() == 'fcstd':
                self.doc.saveAs(str(output_file))
            elif format_type.lower() == 'dxf':
                # Export to DXF (would need additional implementation)
                self.logger.warning("DXF export not yet implemented")
                return False
            
            self.logger.info(f"Model exported to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export model: {e}")
            return False
    
    def close_document(self):
        """Close FreeCAD document."""
        if self.doc:
            FreeCAD.closeDocument(self.doc.Name)
            self.doc = None
