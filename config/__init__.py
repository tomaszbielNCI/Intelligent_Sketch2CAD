"""Configuration module for Intelligent Sketch to CAD."""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_templates(templates_path: str = "config/templates.yaml") -> Dict[str, Any]:
    """Load templates configuration from YAML file."""
    templates_file = Path(templates_path)
    if not templates_file.exists():
        raise FileNotFoundError(f"Templates file not found: {templates_path}")
    
    with open(templates_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
