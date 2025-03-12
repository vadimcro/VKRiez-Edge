"""
VKriez Edge Preprocessors Package for ComfyUI

This package contains specialized edge detection and preprocessing nodes 
for generating high-quality edge maps for ControlNet guidance.

Nodes included:
- VKriez Enhanced Edge Preprocessor: Advanced edge detection with continuous lines
- VKriez Hybrid MTEED Edge Preprocessor: Combines MTEED with advanced edge detection
"""

from .vkriez_enhanced_edge import NODE_CLASS_MAPPINGS as ENHANCED_EDGE_NODES
from .vkriez_hybrid_edge import NODE_CLASS_MAPPINGS as HYBRID_EDGE_NODES

# Merge the node mappings
NODE_CLASS_MAPPINGS = {**ENHANCED_EDGE_NODES, **HYBRID_EDGE_NODES}

# Merge the display name mappings
from .vkriez_enhanced_edge import NODE_DISPLAY_NAME_MAPPINGS as ENHANCED_EDGE_DISPLAY
from .vkriez_hybrid_edge import NODE_DISPLAY_NAME_MAPPINGS as HYBRID_EDGE_DISPLAY

NODE_DISPLAY_NAME_MAPPINGS = {**ENHANCED_EDGE_DISPLAY, **HYBRID_EDGE_DISPLAY}

# Make the merged mappings available for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
