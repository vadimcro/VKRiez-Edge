# VKriez Edge Preprocessors for ComfyUI

A collection of advanced edge detection nodes for ComfyUI that generate high-quality edge maps for ControlNet guidance.

## Overview

This package provides two specialized edge detection nodes:

1. **VKriez Enhanced Edge Preprocessor** - A powerful edge detection solution that produces clean, continuous lines with minimal noise.

2. **VKriez Hybrid MTEED Edge Preprocessor** - Combines MTEED (Multi-Task Transformer Edge and Embedding Detector) with the Enhanced Edge Preprocessor for superior edge detection that understands image content.

## Installation

1. **Install the package**:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/vadimcro/VKRiez-Edge.git
   cd VKRiez-Edge
   pip install -r requirements.txt
   ```

2. **For Hybrid MTEED mode** (optional, but recommended):
   ```bash
   pip install git+https://github.com/Mikubill/sd-webui-controlnet.git
   ```

3. Restart ComfyUI

## Usage

Connect one of the nodes between Load Image node that contains your reference or guidance image and Apply ControlNet node.
The nodes serve as preprocessors for Canny / MistoLine ControlNet Models for contour extraction and guidance.
Additionally you can stick a Preview Bridge node between these two in order to see the results of line extraction.

## Features

### VKriez Enhanced Edge Preprocessor

- Advanced edge detection with clean, continuous lines
- Sophisticated edge linking to connect broken edges
- Noise reduction and small component filtering
- Highly customizable parameters for fine-tuning results

### VKriez Hybrid MTEED Edge Preprocessor

- Combines neural network understanding with classical edge detection
- MTEED understands what's important in the image
- Enhanced Edge Detector adds detail and continuity
- Intelligent blending of both techniques
- Automatic model download from HuggingFace

## Parameter Guide

### VKriez Enhanced Edge Preprocessor Parameters

#### Bilateral Filter Parameters
- **Use Bilateral** - Enable/disable bilateral filtering for noise reduction while preserving edges
- **Bilateral Filter Diameter** - Size of the pixel neighborhood considered. Larger values smooth more but slow down processing.
- **Bilateral Color Sigma** - How much color difference is tolerated. Higher values mix colors more.
- **Bilateral Space Sigma** - How far spatially pixels can influence each other. Higher values create more blurring.

#### CLAHE Parameters (Contrast Limited Adaptive Histogram Equalization)
- **Use CLAHE** - Enable/disable CLAHE processing to enhance local contrast
- **Clip Limit** - Limits contrast enhancement to reduce noise amplification. Higher values increase contrast more.
- **Tile Grid Size** - Size of the grid for histogram equalization. Smaller grids enhance local details more.

#### Canny Edge Detection Parameters
- **Canny Low Threshold** - Lower limit for edge detection. Lower values detect more edges including weak ones.
- **Canny High Threshold** - Upper limit for edge detection. Higher values detect fewer, stronger edges.
- **Canny Aperture Size** - Size of the Sobel kernel used for finding gradients. Larger sizes detect smoother edges.

#### Edge Linking Parameters
- **Use Edge Linking** - Enable/disable edge linking, which connects broken edges
- **Gap Threshold** - Maximum pixel distance between endpoints to connect. Higher values bridge larger gaps.
- **Angle Threshold** - Maximum angular difference to consider edges as related. Higher values connect more diverse edges.

#### Morphological Operation Parameters
- **Use Morphology** - Enable/disable morphological operations for further edge refinement
- **Morphology Kernel Size** - Size of the kernel used for morphological operations. Larger sizes affect broader areas.
- **Morphology Iterations** - Number of times to apply the operation. More iterations create more aggressive results.

#### Component Filtering Parameters
- **Use Component Filter** - Enable/disable filtering of small components
- **Min Component Size** - Minimum size of components to keep (in pixels). Increase to remove more noise and small details.

### VKriez Hybrid MTEED Edge Preprocessor Parameters

Includes all the parameters from the Enhanced Edge Preprocessor, plus:

#### Basic Controls
- **Resolution** - Target resolution for processing. Higher resolutions capture more detail but use more memory.
- **Use MTEED** - Enable/disable the neural network component
- **Use Enhanced Edges** - Enable/disable the enhanced edge detector component

#### Blending Parameters
- **Edge Lower Bound** - Lower intensity threshold for including edges in blending (0.0-1.0)
- **Edge Upper Bound** - Upper intensity threshold for including edges in blending (0.0-1.0)
- **Connectivity** - Connectivity parameter for component analysis (1 for 4-connectivity, 2 for 8-connectivity)

## Examples and Use Cases

### When to use VKriez Enhanced Edge Preprocessor:
- When you need precise control over every aspect of edge detection
- For technical drawings, architectural images, or diagrams
- When fine details and line continuity are important
- When you want to isolate specific edges based on their properties

### When to use VKriez Hybrid MTEED Edge Preprocessor:
- For photographs and complex natural images
- When you want edges that respect object boundaries
- For artistic images where semantic content matters
- When you need a balance between detail and meaningful structure

## Recommended Settings

### Pencil Drawing Style
- Bilateral Filter: Enabled, Diameter=5, Color Sigma=20, Space Sigma=30
- CLAHE: Enabled, Clip Limit=2.0, Tile Size=8
- Canny: Low=50, High=150, Aperture=3
- Edge Linking: Enabled, Gap=3, Angle=30
- Morphology: Enabled, Kernel=3, Iterations=1
- Component Filter: Enabled, Size=20

### Detailed Technical Drawing
- Bilateral Filter: Enabled, Diameter=7, Color Sigma=75, Space Sigma=75
- CLAHE: Enabled, Clip Limit=2.5, Tile Size=8
- Canny: Low=100, High=200, Aperture=3
- Edge Linking: Enabled, Gap=5, Angle=45
- Morphology: Enabled, Kernel=3, Iterations=2
- Component Filter: Enabled, Size=15

### Artistic Outline
- Use Hybrid MTEED Preprocessor
- Resolution: 1280
- MTEED: Enabled
- Enhanced Edges: Enabled with:
  - Bilateral Filter: Enabled, Diameter=9, Color Sigma=100, Space Sigma=100
  - CLAHE: Enabled, Clip Limit=3.0, Tile Size=8
  - Canny: Low=80, High=160, Aperture=3
  - Edge Linking: Enabled, Gap=3, Angle=30
- Blending: Edge Lower=0.1, Edge Upper=1.0, Connectivity=1

## License

MIT License

## Credits

- MTEED model by TheMistoAI: https://github.com/TheMistoAI/ComfyUI-Anyline
- Package development by VKriez
