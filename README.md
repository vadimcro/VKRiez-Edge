# VKriez Edge Preprocessors for ComfyUI

A collection of advanced edge detection nodes for ComfyUI that generate high-quality edge maps for ControlNet guidance.
Currently based on CPU computation, so might be a tid-bit on a slow side. If anyone is willing to refactor the code to GPU computation - Kudos!

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

## Examples

![image](https://github.com/user-attachments/assets/a248ed24-6465-42e4-9e89-03f01f387ebf)

![image](https://github.com/user-attachments/assets/f91da531-1969-4305-be01-0031aa8c1483)

![image](https://github.com/user-attachments/assets/5c481339-d3f0-496e-9d91-c9df5c663453)


## Parameter Guide

# VKriez Edge Preprocessors - Detailed Parameter Guide

This guide provides in-depth explanations of each parameter in the Edge Preprocessors:

## Detailed Parameter Breakdown

### Bilateral Filter Parameters

**Use Bilateral**: Toggles bilateral filtering, which reduces noise while preserving edges.
- *When enabled*: Smooths out noise in flat areas while keeping important edges sharp
- *When disabled*: Preserves more texture detail but can result in noisier edge detection

**Bilateral Filter Diameter**: Controls the size of the pixel neighborhood considered during filtering.
- *Value 5 (Small)*: Minimal smoothing, preserves fine texture and detail
- *Value 7-9 (Medium)*: Balanced smoothing, good for most photos
- *Value 11-15 (Large)*: Strong smoothing, good for noisy images but may lose some detail
- *Example*: For a portrait photo with skin, use larger values (9-11) to smooth skin texture but maintain facial features

**Bilateral Color Sigma**: Determines how much color difference is allowed between pixels when smoothing.
- *Value 10-30 (Low)*: Only very similar colors are blended, preserves color edges
- *Value 50-100 (Medium)*: Moderate color blending, good balance for most images
- *Value 150-200 (High)*: Strong color blending, reduces color noise but may blur colorful details
- *Example*: For a sunset photo with subtle color gradients, use lower values (20-40) to preserve the color transitions

**Bilateral Space Sigma**: Controls how far apart pixels can be to influence each other.
- *Value 10-30 (Low)*: Only nearby pixels affect each other, preserves small details
- *Value 50-100 (Medium)*: Moderate spatial blending, good for most images
- *Value 150-200 (High)*: Wider area blending, creates smoother regions
- *Example*: For an architectural drawing with fine lines, use lower values (30-50) to preserve the thin lines

### CLAHE Parameters (Contrast Limited Adaptive Histogram Equalization)

**Use CLAHE**: Toggles contrast enhancement that works locally in different regions of the image.
- *When enabled*: Improves visibility of edges in both dark and bright areas
- *When disabled*: Maintains original image contrast, which may lose edges in shadows or highlights

**Clip Limit**: Limits how much contrast enhancement is applied, preventing noise amplification.
- *Value 1.0 (Low)*: Subtle enhancement, minimal noise
- *Value 2.0-3.0 (Medium)*: Balanced enhancement, good for most images
- *Value 4.0-10.0 (High)*: Aggressive enhancement, brings out maximum detail but may introduce noise
- *Example*: For a foggy landscape, use higher values (3.0-4.0) to make hidden details more visible

**Tile Grid Size**: Determines the size of local regions for contrast enhancement.
- *Value 2-4 (Small)*: Very localized enhancement, good for images with varied lighting
- *Value 8 (Medium)*: Balanced local/global enhancement, works well for most images
- *Value 12-16 (Large)*: More global enhancement, better for evenly lit images
- *Example*: For a night photo with bright lights and dark shadows, use smaller values (4-6) to enhance details in both areas

### Canny Edge Detection Parameters

**Canny Low Threshold**: Sets the minimum gradient strength to be considered a potential edge.
- *Value 50-80 (Low)*: Detects more edges including faint ones, good for subtle details
- *Value 100-120 (Medium)*: Balanced detection, good for most images
- *Value 150-200 (High)*: Only detects strong edges, reduces noise but may miss subtle details
- *Example*: For a sketch with light pencil lines, use lower values (50-70) to capture the faint strokes

**Canny High Threshold**: Sets the minimum gradient strength to be definitely considered an edge.
- *Value 100-150 (Low)*: More permissive, includes more uncertain edges
- *Value 200 (Medium)*: Standard setting for most images
- *Value 250-255 (High)*: Very strict, only the strongest edges are detected
- *Example*: For a high-contrast graphic with clean lines, use higher values (220-250) for crisp results

**Canny Aperture Size**: Determines the size of the Sobel operator used for finding gradients.
- *Value 3*: Standard setting, detects fine edges and details
- *Value 5*: Medium smoothing, reduces noise but may miss the finest details
- *Value 7*: Stronger smoothing, best for noisy images but blurs fine edges
- *Example*: For a detailed technical drawing with thin lines, use the smallest aperture (3) to preserve fine details
- *Example*: For a noisy photograph or scanned document, use larger aperture (5-7) to reduce the noise impact

### Edge Linking Parameters

**Use Edge Linking**: Toggles the algorithm that connects broken edges to form continuous lines.
- *When enabled*: Creates more connected, continuous outlines, great for illustrations
- *When disabled*: Preserves the raw edge detection output, which may have gaps

**Gap Threshold**: Maximum pixel distance between endpoints to be connected.
- *Value 1-2 (Small)*: Conservative linking, only very close endpoints are connected
- *Value 3-5 (Medium)*: Balanced approach, bridges small gaps without excessive connection
- *Value 6-10 (Large)*: Aggressive linking, bridges larger gaps but may create unwanted connections
- *Example*: For a hand-drawn sketch with slightly broken lines, use medium values (3-4) to complete the lines naturally

**Angle Threshold**: Maximum angular difference between edges to be considered for linking.
- *Value 10-20 (Low)*: Only nearly parallel or continuous edges are linked
- *Value 30-45 (Medium)*: Reasonable angular tolerance, good for most curved shapes
- *Value 60-90 (High)*: High tolerance, allows sharper turns but may create unwanted connections
- *Example*: For geometric shapes with sharp corners, use higher values (45-60) to properly connect at the corners
- *Example*: For flowing, organic shapes, use medium values (30-40) to maintain natural curves

### Morphological Operation Parameters

**Use Morphology**: Toggles morphological operations (closing) to further refine edges.
- *When enabled*: Helps close small gaps and smooth edge contours
- *When disabled*: Preserves original edge structure without additional processing

**Morphology Kernel Size**: Size of the structuring element used for morphological operations.
- *Value 1*: Minimal effect, only the smallest gaps are affected
- *Value 3 (Medium)*: Balanced effect, closes small gaps without excessive thickening
- *Value 5-7 (Large)*: Strong effect, closes larger gaps but may make edges thicker
- *Example*: For a line drawing with small breaks, use a kernel size of 3 to close the gaps while maintaining line thickness

**Morphology Iterations**: Number of times to apply the morphological operation.
- *Value 1*: Single application, subtle effect
- *Value 2*: Moderate effect, good for slightly disconnected edges
- *Value 3*: Strong effect, effectively closes larger gaps but may over-thicken edges
- *Example*: For a dotted line that you want to appear solid, use 2-3 iterations to fully connect the dots

### Component Filtering Parameters

**Use Component Filter**: Toggles filtering of small, isolated edge components.
- *When enabled*: Removes small noise and isolated pixels for cleaner output
- *When disabled*: Preserves all detected edges, including potential noise

**Min Component Size**: Minimum size in pixels for an edge component to be kept.
- *Value 5-15 (Small)*: Minimal filtering, removes only very small speckles
- *Value 20-50 (Medium)*: Moderate filtering, removes small isolated edges
- *Value 100-500 (Large)*: Aggressive filtering, keeps only major edge structures
- *Example*: For a noisy scan with speckling, use higher values (50-100) to eliminate the noise
- *Example*: For detailed line art with fine elements, use lower values (10-20) to preserve small details

### VKriez Hybrid MTEED Edge Preprocessor Specific Parameters

**Resolution**: Target resolution for processing.
- *Value 512-640 (Low)*: Faster processing, good for preview or less detailed output
- *Value 1024 (Medium)*: Good balance of detail and performance for most uses
- *Value 1280-2048 (High)*: Maximum detail, but slower processing and more memory intensive
- *Example*: For creating a detailed ControlNet reference, use 1280 or higher to capture all important details

**Use MTEED**: Toggles the neural network edge detection component.
- *When enabled*: Leverages AI to identify semantically meaningful edges
- *When disabled*: Falls back to traditional edge detection only

**Use Enhanced Edges**: Toggles the enhanced edge detector component.
- *When enabled*: Adds detailed, continuous edges from traditional processing
- *When disabled*: Uses only the MTEED neural output

**Edge Lower Bound**: Lower intensity threshold for including edges in blending.
- *Value 0.0*: Includes all detected edges, even very faint ones
- *Value 0.1-0.3*: Excludes the faintest edges, reducing noise
- *Value 0.4-0.7*: Only includes moderate to strong edges
- *Example*: For clean, minimalist outlines, use higher values (0.3-0.5) to include only stronger edges

**Edge Upper Bound**: Upper intensity threshold for including edges in blending.
- *Value 0.5-0.8*: Excludes the strongest edges, useful for de-emphasizing harsh lines
- *Value 1.0*: Includes all edge intensities up to maximum strength
- *Example*: Almost always keep at 1.0 unless you specifically want to exclude the strongest edges

**Connectivity**: Connectivity parameter for component analysis.
- *Value 1*: 4-connectivity (only cardinal directions), more strict filtering
- *Value 2*: 8-connectivity (includes diagonals), more lenient filtering
- *Example*: For detailed technical drawings, use 2 (8-connectivity) to better preserve diagonal elements


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

## Scenario-Based Examples

### Scenario 1: Technical Blueprint or Architectural Drawing

**Goal**: Extract clean, precise lines while removing noise and maintaining connectivity.

**Recommended Settings**:
- **Enhanced Edge Preprocessor**:
  - Bilateral Filter: Enabled, Diameter=5, Color Sigma=30, Space Sigma=30
  - CLAHE: Enabled, Clip Limit=1.5, Tile Size=8
  - Canny: Low=80, High=160, Aperture=3
  - Edge Linking: Enabled, Gap=2, Angle=20
  - Morphology: Enabled, Kernel=3, Iterations=1
  - Component Filter: Enabled, Size=10

**Why these settings work**: The low bilateral diameter preserves fine lines, while the moderate CLAHE enhances visibility without adding noise. The narrow angle threshold for edge linking ensures precise corners are maintained. Small component filtering removes specks while keeping important small details.

**Visual result**: Clean, continuous lines with sharp corners and preserved details, excellent for ControlNet to understand structural elements.

### Scenario 2: Portrait or Human Figure

**Goal**: Capture facial features and body contours while reducing skin texture noise.

**Recommended Settings**:
- **Hybrid MTEED Edge Preprocessor**:
  - Resolution: 1280
  - MTEED: Enabled
  - Enhanced Edges: Enabled with:
    - Bilateral Filter: Enabled, Diameter=11, Color Sigma=100, Space Sigma=100
    - CLAHE: Enabled, Clip Limit=2.0, Tile Size=8
    - Canny: Low=100, High=200, Aperture=5
    - Edge Linking: Enabled, Gap=3, Angle=40
    - Morphology: Enabled, Kernel=3, Iterations=1
    - Component Filter: Enabled, Size=30
  - Blending: Edge Lower=0.1, Edge Upper=1.0, Connectivity=1

**Why these settings work**: The larger bilateral filter diameter smooths skin texture while MTEED helps identify important facial features. The higher aperture in Canny reduces noise from skin texture. The component filter size of 30 removes small skin detail edges while preserving important facial features.

**Visual result**: Clean outlines of the face and body with important features like eyes, nose, mouth, and hair well-defined, while minimizing skin texture and other distracting details.

### Scenario 3: Hand-Drawn Sketch or Illustration

**Goal**: Enhance and clean up hand-drawn lines while maintaining the artistic style.

**Recommended Settings**:
- **Enhanced Edge Preprocessor**:
  - Bilateral Filter: Enabled, Diameter=7, Color Sigma=50, Space Sigma=50
  - CLAHE: Enabled, Clip Limit=2.5, Tile Size=8
  - Canny: Low=50, High=150, Aperture=3
  - Edge Linking: Enabled, Gap=4, Angle=45
  - Morphology: Enabled, Kernel=3, Iterations=2
  - Component Filter: Enabled, Size=15

**Why these settings work**: The lower Canny thresholds capture lighter pencil/pen strokes, while the edge linking with a gap of 4 helps connect slightly broken lines in the drawing. Multiple morphology iterations help ensure line continuity, and the moderate component filter size removes small smudges while keeping artistic details.

**Visual result**: Cleaned-up, continuous lines that maintain the character of the original drawing, with small imperfections removed.

### Scenario 4: Landscape or Nature Scene

**Goal**: Extract meaningful structural elements while reducing noise from complex textures.

**Recommended Settings**:
- **Hybrid MTEED Edge Preprocessor**:
  - Resolution: 1280
  - MTEED: Enabled
  - Enhanced Edges: Enabled with:
    - Bilateral Filter: Enabled, Diameter=9, Color Sigma=75, Space Sigma=75
    - CLAHE: Enabled, Clip Limit=2.0, Tile Size=4
    - Canny: Low=120, High=240, Aperture=5
    - Edge Linking: Enabled, Gap=3, Angle=30
    - Morphology: Enabled, Kernel=3, Iterations=1
    - Component Filter: Enabled, Size=50
  - Blending: Edge Lower=0.2, Edge Upper=1.0, Connectivity=1

**Why these settings work**: MTEED helps identify meaningful structures like horizons, trees, and buildings, while ignoring less important details. The smaller CLAHE tile size (4) helps enhance details in varied lighting conditions typical in landscapes. The larger component filter size (50) aggressively removes small texture details like grass and leaves that would create noisy edges.

**Visual result**: Clean outlines of major landscape elements like mountains, trees, buildings, and horizons, without excessive texture detail that would confuse ControlNet.

### Scenario 5: Product Photography on Plain Background

**Goal**: Extract precise product outlines with clean edges and minimal background noise.

**Recommended Settings**:
- **Enhanced Edge Preprocessor**:
  - Bilateral Filter: Enabled, Diameter=7, Color Sigma=50, Space Sigma=50
  - CLAHE: Enabled, Clip Limit=1.5, Tile Size=8
  - Canny: Low=100, High=200, Aperture=3
  - Edge Linking: Enabled, Gap=2, Angle=30
  - Morphology: Enabled, Kernel=3, Iterations=1
  - Component Filter: Enabled, Size=25

**Why these settings work**: Moderate bilateral filtering smooths minor texture variations while preserving product details. The conservative CLAHE (1.5) enhances edges without introducing background noise. The component filter size of 25 is large enough to remove background speckles while keeping product details.

**Visual result**: Clean, precise outlines of the product with well-defined features and minimal background noise.

### Scenario 6: Comic Book or Manga Style Art

**Goal**: Extract bold lines and maintain stylistic elements typical in comics.

**Recommended Settings**:
- **Enhanced Edge Preprocessor**:
  - Bilateral Filter: Enabled, Diameter=5, Color Sigma=30, Space Sigma=30
  - CLAHE: Enabled, Clip Limit=3.0, Tile Size=8
  - Canny: Low=80, High=160, Aperture=3
  - Edge Linking: Enabled, Gap=3, Angle=45
  - Morphology: Enabled, Kernel=3, Iterations=1
  - Component Filter: Enabled, Size=10

**Why these settings work**: The higher CLAHE clip limit enhances the bold contrast typical in comics. The moderate gap and angle thresholds in edge linking help maintain the stylistic line quality, while the relatively small component filter size preserves intentional small details like texture lines and hatching.

**Visual result**: Bold, clean lines that preserve the comic style, with good continuity and minimal noise.

### Scenario 7: Old Document or Vintage Photograph

**Goal**: Extract important content while handling aging artifacts, discoloration, and noise.

**Recommended Settings**:
- **Hybrid MTEED Edge Preprocessor**:
  - Resolution: 1280
  - MTEED: Enabled
  - Enhanced Edges: Enabled with:
    - Bilateral Filter: Enabled, Diameter=11, Color Sigma=150, Space Sigma=150
    - CLAHE: Enabled, Clip Limit=3.5, Tile Size=4
    - Canny: Low=70, High=200, Aperture=5
    - Edge Linking: Enabled, Gap=5, Angle=45
    - Morphology: Enabled, Kernel=5, Iterations=2
    - Component Filter: Enabled, Size=60
  - Blending: Edge Lower=0.15, Edge Upper=1.0, Connectivity=1

**Why these settings work**: The large bilateral filter with high sigma values helps smooth out aging artifacts and discoloration. The higher CLAHE clip limit and smaller tile size help recover faded details. The larger morphology kernel and multiple iterations help bridge gaps in faded content, while the aggressive component filtering removes aging artifacts and speckling.

**Visual result**: Recovered content with reduced aging artifacts, improved clarity, and emphasis on the meaningful elements of the document or photograph.

## Troubleshooting Common Issues

### Problem: Too many unwanted edges or noise
- **Solution 1**: Increase bilateral filter diameter and sigma values
- **Solution 2**: Increase Canny high threshold (try 220-250)
- **Solution 3**: Increase minimum component size (try 50-100)
- **Solution 4**: For hybrid mode, increase edge lower bound to 0.2-0.3

### Problem: Important edges are missing
- **Solution 1**: Decrease Canny low threshold (try 50-70)
- **Solution 2**: Increase CLAHE clip limit (try 3.0-4.0)
- **Solution 3**: For hybrid mode, make sure both MTEED and enhanced edges are enabled
- **Solution 4**: Decrease edge lower bound to 0.0

### Problem: Edges have gaps or are disconnected
- **Solution 1**: Increase edge linking gap threshold (try 5-7)
- **Solution 2**: Increase edge linking angle threshold (try 45-60)
- **Solution 3**: Increase morphology kernel size and iterations
- **Solution 4**: Decrease Canny high threshold to detect more potential edges

### Problem: Edges are too thick
- **Solution 1**: Decrease morphology kernel size and iterations
- **Solution 2**: Use smaller bilateral filter diameter
- **Solution 3**: For hybrid mode, try using only MTEED without enhanced edges

### Problem: Processing is too slow
- **Solution 1**: Decrease resolution (try 512-768)
- **Solution 2**: Disable edge linking (most computationally intensive step)
- **Solution 3**: Use smaller bilateral filter diameter
- **Solution 4**: For hybrid mode, try using only enhanced edges without MTEED


## License

MIT License

## Credits

- MTEED model by TheMistoAI: https://github.com/TheMistoAI/ComfyUI-Anyline
- Package development by VKriez
