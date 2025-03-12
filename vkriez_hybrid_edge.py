import cv2
import torch
import numpy as np
from pathlib import Path
import importlib.util
from huggingface_hub import hf_hub_download
from skimage import morphology

# Check if required modules are available
has_controlnet_aux = importlib.util.find_spec("custom_nodes.comfyui_controlnet_aux") is not None

# Import from correct paths if available
if has_controlnet_aux:
    from custom_nodes.comfyui_controlnet_aux.utils import common_annotator_call
    from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.teed import TEDDetector
    from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.teed.ted import TED

def get_intensity_mask(image_array, lower_bound, upper_bound):
    """Apply intensity thresholding to create a mask"""
    mask = image_array[:, :, 0]
    mask = np.where((mask >= lower_bound) & (mask <= upper_bound), mask, 0)
    mask = np.expand_dims(mask, 2).repeat(3, axis=2)
    return mask

def combine_layers(base_layer, top_layer):
    """Blend two layers using screen blend mode with masking"""
    mask = top_layer.astype(bool)
    temp = 1 - (1 - top_layer) * (1 - base_layer)
    result = base_layer * (~mask) + temp * mask
    return result

class VKriezHybridEdgePreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 1280, "min": 64, "max": 2048, "step": 8}),
                
                # MTEED Controls
                "use_mteed": ("BOOLEAN", {"default": True}),
                
                # Enhanced Edge Controls
                "use_enhanced_edges": ("BOOLEAN", {"default": True}),
                
                # Bilateral filter parameters
                "use_bilateral": ("BOOLEAN", {"default": True}),
                "bilateral_d": ("INT", {
                    "default": 7, 
                    "min": 5, 
                    "max": 15, 
                    "step": 2,
                    "display": "Bilateral filter diameter"
                }),
                "bilateral_sigma_color": ("FLOAT", {
                    "default": 75.0, 
                    "min": 10.0, 
                    "max": 200.0, 
                    "step": 5.0,
                    "display": "Bilateral color sigma"
                }),
                "bilateral_sigma_space": ("FLOAT", {
                    "default": 75.0, 
                    "min": 10.0, 
                    "max": 200.0, 
                    "step": 5.0,
                    "display": "Bilateral space sigma"
                }),
                
                # CLAHE parameters
                "use_clahe": ("BOOLEAN", {"default": True}),
                "clip_limit": ("FLOAT", {
                    "default": 2.0, 
                    "min": 0.5, 
                    "max": 10.0, 
                    "step": 0.5,
                    "display": "CLAHE clip limit"
                }),
                "tile_grid_size": ("INT", {
                    "default": 8, 
                    "min": 2, 
                    "max": 16, 
                    "step": 1,
                    "display": "CLAHE tile grid size"
                }),
                
                # Canny parameters
                "canny_low_threshold": ("INT", {
                    "default": 100, 
                    "min": 0, 
                    "max": 255, 
                    "step": 1,
                    "display": "Canny low threshold"
                }),
                "canny_high_threshold": ("INT", {
                    "default": 200, 
                    "min": 0, 
                    "max": 255, 
                    "step": 1,
                    "display": "Canny high threshold"
                }),
                "canny_aperture": ("INT", {
                    "default": 3, 
                    "min": 3, 
                    "max": 7, 
                    "step": 2,  # Only odd values
                    "display": "Canny aperture size"
                }),
                
                # Edge linking parameters
                "use_edge_linking": ("BOOLEAN", {"default": True}),
                "gap_threshold": ("INT", {
                    "default": 3, 
                    "min": 1, 
                    "max": 10, 
                    "step": 1,
                    "display": "Edge linking gap threshold"
                }),
                "angle_threshold": ("INT", {
                    "default": 30, 
                    "min": 5, 
                    "max": 90, 
                    "step": 5,
                    "display": "Edge linking angle threshold"
                }),
                
                # Morphological operation parameters
                "use_morphology": ("BOOLEAN", {"default": True}),
                "morph_kernel_size": ("INT", {
                    "default": 3, 
                    "min": 1, 
                    "max": 7, 
                    "step": 2,  # Only odd values
                    "display": "Morphology kernel size"
                }),
                "morph_iterations": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 3, 
                    "step": 1,
                    "display": "Morphology iterations"
                }),
                
                # Component filtering parameters
                "use_component_filter": ("BOOLEAN", {"default": True}),
                "min_component_size": ("INT", {
                    "default": 36, 
                    "min": 5, 
                    "max": 500, 
                    "step": 5,
                    "display": "Min component size (pixels)"
                }),
                
                # Blending parameters
                "edge_lower_bound": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display": "Edge lower threshold"
                }),
                "edge_upper_bound": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display": "Edge upper threshold"
                }),
                "connectivity": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 3,
                    "display": "Component connectivity"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edges"
    CATEGORY = "VKriez/image/preprocessors"

    def __init__(self):
        self.device = "cpu"
        
        if not has_controlnet_aux:
            print("Warning: comfyui_controlnet_aux not found. MTEED processing will be disabled.")
            print("Please make sure it is installed correctly.")

    def load_model(self):
        """Load the MTEED model from local path or download if not found"""
        subfolder = "Anyline"
        checkpoint_filename = "MTEED.pth"
        checkpoint_dir = Path(__file__).parent.resolve() / "checkpoints" / subfolder
        checkpoint_path = checkpoint_dir / checkpoint_filename
        
        # Download the model if it's not present
        if not checkpoint_path.is_file():
            print("Model not found locally, downloading from HuggingFace...")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = hf_hub_download(
                repo_id="TheMistoAI/MistoLine", 
                filename=checkpoint_filename, 
                subfolder=subfolder, 
                local_dir=checkpoint_dir
            )
        
        # Load the model
        model = TED()
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        
        return model

    def detect_edges(self, image, resolution=1280, 
                     use_mteed=True, use_enhanced_edges=True,
                     use_bilateral=True, bilateral_d=7, bilateral_sigma_color=75.0, bilateral_sigma_space=75.0,
                     use_clahe=True, clip_limit=2.0, tile_grid_size=8, 
                     canny_low_threshold=100, canny_high_threshold=200, canny_aperture=3,
                     use_edge_linking=True, gap_threshold=3, angle_threshold=30, 
                     use_morphology=True, morph_kernel_size=3, morph_iterations=1,
                     use_component_filter=True, min_component_size=36,
                     edge_lower_bound=0.0, edge_upper_bound=1.0, connectivity=1):
        
        # Disable MTEED if custom_controlnet_aux is not available
        if not has_controlnet_aux:
            use_mteed = False
        
        mteed_result = None
        # Step 1: Get MTEED result if enabled
        if use_mteed:
            try:
                # Load and process with MTEED model
                mteed_model = TEDDetector(model=self.load_model()).to(self.device)
                mteed_result = common_annotator_call(mteed_model, image, resolution=resolution)
                mteed_result = mteed_result.squeeze(0).numpy()
                del mteed_model
            except Exception as e:
                print(f"Error loading or processing with MTEED model: {e}")
                import traceback
                traceback.print_exc()
                use_mteed = False
        
        # Step 2: Process with enhanced edge detector if enabled
        enhanced_result = None
        if use_enhanced_edges:
            try:
                # Process with our enhanced edge detector
                enhanced_result = self.process_enhanced_edges(
                    image, use_bilateral, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
                    use_clahe, clip_limit, tile_grid_size, canny_low_threshold, canny_high_threshold, 
                    canny_aperture, use_edge_linking, gap_threshold, angle_threshold, use_morphology, 
                    morph_kernel_size, morph_iterations, use_component_filter, min_component_size,
                    resolution
                )
                
                # Apply intensity thresholding and small component removal
                enhanced_result = get_intensity_mask(enhanced_result, lower_bound=edge_lower_bound, upper_bound=edge_upper_bound)
                cleaned = morphology.remove_small_objects(enhanced_result.astype(bool), min_size=min_component_size, connectivity=connectivity)
                enhanced_result = enhanced_result * cleaned
            except Exception as e:
                print(f"Error in enhanced edge processing: {e}")
                use_enhanced_edges = False
        
        # Step 3: If both methods are disabled or failed, return input image
        if not use_mteed and not use_enhanced_edges:
            print("Both edge detection methods disabled or failed. Returning unprocessed image.")
            return (image,)
        
        # Step 4: If only one method is enabled/successful, return its result
        if use_mteed and not use_enhanced_edges:
            return (torch.tensor(mteed_result).unsqueeze(0),)
        
        if use_enhanced_edges and not use_mteed:
            return (torch.tensor(enhanced_result).unsqueeze(0),)
        
        # Step 5: Blend the results from both methods
        # Resize enhanced_result to match mteed_result dimensions
        mteed_h, mteed_w = mteed_result.shape[:2]
        enhanced_result_resized = cv2.resize(enhanced_result, (mteed_w, mteed_h), interpolation=cv2.INTER_LINEAR)

        # Now combine them
        final_result = combine_layers(mteed_result, enhanced_result_resized)
        
        return (torch.tensor(final_result).unsqueeze(0),)

    def process_enhanced_edges(self, image, use_bilateral, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
                               use_clahe, clip_limit, tile_grid_size, canny_low_threshold, canny_high_threshold, 
                               canny_aperture, use_edge_linking, gap_threshold, angle_threshold, use_morphology, 
                               morph_kernel_size, morph_iterations, use_component_filter, min_component_size,
                               resolution):
        """Process images using our enhanced edge detector"""
        # Convert from tensor to numpy
        img_np = image.squeeze(0).cpu().numpy()
        
        # Convert from float (0-1) to uint8 (0-255)
        img_np = (img_np * 255).astype(np.uint8)
        
        # Resize to target resolution if needed
        h, w = img_np.shape[:2]
        if max(h, w) != resolution:
            # Calculate new dimensions while preserving aspect ratio
            if h > w:
                new_h, new_w = resolution, int(resolution * w / h)
            else:
                new_h, new_w = int(resolution * h / w), resolution
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Call enhanced edge processing function
        processed = self.process_single_image(
            img_np, use_bilateral, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
            use_clahe, clip_limit, tile_grid_size, canny_low_threshold, canny_high_threshold, 
            canny_aperture, use_edge_linking, gap_threshold, angle_threshold, use_morphology, 
            morph_kernel_size, morph_iterations, use_component_filter, min_component_size
        )
        
        # Convert back to float (0-1) range
        processed = processed.astype(np.float32) / 255.0
        
        return processed

    def process_single_image(self, img, use_bilateral, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
                           use_clahe, clip_limit, tile_grid_size, canny_low_threshold, canny_high_threshold, 
                           canny_aperture, use_edge_linking, gap_threshold, angle_threshold, use_morphology, 
                           morph_kernel_size, morph_iterations, use_component_filter, min_component_size):
        """Process a single image using our enhanced edge detector"""
        # Convert RGB to grayscale for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Step 1: Apply bilateral filtering if enabled
        if use_bilateral:
            gray = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

        # Step 2: Apply CLAHE if enabled
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            gray = clahe.apply(gray)
        
        # Step 3: Apply Canny edge detection
        edges = cv2.Canny(gray, 
                         canny_low_threshold, 
                         canny_high_threshold, 
                         apertureSize=canny_aperture,
                         L2gradient=True)
        
        # Step 4: Apply edge linking if enabled
        if use_edge_linking:
            edges = self._apply_edge_linking(edges, gap_threshold, angle_threshold)
        
        # Step 5: Apply morphological operations if enabled
        if use_morphology:
            kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        
        # Step 6: Apply connected component filtering if enabled
        if use_component_filter:
            edges = self._filter_small_components(edges, min_component_size)
        
        # Convert back to 3-channel RGB image (white edges on black background)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return edges_rgb
        
    def _apply_edge_linking(self, edge_image, gap_threshold=3, angle_threshold=30):
        """Links broken edges in the edge detected image to create more continuous lines."""
        # Create a copy of the edge image to work with
        linked_edges = edge_image.copy()
        
        # Find all edge pixels
        edge_points = np.where(edge_image > 0)
        
        # Create a binary image for processing
        binary = (edge_image > 0).astype(np.uint8)
        
        # Calculate gradients for angle information (using Sobel)
        grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        # Find endpoints - pixels that have exactly one neighbor
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        
        # Convolve with the kernel - endpoints will have a value of 11 (10 + 1)
        convolved = cv2.filter2D(binary, -1, kernel)
        endpoints = np.where(convolved == 11)
        
        # For each endpoint, look for nearby endpoints to connect
        for i in range(len(endpoints[0])):
            y1, x1 = endpoints[0][i], endpoints[1][i]
            dir1 = direction[y1, x1]
            
            # Skip if this point has already been processed
            if linked_edges[y1, x1] == 0:
                continue
            
            # Look for other endpoints within the gap threshold
            best_match = None
            min_distance = gap_threshold + 1
            
            for j in range(len(endpoints[0])):
                if i == j:
                    continue
                    
                y2, x2 = endpoints[0][j], endpoints[1][j]
                dir2 = direction[y2, x2]
                
                # Calculate distance
                dist = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                
                # Check if within gap threshold and not already connected
                if dist <= gap_threshold and linked_edges[y2, x2] > 0:
                    # Check angle compatibility (directions should be similar or opposite)
                    angle_diff = np.abs(((dir1 - dir2 + 180) % 360) - 180)
                    
                    if angle_diff <= angle_threshold:
                        if dist < min_distance:
                            min_distance = dist
                            best_match = (y2, x2)
            
            # Connect to the best match if found
            if best_match is not None:
                y2, x2 = best_match
                # Draw a line between the endpoints
                cv2.line(linked_edges, (x1, y1), (x2, y2), 255, 1)
        
        return linked_edges

    def _filter_small_components(self, binary_image, min_size=20):
        """Removes small connected components from a binary image."""
        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        # Create output image
        filtered = np.zeros_like(binary_image)
        
        # Skip label 0 (background)
        for i in range(1, num_labels):
            # If the component size is greater than min_size, keep it
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                filtered[labels == i] = 255
                
        return filtered

# Node registration
NODE_CLASS_MAPPINGS = {
    "VKriezHybridEdgePreprocessor": VKriezHybridEdgePreprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VKriezHybridEdgePreprocessor": "VKriez Hybrid MTEED Edge Preprocessor"
}
