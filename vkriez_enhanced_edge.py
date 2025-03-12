import cv2
import numpy as np
import torch

class VKriezEnhancedEdgePreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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
                    "default": 20, 
                    "min": 5, 
                    "max": 500, 
                    "step": 5,
                    "display": "Min component size (pixels)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edges"
    CATEGORY = "VKriez/image/preprocessors"

    def detect_edges(self, image, use_bilateral, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
                    use_clahe, clip_limit, tile_grid_size, canny_low_threshold, canny_high_threshold, canny_aperture,
                    use_edge_linking, gap_threshold, angle_threshold, use_morphology, morph_kernel_size, 
                    morph_iterations, use_component_filter, min_component_size):
        
        # Process each image in the batch
        batch_size = image.shape[0]
        result = []
        
        for i in range(batch_size):
            # Convert from tensor to numpy
            img_np = image[i].cpu().numpy()
                
            # Convert from float (0-1) to uint8 (0-255)
            img_np = (img_np * 255).astype(np.uint8)
            
            # Call processing function
            processed = self.process_single_image(
                img_np, use_bilateral, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
                use_clahe, clip_limit, tile_grid_size, canny_low_threshold, canny_high_threshold, 
                canny_aperture, use_edge_linking, gap_threshold, angle_threshold, use_morphology, 
                morph_kernel_size, morph_iterations, use_component_filter, min_component_size
            )
            
            # Convert back to float (0-1) range and then to tensor
            processed = processed.astype(np.float32) / 255.0
            processed_tensor = torch.from_numpy(processed)
            result.append(processed_tensor)
        
        # Stack the processed images back into a batch tensor
        return (torch.stack(result, dim=0),)

    def process_single_image(self, img, use_bilateral, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
                           use_clahe, clip_limit, tile_grid_size, canny_low_threshold, canny_high_threshold, 
                           canny_aperture, use_edge_linking, gap_threshold, angle_threshold, use_morphology, 
                           morph_kernel_size, morph_iterations, use_component_filter, min_component_size):
        
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
                         L2gradient=True)  # L2gradient for better accuracy
        
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
        """
        Links broken edges in the edge detected image to create more continuous lines.
        
        Parameters:
        edge_image (numpy.ndarray): Binary edge image from Canny
        gap_threshold (int): Maximum gap size to bridge (in pixels)
        angle_threshold (float): Maximum angle difference (in degrees) for edges to be considered continuous
        
        Returns:
        numpy.ndarray: Edge image with linked edges
        """
        # Create a copy of the edge image to work with
        linked_edges = edge_image.copy()
        
        # Find all edge pixels
        edge_points = np.where(edge_image > 0)
        
        # Create a binary image for processing
        binary = (edge_image > 0).astype(np.uint8)
        
        # Calculate gradients for angle information (using Sobel)
        # These gradients help determine the direction of edges
        grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        # Find endpoints - pixels that have exactly one neighbor
        # These are likely the ends of broken lines
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]], dtype=np.uint8)
        
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
                        # Calculate line equation for the potential connection
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
        """
        Removes small connected components from a binary image.
        
        Parameters:
        binary_image (numpy.ndarray): Binary image
        min_size (int): Minimum component size to keep (in pixels)
        
        Returns:
        numpy.ndarray: Filtered binary image
        """
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

# Node registration - this is needed to make the node available in ComfyUI
NODE_CLASS_MAPPINGS = {
    "VKriezEnhancedEdgePreprocessor": VKriezEnhancedEdgePreprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VKriezEnhancedEdgePreprocessor": "VKriez Enhanced Edge Preprocessor"
}
