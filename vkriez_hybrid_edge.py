import cv2
import torch
import numpy as np
from pathlib import Path
import importlib.util
from huggingface_hub import hf_hub_download
from skimage import morphology

# Check if required modules are available
has_controlnet_aux = importlib.util.find_spec("custom_nodes.comfyui_controlnet_aux") is not None

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
                
                "use_mteed": ("BOOLEAN", {"default": True}),
                "use_enhanced_edges": ("BOOLEAN", {"default": True}),
                
                # GPU Acceleration
                "use_gpu_acceleration": ("BOOLEAN", {"default": True}),
                
                # Bilateral filter
                "use_bilateral": ("BOOLEAN", {"default": True}),
                "bilateral_d": ("INT", {"default": 7, "min": 5, "max": 15, "step": 2}),
                "bilateral_sigma_color": ("FLOAT", {"default": 75.0, "min": 10.0, "max": 200.0, "step": 5.0}),
                "bilateral_sigma_space": ("FLOAT", {"default": 75.0, "min": 10.0, "max": 200.0, "step": 5.0}),
                
                # CLAHE
                "use_clahe": ("BOOLEAN", {"default": True}),
                "clip_limit": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5}),
                "tile_grid_size": ("INT", {"default": 8, "min": 2, "max": 16, "step": 1}),
                
                # Canny
                "canny_low_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "canny_high_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "canny_aperture": ("INT", {"default": 3, "min": 3, "max": 7, "step": 2}),
                
                # Edge linking
                "use_edge_linking": ("BOOLEAN", {"default": True}),
                "gap_threshold": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "angle_threshold": ("INT", {"default": 30, "min": 5, "max": 90, "step": 5}),
                
                # Morphology
                "use_morphology": ("BOOLEAN", {"default": True}),
                "morph_kernel_size": ("INT", {"default": 3, "min": 1, "max": 7, "step": 2}),
                "morph_iterations": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1}),
                
                # Component filtering
                "use_component_filter": ("BOOLEAN", {"default": True}),
                "min_component_size": ("INT", {"default": 36, "min": 5, "max": 500, "step": 5}),
                
                # Blending
                "edge_lower_bound": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_upper_bound": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "connectivity": ("INT", {"default": 1, "min": 1, "max": 3}),
                
                # Adaptive regions
                "use_adaptive_regions": ("BOOLEAN", {"default": True}),
                
                # Enhanced filtering
                "use_enhanced_filtering": ("BOOLEAN", {"default": True}),
                "connectivity_threshold": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1}),
                
                # Edge consistency
                "use_edge_consistency": ("BOOLEAN", {"default": True}),
                "target_edge_thickness": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edges"
    CATEGORY = "VKriez/image/preprocessors"

    def __init__(self):
        # Keep device usage for MTEED
        self.device = "cpu"
        self.has_torch_cuda = torch.cuda.is_available()
        
        if not has_controlnet_aux:
            print("Warning: comfyui_controlnet_aux not found. MTEED processing will be disabled.")

    def load_model(self):
        """Load the MTEED model from local path or download if not found."""
        subfolder = "Anyline"
        checkpoint_filename = "MTEED.pth"
        checkpoint_dir = Path(__file__).parent.resolve() / "checkpoints" / subfolder
        checkpoint_path = checkpoint_dir / checkpoint_filename
        
        if not checkpoint_path.is_file():
            print("Model not found locally, downloading from HuggingFace...")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = hf_hub_download(
                repo_id="TheMistoAI/MistoLine", 
                filename=checkpoint_filename, 
                subfolder=subfolder, 
                local_dir=checkpoint_dir
            )
        
        model = TED()
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        return model

    def detect_edges(
        self, 
        image,
        resolution=1280, 
        use_mteed=True, 
        use_enhanced_edges=True,
        use_gpu_acceleration=True,
        use_bilateral=True, 
        bilateral_d=7, 
        bilateral_sigma_color=75.0, 
        bilateral_sigma_space=75.0,
        use_clahe=True, 
        clip_limit=2.0, 
        tile_grid_size=8, 
        canny_low_threshold=100, 
        canny_high_threshold=200, 
        canny_aperture=3,
        use_edge_linking=True, 
        gap_threshold=3, 
        angle_threshold=30, 
        use_morphology=True, 
        morph_kernel_size=3, 
        morph_iterations=1,
        use_component_filter=True, 
        min_component_size=36,
        edge_lower_bound=0.0, 
        edge_upper_bound=1.0, 
        connectivity=1,
        use_adaptive_regions=True, 
        use_enhanced_filtering=True, 
        use_edge_consistency=True,
        target_edge_thickness=1, 
        connectivity_threshold=2
    ):
        # Disable MTEED if custom_controlnet_aux is not available
        if not has_controlnet_aux:
            use_mteed = False
        
        mteed_result = None
        if use_mteed:
            try:
                mteed_model = TEDDetector(model=self.load_model()).to(self.device)
                mteed_result = common_annotator_call(mteed_model, image, resolution=resolution)
                mteed_result = mteed_result.squeeze(0).numpy()
                del mteed_model
            except Exception as e:
                print(f"Error loading or processing with MTEED model: {e}")
                use_mteed = False
        
        enhanced_result = None
        if use_enhanced_edges:
            try:
                enhanced_result = self.process_enhanced_edges(
                    image, use_gpu_acceleration, use_bilateral, bilateral_d, bilateral_sigma_color,
                    bilateral_sigma_space, use_clahe, clip_limit, tile_grid_size, canny_low_threshold,
                    canny_high_threshold, canny_aperture, use_edge_linking, gap_threshold, angle_threshold,
                    use_morphology, morph_kernel_size, morph_iterations, use_component_filter, 
                    min_component_size, resolution, use_adaptive_regions, use_enhanced_filtering,
                    use_edge_consistency, target_edge_thickness, connectivity_threshold
                )
                
                # Mask out intensity and remove small objects
                enhanced_result = get_intensity_mask(enhanced_result, edge_lower_bound, edge_upper_bound)
                cleaned = morphology.remove_small_objects(
                    enhanced_result.astype(bool),
                    min_size=min_component_size,
                    connectivity=connectivity
                )
                enhanced_result = enhanced_result * cleaned
            except Exception as e:
                print(f"Error in enhanced edge processing: {e}")
                use_enhanced_edges = False
        
        # If neither MTEED nor enhanced edges is enabled/successful
        if not use_mteed and not use_enhanced_edges:
            print("Both edge detection methods disabled or failed. Returning unprocessed image.")
            return (image,)
        
        # If only MTEED is active
        if use_mteed and not use_enhanced_edges:
            return (torch.tensor(mteed_result).unsqueeze(0),)
        
        # If only enhanced edges is active
        if use_enhanced_edges and not use_mteed:
            return (torch.tensor(enhanced_result).unsqueeze(0),)
        
        # Otherwise, blend them
        mteed_h, mteed_w = mteed_result.shape[:2]
        enhanced_result_resized = cv2.resize(
            enhanced_result, (mteed_w, mteed_h), interpolation=cv2.INTER_LINEAR
        )
        final_result = combine_layers(mteed_result, enhanced_result_resized)
        
        return (torch.tensor(final_result).unsqueeze(0),)

    def process_enhanced_edges(
        self, image, use_gpu_acceleration, use_bilateral, bilateral_d, bilateral_sigma_color,
        bilateral_sigma_space, use_clahe, clip_limit, tile_grid_size, canny_low_threshold, 
        canny_high_threshold, canny_aperture, use_edge_linking, gap_threshold, angle_threshold,
        use_morphology, morph_kernel_size, morph_iterations, use_component_filter,
        min_component_size, resolution, use_adaptive_regions=True, use_enhanced_filtering=True,
        use_edge_consistency=True, target_edge_thickness=1, connectivity_threshold=2
    ):
        """Preprocess the image, optionally resizing, then run single-image edge processing."""
        # Convert to numpy, scale to 0-255
        img_np = image.squeeze(0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Resize to requested resolution
        h, w = img_np.shape[:2]
        if max(h, w) != resolution:
            if h > w:
                new_h, new_w = resolution, int(resolution * w / h)
            else:
                new_h, new_w = int(resolution * h / w), resolution
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        processed = self.process_single_image(
            img_np, use_gpu_acceleration, use_bilateral, bilateral_d, 
            bilateral_sigma_color, bilateral_sigma_space, use_clahe, clip_limit, tile_grid_size,
            canny_low_threshold, canny_high_threshold, canny_aperture, use_edge_linking, 
            gap_threshold, angle_threshold, use_morphology, morph_kernel_size, morph_iterations,
            use_component_filter, min_component_size, use_adaptive_regions, 
            use_enhanced_filtering, use_edge_consistency, target_edge_thickness, connectivity_threshold
        )
        
        return processed.astype(np.float32) / 255.0

    def process_single_image(
        self, img, use_gpu_acceleration, use_bilateral, bilateral_d, 
        bilateral_sigma_color, bilateral_sigma_space, use_clahe, clip_limit, tile_grid_size,
        canny_low_threshold, canny_high_threshold, canny_aperture, use_edge_linking, gap_threshold,
        angle_threshold, use_morphology, morph_kernel_size, morph_iterations, use_component_filter,
        min_component_size, use_adaptive_regions=True, use_enhanced_filtering=True,
        use_edge_consistency=True, target_edge_thickness=1, connectivity_threshold=2
    ):
        """Main single-image pipeline similar to the 'enhanced_edge.py' approach, with MTEED retained."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if use_adaptive_regions:
            saliency_mask = self.create_saliency_mask(img)
            fg_mask = saliency_mask
            bg_mask = cv2.bitwise_not(saliency_mask)

        # Bilateral filtering
        if use_bilateral:
            gray = self.bilateral_filter(gray, use_gpu_acceleration, bilateral_d,
                                         bilateral_sigma_color, bilateral_sigma_space)

        # CLAHE
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            gray = clahe.apply(gray)

        # Canny
        edges = cv2.Canny(
            gray, canny_low_threshold, canny_high_threshold, 
            apertureSize=canny_aperture, L2gradient=True
        )

        # Edge linking
        if use_edge_linking:
            edges = self._apply_edge_linking(edges, gap_threshold, angle_threshold)

        # Morph
        if use_morphology:
            k = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=morph_iterations)

        # Adaptive regions
        if use_adaptive_regions:
            fe = edges.copy()
            be = edges.copy()
            if use_component_filter:
                if use_enhanced_filtering:
                    fe = self._enhanced_filter_small_components(
                        fe, min_size=max(5, min_component_size // 2), 
                        connectivity_threshold=connectivity_threshold
                    )
                    be = self._enhanced_filter_small_components(
                        be, min_size=min_component_size * 2, 
                        connectivity_threshold=1
                    )
                else:
                    fe = self._filter_small_components(fe, min_size=max(5, min_component_size // 2))
                    be = self._filter_small_components(be, min_component_size * 2)

            fg_masked = cv2.bitwise_and(fe, fe, mask=fg_mask)
            bg_masked = cv2.bitwise_and(be, be, mask=bg_mask)
            edges = cv2.bitwise_or(fg_masked, bg_masked)

        elif use_component_filter:
            if use_enhanced_filtering:
                edges = self._enhanced_filter_small_components(
                    edges, min_size=min_component_size, connectivity_threshold=connectivity_threshold
                )
            else:
                edges = self._filter_small_components(edges, min_component_size)

        # Edge consistency
        if use_edge_consistency:
            edges = self._enhance_edge_consistency(edges, target_thickness=target_edge_thickness)

        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    def bilateral_filter(self, gray, use_gpu, d, sigma_c, sigma_s):
        """Select CPU or GPU bilateral filtering."""
        if use_gpu and self.has_torch_cuda:
            return self.bilateral_filter_torch(gray, d, sigma_c, sigma_s)
        return cv2.bilateralFilter(gray, d, sigma_c, sigma_s)

    def bilateral_filter_torch(self, gray, d, sigma_c, sigma_s):
        """GPU-based bilateral filter via PyTorch approximation."""
        try:
            timg = torch.from_numpy(gray).float().cuda()
            k = d if d % 2 == 1 else d + 1
            pad = k // 2
            g = torch.arange(k, device='cuda').float() - pad
            gx, gy = torch.meshgrid(g, g, indexing='ij')
            skernel = torch.exp(-(gx**2 + gy**2)/(2*sigma_s**2))
            skernel /= skernel.sum()
            padded = torch.nn.functional.pad(timg[None, None], (pad,pad,pad,pad), mode='reflect')
            blurred = torch.nn.functional.conv2d(padded, skernel[None, None]).squeeze()
            diff = timg - blurred
            iweight = torch.exp(-(diff**2)/(2*sigma_c**2))
            result = blurred * iweight + timg * (1 - iweight)
            return result.clamp(0, 255).byte().cpu().numpy()
        except Exception as e:
            print(f"GPU bilateral failed: {e}, falling back to CPU.")
            return cv2.bilateralFilter(gray, d, sigma_c, sigma_s)

    def _apply_edge_linking(self, edge_image, gap_threshold=3, angle_threshold=30):
        """Optimized local bounding-box approach for endpoint matching."""
        linked = edge_image.copy()
        bin_img = (edge_image > 0).astype(np.uint8)
        gx = cv2.Sobel(bin_img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(bin_img, cv2.CV_64F, 0, 1, ksize=3)
        direction = np.arctan2(gy, gx) * 180 / np.pi
        
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
        conv = cv2.filter2D(bin_img, -1, kernel)
        endpoints = np.where(conv == 11)
        
        endpoint_mask = np.zeros_like(bin_img)
        endpoint_mask[endpoints] = 1

        for i in range(len(endpoints[0])):
            y1, x1 = endpoints[0][i], endpoints[1][i]
            if linked[y1, x1] == 0:
                continue
            dir1 = direction[y1, x1]
            best_match, min_dist = None, gap_threshold + 1
            ylo, yhi = max(0, y1 - gap_threshold), min(bin_img.shape[0], y1 + gap_threshold + 1)
            xlo, xhi = max(0, x1 - gap_threshold), min(bin_img.shape[1], x1 + gap_threshold + 1)
            local_endpoints = np.where(endpoint_mask[ylo:yhi, xlo:xhi] == 1)
            for idx in range(len(local_endpoints[0])):
                ly = local_endpoints[0][idx] + ylo
                lx = local_endpoints[1][idx] + xlo
                if (ly == y1 and lx == x1) or linked[ly, lx] == 0:
                    continue
                dist = np.hypot(ly - y1, lx - x1)
                if dist <= gap_threshold:
                    dir2 = direction[ly, lx]
                    adiff = abs(((dir1 - dir2 + 180) % 360) - 180)
                    if adiff <= angle_threshold and dist < min_dist:
                        min_dist = dist
                        best_match = (ly, lx)
            if best_match:
                cv2.line(linked, (x1, y1), (best_match[1], best_match[0]), 255, 1)
        return linked

    def _filter_small_components(self, binary_image, min_size=20):
        """Remove small connected components."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        out = np.zeros_like(binary_image)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                out[labels == i] = 255
        return out

    def _enhanced_filter_small_components(self, binary_image, min_size=20, connectivity_threshold=2):
        """Improved small component filtering that preserves important structures."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        out = np.zeros_like(binary_image)
        big = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                big.append(i)
                out[labels == i] = 255
        kernel = np.ones((3,3), np.uint8)
        dil_big = cv2.dilate(out, kernel, iterations=connectivity_threshold)
        for i in range(1, n):
            if i not in big:
                small_mask = np.zeros_like(binary_image)
                small_mask[labels == i] = 255
                if np.any(np.logical_and(small_mask, dil_big)):
                    out[labels == i] = 255
        return out

    def _enhance_edge_consistency(self, edge_img, target_thickness=1):
        """Improve edge consistency by skeletonizing, then optionally thickening."""
        bin_img = (edge_img > 0).astype(np.uint8)*255
        kernel = np.ones((3,3), np.uint8)
        dil = cv2.dilate(bin_img, kernel, iterations=1)
        skel = self._skeletonize_cv2(dil)
        if target_thickness > 1:
            return cv2.dilate(skel, kernel, iterations=target_thickness - 1)
        return skel

    def _skeletonize_cv2(self, img):
        """Zhang-Suen thinning for a uniform 1-pixel skeleton."""
        sk = np.zeros(img.shape, np.uint8)
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        size = img.size
        done = False
        temp = img.copy()
        while not done:
            eroded = cv2.erode(temp, elem)
            tmp = cv2.dilate(eroded, elem)
            tmp = cv2.subtract(temp, tmp)
            sk = cv2.bitwise_or(sk, tmp)
            temp = eroded
            if size - cv2.countNonZero(temp) == size:
                done = True
        return sk

    def create_saliency_mask(self, image):
        """Saliency-based foreground mask for adaptive regions."""
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(image_rgb)
        saliency_map = (saliency_map * 255).astype(np.uint8)
        _, thresholded = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return cleaned

NODE_CLASS_MAPPINGS = {
    "VKriezHybridEdgePreprocessor": VKriezHybridEdgePreprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VKriezHybridEdgePreprocessor": "VKriez Hybrid MTEED Edge Preprocessor"
}
