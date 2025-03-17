import cv2
import numpy as np
import torch

class VKriezEnhancedEdgePreprocessor:
    """Enhanced Edge Preprocessor with local endpoint search and simple logging."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_bilateral": ("BOOLEAN", {"default": True}),
                "use_gpu_acceleration": ("BOOLEAN", {"default": True}),
                "bilateral_d": ("INT", {"default": 7, "min": 5, "max": 15, "step": 2}),
                "bilateral_sigma_color": ("FLOAT", {"default": 75.0, "min": 10.0, "max": 200.0, "step": 5.0}),
                "bilateral_sigma_space": ("FLOAT", {"default": 75.0, "min": 10.0, "max": 200.0, "step": 5.0}),
                "use_clahe": ("BOOLEAN", {"default": True}),
                "clip_limit": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5}),
                "tile_grid_size": ("INT", {"default": 8, "min": 2, "max": 16, "step": 1}),
                "canny_low_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "canny_high_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "canny_aperture": ("INT", {"default": 3, "min": 3, "max": 7, "step": 2}),
                "use_edge_linking": ("BOOLEAN", {"default": True}),
                "gap_threshold": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "angle_threshold": ("INT", {"default": 30, "min": 5, "max": 90, "step": 5}),
                "use_morphology": ("BOOLEAN", {"default": True}),
                "morph_kernel_size": ("INT", {"default": 3, "min": 1, "max": 7, "step": 2}),
                "morph_iterations": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1}),
                "use_component_filter": ("BOOLEAN", {"default": True}),
                "min_component_size": ("INT", {"default": 20, "min": 5, "max": 500, "step": 5}),
                "use_adaptive_regions": ("BOOLEAN", {"default": True}),
                "use_enhanced_filtering": ("BOOLEAN", {"default": True}),
                "connectivity_threshold": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1}),
                "use_edge_consistency": ("BOOLEAN", {"default": True}),
                "target_edge_thickness": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edges"
    CATEGORY = "VKriez/image/preprocessors"

    def __init__(self):
        self.has_torch_cuda = torch.cuda.is_available()
        if not self.has_torch_cuda:
            print("Warning: No CUDA support detected. Using CPU.")

    def detect_edges(self, image, **kwargs):
        batch_size = image.shape[0]
        results = []
        print(f"Processing batch of {batch_size} image(s).")
        for i in range(batch_size):
            print(f"  - Start processing image {i+1}/{batch_size}...")
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            processed = self.process_single_image(img_np, **kwargs)
            results.append(torch.from_numpy(processed.astype(np.float32) / 255.0))
            print(f"  - Done processing image {i+1}.")
        print("All images have been processed. Returning result...")
        return (torch.stack(results, dim=0),)

    def bilateral_filter(self, gray, use_gpu, d, sigma_c, sigma_s):
        if use_gpu and self.has_torch_cuda:
            return self.bilateral_filter_torch(gray, d, sigma_c, sigma_s)
        return cv2.bilateralFilter(gray, d, sigma_c, sigma_s)

    def bilateral_filter_torch(self, gray, d, sigma_c, sigma_s):
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

    def process_single_image(
        self, img, use_bilateral, use_gpu_acceleration, bilateral_d, bilateral_sigma_color,
        bilateral_sigma_space, use_clahe, clip_limit, tile_grid_size, canny_low_threshold,
        canny_high_threshold, canny_aperture, use_edge_linking, gap_threshold, angle_threshold,
        use_morphology, morph_kernel_size, morph_iterations, use_component_filter, min_component_size,
        use_adaptive_regions=True, use_enhanced_filtering=True, use_edge_consistency=True,
        target_edge_thickness=1, connectivity_threshold=2
    ):
        print("    Converting to grayscale.")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if use_adaptive_regions:
            print("    Creating saliency mask.")
            saliency_mask = self.create_saliency_mask(img)
            fg_mask = saliency_mask
            bg_mask = cv2.bitwise_not(saliency_mask)

        if use_bilateral:
            print("    Applying bilateral filter.")
            gray = self.bilateral_filter(gray, use_gpu_acceleration, bilateral_d,
                                         bilateral_sigma_color, bilateral_sigma_space)

        if use_clahe:
            print("    Applying CLAHE.")
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            gray = clahe.apply(gray)

        print("    Applying Canny edge detection.")
        edges = cv2.Canny(gray, canny_low_threshold, canny_high_threshold,
                          apertureSize=canny_aperture, L2gradient=True)

        if use_edge_linking:
            print("    Performing edge linking.")
            edges = self._apply_edge_linking(edges, gap_threshold, angle_threshold)

        if use_morphology:
            print("    Performing morphological operations.")
            k = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=morph_iterations)

        if use_adaptive_regions:
            fe = edges.copy()
            be = edges.copy()
            if use_component_filter:
                print("    Filtering components (foreground/background).")
                if use_enhanced_filtering:
                    fe = self._enhanced_filter_small_components(fe, max(5, min_component_size//2), connectivity_threshold)
                    be = self._enhanced_filter_small_components(be, min_component_size*2, 1)
                else:
                    fe = self._filter_small_components(fe, max(5, min_component_size//2))
                    be = self._filter_small_components(be, min_component_size*2)
            fg = cv2.bitwise_and(fe, fe, mask=fg_mask)
            bg = cv2.bitwise_and(be, be, mask=bg_mask)
            edges = cv2.bitwise_or(fg, bg)
        elif use_component_filter:
            print("    Filtering small components.")
            if use_enhanced_filtering:
                edges = self._enhanced_filter_small_components(edges, min_component_size, connectivity_threshold)
            else:
                edges = self._filter_small_components(edges, min_component_size)

        if use_edge_consistency:
            print("    Enhancing edge consistency.")
            edges = self._enhance_edge_consistency(edges, target_edge_thickness)

        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    def _apply_edge_linking(self, edge_image, gap_threshold=3, angle_threshold=30):
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

    def _filter_small_components(self, bin_img, min_size=20):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        out = np.zeros_like(bin_img)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                out[labels == i] = 255
        return out

    def _enhanced_filter_small_components(self, bin_img, min_size=20, conn_thresh=2):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        out = np.zeros_like(bin_img)
        big = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                big.append(i)
                out[labels == i] = 255
        kernel = np.ones((3,3), np.uint8)
        dil_big = cv2.dilate(out, kernel, iterations=conn_thresh)
        for i in range(1, n):
            if i not in big:
                small_mask = np.zeros_like(bin_img)
                small_mask[labels == i] = 255
                if np.any(np.logical_and(small_mask, dil_big)):
                    out[labels == i] = 255
        return out

    def _enhance_edge_consistency(self, edge_img, target_thickness=1):
        bin_img = (edge_img > 0).astype(np.uint8)*255
        kernel = np.ones((3,3), np.uint8)
        dil = cv2.dilate(bin_img, kernel, iterations=1)
        skel = self._skeletonize_cv2(dil)
        if target_thickness > 1:
            return cv2.dilate(skel, kernel, iterations=target_thickness-1)
        return skel

    def _skeletonize_cv2(self, img):
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
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = image
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, sal_map = sal.computeSaliency(rgb)
        sal_map = (sal_map * 255).astype(np.uint8)
        _, th = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k)
        return cleaned

NODE_CLASS_MAPPINGS = {
    "VKriezEnhancedEdgePreprocessor": VKriezEnhancedEdgePreprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VKriezEnhancedEdgePreprocessor": "VKriez Enhanced Edge Preprocessor"
}
