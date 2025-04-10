
#### Third Party Lib's #####
import torch
import numpy as np

#### Services #####
from EUP.services.tensor import TensorService

class DenoiseService():

    def __init__(self):
        self.tensorService = TensorService()
    
    ### SMH - Smooth Denoinse Strategy 
    def generateDenoiseMaskforSMHStrategy(self, tileSize, min_value=0.95, max_value=1.0, falloff=0.30, device='cpu'):
        """
        Generates a denoise mask with a reversed radial gradient, where the borders are min_value and
        the center is max_value, with a smooth transition.

        Parameters:
            tile_h (int): The height of the tile.
            tile_w (int): The width of the tile.
            min_value (float): The minimum value for the mask (border).
            max_value (float): The maximum value for the mask (center).
            falloff (float): Controls how smoothly the mask transitions from border to center.
            device (str): The device to place the tensor on ('cpu' or 'cuda').

        Returns:
            torch.Tensor: The generated denoise mask.
        """
        w, h = tileSize
        center_x, center_y = w // 2, h // 2

        # Create grid for the tile's coordinates
        y, x = np.ogrid[:h, :w]

        # Calculate the distance from each point to the center
        dist_x = np.abs(x - center_x)
        dist_y = np.abs(y - center_y)
        dist = np.maximum(dist_x, dist_y)

        # Normalize the distance (max_dist is half the size of the larger dimension)
        max_dist = np.max([w, h]) / 2
        normalized_dist = dist / max_dist

        # Create the radial gradient where the center is max_value and the borders are min_value
        gradient = max_value - (normalized_dist ** falloff) * (max_value - min_value)

        # Convert to a torch tensor
        mask_tensor = torch.tensor(gradient, dtype=torch.float32, device=device)

        return mask_tensor
    
    ### ASADP - Aspect Adaption Denoinse Strategy
    def generateDenoiseMaskforASADPStrategy(self, tileSize, min_value=0.95, max_value=1.0,
                                            base_falloff=0.30, device='cpu'):
        """
        Generates a denoise mask with a reversed radial gradient, and aspect-ratio-aware adaptation
        for thin tiles. The center is max_value, the borders are min_value, with smooth adaptive transition.

        Parameters:
            tile_h (int): Height of the tile.
            tile_w (int): Width of the tile.
            min_value (float): Minimum value at borders.
            max_value (float): Maximum value at center.
            base_falloff (float): Base falloff exponent for the radial gradient.
            device (str): Device for the returned tensor ('cpu' or 'cuda').

        Returns:
            torch.Tensor: A [1, 1, H, W] shaped denoise mask tensor.
        """
        w, h = tileSize
        center_x, center_y = w // 2, h // 2

        # Grid for pixel coordinates
        y, x = np.ogrid[:h, :w]

        # Chebyshev distance from center (squared look, fast computation)
        dist_x = np.abs(x - center_x)
        dist_y = np.abs(y - center_y)
        dist = np.maximum(dist_x, dist_y)

        # Normalize distance
        max_dist = max(w, h) / 2
        normalized_dist = dist / max_dist

        # Adjust falloff based on aspect ratio
        aspect_ratio = w / h
        if aspect_ratio > 2.0 or aspect_ratio < 0.5:
            falloff = base_falloff * 0.7  # Smoothen transition more for thin tiles
        else:
            falloff = base_falloff

        # Apply reversed radial formula
        gradient = max_value - (normalized_dist ** falloff) * (max_value - min_value)
        gradient = np.clip(gradient, min_value, max_value)

        # Convert to torch tensor with shape [1, 1, H, W]
        mask_tensor = torch.tensor(gradient, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        return mask_tensor

        
    def generateDenoiseMask_atBoundary(self, latentImage, tileSize, mask, device='cpu'):
        
        latentTensor = self.tensorService.getTensorfromLatentImage(latentImage)
        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        x, y, tile_w, tile_h = tileSize
        
        # Check if a mask is needed at the boundary
        if (tile_h == tile_h or tile_h == H) and (tile_w == tile_w or tile_w == W):
            return ((x, y, tile_w, tile_h), mask)
        
        # Calculate the offsets
        h_offset = min(0, H - (y + tile_h))
        w_offset = min(0, W - (x + tile_w))
        
        # Create a new mask tensor
        new_mask = torch.zeros((B, C, tile_h, tile_w), dtype=latentTensor.dtype, device=device)

        # Apply the mask or create a new one if none is provided
        new_mask[:, :, -h_offset:tile_h if h_offset == 0 else tile_h, -w_offset:tile_w if w_offset == 0 else tile_w] = 1.0 if mask is None else mask
        
        return ((x + w_offset, y + h_offset, tile_w, tile_h), new_mask)