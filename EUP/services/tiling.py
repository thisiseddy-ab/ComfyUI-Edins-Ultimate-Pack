#### Comfy Lib's #####
import comfy.sample
import comfy.sampler_helpers

#### Third Party Lib's #####
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import sobel, gaussian_filter

#### Services #####
from EUP.services.tensor import TensorService

class Tiling_Strategy_Base():

    def __init__(self):
        self.tensorService = TensorService()

    def getTilesfromTilePos(self, tensor, tile_pos):
        allTiles = []
        for t_pos_pass in tile_pos:
            tile_pass_l = []
            for t_pos_group in t_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in t_pos_group:
                    tile_group_l.append(self.tensorService.getSlice(tensor, y, tile_h, x, tile_w))
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    def getNoiseTilesfromTilePos(self, noise_tensor , tile_positions):
        return self.getTilesfromTilePos(noise_tensor, tile_positions)

    def generateRandomNoise(self, latent, seed, disable_noise: bool):
        samples = latent.get("samples")
        shape = samples.shape
        B, C, H, W = shape
        
        if disable_noise:
            return torch.zeros((B, C, H, W), dtype=samples.dtype, layout=samples.layout, device="cpu")
        else:
            batch_inds = latent.get("batch_index")
            return comfy.sample.prepare_noise(samples, seed, batch_inds)
        
    def generateNoiseMask(self, latent, noise):
        noise_mask = latent.get("noise_mask")
        if noise_mask is not None:
            noise_mask = comfy.sampler_helpers.prepare_mask(noise_mask, noise.shape, device='cpu')
        return noise_mask

    def generateGenBlendMask(self, batch_size, tile_h, tile_w,):
            """Creates a smooth blending mask for latent tiles."""
            
            # Grid for radial blending
            grid_x, grid_y = torch.meshgrid(
                torch.linspace(-1, 1, tile_h, device="cpu"),
                torch.linspace(-1, 1, tile_w, device="cpu"),
                indexing="ij",
            )

            # Compute radial distance
            radius = torch.sqrt(grid_x**2 + grid_y**2)
            
            # Apply a cosine-based fade (soft at edges, sharp in center)
            fade_radial = 0.5 * (1 - torch.cos(torch.clamp(radius * torch.pi, 0, torch.pi)))
            
            # Directional fades to prevent excessive center softening
            fade_x = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, tile_w, device="cpu"))).view(1, 1, 1, -1)
            fade_y = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, tile_h, device="cpu"))).view(1, 1, -1, 1)
            
            # Merge both blending styles
            fade = fade_radial * fade_x * fade_y
            
            # Adjust the mask to avoid too much softening at the center
            fade = torch.clamp(fade, min=0.01)  # Prevent total blackness
            
            # Ensure proper batch size
            mask = fade.expand(batch_size, 1, tile_h, tile_w)
            
            return mask
    
    def generateGenDenoiseMask_Old(self, tile_h, tile_w, border_strength=0.65, falloff=0.75, device='cpu'):
        """
        Generates a general denoise mask that reduces changes at the borders to prevent seams.

        Parameters:
            tile_h (int): The height of the tile.
            tile_w (int): The width of the tile.
            border_strength (float): The intensity of the border preservation (higher = less change at borders).
            falloff (float): Controls how smoothly the mask transitions from border to center.
            device (str): The device to place the tensor on ('cpu' or 'cuda').

        Returns:
            torch.Tensor: The generated denoise mask.
        """
        h = tile_h
        w = tile_w
        y, x = np.ogrid[:h, :w]

        # Distance from borders
        dist_x = np.minimum(x, w - 1 - x)
        dist_y = np.minimum(y, h - 1 - y)
        dist = np.minimum(dist_x, dist_y)

        # Normalize distance to [0, 1] range
        max_dist = np.min([w, h]) / 2
        normalized_dist = dist / max_dist

        # Apply border_strength and falloff to control the mask
        mask = np.clip((normalized_dist ** falloff) * border_strength, 0, 1)

        # Convert to torch tensor
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

        return mask_tensor
    
    def generateGenDenoiseMask(self, tile_h, tile_w, min_value=0.25, max_value=0.85, falloff=0.30, device='cpu'):
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
        h = tile_h
        w = tile_w
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

        # Print the top border (first few rows)
        #print("Top Border:")
        #print(gradient[:5, :])

        # Print the bottom border (last few rows)
       # print("Bottom Border:")
        #print(gradient[-5:, :])

        # Print the left border (first few columns)
        #print("Left Border:")
        #print(gradient[:, :5])

        # Print the right border (last few columns)
        #print("Right Border:")
        #print(gradient[:, -5:])

        # Print the center region (middle of the mask)
        #print("Center Region (middle part of the mask):")
        #print(gradient[gradient.shape[0]//2-5:gradient.shape[0]//2+5, gradient.shape[1]//2-5:gradient.shape[1]//2+5])

        # Convert to a torch tensor
        mask_tensor = torch.tensor(gradient, dtype=torch.float32, device=device)

        return mask_tensor

        
    def generateDenoiseMask_atBoundary(self, h, h_len, w, w_len, tile_w, tile_h, latent_size_h, latent_size_w, mask, device='cpu'):
        # Check if a mask is needed at the boundary
        if (h_len == tile_h or h_len == latent_size_h) and (w_len == tile_w or w_len == latent_size_w):
            return ((w, h, w_len, h_len), mask)
        
        # Calculate the offsets
        h_offset = min(0, latent_size_h - (h + tile_h))
        w_offset = min(0, latent_size_w - (w + tile_w))
        
        # Create a new mask tensor
        new_mask = torch.zeros((1, 1, tile_h, tile_w), dtype=torch.float32, device=device)

        # Apply the mask or create a new one if none is provided
        new_mask[:, :, -h_offset:h_len if h_offset == 0 else tile_h, -w_offset:w_len if w_offset == 0 else tile_w] = 1.0 if mask is None else mask
        
        return ((w + w_offset, h + h_offset, tile_w, tile_h), new_mask)

    def getDenoiseMaskTiles(self, noise_mask, tile_pos,):
        allTiles = []
        for tile_pos_pass in tile_pos:
            tile_pass_l = []
            for tile_pos_group in tile_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in tile_pos_group:
                    tile_mask = self.generateGenDenoiseMask(tile_h, tile_w)
                    tiled_mask = None
                    if noise_mask is not None:
                        tiled_mask = self.tensorService.getSlice(noise_mask, y, tile_h, x, tile_w)
                    if tile_mask is not None:
                        tiled_mask = tiled_mask * tile_mask if tiled_mask is not None else tile_mask
                    tile_group_l.append(tiled_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles

    def getBlendMaskTiles(self, tile_pos, B):
        allTiles = []
        for t_pos_pass in tile_pos:
            tile_pass_l = []
            for t_pos_group in t_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in t_pos_group:
                    blend_mask = self.generateGenBlendMask(B, tile_h, tile_w)
                    tile_group_l.append(blend_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
###### Singel Pass - Simple Tiling Strategy ######  
class STService(Tiling_Strategy_Base):
    
    def __init__(self):
        super().__init__()

    def generatePosforSTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int):
        """
        Generates a single pass of simple tiling with the same nested structure as multi-pass.
        """

        # Ensure tile sizes fit the dimensions
        num_tiles_x = max(1, latent_width // tile_width)
        num_tiles_y = max(1, latent_height // tile_height)

        adjusted_tile_width = latent_width // num_tiles_x
        adjusted_tile_height = latent_height // num_tiles_y

        # Base positions for normal tiling
        x_positions = list(range(0, latent_width, adjusted_tile_width))
        y_positions = list(range(0, latent_height, adjusted_tile_height))

        # Generate tile positions in the same structure as multi-pass
        pass_tiles = [[(x, y, adjusted_tile_width, adjusted_tile_height) for x in x_positions for y in y_positions]]

        return [pass_tiles]

    def getTilesforSTStrategy(self,samples, tile_positions):
        return self.getTilesfromTilePos(samples, tile_positions)
    
    def getNoiseTilesforSTStrategy(self, noise_tensor , tile_positions):
        return self.getTilesfromTilePos(noise_tensor, tile_positions)
    
    def generateDenoiseMaskforSTStrategy(self, tile_h, tile_w, border_strength=1.0, falloff=0.5 ):
        return self.generateGenDenoiseMask(tile_h, tile_w, border_strength, falloff)
    
    def getDenoiseMaskTilesforSTStrategy(self, noise_mask, tile_pos):
        return self.getDenoiseMaskTiles(noise_mask, tile_pos)

    def getBlendMaskTilesforSTStrategy(self, tile_pos, B):
        return self.getBlendMaskTiles(tile_pos, B)
    
###### Multi-Pass - Simple Tiling Strategy ######   
class MP_STService(Tiling_Strategy_Base):
    
    def __init__(self):
        super().__init__()

        # st - Stands for Simple Tiling Streteg
        self.st_Service = STService()

    def generatePosforMP_STStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, passes: int = 2):
        """
        Generates multiple passes of simple tiling with alternating shifts.
        """

        allTiles = []
        for i in range(passes):
            tile_pos = self.st_Service.generatePosforSTStrategy(latent_width, latent_height, tile_width, tile_height)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles 
    
    def getTilesforMP_STStrategy(self, samples, tile_pos_and_steps):
        return self.st_Service.getTilesforSTStrategy(samples, tile_pos_and_steps)
    
    def getNoiseTilesforMP_STStrategy(self, noise_tensor , tile_pos_and_steps):
        return self.st_Service.getNoiseTilesforSTStrategy(noise_tensor, tile_pos_and_steps)
    
    def getDenoiseMaskTilesforMP_STStrategy(self, noise_mask, tile_pos):
        return self.st_Service.getDenoiseMaskTilesforSTStrategy(noise_mask, tile_pos)
    
    def getBlendMaskTilesforMP_STStrategy(self, tile_pos, B):
        return self.st_Service.getBlendMaskTilesforSTStrategy(tile_pos, B)

###### Singel Pass - Random Tiling Strategy ######   
class RTService(Tiling_Strategy_Base):
    
    def __init__(self):
        super().__init__()

    def generatePosforRTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, seed):
        generator: torch.Generator = torch.manual_seed(seed)

        # Generate random jitter offsets
        rands = torch.rand((2,), dtype=torch.float32, generator=generator).numpy()
        jitter_w = int(rands[0] * tile_width)
        jitter_h = int(rands[1] * tile_height)

        # Compute tile positions with jittered offsets
        tiles_h = self.calcCoords(latent_height, tile_height, jitter_h)
        tiles_w = self.calcCoords(latent_width, tile_width, jitter_w)

        # Ensure the same nested structure: [[[ (x, y, w, h), (x, y, w, h), ... ]]]
        tile_positions = [[[(w_start, h_start, w_size, h_size) 
                            for h_start, h_size in tiles_h 
                            for w_start, w_size in tiles_w]]]

        return tile_positions
    
    def calcCoords(self, latent_size, tile_size, jitter):
        tile_coords = int((latent_size + jitter - 1) // tile_size + 1)
        tile_coords = [np.clip(tile_size * c - jitter, 0, latent_size) for c in range(tile_coords + 1)]
        tile_coords = [(c1, c2 - c1) for c1, c2 in zip(tile_coords, tile_coords[1:])]
        return tile_coords
    
    def getTilesforRTStrategy(self,samples, tile_pos):
        return self.getTilesfromTilePos(samples, tile_pos)
    
    def getNoiseTilesforRTStrategy(self, noise_tensor, tile_pos):
        return self.getTilesfromTilePos(noise_tensor, tile_pos)
    
    
    def generateDenoiseMaskforRTStrategy(self, tile_h, tile_w, border_strength=0.85, falloff=0.45, device='cpu'):
        """
        Generates a denoise mask that dynamically adjusts the strength for thin rectangular tiles and preserves borders to avoid seams.

        Parameters:
            tile_h (int): Tile height.
            tile_w (int): Tile width.
            border_strength (float): Border preservation strength (higher = less change at borders).
            falloff (float): Smoothness of transition from border to center.
            device (str): Device to place the tensor on ('cpu' or 'cuda').

        Returns:
            torch.Tensor: The denoise mask with shape [1, 1, tile_h, tile_w].
        """
        h = tile_h
        w = tile_w
        y, x = np.ogrid[:h, :w]

        # Distance from borders
        dist_x = np.minimum(x, w - 1 - x)
        dist_y = np.minimum(y, h - 1 - y)
        dist = np.minimum(dist_x, dist_y)

        # Normalize distance to [0, 1] range
        max_dist = np.min([w, h]) / 2
        normalized_dist = dist / max_dist

        # Apply border_strength and falloff to control the mask
        base_mask = np.clip((normalized_dist ** falloff) * border_strength, 0, 1)

        # Calculate aspect ratio of the tile (width / height)
        aspect_ratio = w / h

        # Adjust denoise strength based on aspect ratio:
        # If the tile is thin (high aspect ratio), reduce the denoise strength
        if aspect_ratio > 2.0:  # Horizontal tiles (aspect ratio > 2)
            scale_factor = 0.5  # Lower the denoise strength for thin tiles
        elif aspect_ratio < 0.5:  # Vertical tiles (aspect ratio < 0.5)
            scale_factor = 0.5  # Lower the denoise strength for thin tiles
        else:
            scale_factor = 1.0  # No adjustment for more square or balanced tiles

        # Apply dynamic scaling to adjust mask strength for thin tiles
        adjusted_mask = base_mask * scale_factor

        # Convert to torch tensor and expand dimensions to [1, 1, tile_h, tile_w]
        mask_tensor = torch.tensor(adjusted_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        return mask_tensor
    
    def getDenoiseMaskTilesforRTStrategy(self, noise_mask, tile_pos):
        allTiles = []
        for tile_pos_pass in tile_pos:
            tile_pass_l = []
            for tile_pos_group in tile_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in tile_pos_group:
                    tile_mask = self.generateDenoiseMaskforRTStrategy(tile_h, tile_w)
                    tiled_mask = None
                    if noise_mask is not None:
                        tiled_mask = self.tensorService.getSlice(noise_mask, y, tile_h, x, tile_w)
                    if tile_mask is not None:
                        if tiled_mask is not None:
                            tiled_mask *= tiled_mask
                        else:
                            tiled_mask = tile_mask      
                    tile_group_l.append(tiled_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    def getBlendMaskTilesforRTStrategy(self, tile_pos, B):
        return self.getBlendMaskTiles(tile_pos, B)
    
###### Multi-Pass - Random Tiling Strategy ######   
class MP_RTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()
        self.rt_Service = RTService() 

    def generatePos_forMP_RTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, seed, passes: int = 2):
        """
        Generates multiple passes of random tiling with alternating jitter logic.
        Matches the nested list structure of generatePos_forMP_CPTStrategy().
        """
        allTiles = []
        for i in range(passes):
            tile_pos = self.rt_Service.generatePosforRTStrategy(latent_width, latent_height, tile_width, tile_height, seed)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles 

    def getTilesforMP_RTStrategy(self, samples, tile_pos):
        return self.rt_Service.getTilesforRTStrategy(samples, tile_pos)
    
    def getNoiseTilesforMP_RTStrategy(self, noise_tensor, tile_pos):
        return self.rt_Service.getNoiseTilesforRTStrategy(noise_tensor, tile_pos)
    
    def getDenoiseMaskTilesforMP_RTStrategy(self, noise_mask, tile_pos):
        return self.rt_Service.getDenoiseMaskTilesforRTStrategy(noise_mask, tile_pos)
    
    def getBlendMaskTilesforMP_RTStrategy(self, tile_pos, B):
        return self.rt_Service.getBlendMaskTilesforRTStrategy(tile_pos, B)

###### Singel Pass - Padded Tiling Strategy ######
class PTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

    def generatePosforPTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, padding: int):
        # Calculate how many tiles fit in each dimension
        num_tiles_x = latent_width // tile_width
        num_tiles_y = latent_height // tile_height

        # Adjust tile size to fit perfectly
        adjusted_tile_width = latent_width // num_tiles_x
        adjusted_tile_height = latent_height // num_tiles_y

        tile_positions = []

        for j in range(num_tiles_y):
            row_positions = []
            for i in range(num_tiles_x):
                x = i * adjusted_tile_width
                y = j * adjusted_tile_height

                # Apply padding to all sides
                left_pad = padding
                right_pad = padding
                top_pad = padding
                bottom_pad = padding

                # Adjust tile size with padding
                padded_tile_width = adjusted_tile_width + left_pad + right_pad
                padded_tile_height = adjusted_tile_height + top_pad + bottom_pad

                # Clamping the positions to ensure tiles stay within bounds and prevent overlap
                adjusted_x = max(0, min(x - left_pad, latent_width - padded_tile_width))
                adjusted_y = max(0, min(y - top_pad, latent_height - padded_tile_height))

                row_positions.append((adjusted_x, adjusted_y, padded_tile_width, padded_tile_height))

            tile_positions.append(row_positions)

        return [tile_positions]

    def getTilesforPTStrategy(self, samples, tile_positions):
        return self.getTilesfromTilePos(samples, tile_positions)

    def applyPaddingforPTStrategy(self, tile, padding, strategy, pos_x, pos_y, tile_positions):
        """
        Applies padding to a tile using the multi-pass PTS strategy.
        """
        # Apply padding uniformly to all sides
        left_pad = padding
        right_pad = padding
        top_pad = padding
        bottom_pad = padding

        # Apply padding using the chosen strategy
        if strategy in ["circular", "reflect", "replicate"]:
            return F.pad(tile, (left_pad, right_pad, top_pad, bottom_pad), mode=strategy)
        elif strategy == "organic":
            return self.applyOrganicPadding(tile, (left_pad, right_pad, top_pad, bottom_pad))
        else:
            return F.pad(tile, (left_pad, right_pad, top_pad, bottom_pad), mode="constant", value=0)

    def getPaddedTilesforPTStrategy(self, tiled_tiles, tile_pos, padding_strategy, padding):
        """
        Applies padding to all tiles using the multi-pass PTS strategy.
        """
        allTiles = []
        for tile_pos_pass, tiles_pass in zip(tile_pos, tiled_tiles):
            tile_pass_l = []
            for tile_pos_group, tiles_group in zip(tile_pos_pass, tiles_pass):
                tile_positions = [pos for pos in tile_pos_group] 
                tile_group_l = []
                for (x, y, tile_w, tile_h), tile in zip(tile_pos_group, tiles_group):
                    padded_tile = self.applyPaddingforPTStrategy(tile, padding, padding_strategy, x, y, tile_positions)
                    tile_group_l.append(padded_tile)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)

        return allTiles
    
    def applyOrganicPadding(self, tile, pad):
        """Applies seamless mirrored padding with soft blending and natural variation, avoiding boxy artifacts."""
        B, C, H, W = tile.shape
        
        # Unpack padding for all sides
        left, right, top, bottom = pad

        # Create the padded tile
        padded_tile = torch.zeros((B, C, H + top + bottom, W + left + right), device=tile.device, dtype=tile.dtype)

        # Place the original tile in the center
        padded_tile[:, :, top:top + H, left:left + W] = tile

        # Apply mirrored padding for all sides
        if left > 0:
            left_patch = self.softBlend(tile[:, :, :, :left], (B, C, H, left), flip_dims=[3], device=tile.device, blend_factor=0.55)
            padded_tile[:, :, top:top + H, :left] = left_patch

        if right > 0:
            right_patch = self.softBlend(tile[:, :, :, -right:], (B, C, H, right), flip_dims=[3], device=tile.device, blend_factor=0.55)
            padded_tile[:, :, top:top + H, -right:] = right_patch

        if top > 0:
            top_patch = self.softBlend(tile[:, :, :top, :], (B, C, top, W), flip_dims=[2], device=tile.device, blend_factor=0.55)
            padded_tile[:, :, :top, left:left + W] = top_patch

        if bottom > 0:
            bottom_patch = self.softBlend(tile[:, :, -bottom:, :], (B, C, bottom, W), flip_dims=[2], device=tile.device, blend_factor=0.55)
            padded_tile[:, :, -bottom:, left:left + W] = bottom_patch

        # Corner regions (blend both X and Y for smooth edges)
        if left > 0 and top > 0:
            padded_tile[:, :, :top, :left] = self.softBlend(tile[:, :, :top, :left], (B, C, top, left), flip_dims=[2, 3], device=tile.device, blend_factor=0.35)

        if right > 0 and top > 0:
            padded_tile[:, :, :top, -right:] = self.softBlend(tile[:, :, :top, -right:], (B, C, top, right), flip_dims=[2, 3], device=tile.device, blend_factor=0.35)

        if left > 0 and bottom > 0:
            padded_tile[:, :, -bottom:, :left] = self.softBlend(tile[:, :, -bottom:, :left], (B, C, bottom, left), flip_dims=[2, 3], device=tile.device, blend_factor=0.35)

        if right > 0 and bottom > 0:
            padded_tile[:, :, -bottom:, -right:] = self.softBlend(tile[:, :, -bottom:, -right:], (B, C, bottom, right), flip_dims=[2, 3], device=tile.device, blend_factor=0.35)

        return padded_tile
    
    def softBlend(self, patch, expand_shape, flip_dims, device,  blend_factor=0.7):
        """Mirrors a patch and blends it softly with adaptive randomness."""
        mirrored = patch
        for flip_dim in flip_dims:
            mirrored = torch.flip(mirrored, dims=[flip_dim])  # Mirror flip in both directions

        # Adaptive blending factor (less boxy)
        blend_factor = torch.tensor(blend_factor, device=device).expand_as(patch)

        # Perlin-style noise (natural variation, no harsh edges)
        noise = (torch.rand_like(patch) - 0.5) * 1.5  # Random noise in [-0.5, 0.5]
        blend_factor = torch.clamp(blend_factor + noise, 0.4, 0.5)  # Adaptive range

        return (patch * blend_factor + mirrored * (1 - blend_factor)).expand(expand_shape)
    
    def getNoiseTilesforPTStrategy(self, latent, padded_tiles, seed, disable_noise: bool):
        samples = latent.get("samples")
        shape = samples.shape
        B, C, H, W = shape
        allTiles = []
        for tiles_pass in padded_tiles:
            tile_pass_l = []
            for tiles_group in tiles_pass:
                tile_group_l = []
                for padded_tile in tiles_group:
                    tile_shape = padded_tile.shape
                    x, y, h, w = tile_shape
                    if disable_noise:
                        tile_group_l.append(torch.zeros((B, C, h, w), dtype=samples.dtype, layout=samples.layout, device="cpu"))
                    else:
                        tile_group_l.append(comfy.sample.prepare_noise(padded_tile, seed, B))
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    def getDenoiseMaskTilesforPTStrategy(self, noise_mask, tile_pos, padded_tiles):
        allTiles = []
        for tile_pos_pass, tiles_pass in zip(tile_pos, padded_tiles):
            tile_pass_l = []
            for tile_pos_group, tiles_group in zip(tile_pos_pass, tiles_pass):
                tile_group_l = []
                for (x, y, tile_w, tile_h), padded_tile in zip(tile_pos_group, tiles_group):
                    padded_h = padded_tile.shape[-2]
                    padded_w = padded_tile.shape[-1]       
                    tile_mask = self.generateGenDenoiseMask(padded_h, padded_w)
                    tiled_mask = None
                    if noise_mask is not None:
                        tiled_mask = self.tensorService.getSlice(noise_mask, y, padded_h, x, padded_w)
                    if tile_mask is not None:
                        if tiled_mask is not None:
                            tiled_mask *= tiled_mask
                        else:
                            tiled_mask = tile_mask      
                    tile_group_l.append(tiled_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    def getBlendMaskTilesforPTStrategy(self, padded_tiles, B):
        allTiles = []
        for tiles_pass in padded_tiles:
            tile_pass_l = []
            for tiles_group in tiles_pass:
                tile_group_l = []
                for padded_tile in tiles_group:
                    tile_shape = padded_tile.shape
                    x, y, h, w = tile_shape
                    tile_group_l.append(self.generateGenBlendMask(B, h, w))
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles

###### Multi-Pass - Padded Tiling Strategy ######  
class MP_PTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()
        self.pt_Service = PTService()

    def generatePos_forMP_PTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, padding: int, passes: int = 2):
        allTiles = []
        for i in range(passes):
            tile_pos = self.pt_Service.generatePosforPTStrategy(latent_width, latent_height, tile_width, tile_height, padding)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles

    def getTilesforMP_PTStrategy(self, samples, tile_pos_and_steps):
        return self.pt_Service.getTilesforPTStrategy(samples, tile_pos_and_steps)

    def getPaddedTilesforMP_PTStrategy(self, tiled_tiles, tile_pos, padding_strategy, padding):
        return self.pt_Service.getPaddedTilesforPTStrategy(tiled_tiles, tile_pos, padding_strategy, padding)

    def getNoiseTilesforMP_PTStrategy(self, latent, padded_tiles, seed, disable_noise: bool):
        return self.pt_Service.getNoiseTilesforPTStrategy(latent, padded_tiles, seed, disable_noise)

    def getDenoiseMaskTilesforMP_PTStrategy(self, noise_mask, tile_pos, padded_tiles):
        return self.pt_Service.getDenoiseMaskTilesforPTStrategy(noise_mask, tile_pos, padded_tiles)

    def getBlendMaskTilesforMP_PTStrategy(self, padded_tiles, B):
        return self.pt_Service.getBlendMaskTilesforPTStrategy(padded_tiles, B)

###### Singel Pass - Adjacency Padded Tiling Strategy ######
class APTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()
    
    def generatePosforAPTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, padding: int):
        # Calculate how many tiles fit in each dimension
        num_tiles_x = latent_width // tile_width
        num_tiles_y = latent_height // tile_height

        # Adjust tile size to fit perfectly
        adjusted_tile_width = latent_width // num_tiles_x
        adjusted_tile_height = latent_height // num_tiles_y

        tile_positions = []

        for j in range(num_tiles_y):
            row_positions = []
            for i in range(num_tiles_x):
                x = i * adjusted_tile_width
                y = j * adjusted_tile_height

                # Adjust for padding
                left_pad = padding if i != 0 else 0  # Padding if not the left-most tile
                right_pad = padding if i != num_tiles_x - 1 else 0  # Padding if not the right-most tile
                top_pad = padding if j != 0 else 0  # Padding if not the top-most tile
                bottom_pad = padding if j != num_tiles_y - 1 else 0  # Padding if not the bottom-most tile

                # Ensure no overflow occurs at the edges (both horizontally and vertically)
                padded_tile_width = adjusted_tile_width + left_pad + right_pad
                padded_tile_height = adjusted_tile_height + top_pad + bottom_pad

                # Clamping the positions to ensure tiles stay within bounds and prevent overlap
                adjusted_x = max(0, min(x - left_pad, latent_width - padded_tile_width))
                adjusted_y = max(0, min(y - top_pad, latent_height - padded_tile_height))

                row_positions.append((adjusted_x, adjusted_y, padded_tile_width, padded_tile_height))

            tile_positions.append(row_positions)  # Maintain the correct nested structure

        return [tile_positions]
    
    def getTilesforAPTStrategy(self, samples, tile_positions):
        return self.getTilesfromTilePos(samples, tile_positions)

    def applyPaddingforAPTStrategy(self, tile, padding, strategy, pos_x, pos_y, tile_positions):
        """
        Applies padding to a tile using the multi-pass PTS strategy.
        """
        # Calculate the number of tiles in x and y directions based on tile_positions
        num_tiles_x = max((x + tile_w) for (x, _, tile_w, _) in tile_positions) // tile.shape[-1] + 1
        num_tiles_y = max((y + tile_h) for (_, y, _, tile_h) in tile_positions) // tile.shape[-2] + 1

        if padding > 0:
            # Add padding for each side based on the tile's position
            left_pad = padding if pos_x != 0 else 0  # Only add padding if the tile is not on the left edge
            right_pad = padding if pos_x != num_tiles_x - 1 else 0  # Only add padding if the tile is not on the right edge
            top_pad = padding if pos_y != 0 else 0  # Only add padding if the tile is not on the top edge
            bottom_pad = padding if pos_y != num_tiles_y - 1 else 0  # Only add padding if the tile is not on the bottom edge

            # Apply padding using the chosen strategy
            if strategy in ["circular", "reflect", "replicate"]:
                return F.pad(tile, (left_pad, right_pad, top_pad, bottom_pad), mode=strategy)
            elif strategy == "organic":
                return self.applyOrganicPadding(tile, (left_pad, right_pad, top_pad, bottom_pad))
            else:
                return F.pad(tile, (left_pad, right_pad, top_pad, bottom_pad), mode="constant", value=0)

        return tile

    def getPaddedTilesforAPTStrategy(self, tiled_tiles, tile_pos, padding_strategy, padding):
        """
        Applies padding to all tiles using the multi-pass PTS strategy.
        """
        allTiles = []
        for tile_pos_pass, tiles_pass in zip(tile_pos, tiled_tiles):
            tile_pass_l = []
            for tile_pos_group, tiles_group in zip(tile_pos_pass, tiles_pass):
                tile_positions = [pos for pos in tile_pos_group] 
                tile_group_l = []
                for (x, y, tile_w, tile_h), tile in zip(tile_pos_group, tiles_group):
                    padded_tile = self.applyPaddingforAPTStrategy(tile, padding, padding_strategy, x, y, tile_positions)
                    tile_group_l.append(padded_tile)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)

        return allTiles
    
    def applyOrganicPadding(self, tile, pad):
        """Applies seamless mirrored padding with soft blending and natural variation, avoiding boxy artifacts."""
        B, C, H, W = tile.shape
        
        # Unpack padding only for the sides where padding exists (values > 0)
        pad_len = len(pad)
        
        # Default to 0 for missing sides (we could append missing sides as 0, so no padding is applied)
        left = pad[0] if pad_len > 0 else 0
        right = pad[1] if pad_len > 1 else 0
        top = pad[2] if pad_len > 2 else 0
        bottom = pad[3] if pad_len > 3 else 0

        # Create the padded tile
        padded_tile = torch.zeros((B, C, H + top + bottom, W + left + right), device=tile.device, dtype=tile.dtype)

        # Place the original tile in the center
        padded_tile[:, :, top:top + H, left:left + W] = tile

        # Left padding (soft blended from left edge)
        if left > 0:
            left_patch = self.softBlend(tile[:, :, :, :left], (B, C, H, left), flip_dims=[3], device=tile.device, blend_factor=0.55)
            padded_tile[:, :, top:top + H, :left] = left_patch

        # Right padding
        if right > 0:
            right_patch = self.softBlend(tile[:, :, :, -right:], (B, C, H, right), flip_dims=[3], device=tile.device, blend_factor=0.55)
            padded_tile[:, :, top:top + H, -right:] = right_patch

        # Top padding
        if top > 0:
            top_patch = self.softBlend(tile[:, :, :top, :], (B, C, top, W), flip_dims=[2], device=tile.device, blend_factor=0.55)
            padded_tile[:, :, :top, left:left + W] = top_patch

        # Bottom padding
        if bottom > 0:
            bottom_patch = self.softBlend(tile[:, :, -bottom:, :], (B, C, bottom, W), flip_dims=[2], device=tile.device, blend_factor=0.55)
            padded_tile[:, :, -bottom:, left:left + W] = bottom_patch

        # Corner regions (blend both X and Y for smooth edges)
        if left > 0 and top > 0:
            padded_tile[:, :, :top, :left] = self.softBlend(tile[:, :, :top, :left], (B, C, top, left), flip_dims=[2, 3], device=tile.device, blend_factor=0.35)

        if right > 0 and top > 0:
            padded_tile[:, :, :top, -right:] = self.softBlend(tile[:, :, :top, -right:], (B, C, top, right), flip_dims=[2, 3], device=tile.device, blend_factor=0.35)

        if left > 0 and bottom > 0:
            padded_tile[:, :, -bottom:, :left] = self.softBlend(tile[:, :, -bottom:, :left], (B, C, bottom, left), flip_dims=[2, 3], device=tile.device, blend_factor=0.35)

        if right > 0 and bottom > 0:
            padded_tile[:, :, -bottom:, -right:] = self.softBlend(tile[:, :, -bottom:, -right:], (B, C, bottom, right), flip_dims=[2, 3], device=tile.device, blend_factor=0.35)

        return padded_tile
        
    def softBlend(self, patch, expand_shape, flip_dims, device,  blend_factor=0.7):
        """Mirrors a patch and blends it softly with adaptive randomness."""
        mirrored = patch
        for flip_dim in flip_dims:
            mirrored = torch.flip(mirrored, dims=[flip_dim])  # Mirror flip in both directions

        # Adaptive blending factor (less boxy)
        blend_factor = torch.tensor(blend_factor, device=device).expand_as(patch)

        # Perlin-style noise (natural variation, no harsh edges)
        noise = (torch.rand_like(patch) - 0.5) * 1.5  # Random noise in [-0.5, 0.5]
        blend_factor = torch.clamp(blend_factor + noise, 0.4, 0.5)  # Adaptive range

        return (patch * blend_factor + mirrored * (1 - blend_factor)).expand(expand_shape)
    
    def getNoiseTilesforAPTStrategy(self, latent, padded_tiles, seed, disable_noise: bool):
        samples = latent.get("samples")
        shape = samples.shape
        B, C, H, W = shape
        allTiles = []
        for tiles_pass in padded_tiles:
            tile_pass_l = []
            for tiles_group in tiles_pass:
                tile_group_l = []
                for padded_tile in tiles_group:
                    tile_shape = padded_tile.shape
                    x, y, h, w = tile_shape
                    if disable_noise:
                        tile_group_l.append(torch.zeros((B, C, h, w), dtype=samples.dtype, layout=samples.layout, device="cpu"))
                    else:
                        tile_group_l.append(comfy.sample.prepare_noise(padded_tile, seed, B))
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    def getDenoiseMaskTilesforAPTStrategy(self, noise_mask, tile_pos, padded_tiles):
        allTiles = []
        for tile_pos_pass, tiles_pass in zip(tile_pos, padded_tiles):
            tile_pass_l = []
            for tile_pos_group, tiles_group in zip(tile_pos_pass, tiles_pass):
                tile_group_l = []
                for (x, y, tile_w, tile_h), padded_tile in zip(tile_pos_group, tiles_group):
                    padded_h = padded_tile.shape[-2]
                    padded_w = padded_tile.shape[-1]       
                    tile_mask = self.generateGenDenoiseMask(padded_h, padded_w)
                    tiled_mask = None
                    if noise_mask is not None:
                        tiled_mask = self.tensorService.getSlice(noise_mask, y, padded_h, x, padded_w)
                    if tile_mask is not None:
                        if tiled_mask is not None:
                            tiled_mask *= tiled_mask
                        else:
                            tiled_mask = tile_mask      
                    tile_group_l.append(tiled_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    def getBlendMaskTilesforAPTStrategy(self, padded_tiles, B):
        allTiles = []
        for tiles_pass in padded_tiles:
            tile_pass_l = []
            for tiles_group in tiles_pass:
                tile_group_l = []
                for padded_tile in tiles_group:
                    tile_shape = padded_tile.shape
                    x, y, h, w = tile_shape
                    tile_group_l.append(self.generateGenBlendMask(B, h, w))
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    
###### Multi-Pass - Adjacency Padded Tiling Strategy ######   
class MP_APTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

        # pts - Stands for Padded Tiling Streteg
        self.apt_Service = APTService()

    def generatePos_forMP_APTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, padding: int, passes: int = 2):
        allTiles = []
        for i in range(passes):
            tile_pos = self.apt_Service.generatePosforAPTStrategy(latent_width, latent_height, tile_width, tile_height, padding)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles
    
    def getTilesforMP_APTStrategy(self, samples, tile_pos_and_steps):
        return self.apt_Service.getTilesforAPTStrategy(samples, tile_pos_and_steps)

    def getPaddedTilesforMP_APTStrategy(self, tiled_tiles, tile_pos, padding_strategy, padding):
        return self.apt_Service.getPaddedTilesforAPTStrategy(tiled_tiles, tile_pos, padding_strategy, padding)
    
    def getNoiseTilesforMP_APTStrategy(self, latent, padded_tiles, seed, disable_noise: bool):
        return self.apt_Service.getNoiseTilesforAPTStrategy(latent, padded_tiles, seed, disable_noise)
    
    def getDenoiseMaskTilesforMP_APTStrategy(self, noise_mask, tile_pos, padded_tiles):
        return self.apt_Service.getDenoiseMaskTilesforAPTStrategy(noise_mask, tile_pos, padded_tiles)
    
    def getBlendMaskTilesforMP_APTStrategy(self, padded_tiles, B):
        return self.apt_Service.getBlendMaskTilesforAPTStrategy(padded_tiles, B)


###### Singel Pass - Context-Padded Tiling Strategy ######   
class CPTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

    def generatePosforCPTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int):
        """
        Single-Pass Context-Padded Tiling Strategy.
        Generates only one pass without shift-based multi-pass logic.
        """
        tile_size_h = min(latent_height, max(4, (tile_width // 4) * 4))  # Use tile_width
        tile_size_w = min(latent_width, max(4, (tile_height // 4) * 4))  # Use tile_height

        h = np.arange(0, latent_height, tile_size_h)
        w = np.arange(0, latent_width, tile_size_w)

        def create_tile(i, j):
            h_start = int(h[i])
            w_start = int(w[j])
            h_size = min(tile_size_h, latent_height - h_start)
            w_size = min(tile_size_w, latent_width - w_start)

            return (w_start, h_start, w_size, h_size)


        passes = [
            [[create_tile(i, j) for i in range(len(h)) for j in range(len(w))]],
        ]
        
        return passes
    
    def getTilesforCPTStrategy(self, samples, tile_positions):
        return self.getTilesfromTilePos(samples, tile_positions)
    
    def getNoiseTilesforCPTStrategy(self, noise_tensor , tile_positions):
        return self.getTilesfromTilePos(noise_tensor, tile_positions)
    
    def getDenoiseMaskTilesforCPTStrategy(self, noise_mask, tile_pos, B, latent_height, latent_width):
        allTiles = []
        for tile_pos_pass in tile_pos:
            tile_pass_l = []
            for tile_pos_group in tile_pos_pass:
                tile_group_l = []
                for i, (x, y, tile_w, tile_h) in enumerate(tile_pos_group):
                    tile_mask = self.generateGenDenoiseMask(tile_h, tile_w)
                    tiled_mask = None
                    if noise_mask is not None:
                        tiled_mask = self.tensorService.getSlice(noise_mask, y, tile_h, x, tile_w)
                    if tile_mask is not None:
                        tiled_mask = tiled_mask * tile_mask if tiled_mask is not None else tile_mask
                    (x, y, tile_w, tile_h), tiled_mask = self.generateDenoiseMask_atBoundary(y, tile_h, x, tile_w, tile_h, tile_w, latent_height, latent_width, tiled_mask)
                    tile_pos_group[i] = (x, y, tile_w, tile_h)
                    tile_group_l.append(tiled_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    def getBlendeMaskTilesforCPTStrategy(self, tile_pos, B):
        return self.getBlendMaskTiles(tile_pos, B)

###### Multi-Pass - Context-Padded Tiling Strategy ######   
class MP_CPTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()
        # cpt - Stands for Context-Padded Tiling Stretegy
        self.cpt_Service = CPTService()


    def generatePosforMP_CPTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int):
        tile_size_h = min(latent_height, max(4, (tile_width // 4) * 4))
        tile_size_w = min(latent_width, max(4, (tile_height // 4) * 4))

        h = np.arange(0, latent_height, tile_size_h)
        h_shift = np.arange(tile_size_h // 2, latent_height - tile_size_h // 2, tile_size_h)
        w = np.arange(0, latent_width, tile_size_w)
        w_shift = np.arange(tile_size_w // 2, latent_width - tile_size_w // 2, tile_size_w)

        def create_tile(hs, ws, i, j):
            h = int(hs[i])
            w = int(ws[j])
            h_len = min(tile_size_h, latent_height - h)
            w_len = min(tile_size_w, latent_width - w)

            return (h, w, h_len, w_len)
        passes = [
            [[create_tile(h,       w,       i, j) for i in range(len(h))       for j in range(len(w))]],
            [[create_tile(h_shift, w,       i, j) for i in range(len(h_shift)) for j in range(len(w))]],
            [[create_tile(h,       w_shift, i, j) for i in range(len(h))       for j in range(len(w_shift))]],
            [[create_tile(h_shift, w_shift, i, j) for i in range(len(h_shift)) for j in range(len(w_shift))]],
        ]
        
        return passes
    
    def getTilesforMP_CPTStrategy(self, samples, tile_pos):
        return self.cpt_Service.getTilesforCPTStrategy(samples, tile_pos)
    
    def getNoiseTilesforMP_CPTStrategy(self, noise_tensor , tile_pos):
        return self.cpt_Service.getNoiseTilesforCPTStrategy(noise_tensor, tile_pos)
    
    def getDenoiseMaskTilesforMP_CPTStrategy(self, noise_mask, tile_pos, B, latent_height, latent_width):
        return self.cpt_Service.getDenoiseMaskTilesforCPTStrategy(noise_mask, tile_pos, B, latent_height, latent_width)
    
    def getBlendeMaskTilesforMP_CPTStrategy(self, tile_pos, B):
        return self.cpt_Service.getBlendeMaskTilesforCPTStrategy(tile_pos, B)

###### Single Pass - Overlap Tiling Strategy ######   
class OVPStrategy(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

    def generatePosforOVPStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, overlap: int):
        """
        Generate tile positions with overlap while ensuring full image coverage.

        :param latent_width: Total width of the latent image.
        :param latent_height: Total height of the latent image.
        :param tile_width: Base width of each tile (latent space).
        :param tile_height: Base height of each tile (latent space).
        :param overlap: Overlap in latent space (NOT a percentage).
        :return: List of tile positions with structure [[(x, y, w, h), ...], ...].
        """
        # Compute effective stride
        stride_x = max(1, tile_width - overlap)  
        stride_y = max(1, tile_height - overlap)  

        # Calculate number of tiles, ensuring at least one tile
        num_tiles_x = max(1, (latent_width - overlap) // stride_x)
        num_tiles_y = max(1, (latent_height - overlap) // stride_y)

        tile_positions = []

        for j in range(num_tiles_y + 1):  # +1 ensures full coverage at the bottom
            row_positions = []
            for i in range(num_tiles_x + 1):  # +1 ensures full coverage at the right
                x = i * stride_x
                y = j * stride_y

                # Ensure last tile reaches the boundary
                if x + tile_width > latent_width:
                    x = latent_width - tile_width  # Align last column with the right edge
                if y + tile_height > latent_height:
                    y = latent_height - tile_height  # Align last row with the bottom edge

                row_positions.append((x, y, tile_width, tile_height))

            tile_positions.append(row_positions)

        return [tile_positions]
    
    def getTilesforOVPStrategy(self, samples, tile_positions):
        return self.getTilesfromTilePos(samples, tile_positions)
    
    def getNoiseTilesforOVPStrategy(self, noise_tensor , tile_positions):
        return self.getTilesfromTilePos(noise_tensor, tile_positions)
    
    def getDenoiseMaskTilesforOVPStrategy(self, noise_mask, tile_pos,):
        return self.getDenoiseMaskTiles(noise_mask, tile_pos)
    
    def getBlendeMaskTilesforOVPStrategy(self, tile_pos, B):
        return self.getBlendMaskTiles(tile_pos, B)

###### Multi-Pass - Overlap Tiling Strategy ######   
class MP_OVPStrategy(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

        self.ovp_Service = OVPStrategy()

    def generatePosforMP_OVPStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, overlap: float, passes: int = 2):
        """
        Generate tile positions with overlap while keeping array depth like PT strategy.

        :param latent_width: Total width of the latent image.
        :param latent_height: Total height of the latent image.
        :param tile_width: Base width of each tile.
        :param tile_height: Base height of each tile.
        :param overlap: Overlap ratio (e.g., 0.2 for 20% overlap).
        :return: List of tile positions with structure [[(x, y, w, h), ...], ...].
        """
        allTiles = []
        for i in range(passes):
            tile_pos = self.ovp_Service.generatePosforOVPStrategy(latent_width, latent_height, tile_width, tile_height, overlap)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles
    
    def getTilesforMP_OVPStrategy(self, samples, tile_positions):
        return self.ovp_Service.getTilesforOVPStrategy(samples, tile_positions)
    
    def getNoiseTilesforMP_OVPStrategy(self, noise_tensor , tile_positions):
        return self.ovp_Service.getNoiseTilesforOVPStrategy(noise_tensor, tile_positions)
    
    def getDenoiseMaskTilesforMP_OVPStrategy(self, noise_mask, tile_pos):
        return self.ovp_Service.getDenoiseMaskTilesforOVPStrategy(noise_mask, tile_pos)
    
    def getBlendeMaskTilesforMP_OVPStrategy(self, tile_pos, B):
        return self.ovp_Service.getBlendeMaskTilesforOVPStrategy(tile_pos, B)

###### Single Pass - Adaptive Tiling Strategy ######   
class ADPService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

    def calculateEntropy(self, tile):
        """
        Calculate entropy for a given tile.
        Entropy measures the disorder or randomness in the image patch.
        """
        flattened_tile = tile.reshape(-1)  
        hist = torch.histc(flattened_tile.float(), bins=256, min=0, max=255)
        hist = hist / hist.sum() 
        entropy = -torch.sum(hist * torch.log2(hist + 1e-8)) 
        return entropy

    def smoothComplexityMap(self, complexity_map, kernel_size=2):
        """
        Smooth the complexity map to reduce abrupt changes between adjacent tiles.
        We apply a smaller smoothing kernel here to retain more fine-grained entropy.
        """
        # Apply a simple smoothing filter like a box blur
        padding = kernel_size // 2
        kernel = torch.ones((kernel_size, kernel_size), device=complexity_map.device) / (kernel_size**2)
        smoothed_map = torch.nn.functional.conv2d(complexity_map.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=padding)
        return smoothed_map.squeeze(0).squeeze(0)

    def calculateComplexity(self, latent_tensor, tile_size=32):
        """
        Compute entropy for each tile to estimate complexity.
        Higher entropy → high complexity (edges, textures)
        Lower entropy → smooth regions
        """
        B, C, H, W = latent_tensor.shape
        complexity_map = torch.zeros((H, W), device=latent_tensor.device)

        for i in range(0, H, tile_size):
            for j in range(0, W, tile_size):
                tile = latent_tensor[:, :, i:i+tile_size, j:j+tile_size]
                entropy = self.calculateEntropy(tile)
                complexity_map[i:i+tile_size, j:j+tile_size] = entropy

        # Apply lighter smoothing to reduce extreme transitions
        complexity_map = self.smoothComplexityMap(complexity_map, kernel_size=2)
        return complexity_map

    def generatePosforADPStrategy(self, latent_tensor, tile_width=64, tile_height=64):
        """
        Generate tile positions based on complexity.
        Dynamically adjusts tile size based on the base_tile_size.
        """
        B, C, latent_height, latent_width = latent_tensor.shape
        
        # Scale factor based on base_tile_size
        average_tile_size = int((tile_width + tile_height) / 2) 
        scale_factor = average_tile_size / 64 
        
        # Dynamically scale the tile size for complexity calculation
        dynamic_tile_size = int(32 * scale_factor) 

        # Calculate complexity using dynamically adjusted tile size
        complexity_map = self.calculateComplexity(latent_tensor, tile_size=dynamic_tile_size)

        def create_tile(x, y):
            complexity = complexity_map[y, x].item()
            tile_size = int(average_tile_size * (1.5 - complexity / torch.max(complexity_map)))

            # Ensure tile size stays within reasonable limits
            min_tile_size = 16
            max_tile_size = 128
            tile_size = min(max(tile_size, min_tile_size), max_tile_size)

            tile_w = min(tile_size, latent_width - x)
            tile_h = min(tile_size, latent_height - y)
            return (x, y, tile_w, tile_h)

        tile_positions = [[[]]] 

        y = 0
        while y < latent_height:
            x = 0
            while x < latent_width:
                tile_positions[0][0].append(create_tile(x, y))
                x += tile_positions[0][0][-1][2]  # Move x by tile width
            y += tile_positions[0][0][-1][3]  # Move y by tile height

        return tile_positions
    
    def getTilesforADPStrategy(self, samples, tile_positions):
        return self.getTilesfromTilePos(samples, tile_positions)
    
    def getNoiseTilesforADPStrategy(self, noise_tensor , tile_positions):
        return self.getTilesfromTilePos(noise_tensor, tile_positions)
    
    def getDenoiseMaskTilesforADPStrategy(self, noise_mask, tile_pos,):
        return self.getDenoiseMaskTiles(noise_mask, tile_pos)
    
    def getBlendeMaskTilesforADPStrategy(self, tile_pos, B):
        return self.getBlendMaskTiles(tile_pos, B)

###### Multi-Pass - Adaptive Tiling Strategy ######   
class MP_ADPService(Tiling_Strategy_Base):
    def __init__(self):
        super().__init__()

        # adp - Stands for Adaptive Tiling Strategy
        self.adp_Service = ADPService()

    def generatePos_forMP_ADPStrategy(self, latent_tensor, tile_width: int = 64, tile_height: int = 64, passes: int = 2):
        """
        Multi-pass tiling: Starts large, refines complex areas in later passes.
        Returns depth array structure: [[[pass_1_tiles]], [[pass_2_tiles]], ...]
        """
        average_tile_size = int((tile_width + tile_height) / 2) 
        tile_positions = self.adp_Service.generatePosforADPStrategy(latent_tensor, average_tile_size)

        for pass_idx in range(passes): 
            refined_positions = []
            num_new_tiles = 0
            max_new_tiles = len(tile_positions[pass_idx - 1][0]) * 2  

            for (x, y, w, h) in tile_positions[pass_idx - 1][0]:
                region = latent_tensor[:, :, y:y + h, x:x + w]
                complexity = torch.std(region).item()

                # Refine only if complexity is high and we haven’t exceeded the max new tile count
                if complexity > 0.1 and num_new_tiles < max_new_tiles and w > 16 and h > 16:
                    refined_positions.append(self.adjustTile(x, y, w, h))
                    num_new_tiles += 1  
                else:
                    refined_positions.append((x, y, w, h)) 

            # Append the refined positions for the next pass
            tile_positions.append([refined_positions])

        return tile_positions

    def adjustTile(self, x, y, w, h):
        new_w = max(16, w // 2)
        new_h = max(16, h // 2)
        return (x, y, new_w, new_h)
    
    def getTilesforMP_ADPStrategy(self, samples, tile_positions):
        return self.adp_Service.getTilesforADPStrategy(samples, tile_positions)
    
    def getNoiseTilesforMP_ADPStrategy(self, noise_tensor , tile_positions):
        return self.adp_Service.getNoiseTilesforADPStrategy(noise_tensor, tile_positions)
    
    def getDenoiseMaskTilesforMP_ADPStrategy(self, noise_mask, tile_pos,):
        return self.adp_Service.getDenoiseMaskTilesforADPStrategy(noise_mask, tile_pos)
    
    def getBlendeMaskTilesforMP_ADPStrategy(self, tile_pos, B):
        return self.adp_Service.getBlendeMaskTilesforADPStrategy(tile_pos, B)

###### Single Pass - Hierarchical Tiling Strategy ######  
class HRCService(Tiling_Strategy_Base):
    def __init__(self):
        super().__init__()

    def calculateEntropy(self, tile):
        """
        Calculate entropy (complexity) of a tile.
        """
        flattened_tile = tile.reshape(-1)
        hist = torch.histc(flattened_tile.float(), bins=256, min=0, max=255)
        hist = hist / hist.sum() 
        entropy = -torch.sum(hist * torch.log2(hist + 1e-8))
        return entropy.item()

    def generatePosforHRCStrategy(self, latent_tensor, min_tile_size=16, complexity_threshold=0.1):
        """
        Fully adaptive hierarchical tiling: Recursively refines high-complexity tiles.
        Ensures that valid tiles are always generated to avoid empty sequences.
        """
        B, C, H, W = latent_tensor.shape
        tile_queue = [(0, 0, W, H)]  # Start with the full image
        final_tiles = []

        ## Some Fixes So i do not get bit tile sizes
        average_latent_size = (H + W) // 2
        
        if average_latent_size >= 256:
            min_tile_size = 24
        elif average_latent_size >= 512:
            min_tile_size = 32
        elif average_latent_size >= 1024:
            min_tile_size = 48
        elif average_latent_size >= 2048:
            min_tile_size = 64
        elif average_latent_size >= 4096:
            min_tile_size = 96
        elif average_latent_size >= 8192:
            min_tile_size = 128
        elif average_latent_size >= 16384:
            min_tile_size = 192
        
        '''
        if average_latent_size >= 512:
            min_tile_size = 24
        elif average_latent_size >= 1024:
            min_tile_size = 32
        elif average_latent_size >= 2048:
            min_tile_size = 48
        elif average_latent_size >= 4096:
            min_tile_size = 64
        elif average_latent_size >= 8192:
            min_tile_size = 96
        elif average_latent_size >= 16384:
            min_tile_size = 128

        '''

        while tile_queue:
            x, y, w, h = tile_queue.pop(0)
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Ensure valid tile size within image bounds
            w = min(w, W - x)
            h = min(h, H - y)

            if w < min_tile_size or h < min_tile_size:
                final_tiles.append((x, y, max(w, min_tile_size), max(h, min_tile_size)))
                continue

            # Compute tile complexity
            tile = latent_tensor[:, :, y:y+h, x:x+w]
            complexity = self.calculateEntropy(tile)

            if complexity > complexity_threshold and w > min_tile_size * 2 and h > min_tile_size * 2:
                # Split into 4 smaller tiles
                new_w, new_h = max(w // 2, min_tile_size), max(h // 2, min_tile_size)

                tile_queue.extend([
                    (x, y, new_w, new_h), (x + new_w, y, new_w, new_h),
                    (x, y + new_h, new_w, new_h), (x + new_w, y + new_h, new_w, new_h)
                ])
            else:
                final_tiles.append((x, y, w, h))

        # Ensure at least one tile exists
        if not final_tiles:
            final_tiles.append((0, 0, min_tile_size, min_tile_size))

        return [[final_tiles]]

    def getTilesforHRCStrategy(self, samples, tile_positions):
        return self.getTilesfromTilePos(samples, tile_positions)
    
    def getNoiseTilesforHRCStrategy(self, noise_tensor , tile_positions):
        return self.getTilesfromTilePos(noise_tensor, tile_positions)
    
    def getDenoiseMaskTilesforHRCStrategy(self, noise_mask, tile_pos,):
        return self.getDenoiseMaskTiles(noise_mask, tile_pos)
    
    def getBlendMaskTilesforHRCStrategy(self, tile_pos, B):
        return self.getBlendMaskTiles(tile_pos, B)

###### Single Pass - Hierarchical Tiling Strategy ######  
class MP_HRCService(Tiling_Strategy_Base):
    
    def __init__(self, ):
        super().__init__()

        self.hrc_Service = HRCService()

    def generatePos_forMP_HRCStrategy(self, latent_tensor, passes: int = 2, min_tile_size: int = 16, base_complexity_threshold: float = 0.1, complexity_variation: float = 0.05):
        """
        Multi-pass adaptive hierarchical tiling. Varies complexity threshold across passes.
        """
        allTiles = []
        for i in range(passes):
            variation = (i - passes // 2) * complexity_variation
            complexity_threshold = max(0.01, base_complexity_threshold + variation)
            tile_pos = self.hrc_Service.generatePosforHRCStrategy(latent_tensor, min_tile_size, complexity_threshold)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles

    def getTilesforMP_HRCStrategy(self, samples, tile_positions):
        return self.hrc_Service.getTilesforHRCStrategy(samples, tile_positions)

    def getNoiseTilesforMP_HRCStrategy(self, noise_tensor, tile_positions):
        return self.hrc_Service.getNoiseTilesforHRCStrategy(noise_tensor, tile_positions)

    def getDenoiseMaskTilesforMP_HRCStrategy(self, noise_mask, tile_pos):
        return self.hrc_Service.getDenoiseMaskTilesforHRCStrategy(noise_mask, tile_pos)

    def getBlendMaskTilesforMP_HRCStrategy(self, tile_pos, B):
        return self.hrc_Service.getBlendMaskTilesforHRCStrategy(tile_pos, B)

###### Single-Pass - Non-Uniform Tiling Strategy ###### 
class NUService(Tiling_Strategy_Base): 
    
    def __init__(self):
        super().__init__()

    def computeDetailMap(self, latent_tensor):
        """
        Compute a more robust detail/complexity map by combining low- and high-frequency information.
        This version uses edge detection and local variance to better capture image complexity.
        """
        latent_np = latent_tensor.squeeze().cpu().numpy()  # Convert tensor to NumPy (C, H, W)
        C, H, W = latent_np.shape

        # Step 1: Compute gradient magnitude (edge detection)
        grad_x = sobel(latent_np, axis=2)  # Horizontal edges
        grad_y = sobel(latent_np, axis=1)  # Vertical edges
        grad_mag = np.hypot(grad_x, grad_y)  # Magnitude of gradient

        # Step 2: Normalize gradient magnitude for better control over edge data
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

        # Step 3: Compute local variance (texture complexity)
        texture_map = gaussian_filter(latent_np, sigma=2)  # Adjusted Gaussian for capturing larger features
        local_variance = np.var(texture_map, axis=(1, 2))  # Compute variance for each channel (C,)

        # Step 4: Normalize variance map to match range [0, 1]
        local_variance = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min() + 1e-8)

        # Reshape local_variance to match gradient map's shape (broadcast it)
        local_variance_map = local_variance[:, np.newaxis, np.newaxis]  # Reshape to (C, 1, 1)
        local_variance_map = np.broadcast_to(local_variance_map, grad_mag.shape)  # Broadcast to match (C, H, W)

        # Step 5: Combine edge map and local variance map
        detail_map = grad_mag + local_variance_map
        detail_map = np.clip(detail_map, 0, 1)  # Ensure the map is between 0 and 1

        return torch.tensor(detail_map, device=latent_tensor.device)


    def generatePosforNUStrategy(self, latent_tensor, tile_width=64, tile_height=64, min_tile_size=16, scale_min=0.25, scale_max=2.0):

        B, C, latent_height, latent_width = latent_tensor.shape
        average_tile_size = int((tile_width + tile_height) / 2)

        ## Some Fixes So i do not get To much Tiles, and bigger tile than i can generate
        if average_tile_size >= 128:
            average_tile_size = 128
            min_tile_size = 32
            
        # Compute complexity map
        complexity_map = self.computeDetailMap(latent_tensor)  # Get the map

        # Ensure complexity_map is (C, H, W)
        if complexity_map.dim() != 3:
            print("Unexpected complexity map dimensions:", complexity_map.shape)
            return []

        # Explicitly normalize the complexity map to a [0, 1] range
        complexity_map = (complexity_map - complexity_map.min()) / (complexity_map.max() - complexity_map.min() + 1e-8)

        def create_tile(x, y):
            """
            Adjusts tile size based on complexity.
            """
            # Get the complexity value for the current tile location (mean across channels)
            complexity = complexity_map[:, y, x].mean().item()

            # Calculate tile size based on complexity, scaling between min and max
            tile_size = int(average_tile_size * (scale_min + (scale_max - scale_min) * complexity))  # Dynamic scaling

            # Force a larger random variation to improve the diversity of tile sizes
            tile_size = tile_size + np.random.randint(-15, 15)  # Randomly perturb the tile size by a larger amount

            # Ensure tile size stays within reasonable limits
            tile_size = min(max(tile_size, min_tile_size), latent_width)

            tile_w = min(tile_size, latent_width - x)  # Width of tile, ensuring it doesn't exceed image width
            tile_h = min(tile_size, latent_height - y)  # Height of tile, ensuring it doesn't exceed image height
            return (x, y, tile_w, tile_h)

        # Output in depth array format
        tile_positions = [[[]]]

        y = 0
        while y < latent_height:
            x = 0
            while x < latent_width:
                tile_positions[0][0].append(create_tile(x, y))
                # Dynamically adjust the step size based on tile width and height
                x += tile_positions[0][0][-1][2]  # Advance by the width of the last created tile

            # Add random variation to the vertical step size as well
            y += np.random.randint(min_tile_size, tile_height)  # Adding randomness to the vertical step size

        return tile_positions
        
    def getTilesforNUStrategy(self, samples, tile_positions):
        return self.getTilesfromTilePos(samples, tile_positions)
    
    def getNoiseTilesforNUStrategy(self, noise_tensor , tile_positions):
        return self.getTilesfromTilePos(noise_tensor, tile_positions)
    
    def getDenoiseMaskTilesforNUStrategy(self, noise_mask, tile_pos,):
        return self.getDenoiseMaskTiles(noise_mask, tile_pos)
    
    def getBlendeMaskTilesforNUStrategy(self, tile_pos, B):
        return self.getBlendMaskTiles(tile_pos, B)

###### Multi-Pass - Non-Uniform Tiling Strategy ######  
class MP_NUService(Tiling_Strategy_Base): 
    def __init__(self):
        super().__init__()
        
        # nu - Stands for Non-Uniform Tiling Strategy
        self.nu_Service = NUService()

    def generatePosforMP_NUStrategy(self, latent_tensor, passes: int = 2, tile_width: int = 64, tile_height: int = 64, min_tile_size: int = 16):
        """
        Generates non-uniform tiles using a depth-array format: [[[pass_1_tiles]]]
        """
        allTiles = []
        scale_patterns = [
            (0.25, 1.5),  # Small tiles
            (0.5, 1.0),   # Medium balance
            (0.3, 2.0)    # Larger tiles, more variation
        ]

        for i in range(passes):
            # Cycle through the scale patterns based on pass index
            scale_min, scale_max = scale_patterns[i % len(scale_patterns)]

            # Generate position strategy with dynamic scale_min and scale_max
            tile_pos = self.nu_Service.generatePosforNUStrategy(latent_tensor, tile_width, tile_height, min_tile_size, scale_min, scale_max)
            
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        
        return allTiles
    
    def getTilesforMP_NUStrategy(self, samples, tile_positions):
        return self.nu_Service.getTilesforNUStrategy(samples, tile_positions)
    
    def getNoiseTilesforMP_NUStrategy(self, noise_tensor, tile_positions):
        return self.nu_Service.getTilesforNUStrategy(noise_tensor, tile_positions)
    
    def getDenoiseMaskTilesforMP_NUStrategy(self, noise_mask, tile_pos):
        return self.nu_Service.getDenoiseMaskTilesforNUStrategy(noise_mask, tile_pos)
    
    def getBlendeMaskTilesforMP_NUStrategy(self, tile_pos, B):
        return self.nu_Service.getBlendeMaskTilesforNUStrategy(tile_pos, B)

class TilingService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

        # st - Stands for Simple Tiling Strategy
        self.st_Service = STService()
        # rt - Stands for Random Tiling Strategy
        self.rt_Service = RTService()
        # pt - Stands for Padded Tiling Strategy
        self.pt_Service = PTService()
        # apt - Stands for Adjency Padded Tiling Strategy
        self.apt_Service = APTService()
        # cpt - Stands for Context-Padded Tiling Strategy
        self.cpt_Service = CPTService()
        # ovp - Stands for Overlap Padded Tiling Strategy
        self.ovp_Service = OVPStrategy()
        # adp - Stands for Adaptive Tiling Strategy
        self.adp_Service = ADPService()
        # hrc - Stands for Hierarchical Tiling Strategy
        self.hrc_Service = HRCService()
        # nu - Stands for Non-Uniform Tiling Strategy
        self.nu_Service = NUService()
        
        # mp_st - Stands for Multi-Pass Simple Tiling Strategy
        self.mp_st_Service = MP_STService()
        # mp_rt - Stands for Multi-Pass Random Tiling Strategy
        self.mp_rt_Service = MP_RTService()
        # mp_pt - Stands for Multi-Pass Padding Tiling Strategy
        self.mp_pt_Service = MP_PTService()
        # mp_apt - Stands for Multi-Pass Adjency Padding Tiling Strategy
        self.mp_apt_Service = MP_APTService()
        # mp_cpt - Stands for Multi-Pass Context-Padded Tiling Stretegy
        self.mp_cpt_Service = MP_CPTService()
        # mp_ovp - Stands for Overlapy Padded Tiling Strategy
        self.mp_ovp_Service = MP_OVPStrategy()
        # mp_adp - Stands for Multi-Pass Adaptive Tiling Strategy
        self.mp_adp_Service = MP_ADPService()
        # mp_hrc - Stands for Multi-Pass Hierarchical Tiling Strategy
        self.mp_hrc_Service = MP_HRCService()
        # mp_nu - Stands for Multi-Pass Non-Uniform Tiling Strategy
        self.mp_nu_Service = MP_NUService()
