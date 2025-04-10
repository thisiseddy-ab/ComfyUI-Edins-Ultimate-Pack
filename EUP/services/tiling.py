import random 

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
from EUP.services.noise import NoiseService
from EUP.services.denoise import DenoiseService
from EUP.services.blend import BlendService
from EUP.services.base_model_res  import BaseModelResolutionService

class Tiling_Strategy_Base():

    def __init__(self):
        self.tensorService = TensorService()
        self.noiseService = NoiseService()
        self.denoiseService = DenoiseService()
        self.blendService = BlendService()

    def getTilesfromTilePos(self, latentImage, tilePos):
        latentTensor = self.tensorService.getTensorfromLatentImage(latentImage)
        allTiles = []
        for t_pos_pass in tilePos:
            tile_pass_l = []
            for t_pos_group in t_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in t_pos_group:
                    tile_group_l.append(self.tensorService.getSlice(latentTensor, (x, y, tile_w, tile_h)))
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    def choseNoiseStrategy(self, latentImage, seed: int, disable_noise: bool, noise_strategy, noise_strategy_pars, device: str = 'cpu'):
        if noise_strategy == "random":
            return self.noiseService.generateNoiseforRDMStrategy(latentImage,seed, disable_noise, device)
        else:
            raise ValueError(f"Unknown noise strategy: {noise_strategy}")
        
    def choseNoiseMaskStrategy(self, latentImage, noiseTensor, noise_mask_strategy, noise_mask_strategy_pars, device: str = 'cpu'):
        if noise_mask_strategy == "standart":
            return self.noiseService.generateNoiseMaskforSDAStrategy(latentImage, noiseTensor, device)
        else:
            raise ValueError(f"Unknown noise mask strategy: {noise_mask_strategy}")
    
    def getNoiseTilesfromTilePos(self, noiseTensor, tilePos):
        allTiles = []
        for t_pos_pass in tilePos:
            tile_pass_l = []
            for t_pos_group in t_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in t_pos_group:
                    tile_group_l.append(self.tensorService.getSlice(noiseTensor, (x, y, tile_w, tile_h)))
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles

    def choseDenoiseStrategy(self, tileSize, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        if denoise_strategy == "none":
            return None
        elif denoise_strategy == "smooth":
            min_value, max_value, falloff = denoise_strategy_pars
            return self.denoiseService.generateDenoiseMaskforSMHStrategy(tileSize=tileSize, min_value=min_value, max_value=max_value, falloff=falloff, device=device)
        elif denoise_strategy == "aspect-adaption":
            min_value, max_value, base_falloff = denoise_strategy_pars
            return self.denoiseService.generateDenoiseMaskforASADPStrategy(tileSize=tileSize, min_value=min_value, max_value=max_value, base_falloff=base_falloff, device=device)
        else:
            raise ValueError(f"Unknown denoise strategy: {denoise_strategy}")
        
    def getDenoiseMaskTiles(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        allTiles = []
        for tile_pos_pass in tilePos:
            tile_pass_l = []
            for tile_pos_group in tile_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in tile_pos_group:
                    tile_mask = self.choseDenoiseStrategy((tile_w, tile_h), denoise_strategy, denoise_strategy_pars, device)
                    tiled_mask = None
                    if noiseMask is not None:
                        tiled_mask = self.tensorService.getSlice(noiseMask, (x, y, tile_w, tile_h))
                    if tile_mask is not None:
                        tiled_mask = tiled_mask * tile_mask if tiled_mask is not None else tile_mask
                    tile_group_l.append(tiled_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
     
    def choseBlendStrategy(self, latentImage, tileSize, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        if blend_strategy == "none":
            return None
        elif blend_strategy == "soft-focus":
            radial_strength, directional_strength, min_fade, blend_power, fade_offset, fade_scale = blend_strategy_pars
            return self.blendService.generateBlendMaskforSFStrategy(latentImage, tileSize, radial_strength, directional_strength, min_fade, blend_power, fade_offset, fade_scale, device)
        else:
            raise ValueError(f"Unknown blend strategy: {blend_strategy}")

    def getBlendMaskTiles(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        allTiles = []
        for t_pos_pass in tilePos:
            tile_pass_l = []
            for t_pos_group in t_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in t_pos_group:
                    blend_mask = self.choseBlendStrategy(latentImage, (tile_w, tile_h), blend_strategy, blend_strategy_pars, device)
                    tile_group_l.append(blend_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
###### Singel Pass - Simple Tiling Strategy ######  
class STService(Tiling_Strategy_Base):
    
    def __init__(self):
        super().__init__()

    def generatePosforSTStrategy(self, latentImage, tileSize):
        """
        Generates a single pass of simple tiling with the same nested structure as multi-pass.
        """

        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        tile_w, tile_h = tileSize

        # Ensure tile sizes fit the dimensions
        num_tiles_x = max(1, W // tile_w)
        num_tiles_y = max(1, H // tile_h)

        adjusted_tile_width = W // num_tiles_x
        adjusted_tile_height = H // num_tiles_y

        # Base positions for normal tiling
        x_positions = list(range(0, W, adjusted_tile_width))
        y_positions = list(range(0, H, adjusted_tile_height))

        # Generate tile positions in the same structure as multi-pass
        pass_tiles = [[(x, y, adjusted_tile_width, adjusted_tile_height) for x in x_positions for y in y_positions]]

        return [pass_tiles]

    def getTilesforSTStrategy(self, latentImage, tilePos):
        return self.getTilesfromTilePos(latentImage, tilePos)
    
    def getNoiseTilesforSTStrategy(self, noiseTensor, tilePos):
        return self.getNoiseTilesfromTilePos(noiseTensor, tilePos)
    
    def getDenoiseMaskTilesforSTStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        return self.getDenoiseMaskTiles(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)

    def getBlendMaskTilesforSTStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        return self.getBlendMaskTiles(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)
    
###### Multi-Pass - Simple Tiling Strategy ######   
class MP_STService(Tiling_Strategy_Base):
    
    def __init__(self):
        super().__init__()

        # st - Stands for Simple Tiling Streteg
        self.st_Service = STService()

    def generatePosforMP_STStrategy(self, latentImage, tileSize, passes: int):
        allTiles = []
        for i in range(passes):
            tile_pos = self.st_Service.generatePosforSTStrategy(latentImage, tileSize)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles 
    
    def getTilesforMP_STStrategy(self, latentImage, tilePos):
        return self.st_Service.getTilesforSTStrategy(latentImage, tilePos)
    
    def getNoiseTilesforMP_STStrategy(self, noiseTensor, tilePos):
        return self.st_Service.getNoiseTilesforSTStrategy(noiseTensor, tilePos)
    
    def getDenoiseMaskTilesforMP_STStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str ='cpu'):
        return self.st_Service.getDenoiseMaskTilesforSTStrategy(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendMaskTilesforMP_STStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str ='cpu'):
        return self.st_Service.getBlendMaskTilesforSTStrategy(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Singel Pass - Random Tiling Strategy ######   
class RTService(Tiling_Strategy_Base):
    
    def __init__(self):
        super().__init__()

    def generatePosforRTStrategy(self, latentImage, tileSize, seed: int):
        
        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        tile_w, tile_h = tileSize
        
        generator: torch.Generator = torch.manual_seed(seed)

        # Generate random jitter offsets
        rands = torch.rand((2,), dtype=torch.float32, generator=generator).numpy()
        jitter_w = int(rands[0] * tile_w)
        jitter_h = int(rands[1] * tile_h)

        # Compute tile positions with jittered offsets
        tiles_h = self.calcCoords(H, tile_h, jitter_h)
        tiles_w = self.calcCoords(W, tile_w, jitter_w)

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
    
    def getTilesforRTStrategy(self,latentImage, tilePos):
        return self.getTilesfromTilePos(latentImage, tilePos)
    
    def getNoiseTilesforRTStrategy(self, noiseTensor, tilePos):
        return self.getNoiseTilesfromTilePos(noiseTensor, tilePos)
    
    def getDenoiseMaskTilesforRTStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str ='cpu'):
        return self.getDenoiseMaskTiles(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendMaskTilesforRTStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str ='cpu'):
        return self.getBlendMaskTiles(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)
    
###### Multi-Pass - Random Tiling Strategy ######   
class MP_RTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()
        self.rt_Service = RTService() 

    def generatePos_forMP_RTStrategy(self, latentImage, tileSize, seed: int, passes: int):
        allTiles = []
        for i in range(passes):
            tile_pos = self.rt_Service.generatePosforRTStrategy(latentImage, tileSize, seed)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles 

    def getTilesforMP_RTStrategy(self, latentImage, tilePos):
        return self.rt_Service.getTilesforRTStrategy(latentImage, tilePos)
    
    def getNoiseTilesforMP_RTStrategy(self, noiseTensor, tilePos):
        return self.rt_Service.getNoiseTilesforRTStrategy(noiseTensor, tilePos)
    
    def getDenoiseMaskTilesforMP_RTStrategy(self, noise_mask, tilePos, denoise_strategy, denoise_strategy_pars, device: str ='cpu'):
        return self.rt_Service.getDenoiseMaskTilesforRTStrategy(noise_mask, tilePos, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendMaskTilesforMP_RTStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str ='cpu'):
        return self.rt_Service.getBlendMaskTilesforRTStrategy(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Singel Pass - Padded Tiling Strategy ######
class PTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

    def generatePosforPTStrategy(self, latentImage, tileSize, padding: int):

        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        tile_w, tile_h = tileSize

        # Calculate how many tiles fit in each dimension
        num_tiles_x = W // tile_w
        num_tiles_y = H // tile_h

        # Adjust tile size to fit perfectly
        adjusted_tile_width = W // num_tiles_x
        adjusted_tile_height = H // num_tiles_y

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
                adjusted_x = max(0, min(x - left_pad, W - padded_tile_width))
                adjusted_y = max(0, min(y - top_pad, H - padded_tile_height))

                row_positions.append((adjusted_x, adjusted_y, padded_tile_width, padded_tile_height))

            tile_positions.append(row_positions)

        return [tile_positions]

    def getTilesforPTStrategy(self, latentImage, tilePos):
        return self.getTilesfromTilePos(latentImage, tilePos)

    def applyPaddingforPTStrategy(self, tensorTile, paddingStrategy, padding, device: str = 'cpu'):

        # Apply padding uniformly to all sides
        left_pad = padding
        right_pad = padding
        top_pad = padding
        bottom_pad = padding

        # Apply padding using the chosen strategy
        if paddingStrategy in ["circular", "reflect", "replicate"]:
            return F.pad(tensorTile, (left_pad, right_pad, top_pad, bottom_pad), mode=paddingStrategy)
        elif paddingStrategy == "organic":
            return self.applyOrganicPadding(tensorTile, (left_pad, right_pad, top_pad, bottom_pad), device)
        else:
            return F.pad(tensorTile, (left_pad, right_pad, top_pad, bottom_pad), mode="constant", value=0)

    def getPaddedTilesforPTStrategy(self, tiledTiles, tilePos, paddingStrategy, padding, device: str = 'cpu'):

        allTiles = []
        for tile_pos_pass, tiles_pass in zip(tilePos, tiledTiles):
            tile_pass_l = []
            for tile_pos_group, tiles_group in zip(tile_pos_pass, tiles_pass):
                tile_group_l = []
                for (x, y, tile_w, tile_h), tile in zip(tile_pos_group, tiles_group):
                    padded_tile = self.applyPaddingforPTStrategy(tile, paddingStrategy, padding, device)
                    tile_group_l.append(padded_tile)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)

        return allTiles
    
    def applyOrganicPadding(self, tensorTile, padddings, device: str = 'cpu'):
        """Applies seamless mirrored padding with soft blending and natural variation, avoiding boxy artifacts."""
        B, C, H, W = tensorTile.shape
        
        # Unpack padding for all sides
        left, right, top, bottom = padddings

        # Create the padded tile
        padded_tile = torch.zeros((B, C, H + top + bottom, W + left + right), device=device, dtype=tensorTile.dtype)

        # Place the original tile in the center
        padded_tile[:, :, top:top + H, left:left + W] = tensorTile

        # Apply mirrored padding for all sides
        if left > 0:
            left_patch = self.softBlend(tensorTile[:, :, :, :left], (B, C, H, left), flip_dims=[3], blend_factor=0.55, device=device)
            padded_tile[:, :, top:top + H, :left] = left_patch

        if right > 0:
            right_patch = self.softBlend(tensorTile[:, :, :, -right:], (B, C, H, right), flip_dims=[3], blend_factor=0.55, device=device)
            padded_tile[:, :, top:top + H, -right:] = right_patch

        if top > 0:
            top_patch = self.softBlend(tensorTile[:, :, :top, :], (B, C, top, W), flip_dims=[2], blend_factor=0.55, device=device,)
            padded_tile[:, :, :top, left:left + W] = top_patch

        if bottom > 0:
            bottom_patch = self.softBlend(tensorTile[:, :, -bottom:, :], (B, C, bottom, W), flip_dims=[2], blend_factor=0.55, device=device)
            padded_tile[:, :, -bottom:, left:left + W] = bottom_patch

        # Corner regions (blend both X and Y for smooth edges)
        if left > 0 and top > 0:
            padded_tile[:, :, :top, :left] = self.softBlend(tensorTile[:, :, :top, :left], (B, C, top, left), flip_dims=[2, 3], blend_factor=0.35, device=device)

        if right > 0 and top > 0:
            padded_tile[:, :, :top, -right:] = self.softBlend(tensorTile[:, :, :top, -right:], (B, C, top, right), flip_dims=[2, 3], blend_factor=0.35, device=device)

        if left > 0 and bottom > 0:
            padded_tile[:, :, -bottom:, :left] = self.softBlend(tensorTile[:, :, -bottom:, :left], (B, C, bottom, left), flip_dims=[2, 3], blend_factor=0.35, device=device)

        if right > 0 and bottom > 0:
            padded_tile[:, :, -bottom:, -right:] = self.softBlend(tensorTile[:, :, -bottom:, -right:], (B, C, bottom, right), flip_dims=[2, 3], blend_factor=0.35, device=device)

        return padded_tile
    
    def softBlend(self, patch, expandShape, flipDims, blend_factor=0.7, device: str = 'cpu'):
        """Mirrors a patch and blends it softly with adaptive randomness."""
        mirrored = patch
        for flip_dim in flipDims:
            mirrored = torch.flip(mirrored, dims=[flip_dim])

        # Adaptive blending factor (less boxy)
        blend_factor = torch.tensor(blend_factor, device=device).expand_as(patch)

        # Perlin-style noise (natural variation, no harsh edges)
        noise = (torch.rand_like(patch) - 0.5) * 1.5 
        blend_factor = torch.clamp(blend_factor + noise, 0.4, 0.5) 

        return (patch * blend_factor + mirrored * (1 - blend_factor)).expand(expandShape)
    
    def getNoiseTilesforPTStrategy(self, latentImage, paddedTiles, seed: int, disableNoise: bool, device: str = 'cpu'):
        
        latentTensor = self.tensorService.getTensorfromLatentImage(latentImage)
        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)

        allTiles = []
        for tiles_pass in paddedTiles:
            tile_pass_l = []
            for tiles_group in tiles_pass:
                tile_group_l = []
                for padded_tile in tiles_group:
                    x, y, h, w = padded_tile.shape
                    if disableNoise:
                        tile_group_l.append(torch.zeros((B, C, h, w), dtype=latentTensor.dtype, layout=latentTensor.layout, device=device))
                    else:
                        batchIndex = self.tensorService.getBatchIndexfromLatentTensor(latentTensor)
                        tile_group_l.append(comfy.sample.prepare_noise(padded_tile, seed, batchIndex))
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    def getDenoiseMaskTilesforPTStrategy(self, noiseMask, tilePos, paddedTiles, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        allTiles = []
        for tile_pos_pass, tiles_pass in zip(tilePos, paddedTiles):
            tile_pass_l = []
            for tile_pos_group, tiles_group in zip(tile_pos_pass, tiles_pass):
                tile_group_l = []
                for (x, y, tile_w, tile_h), padded_tile in zip(tile_pos_group, tiles_group): 
                    b, c, padded_h, padded_w = padded_tile.shape
                    tile_mask = self.choseDenoiseStrategy((padded_w, padded_h), denoise_strategy, denoise_strategy_pars, device)
                    tiled_mask = None
                    if noiseMask is not None:
                        tiled_mask = self.tensorService.getSlice(noiseMask, (x, y, padded_w, padded_h))
                                                        
                    if tile_mask is not None:
                        if tiled_mask is not None:
                            tiled_mask *= tiled_mask
                        else:
                            tiled_mask = tile_mask      
                    tile_group_l.append(tiled_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    def getBlendMaskTilesforPTStrategy(self, latentImage, paddedTiles, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        allTiles = []
        for tiles_pass in paddedTiles:
            tile_pass_l = []
            for tiles_group in tiles_pass:
                tile_group_l = []
                for padded_tile in tiles_group:
                    tile_shape = padded_tile.shape
                    b, c, h, w = tile_shape
                    tile_group_l.append(self.choseBlendStrategy(latentImage, (w, h), blend_strategy, blend_strategy_pars, device))
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles

###### Multi-Pass - Padded Tiling Strategy ######  
class MP_PTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()
        self.pt_Service = PTService()

    def generatePos_forMP_PTStrategy(self, latentImage, tileSize, padding, passes: int):
        allTiles = []
        for i in range(passes):
            tile_pos = self.pt_Service.generatePosforPTStrategy(latentImage, tileSize, padding)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles

    def getTilesforMP_PTStrategy(self, latentImage, tilePos):
        return self.pt_Service.getTilesforPTStrategy(latentImage, tilePos)

    def getPaddedTilesforMP_PTStrategy(self, tiledTiles, tilePos, padding_strategy, padding, device: str = 'cpu'):
        return self.pt_Service.getPaddedTilesforPTStrategy(tiledTiles, tilePos, padding_strategy, padding, device)

    def getNoiseTilesforMP_PTStrategy(self, latentImage, paddedTiles, seed: int, disableNoise: bool, device: str = 'cpu'):
        return self.pt_Service.getNoiseTilesforPTStrategy(latentImage, paddedTiles, seed, disableNoise, device)

    def getDenoiseMaskTilesforMP_PTStrategy(self, noiseMask, tilePos, paddedTiles, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        return self.pt_Service.getDenoiseMaskTilesforPTStrategy(noiseMask, tilePos, paddedTiles, denoise_strategy, denoise_strategy_pars, device)

    def getBlendMaskTilesforMP_PTStrategy(self, latentImage, paddedTiles, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        return self.pt_Service.getBlendMaskTilesforPTStrategy(latentImage, paddedTiles, blend_strategy, blend_strategy_pars, device)

###### Singel Pass - Adjacency Padded Tiling Strategy ######
class APTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()
        self.pt_Service = PTService()
    
    def generatePosforAPTStrategy(self, latentImage, tileSize, padding: int):

        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        tile_w, tile_h = tileSize

        # Calculate how many tiles fit in each dimension
        num_tiles_x = W // tile_w
        num_tiles_y = H // tile_h

        # Adjust tile size to fit perfectly
        adjusted_tile_width = W // num_tiles_x
        adjusted_tile_height = H // num_tiles_y

        tile_positions = []

        for j in range(num_tiles_y):
            row_positions = []
            for i in range(num_tiles_x):
                x = i * adjusted_tile_width
                y = j * adjusted_tile_height

                # Adjust for padding
                left_pad = padding if i != 0 else 0 
                right_pad = padding if i != num_tiles_x - 1 else 0 
                top_pad = padding if j != 0 else 0 
                bottom_pad = padding if j != num_tiles_y - 1 else 0 

                # Ensure no overflow occurs at the edges (both horizontally and vertically)
                padded_tile_width = adjusted_tile_width + left_pad + right_pad
                padded_tile_height = adjusted_tile_height + top_pad + bottom_pad

                # Clamping the positions to ensure tiles stay within bounds and prevent overlap
                adjusted_x = max(0, min(x - left_pad, W - padded_tile_width))
                adjusted_y = max(0, min(y - top_pad, H - padded_tile_height))

                row_positions.append((adjusted_x, adjusted_y, padded_tile_width, padded_tile_height))

            tile_positions.append(row_positions)  # Maintain the correct nested structure

        return [tile_positions]
    
    def getTilesforAPTStrategy(self, latentImage, tilePos):
        return self.getTilesfromTilePos(latentImage, tilePos)

    def applyPaddingforAPTStrategy(self, tileTensor, paddingStrategy, padding, tilePos):

        pos_x, pos_y = tilePos
        # Calculate the number of tiles in x and y directions based on tile_positions
        num_tiles_x = max((x + tile_w) for (x, _, tile_w, _) in paddingStrategy) // tileTensor.shape[-1] + 1
        num_tiles_y = max((y + tile_h) for (_, y, _, tile_h) in paddingStrategy) // tileTensor.shape[-2] + 1

        if padding > 0:
            # Add padding for each side based on the tile's position
            left_pad = padding if pos_x != 0 else 0  # Only add padding if the tile is not on the left edge
            right_pad = padding if pos_x != num_tiles_x - 1 else 0  # Only add padding if the tile is not on the right edge
            top_pad = padding if pos_y != 0 else 0  # Only add padding if the tile is not on the top edge
            bottom_pad = padding if pos_y != num_tiles_y - 1 else 0  # Only add padding if the tile is not on the bottom edge

            # Apply padding using the chosen strategy
            if paddingStrategy in ["circular", "reflect", "replicate"]:
                return F.pad(tileTensor, (left_pad, right_pad, top_pad, bottom_pad), mode=paddingStrategy)
            elif paddingStrategy == "organic":
                return self.applyOrganicPadding(tileTensor, (left_pad, right_pad, top_pad, bottom_pad))
            else:
                return F.pad(tileTensor, (left_pad, right_pad, top_pad, bottom_pad), mode="constant", value=0)

        return tileTensor

    def getPaddedTilesforAPTStrategy(self, tiledTiles, tilePos, paddingStrategy, padding: int, device: str = 'cpu'):
        """
        Applies padding to all tiles using the multi-pass PTS strategy.
        """
        allTiles = []
        for tile_pos_pass, tiles_pass in zip(tilePos, tiledTiles):
            tile_pass_l = []
            for tile_pos_group, tiles_group in zip(tile_pos_pass, tiles_pass):
                tile_group_l = []
                for (x, y, tile_w, tile_h), tileTensor in zip(tile_pos_group, tiles_group):
                    padded_tile = self.applyPaddingforAPTStrategy(tileTensor, paddingStrategy, padding, (x, y), device)
                    tile_group_l.append(padded_tile)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)

        return allTiles
    
    def applyOrganicPadding(self, tileTensor, paddings, device: str = 'cpu'):

        B, C, H, W = tileTensor.shape
        
        # Unpack padding only for the sides where padding exists (values > 0)
        pad_len = len(paddings)
        
        # Default to 0 for missing sides (we could append missing sides as 0, so no padding is applied)
        left = paddings[0] if pad_len > 0 else 0
        right = paddings[1] if pad_len > 1 else 0
        top = paddings[2] if pad_len > 2 else 0
        bottom = paddings[3] if pad_len > 3 else 0

        # Create the padded tile
        padded_tile = torch.zeros((B, C, H + top + bottom, W + left + right), dtype=tileTensor.dtype, device=device)

        # Place the original tile in the center
        padded_tile[:, :, top:top + H, left:left + W] = tileTensor

        # Left padding (soft blended from left edge)
        if left > 0:
            left_patch = self.softBlend(tileTensor[:, :, :, :left], (B, C, H, left), flip_dims=[3], blend_factor=0.55, device=device)
            padded_tile[:, :, top:top + H, :left] = left_patch

        # Right padding
        if right > 0:
            right_patch = self.softBlend(tileTensor[:, :, :, -right:], (B, C, H, right), flip_dims=[3], blend_factor=0.55, device=device)
            padded_tile[:, :, top:top + H, -right:] = right_patch

        # Top padding
        if top > 0:
            top_patch = self.softBlend(tileTensor[:, :, :top, :], (B, C, top, W), flip_dims=[2], blend_factor=0.55, device=device)
            padded_tile[:, :, :top, left:left + W] = top_patch

        # Bottom padding
        if bottom > 0:
            bottom_patch = self.softBlend(tileTensor[:, :, -bottom:, :], (B, C, bottom, W), flip_dims=[2], blend_factor=0.55, device=device)
            padded_tile[:, :, -bottom:, left:left + W] = bottom_patch

        # Corner regions (blend both X and Y for smooth edges)
        if left > 0 and top > 0:
            padded_tile[:, :, :top, :left] = self.softBlend(tileTensor[:, :, :top, :left], (B, C, top, left), flip_dims=[2, 3], blend_factor=0.35, device=device)

        if right > 0 and top > 0:
            padded_tile[:, :, :top, -right:] = self.softBlend(tileTensor[:, :, :top, -right:], (B, C, top, right), flip_dims=[2, 3], blend_factor=0.35, device=device)

        if left > 0 and bottom > 0:
            padded_tile[:, :, -bottom:, :left] = self.softBlend(tileTensor[:, :, -bottom:, :left], (B, C, bottom, left), flip_dims=[2, 3], blend_factor=0.35, device=device)

        if right > 0 and bottom > 0:
            padded_tile[:, :, -bottom:, -right:] = self.softBlend(tileTensor[:, :, -bottom:, -right:], (B, C, bottom, right), flip_dims=[2, 3], blend_factor=0.35, device=device)

        return padded_tile
        
    def softBlend(self, patch, expandShape, flipDims, blend_factor=0.7, device: str = 'cpu'):

        mirrored = patch
        for flip_dim in flipDims:
            mirrored = torch.flip(mirrored, dims=[flip_dim])

        # Adaptive blending factor (less boxy)
        blend_factor = torch.tensor(blend_factor, device=device).expand_as(patch)

        # Perlin-style noise (natural variation, no harsh edges)
        noise = (torch.rand_like(patch) - 0.5) * 1.5 
        blend_factor = torch.clamp(blend_factor + noise, 0.4, 0.5)

        return (patch * blend_factor + mirrored * (1 - blend_factor)).expand(expandShape)
    
    def getNoiseTilesforAPTStrategy(self, latentImage, paddedTiles, seed: int, disableNoise: bool, device: str = 'cpu'):
        return self.pt_Service.getNoiseTilesforPTStrategy(latentImage, paddedTiles, seed, disableNoise, device)

    def getDenoiseMaskTilesforAPTStrategy(self, noiseMask, tilePos, paddedTiles, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        return self.pt_Service.getDenoiseMaskTilesforPTStrategy(noiseMask, tilePos, paddedTiles, denoise_strategy, denoise_strategy_pars, device)

    def getBlendMaskTilesforAPTStrategy(self, latentImage, paddedTiles, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        return self.pt_Service.getBlendMaskTilesforPTStrategy(latentImage, paddedTiles, blend_strategy, blend_strategy_pars, device)
    
    
###### Multi-Pass - Adjacency Padded Tiling Strategy ######   
class MP_APTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

        # pts - Stands for Padded Tiling Streteg
        self.apt_Service = APTService()

    def generatePos_forMP_APTStrategy(self, latentImage, tileSize, padding: int, passes: int):
        allTiles = []
        for i in range(passes):
            tile_pos = self.apt_Service.generatePosforAPTStrategy(latentImage, tileSize, padding)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles
    
    def getTilesforMP_APTStrategy(self, latentImage, tilePos):
        return self.apt_Service.getTilesforAPTStrategy(latentImage, tilePos)

    def getPaddedTilesforMP_APTStrategy(self, tiledTiles, tilePos, padding_strategy, padding: int, device: str = 'cpu'):
        return self.apt_Service.getPaddedTilesforAPTStrategy(tiledTiles, tilePos, padding_strategy, padding, device)
    
    def getNoiseTilesforMP_APTStrategy(self, latentImage, paddedTiles, seed: int, disableNoise: bool, device: str = 'cpu'):
        return self.apt_Service.getNoiseTilesforAPTStrategy(latentImage, paddedTiles, seed, disableNoise, device)
    
    def getDenoiseMaskTilesforMP_APTStrategy(self, noiseMask, tilePos, paddedTiles, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        return self.apt_Service.getDenoiseMaskTilesforAPTStrategy(noiseMask, tilePos, paddedTiles, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendMaskTilesforMP_APTStrategy(self, latentImage, padded_tiles, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        return self.apt_Service.getBlendMaskTilesforAPTStrategy(latentImage, padded_tiles, blend_strategy, blend_strategy_pars, device)


###### Singel Pass - Context-Padded Tiling Strategy ######   
class CPTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

    def generatePosforCPTStrategy(self, latentImage, tileSize):
        """
        Single-Pass Context-Padded Tiling Strategy.
        Generates only one pass without shift-based multi-pass logic.
        """
        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        tile_w, tile_h = tileSize

        tile_size_h = min(H, max(4, (tile_w // 4) * 4))  # Use tile_width
        tile_size_w = min(W, max(4, (tile_h // 4) * 4))  # Use tile_height

        h = np.arange(0, H, tile_size_h)
        w = np.arange(0, W, tile_size_w)

        def create_tile(i, j):
            h_start = int(h[i])
            w_start = int(w[j])
            h_size = min(tile_size_h, H - h_start)
            w_size = min(tile_size_w, W - w_start)

            return (w_start, h_start, w_size, h_size)


        passes = [
            [[create_tile(i, j) for i in range(len(h)) for j in range(len(w))]],
        ]
        
        return passes
    
    def getTilesforCPTStrategy(self, latentImage, tilePos):
        return self.getTilesfromTilePos(latentImage, tilePos)
    
    def getNoiseTilesforCPTStrategy(self, latentImage, tilePos):
        return self.getNoiseTilesfromTilePos(latentImage, tilePos)
    
    def getDenoiseMaskTilesforCPTStrategy(self, latentImage, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        allTiles = []
        for tile_pos_pass in tilePos:
            tile_pass_l = []
            for tile_pos_group in tile_pos_pass:
                tile_group_l = []
                for i, (x, y, tile_w, tile_h) in enumerate(tile_pos_group):
                    tile_mask = self.choseDenoiseStrategy((tile_w, tile_h), denoise_strategy, denoise_strategy_pars, device)
                    tiled_mask = None
                    if noiseMask is not None:
                        tiled_mask = self.tensorService.getSlice(noiseMask, (x, y, tile_w, tile_h))
                    if tile_mask is not None:
                        tiled_mask = tiled_mask * tile_mask if tiled_mask is not None else tile_mask
                    (x, y, tile_w, tile_h), tiled_mask = self.denoiseService.generateDenoiseMask_atBoundary(latentImage, (x, y, tile_w, tile_h), tiled_mask, device)
                    tile_pos_group[i] = (x, y, tile_w, tile_h)
                    tile_group_l.append(tiled_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
    def getBlendeMaskTilesforCPTStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        return self.getBlendMaskTiles(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Multi-Pass - Context-Padded Tiling Strategy ######   
class MP_CPTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()
        # cpt - Stands for Context-Padded Tiling Stretegy
        self.cpt_Service = CPTService()


    def generatePosforMP_CPTStrategy(self, latentImage, tileSize):

        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        tile_w, tile_h = tileSize

        tile_size_h = min(H, max(4, (tile_w // 4) * 4))
        tile_size_w = min(W, max(4, (tile_h // 4) * 4))

        h = np.arange(0, H, tile_size_h)
        h_shift = np.arange(tile_size_h // 2, H - tile_size_h // 2, tile_size_h)
        w = np.arange(0, W, tile_size_w)
        w_shift = np.arange(tile_size_w // 2, W - tile_size_w // 2, tile_size_w)

        def create_tile(hs, ws, i, j):
            h = int(hs[i])
            w = int(ws[j])
            h_len = min(tile_size_h, H - h)
            w_len = min(tile_size_w, W - w)

            return (h, w, h_len, w_len)
        passes = [
            [[create_tile(h,       w,       i, j) for i in range(len(h))       for j in range(len(w))]],
            [[create_tile(h_shift, w,       i, j) for i in range(len(h_shift)) for j in range(len(w))]],
            [[create_tile(h,       w_shift, i, j) for i in range(len(h))       for j in range(len(w_shift))]],
            [[create_tile(h_shift, w_shift, i, j) for i in range(len(h_shift)) for j in range(len(w_shift))]],
        ]
        
        return passes
    
    def getTilesforMP_CPTStrategy(self, latentImage, tilePos):
        return self.cpt_Service.getTilesforCPTStrategy(latentImage, tilePos)
    
    def getNoiseTilesforMP_CPTStrategy(self, latentImage, tilePos):
        return self.cpt_Service.getNoiseTilesforCPTStrategy(latentImage, tilePos)
    
    def getDenoiseMaskTilesforMP_CPTStrategy(self, latentImage, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        return self.cpt_Service.getDenoiseMaskTilesforCPTStrategy(latentImage, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendeMaskTilesforMP_CPTStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        return self.cpt_Service.getBlendeMaskTilesforCPTStrategy(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Single Pass - Overlap Tiling Strategy ######   
class OVPStrategy(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

    def generatePosforOVPStrategy(self, latentImage, tileSize, overlap: int):

        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        tile_w, tile_h = tileSize

        # Compute effective stride
        stride_x = max(1, tile_w - overlap)  
        stride_y = max(1, tile_h - overlap)  

        # Calculate number of tiles, ensuring at least one tile
        num_tiles_x = max(1, (W - overlap) // stride_x)
        num_tiles_y = max(1, (H - overlap) // stride_y)

        tile_positions = []

        for j in range(num_tiles_y + 1):
            row_positions = []
            for i in range(num_tiles_x + 1):
                x = i * stride_x
                y = j * stride_y

                if x + tile_w > W:
                    x = W - tile_w
                if y + tile_h > H:
                    y = H - tile_h

                row_positions.append((x, y, tile_w, tile_h))

            tile_positions.append(row_positions)

        return [tile_positions]
    
    def getTilesforOVPStrategy(self, latentImage, tilePos):
        return self.getTilesfromTilePos(latentImage, tilePos)
    
    def getNoiseTilesforOVPStrategy(self, latentImage, tilePos):
        return self.getTilesfromTilePos(latentImage, tilePos)
    
    def getDenoiseMaskTilesforOVPStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        return self.getDenoiseMaskTiles(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendeMaskTilesforOVPStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        return self.getBlendMaskTiles(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Multi-Pass - Overlap Tiling Strategy ######   
class MP_OVPStrategy(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

        self.ovp_Service = OVPStrategy()

    def generatePosforMP_OVPStrategy(self, latentImage, tileSize, overlap: float, passes: int):

        allTiles = []
        for i in range(passes):
            tile_pos = self.ovp_Service.generatePosforOVPStrategy(latentImage, tileSize, overlap)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles
    
    def getTilesforMP_OVPStrategy(self, latentImage, tilePos):
        return self.ovp_Service.getTilesforOVPStrategy(latentImage, tilePos)
    
    def getNoiseTilesforMP_OVPStrategy(self, latentImage, tilePos):
        return self.ovp_Service.getNoiseTilesforOVPStrategy(latentImage, tilePos)
    
    def getDenoiseMaskTilesforMP_OVPStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        return self.ovp_Service.getDenoiseMaskTilesforOVPStrategy(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendeMaskTilesforMP_OVPStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        return self.ovp_Service.getBlendeMaskTilesforOVPStrategy(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Single Pass - Adaptive Tiling Strategy ######   
class ADPService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

        self.baseModelResService = BaseModelResolutionService()

    def calculateEntropy(self, tileTensor):
        """
        Calculate entropy for a given tile.
        Entropy measures the disorder or randomness in the image patch.
        """
        flattened_tile = tileTensor.reshape(-1)  
        hist = torch.histc(flattened_tile.float(), bins=256, min=0, max=255)
        hist = hist / hist.sum() 
        entropy = -torch.sum(hist * torch.log2(hist + 1e-8)) 
        return entropy

    def smoothComplexityMap(self, complexityMap, kernel_size: int = 2, device: str = 'cpu'):
        """
        Smooth the complexity map to reduce abrupt changes between adjacent tiles.
        We apply a smaller smoothing kernel here to retain more fine-grained entropy.
        """
        # Apply a simple smoothing filter like a box blur
        padding = kernel_size // 2
        kernel = torch.ones((kernel_size, kernel_size), device=device) / (kernel_size**2)
        smoothed_map = torch.nn.functional.conv2d(complexityMap.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=padding)
        return smoothed_map.squeeze(0).squeeze(0)

    def calculateComplexity(self, latentTensor, tile_size: int = 32, device: str = 'cpu'): 
        """
        Compute entropy for each tile to estimate complexity.
        Higher entropy → high complexity (edges, textures)
        Lower entropy → smooth regions
        """
        B, C, H, W = latentTensor.shape
        complexity_map = torch.zeros((H, W), device=device)

        for i in range(0, H, tile_size):
            for j in range(0, W, tile_size):
                tileTensor = latentTensor[:, :, i:i+tile_size, j:j+tile_size]
                entropy = self.calculateEntropy(tileTensor)
                complexity_map[i:i+tile_size, j:j+tile_size] = entropy

        # Apply lighter smoothing to reduce extreme transitions
        complexity_map = self.smoothComplexityMap(complexity_map, kernel_size=2, device=device)
        return complexity_map

    def generatePosforADPStrategy(self, latentImage, tileSize, base_model : str, VRAM : str, precision: str, tile_growth_exponent: float = 0.65, device: str = 'cpu'):
        """
        Generate tile positions based on complexity.
        Dynamically adjusts tile size based on the base_tile_size.
        """
        latentTensor = self.tensorService.getTensorfromLatentImage(latentImage)
        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        tile_w, tile_h = tileSize

        average_latent_size = (H + W) // 2
        average_tile_size = int((tile_w + tile_h) / 2) 
        latent_downscale = 8  # Usually 8 for SD1.x, 4 for SDXL
        
        # Get the max resolution and convert it to latent space
        max_resolution_px = self.baseModelResService.getPracticalMaxAverage(base_model, precision, VRAM)
        max_resolution_latent = max_resolution_px // latent_downscale if max_resolution_px else 128

        # Calculate scale ratio and adaptive min tile size
        scale_ratio = average_latent_size / max_resolution_latent
        min_tile_size = int(min(max_resolution_latent, 16 * (1 + scale_ratio**tile_growth_exponent)))

        complexity_map = self.calculateComplexity(latentTensor, tile_size=min_tile_size, device=device)

        def create_tile(x, y):
            complexity = complexity_map[y, x].item()
            
            # Instead of clamping, dynamically adjust the tile size
            tile_size = int(average_tile_size * (1.5 - complexity / torch.max(complexity_map)))

            # Adjust the tile size within a dynamic range
            tile_size = min(max(tile_size, min_tile_size), max_resolution_latent)

            tile_w = min(tile_size, W - x)
            tile_h = min(tile_size, H - y)
            return (x, y, tile_w, tile_h)

        tile_positions = [[[]]] 

        y = 0
        while y < H:
            x = 0
            while x < W:
                tile_positions[0][0].append(create_tile(x, y))
                x += tile_positions[0][0][-1][2]  # Move x by tile width
            y += tile_positions[0][0][-1][3]  # Move y by tile height

        return tile_positions
    
    def getTilesforADPStrategy(self, latentImage, tilePos):
        return self.getTilesfromTilePos(latentImage, tilePos)
    
    def getNoiseTilesforADPStrategy(self, latentImage, tilePos):
        return self.getNoiseTilesfromTilePos(latentImage, tilePos)
    
    def getDenoiseMaskTilesforADPStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device='cpu'):
        return self.getDenoiseMaskTiles(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendeMaskTilesforADPStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device='cpu'):
        return self.getBlendMaskTiles(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Multi-Pass - Adaptive Tiling Strategy ######   
class MP_ADPService(Tiling_Strategy_Base):
    def __init__(self):
        super().__init__()

        # adp - Stands for Adaptive Tiling Strategy
        self.adp_Service = ADPService()

    def generatePos_forMP_ADPStrategy(self, latentImage, tileSize, passes: int, base_model : str, VRAM : str, precision: str, tile_growth_exponent: float = 0.65):

        latentTensor = self.tensorService.getTensorfromLatentImage(latentImage)
        tile_positions = self.adp_Service.generatePosforADPStrategy(latentImage, tileSize, base_model, precision, VRAM, tile_growth_exponent)

        for pass_idx in range(passes): 
            refined_positions = []
            num_new_tiles = 0
            max_new_tiles = len(tile_positions[pass_idx - 1][0]) * 2  

            for (x, y, w, h) in tile_positions[pass_idx - 1][0]:
                region = latentTensor[:, :, y:y + h, x:x + w]
                complexity = torch.std(region).item()

                # Refine only if complexity is high and we haven’t exceeded the max new tile count
                if complexity > 0.1 and num_new_tiles < max_new_tiles and w > 16 and h > 16:
                    refined_positions.append(self.adjustTile(x, y, w, h))
                    num_new_tiles += 1  
                else:
                    refined_positions.append((x, y, w, h)) 

            tile_positions.append([refined_positions])

        return tile_positions

    def adjustTile(self, x, y, w, h):
        new_w = max(16, w // 2)
        new_h = max(16, h // 2)
        return (x, y, new_w, new_h)
    
    def getTilesforMP_ADPStrategy(self, latentImage, tilePos):
        return self.adp_Service.getTilesforADPStrategy(latentImage, tilePos)
    
    def getNoiseTilesforMP_ADPStrategy(self, latentImage, tilePos):
        return self.adp_Service.getNoiseTilesforADPStrategy(latentImage, tilePos)
    
    def getDenoiseMaskTilesforMP_ADPStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        return self.adp_Service.getDenoiseMaskTilesforADPStrategy(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendeMaskTilesforMP_ADPStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        return self.adp_Service.getBlendeMaskTilesforADPStrategy(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Single Pass - Hierarchical Tiling Strategy ######  
class HRCService(Tiling_Strategy_Base):
    def __init__(self):
        super().__init__()

        self.baseModelResService = BaseModelResolutionService()

    def calculateEntropy(self, tileTensor):
        """
        Calculate entropy (complexity) of a tile.
        """
        flattened_tile = tileTensor.reshape(-1)
        hist = torch.histc(flattened_tile.float(), bins=256, min=0, max=255)
        hist = hist / hist.sum() 
        entropy = -torch.sum(hist * torch.log2(hist + 1e-8))
        return entropy.item()

    def generatePosforHRCStrategy(self, latentImage, base_model: str, VRAM: str, precision : str, tile_growth_exponent: float = 0.65, complexity_threshold: float = 0.1):
        """
        Fully adaptive hierarchical tiling: Recursively refines high-complexity tiles.
        Ensures that valid tiles are always generated to avoid empty sequences.
        """
        latentTensor = self.tensorService.getTensorfromLatentImage(latentImage)
        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        tile_queue = [(B, C, W, H)]
        final_tiles = []

        average_latent_size = (H + W) // 2
        # Get the max resolution and convert it to latent space
        latent_downscale = 8  # Usually 8 for SD1.x, 4 for SDXL
        max_resolution_px = self.baseModelResService.getPracticalMaxAverage(base_model, precision, VRAM)
        max_resolution_latent = max_resolution_px // latent_downscale if max_resolution_px else 128  # fallback
        print("max_resolution_latent:", max_resolution_latent)

        # Calculate scale ratio and adaptive min tile size
        scale_ratio = average_latent_size / max_resolution_latent
        min_tile_size = int(min(max_resolution_latent, 16 * (1 + scale_ratio**tile_growth_exponent)))
        print("min_tile_size:", min_tile_size)
        

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
            tileTensor = latentTensor[:, :, y:y+h, x:x+w]
            complexity = self.calculateEntropy(tileTensor)

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

    def getTilesforHRCStrategy(self, latentImage, tilePos):
        return self.getTilesfromTilePos(latentImage, tilePos)
    
    def getNoiseTilesforHRCStrategy(self, latentImage, tilePos):
        return self.getNoiseTilesfromTilePos(latentImage, tilePos)
    
    def getDenoiseMaskTilesforHRCStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str ='cpu'):
        return self.getDenoiseMaskTiles(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendMaskTilesforHRCStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str ='cpu'):
        return self.getBlendMaskTiles(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Multiple Pass - Hierarchical Tiling Strategy ######  
class MP_HRCService(Tiling_Strategy_Base):
    
    def __init__(self, ):
        super().__init__()

        self.hrc_Service = HRCService()

    def generatePos_forMP_HRCStrategy(self, latentImage, passes: int , base_model: str, VRAM: str, precision : str, tile_growth_exponent: float = 0.65, complexity_threshold: float = 0.1, complexity_variation: float = 0.05):
        """
        Multi-pass adaptive hierarchical tiling. Varies complexity threshold across passes.
        """
        allTiles = []
        for i in range(passes):
            variation = (i - passes // 2) * complexity_variation
            complexity_threshold = max(0.01, complexity_threshold + variation)
            tile_pos = self.hrc_Service.generatePosforHRCStrategy(latentImage, base_model, VRAM, precision, tile_growth_exponent, complexity_threshold)
            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)
        return allTiles

    def getTilesforMP_HRCStrategy(self, latentImage, tilePos):
        return self.hrc_Service.getTilesforHRCStrategy(latentImage, tilePos)

    def getNoiseTilesforMP_HRCStrategy(self, latentImage, tilePos):
        return self.hrc_Service.getNoiseTilesforHRCStrategy(latentImage, tilePos)

    def getDenoiseMaskTilesforMP_HRCStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str = 'cpu'):
        return self.hrc_Service.getDenoiseMaskTilesforHRCStrategy(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)

    def getBlendMaskTilesforMP_HRCStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str = 'cpu'):
        return self.hrc_Service.getBlendMaskTilesforHRCStrategy(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)


###### Single Pass - Random-Hierarchical Tiling Strategy ######  
class RTHRCService(Tiling_Strategy_Base):
    
    def __init__(self):
        super().__init__()
        
        self.baseModelResService = BaseModelResolutionService()

    def calculateEntropy(self, tileTensor):
        """
        Calculate entropy (complexity) of a tile.
        """
        flattened_tile = tileTensor.reshape(-1)
        hist = torch.histc(flattened_tile.float(), bins=256, min=0, max=255)
        hist = hist / hist.sum() 
        entropy = -torch.sum(hist * torch.log2(hist + 1e-8))
        return entropy.item()

    def generateRandomRegionsCoveringWholeArea(self, latentImage, seed: int, min_size: int = 64, max_size: int = 256):
        random.seed(seed)

        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)

        def generate_split_points(total, min_size, max_size):
            points = [0]
            while points[-1] < total:
                remaining = total - points[-1]
                if remaining < min_size:
                    break
                max_allowed = min(max_size, remaining)
                jittered = random.randint(min_size, max_allowed)
                points.append(points[-1] + jittered)
            if points[-1] < total:
                points.append(total)  
            return [points[i+1] - points[i] for i in range(len(points) - 1)]

        tile_widths = generate_split_points(W, min_size, max_size)
        tile_heights = generate_split_points(H, min_size, max_size)

        regions = []
        y = 0
        for h in tile_heights:
            x = 0
            for w in tile_widths:
                regions.append((x, y, w, h))
                x += w
            y += h

        return regions

    def generateSubtilesFromRegion(self, latentImage, tilePos, min_tile_size, complexity_threshold):

        latentTensor = self.tensorService.getTensorfromLatentImage(latentImage)
        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        x, y, w, h = tilePos
        
        tile_queue = [(x, y, w, h)]
        subtiles = []
        max_jitter_ratio = 0.1  # Jitter up to 10% of the dimension

        while tile_queue:
            tx, ty, tw, th = tile_queue.pop(0)

            # Clamp to image bounds
            tw = min(tw, W - tx)
            th = min(th, H - ty)

            if tw < min_tile_size or th < min_tile_size:
                subtiles.append((tx, ty, tw, th))  # use clamped size directly
                continue

            tileTensor = latentTensor[:, :, ty:ty+th, tx:tx+tw]
            entropy = self.calculateEntropy(tileTensor)

            if entropy > complexity_threshold and tw > min_tile_size * 2 and th > min_tile_size * 2:
                # Random split ratios
                w_ratio = random.uniform(0.3, 0.7)
                h_ratio = random.uniform(0.3, 0.7)

                # Split positions with jitter
                w_split = int(tw * w_ratio)
                h_split = int(th * h_ratio)

                w_jitter = int(tw * max_jitter_ratio)
                h_jitter = int(th * max_jitter_ratio)

                w_split += random.randint(-w_jitter, w_jitter)
                h_split += random.randint(-h_jitter, h_jitter)

                # Clamp splits to safe range
                w_split = max(min_tile_size, min(tw - min_tile_size, w_split))
                h_split = max(min_tile_size, min(th - min_tile_size, h_split))

                tile_queue.extend([
                    (tx,         ty,         w_split, h_split),
                    (tx+w_split, ty,         tw-w_split, h_split),
                    (tx,         ty+h_split, w_split, th-h_split),
                    (tx+w_split, ty+h_split, tw-w_split, th-h_split),
                ])
            else:
                subtiles.append((tx, ty, tw, th))

        return subtiles

    def generatePosforRTHRCStrategy(self, latentImage, seed: int, base_model: str, VRAM: str, precision: str, tile_growth_exponent=0.65, complexity_threshold=0.1):
        
        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        
        latent_downscale = 8
        max_res = self.baseModelResService.getPracticalMaxAverage(base_model, precision, VRAM)
        max_latent = max_res // latent_downscale if max_res else 128
        avg_latent = (W + H) // 2 
        scale_ratio = avg_latent / max_latent
        min_tile_size = max(16, int(round((16 * (1 + scale_ratio**tile_growth_exponent)) / 16) * 16))

        minRegionSize = 64  
        regions = self.generateRandomRegionsCoveringWholeArea(latentImage, seed, min_size=minRegionSize, max_size=max_latent)

        final_tiles = []
        for x, y, w, h in regions:
            subtiles = self.generateSubtilesFromRegion(latentImage, (x, y, w, h), min_tile_size, complexity_threshold)
            final_tiles.extend(subtiles)

        return [[final_tiles]] 

    def getTilesforRTHRCStrategy(self, latentImage, tilePos):
        return self.getTilesfromTilePos(latentImage, tilePos)

    def getNoiseTilesforRTHRCStrategy(self, latentImage, tilePos):
        return self.getNoiseTilesfromTilePos(latentImage, tilePos)

    def getDenoiseMaskTilesforRTHRCStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device='cpu'):
        return self.getDenoiseMaskTiles(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)

    def getBlendMaskTilesforRTHRCStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device='cpu'):
        return self.getBlendMaskTiles(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Multi-Pass - Random-Hierarchical Tiling Strategy ######
class MP_RTHRCService(Tiling_Strategy_Base):
    
    def __init__(self):
        super().__init__()
        self.rthrc_Service = RTHRCService()

    def generatePos_forMP_RTHRCStrategy(self, latentImage, seed: int, passes: int, base_model: str, VRAM: str, precision: str,
                                        tile_growth_exponent: float = 0.65, complexity_threshold: float = 0.1, complexity_variation: float = 0.05,):
        """
        Multi-pass version of RTHRC strategy with variable complexity threshold and seed per pass.
        """
        all_tiles = []

        for i in range(passes):
            variation = (i - passes // 2) * complexity_variation
            adjusted_threshold = max(0.01, complexity_threshold + variation)
            adjusted_seed = seed + i * 173  # vary seed per pass to shuffle regions

            tile_pos = self.rthrc_Service.generatePosforRTHRCStrategy(
                latentImage,
                adjusted_seed,
                base_model,
                VRAM,
                precision,
                tile_growth_exponent,
                adjusted_threshold
            )

            all_tiles.append(tile_pos[0])  # flatten outer list structure [[tiles]]

        return all_tiles
    
    def getTilesforMP_RTHRCStrategy(self, latentImage, tilePos):
        return self.rthrc_Service.getTilesforRTHRCStrategy(latentImage, tilePos)

    def getNoiseTilesforMP_RTHRCStrategy(self, latentImage, tilePos):
        return self.rthrc_Service.getNoiseTilesforRTHRCStrategy(latentImage, tilePos)

    def getDenoiseMaskTilesforMP_RTHRCStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str ='cpu'):
        return self.rthrc_Service.getDenoiseMaskTilesforRTHRCStrategy(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)

    def getBlendMaskTilesforMP_RTHRCStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str ='cpu'):
        return self.rthrc_Service.getBlendMaskTilesforRTHRCStrategy(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Single-Pass - Non-Uniform Tiling Strategy ###### 
class NUService(Tiling_Strategy_Base): 
    
    def __init__(self):
        super().__init__()
        
        self.baseModelResService = BaseModelResolutionService()

    def computeDetailMap(self, latentImage, device: str = 'cpu'):
        """
        Compute a more robust detail/complexity map by combining low- and high-frequency information.
        This version uses edge detection and local variance to better capture image complexity.
        """
        latentTensor = self.tensorService.getTensorfromLatentImage(latentImage)
        B, _, _, _ = self.tensorService.getShapefromLatentImage(latentImage)
        
        latent_np = latentTensor.squeeze().cpu().numpy()

        # Step 1: Compute gradient magnitude (edge detection)
        grad_x = sobel(latent_np, axis=2)
        grad_y = sobel(latent_np, axis=1) 
        grad_mag = np.hypot(grad_x, grad_y)

        # Step 2: Normalize gradient magnitude for better control over edge data
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

        # Step 3: Compute local variance (texture complexity)
        texture_map = gaussian_filter(latent_np, sigma=2)
        local_variance = np.var(texture_map, axis=(1, 2))  

        # Step 4: Normalize variance map to match range [0, 1]
        local_variance = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min() + 1e-8)

        # Reshape local_variance to match gradient map's shape (broadcast it)
        local_variance_map = local_variance[:, np.newaxis, np.newaxis] 
        local_variance_map = np.broadcast_to(local_variance_map, grad_mag.shape)  

        # Step 5: Combine edge map and local variance map
        detail_map = grad_mag + local_variance_map
        detail_map = np.clip(detail_map, 0, 1)

        return torch.tensor(detail_map, dtype=latentTensor.dtype, device=device).unsqueeze(0).expand(B, -1, -1, -1)


    def generatePosforNUStrategy(self, latentImage, base_model: str, VRAM: str, precision: str, tile_growth_exponent: float = 0.65, 
                                scale_min: float = 0.25, scale_max: float = 2.0, device: str = 'cpu'
                                ):

        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)

        latent_downscale = 8
        max_res = self.baseModelResService.getPracticalMaxAverage(base_model, precision, VRAM)
        max_latent = max_res // latent_downscale if max_res else 128
        avg_latent = (W + H) // 2
        scale_ratio = avg_latent / max_latent

        # Compute minimum adaptive tile size (rounded to multiple of 16)
        min_tile_size = max(16, int(round((16 * (1 + scale_ratio**tile_growth_exponent)) / 16) * 16))
        base_tile_size = 96 
        max_tile_size = min(128, max(W, H))

        # Compute complexity map
        complexity_map = self.computeDetailMap(latentImage, device)

        if complexity_map.dim() != 3:
            print("Unexpected complexity map dimensions:", complexity_map.shape)
            return []

        complexity_map = (complexity_map - complexity_map.min()) / (complexity_map.max() - complexity_map.min() + 1e-8)

        def create_tile(x, y):
            complexity = complexity_map[:, y, x].mean().item()
            tile_size = int(base_tile_size * (scale_min + (scale_max - scale_min) * complexity))
            tile_size += np.random.randint(-15, 15)
            tile_size = max(min_tile_size, min(tile_size, max_tile_size))
            tile_w = min(tile_size, W - x)
            tile_h = min(tile_size, H - y)
            return (x, y, tile_w, tile_h)

        tile_positions = [[[]]]
        y = 0
        while y < H:
            x = 0
            while x < W:
                tile = create_tile(x, y)
                tile_positions[0][0].append(tile)
                x += tile[2]

            # Safe step calculation to avoid low >= high in randint
            remaining_height = H - y
            step_limit = min(min_tile_size + 32, remaining_height)
            if step_limit > min_tile_size:
                y += np.random.randint(min_tile_size, step_limit)
            else:
                y += step_limit

        return tile_positions
            
    def getTilesforNUStrategy(self, latentImage, tilePos):
        return self.getTilesfromTilePos(latentImage, tilePos)
    
    def getNoiseTilesforNUStrategy(self, latentImage, tilePos):
        return self.getNoiseTilesfromTilePos(latentImage, tilePos)
    
    def getDenoiseMaskTilesforNUStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device: str ='cpu'):
        return self.getDenoiseMaskTiles(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendeMaskTilesforNUStrategy(self, latentImage, tilePos, blend_strategy, blend_strategy_pars, device: str ='cpu'):
        return self.getBlendMaskTiles(latentImage, tilePos, blend_strategy, blend_strategy_pars, device)

###### Multi-Pass - Non-Uniform Tiling Strategy ######  
class MP_NUService(Tiling_Strategy_Base): 
    def __init__(self):
        super().__init__()
        
        # nu - Stands for Non-Uniform Tiling Strategy
        self.nu_Service = NUService()

    def generatePosforMP_NUStrategy(self, latentImage, passes: int, base_model: str, VRAM: str, precision: str, 
                                 tile_growth_exponent: float = 0.65, scale_min: float = 0.25, scale_max: float = 2.0):
        """
        Generates non-uniform tiles using a depth-array format: [[[pass_1_tiles]]]
        with dynamically generated scale patterns based on input min/max.
        """
        allTiles = []

        def getScaledPattern(pass_idx: int):
            """
            Varies the scale pattern based on the pass number.
            This creates diversity across passes using input scale_min and scale_max as bounds.
            """
            range_span = scale_max - scale_min
            # Cycle bias: small, balanced, large
            if pass_idx % 3 == 0:  # Small-focused
                return (scale_min, scale_min + range_span * 0.6)
            elif pass_idx % 3 == 1:  # Medium-balanced
                mid = (scale_min + scale_max) / 2
                return (mid - range_span * 0.25, mid + range_span * 0.25)
            else:  # Large-focused
                return (scale_min + range_span * 0.2, scale_max)

        for i in range(passes):
            cur_scale_min, cur_scale_max = getScaledPattern(i)

            tile_pos = self.nu_Service.generatePosforNUStrategy(latentImage, base_model, VRAM, precision, tile_growth_exponent=tile_growth_exponent, 
                                                                scale_min=cur_scale_min, scale_max=cur_scale_max
            )

            for tile_pos_pass in tile_pos:
                tile_pass_l = []
                for tile_pos_group in tile_pos_pass:
                    tile_pass_l.append(tile_pos_group)
                allTiles.append(tile_pass_l)

        return allTiles
    
    def getTilesforMP_NUStrategy(self, latentImage, tilePos):
        return self.nu_Service.getTilesforNUStrategy(latentImage, tilePos)
    
    def getNoiseTilesforMP_NUStrategy(self, latentImage, tilePos):
        return self.nu_Service.getTilesforNUStrategy(latentImage, tilePos)
    
    def getDenoiseMaskTilesforMP_NUStrategy(self, noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device='cpu'):
        return self.nu_Service.getDenoiseMaskTilesforNUStrategy(noiseMask, tilePos, denoise_strategy, denoise_strategy_pars, device)
    
    def getBlendeMaskTilesforMP_NUStrategy(self, latent_image, tiled_positions, blend_strategy, blend_strategy_pars, device='cpu'):
        return self.nu_Service.getBlendeMaskTilesforNUStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars, device)

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
        # rthrc - Stands Random-Hierarchical Tiling Strategy 
        self.rthrc_Service  = RTHRCService()
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
        # rthrc - Stands Random-Hierarchical Tiling Strategy 
        self.mp_rthrc_Service  = MP_RTHRCService()
        # mp_nu - Stands for Multi-Pass Non-Uniform Tiling Strategy
        self.mp_nu_Service = MP_NUService()
