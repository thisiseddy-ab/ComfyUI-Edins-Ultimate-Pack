#### Build In Lib's #####
import sys
import os
import random
from typing import List, Tuple


import comfy.sampler_helpers
import nodes
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.controlnet

from nodes import MAX_RESOLUTION

#### My Lib's #####
from EUP.services import basic_utils

#### Third Party Lib's #####
import torch
import torch.nn.functional as F
import numpy as np
import latent_preview
from tqdm.auto import tqdm

#### Custom Nodes ####
from custom_nodes.ComfyUi_NNLatentUpscale import nn_upscale

class LatentUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
                "upscale_version": (["SDXL", "SD 1.x"],),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale_latent"

    def upscale_latent(self, latent, scale_factor, upscale_version):
        nn_upscaler = nn_upscale.NNLatentUpscale()
        return nn_upscaler.upscale(latent=latent, version=upscale_version, upscale=scale_factor)

def recursion_to_list(obj, attr):
    current = obj
    yield current
    while True:
        current = getattr(current, attr, None)
        if current is not None:
            yield current
        else:
            return


class ControlNetService():

    def __init__(self):
        self.tensorService = TensorService()
    
    def extractCnet(self, positive, negative):
        cnets = [c['control'] for (_, c) in positive + negative if 'control' in c]
        cnets = list(set([x for m in cnets for x in recursion_to_list(m, "previous_controlnet")]))  # Recursive extraction
        return [x for x in cnets if isinstance(x, comfy.controlnet.ControlNet)]
    
    def prepareCnet_imgs(self, cnets, shape):
        return [
            torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
            if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 else None
            for m in cnets
    ]

    def sliceCnet(self, h, h_len, w, w_len, model:comfy.controlnet.ControlBase, img):
        if img is None:
            img = model.cond_hint_original
        hint = self.tensorService.getSlice(img, h*8, h_len*8, w*8, w_len*8)
        if isinstance(model, comfy.controlnet.ControlLora):
            model.cond_hint = hint.float().to(model.device)
        else:
            model.cond_hint = hint.to(model.control_model.dtype).to(model.control_model.device)
    
    def prepareSlicedCnets(self, pos, neg, cnets, cnet_imgs, tile_h, tile_h_len, tile_w, tile_w_len):
        for idx, (m, img) in enumerate(zip(cnets, cnet_imgs)):
            self.sliceCnet(tile_h, tile_h_len, tile_w, tile_w_len, m, img)
            
            # Replace the original 'control' key in the corresponding positions
            if idx < len(pos):
                pos[idx] = (None, {'control': m})
            else:
                pos.append((None, {'control': m}))
            
            if idx < len(neg):
                neg[idx] = (None, {'control': m})
            else:
                neg.append((None, {'control': m}))
        
        # Return the modified pos and neg lists
        return pos, neg


class T2IService():

    def __init__(self):
        self.tensorService = TensorService()

    def extract_T2I(self, positive, negative):
        t2is = [c['control'] for (_, c) in positive + negative if 'control' in c]
        t2is = [x for m in t2is for x in recursion_to_list(m, "previous_controlnet")]  # Recursive extraction
        return [x for x in t2is if isinstance(x, comfy.controlnet.T2IAdapter)]
    
    def prepareT2I_imgs(self, T2Is, shape):
        return [
            torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
            if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 or (m.channels_in == 1 and m.cond_hint_original.shape[1] != 1) else None
            for m in T2Is
        ]
    
    def slices_T2I(self, h, h_len, w, w_len, model:comfy.controlnet.ControlBase, img):
        model.control_input = None
        if img is None:
            img = model.cond_hint_original
        model.cond_hint = self.tensorService.getSlice(img, h*8, h_len*8, w*8, w_len*8).float().to(model.device)

    def prepareSlicedT2Is(self, pos, neg, T2Is, T2I_imgs, tile_h, tile_h_len, tile_w, tile_w_len):
        for idx, (m, img) in enumerate(zip(T2Is, T2I_imgs)):
            self.slices_T2I(tile_h, tile_h_len, tile_w, tile_w_len, m, img)
            
            # Replace the original 'control' key in the corresponding positions
            if idx < len(pos):
                pos[idx] = (None, {'control': m})
            else:
                pos.append((None, {'control': m}))
            
            if idx < len(neg):
                neg[idx] = (None, {'control': m})
            else:
                neg.append((None, {'control': m}))
        
        # Return the modified pos and neg lists
        return pos, neg

class GligenService():

    def extractGligen(self, positive, negative):
        gligen_pos = [
            c[1]['gligen'] if 'gligen' in c[1] else None
            for c in positive
        ]
        gligen_neg = [
            c[1]['gligen'] if 'gligen' in c[1] else None
            for c in negative
        ]
        return gligen_pos, gligen_neg
    
    def sliceGligen(self, tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen):
        tile_h_end = tile_h + tile_h_len
        tile_w_end = tile_w + tile_w_len
        if gligen is None:
            return
        gligen_type = gligen[0]
        gligen_model = gligen[1]
        gligen_areas = gligen[2]
        
        gligen_areas_new = []
        for emb, h_len, w_len, h, w in gligen_areas:
            h_end = h + h_len
            w_end = w + w_len
            if h < tile_h_end and h_end > tile_h and w < tile_w_end and w_end > tile_w:
                new_h = max(0, h - tile_h)
                new_w = max(0, w - tile_w)
                new_h_end = min(tile_h_end, h_end - tile_h)
                new_w_end = min(tile_w_end, w_end - tile_w)
                gligen_areas_new.append((emb, new_h_end - new_h, new_w_end - new_w, new_h, new_w))

        if len(gligen_areas_new) == 0:
            del cond['gligen']
        else:
            cond['gligen'] = (gligen_type, gligen_model, gligen_areas_new)

    def prepsreSlicedGligen(self, pos, neg, gligen_pos, gligen_neg, tile_h, tile_h_len, tile_w, tile_w_len):
        # Modify 'pos' and 'neg' lists in place by slicing the gligen conditions
        for cond, gligen in zip(pos, gligen_pos):
            self.sliceGligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)
        for cond, gligen in zip(neg, gligen_neg):
            self.sliceGligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)
        
        # Return the modified 'pos' and 'neg' lists
        return pos, neg

class Spatial_Condition_Service():
    def __init__(self):
        self.tensorService = TensorService()

    def extractSpatialConds(self, positive, negative, shape, device):
        spatial_conds_pos = [
            (c[1]['area'] if 'area' in c[1] else None, 
                comfy.sampler_helpers.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
            for c in positive
        ]
        spatial_conds_neg = [
            (c[1]['area'] if 'area' in c[1] else None, 
                comfy.sampler_helpers.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
            for c in negative
        ]
        return (spatial_conds_pos, spatial_conds_neg)
    
    def sliceSpatialConds(self, tile_h, tile_h_len, tile_w, tile_w_len, cond, area):
        tile_h_end = tile_h + tile_h_len
        tile_w_end = tile_w + tile_w_len
        coords = area[0] #h_len, w_len, h, w,
        mask = area[1]
        if coords is not None:
            h_len, w_len, h, w = coords
            h_end = h + h_len
            w_end = w + w_len
            if h < tile_h_end and h_end > tile_h and w < tile_w_end and w_end > tile_w:
                new_h = max(0, h - tile_h)
                new_w = max(0, w - tile_w)
                new_h_end = min(tile_h_end, h_end - tile_h)
                new_w_end = min(tile_w_end, w_end - tile_w)
                cond[1]['area'] = (new_h_end - new_h, new_w_end - new_w, new_h, new_w)
            else:
                return (cond, True)
        if mask is not None:
            new_mask = self.tensorService.getSlice(mask, tile_h, tile_h_len, tile_w, tile_w_len)
            if new_mask.sum().cpu() == 0.0 and 'mask' in cond[1]:
                return (cond, True)
            else:
                cond[1]['mask'] = new_mask
        return (cond, False)
    
    def prepareSlicedConds(self, pos, neg, spatial_conds_pos, spatial_conds_neg, tile_h, tile_h_len, tile_w, tile_w_len):
        pos = [self.sliceSpatialConds(tile_h, tile_h_len, tile_w, tile_w_len, c, area) for c, area in zip(pos, spatial_conds_pos)]
        pos = [c for c, ignore in pos if not ignore]
        neg = [self.sliceSpatialConds(tile_h, tile_h_len, tile_w, tile_w_len, c, area) for c, area in zip(neg, spatial_conds_neg)]
        neg = [c for c, ignore in neg if not ignore]
        return (pos, neg)
    
class Condition_Service():

    # sptcond - Stands for Spatial Condition

    def __init__(self):
        self.sptcondService = Spatial_Condition_Service()

class TensorService(): 

    def getSlice(self, tensor, h, h_len, w, w_len):
        """
        Extract a slice from the input tensor at the specified location and size.
        
        Args:
            tensor: The input tensor (e.g., a latent tensor).
            h: The starting height (y-coordinate) of the slice.
            h_len: The height length of the slice.
            w: The starting width (x-coordinate) of the slice.
            w_len: The width length of the slice.
        
        Returns:
            A slice of the input tensor.
        """
        return tensor[:, :, h:h + h_len, w:w + w_len]


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
    
    def generateGenDenoiseMask(self, tile_h, tile_w, border_strength=0.85, falloff=0.45, device='cpu'):
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
    
###### Singel Pass - Simple Tiling Strategy ######  
class STService(Tiling_Strategy_Base):
    
    def __init__(self):
        super().__init__()

    def generatePos_forSTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int):
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
        allTiles = []
        for t_pos_pass in tile_pos:
            tile_pass_l = []
            for t_pos_group in t_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in t_pos_group:
                    tile_mask = self.generateDenoiseMaskforSTStrategy(tile_h, tile_w)
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

    def getBlendMaskTilesforSTStrategy(self, tile_pos, B):
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
    
###### Multi-Pass - Simple Tiling Strategy ######   
class MP_STService(Tiling_Strategy_Base):
    
    def __init__(self, passes: int = 2):
        super().__init__()
        self.passes = passes

        # st - Stands for Simple Tiling Streteg
        self.st_Service = STService()

    def generatePos_forMP_STStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int):
        """
        Generates multiple passes of simple tiling with alternating shifts.
        """

        allTiles = []
        for i in range(self.passes):
            tile_pos = self.st_Service.generatePos_forSTStrategy(latent_width, latent_height, tile_width, tile_height)
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

    def generatePos_forRTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, seed):
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
        allTiles = []
        for tile_pos_pass in tile_pos:
            tile_pass_l = []
            for tile_pos_group in tile_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in tile_pos_group:
                    blend_mask = self.generateGenBlendMask(B, tile_h, tile_w)
                    tile_group_l.append(blend_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles
    
###### Multi-Pass - Random Tiling Strategy ######   
class MP_RTService(Tiling_Strategy_Base):

    def __init__(self, passes=2):
        super().__init__()
        self.rt_Service = RTService() 
        self.passes = passes

    def generatePos_forMP_RTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, seed):
        """
        Generates multiple passes of random tiling with alternating jitter logic.
        Matches the nested list structure of generatePos_forMP_CPTStrategy().
        """
        allTiles = []
        for i in range(self.passes):
            tile_pos = self.rt_Service.generatePos_forRTStrategy(latent_width, latent_height, tile_width, tile_height, seed)
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

###### Singel Pass - Adjacency Padded Tiling Strategy ######
class APTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()
    
    def generatePos_forAPTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, padding: int):
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

    def applyPadding_forAPTStrategy(self, tile, padding, strategy, pos_x, pos_y, tile_positions):
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
                    padded_tile = self.applyPadding_forAPTStrategy(tile, padding, padding_strategy, x, y, tile_positions)
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
    
    def generateDenoiseMaskforAPTStrategy(self, tile_h, tile_w, border_strength=1.0, falloff=0.5 ):
        return self.generateGenDenoiseMask(tile_h, tile_w, border_strength, falloff)
    
    def getDenoiseMaskTilesforAPTStrategy(self, noise_mask, tile_pos, padded_tiles):
        allTiles = []
        for tile_pos_pass, tiles_pass in zip(tile_pos, padded_tiles):
            tile_pass_l = []
            for tile_pos_group, tiles_group in zip(tile_pos_pass, tiles_pass):
                tile_group_l = []
                for (x, y, tile_w, tile_h), padded_tile in zip(tile_pos_group, tiles_group):
                    padded_h = padded_tile.shape[-2]
                    padded_w = padded_tile.shape[-1]       
                    tile_mask = self.generateDenoiseMaskforAPTStrategy(padded_h, padded_w)
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
    
    
###### Multi-Pass - Padded Tiling Strategy ######   
class MP_APTService(Tiling_Strategy_Base):

    def __init__(self, passes: int = 2):
        super().__init__()
        self.passes = passes

        # pts - Stands for Padded Tiling Streteg
        self.apt_Service = APTService()

    def generatePos_forMP_APTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, padding: int):
        allTiles = []
        for i in range(self.passes):
            tile_pos = self.apt_Service.generatePos_forAPTStrategy(latent_width, latent_height, tile_width, tile_height, padding)
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

    def generatePos_forCPTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int):
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
        allTiles = []
        for tile_pos_pass in tile_pos:
            tile_pass_l = []
            for tile_pos_group in tile_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in tile_pos_group:
                    blend_mask = self.generateGenBlendMask(B, tile_h, tile_w)
                    tile_group_l.append(blend_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles

###### Multi-Pass - Context-Padded Tiling Strategy ######   
class MP_CPTService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()
        # cpt - Stands for Context-Padded Tiling Stretegy
        self.cpt_Service = CPTService()


    def generatePos_forMP_CPTStrategy(self, latent_width: int, latent_height: int, tile_width: int, tile_height: int, steps):
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

    def generatePos_forADPStrategy(self, latent_tensor, tile_width=64, tile_height=64):
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
    
    def getBlendeMaskTilesforADPStrategy(self, tile_pos, B):
        allTiles = []
        for tile_pos_pass in tile_pos:
            tile_pass_l = []
            for tile_pos_group in tile_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in tile_pos_group:
                    blend_mask = self.generateGenBlendMask(B, tile_h, tile_w)
                    tile_group_l.append(blend_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles

###### Multi-Pass - Adaptive Tiling Strategy ######   
class MP_ADPService(Tiling_Strategy_Base):
    def __init__(self, passes=2):
        """
        passes: Number of refinement passes.
        max_new_tiles_factor: Limits how many new tiles can be created per pass.
        """
        super().__init__()
        self.passes = passes

        # adp - Stands for Adaptive Tiling Strategy
        self.adp_Service = ADPService()

    def generatePos_forMP_ADPStrategy(self, latent_tensor, tile_width=64, tile_height=64):
        """
        Multi-pass tiling: Starts large, refines complex areas in later passes.
        Returns depth array structure: [[[pass_1_tiles]], [[pass_2_tiles]], ...]
        """
        average_tile_size = int((tile_width + tile_height) / 2) 
        tile_positions = self.adp_Service.generatePos_forADPStrategy(latent_tensor, average_tile_size)

        for pass_idx in range(1, self.passes): 
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

    def generatePos_forHRCStrategy(self, latent_tensor, min_tile_size=16, complexity_threshold=0.1):
        """
        Fully adaptive hierarchical tiling: Recursively refines high-complexity tiles.
        Ensures that valid tiles are always generated to avoid empty sequences.
        """
        B, C, H, W = latent_tensor.shape
        tile_queue = [(0, 0, W, H)]  # Start with the full image
        final_tiles = []

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
    
    def getBlendeMaskTilesforHRCStrategy(self, tile_pos, B):
        allTiles = []
        for tile_pos_pass in tile_pos:
            tile_pass_l = []
            for tile_pos_group in tile_pos_pass:
                tile_group_l = []
                for (x, y, tile_w, tile_h) in tile_pos_group:
                    blend_mask = self.generateGenBlendMask(B, tile_h, tile_w)
                    tile_group_l.append(blend_mask)
                tile_pass_l.append(tile_group_l)
            allTiles.append(tile_pass_l)
        return allTiles

class MP_HRCService(Tiling_Strategy_Base):
    
    def __init__(self, passes=2):
        super().__init__()
        self.passes = passes
        self.hrc_Service = HRCService()

    def generatePos_forMP_HRCStrategy(self, latent_tensor, min_tile_size=16, base_complexity_threshold=0.1, complexity_variation=0.05):
        """
        Multi-pass adaptive hierarchical tiling. Varies complexity threshold across passes.
        """
        allTiles = []
        for i in range(self.passes):
            variation = (i - self.passes // 2) * complexity_variation
            complexity_threshold = max(0.01, base_complexity_threshold + variation)
            tile_pos = self.hrc_Service.generatePos_forHRCStrategy(latent_tensor, min_tile_size, complexity_threshold)
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

    def getBlendeMaskTilesforMP_HRCStrategy(self, tile_pos, B):
        return self.hrc_Service.getBlendeMaskTilesforHRCStrategy(tile_pos, B)

       
class TilingService(Tiling_Strategy_Base):

    def __init__(self):
        super().__init__()

        # st - Stands for Simple Tiling Strategy
        self.st_Service = STService()
        # rt - Stands for Random Tiling Strategy
        self.rt_Service = RTService()
        # apt - Stands for Padded Tiling Strategy
        self.apt_Service = APTService()
        # cpt - Stands for Context-Padded Tiling Strategy
        self.cpt_Service = CPTService()
        # adp - Stands for Adaptive Tiling Strategy
        self.adp_Service = ADPService()
        # hrc - Stands for Hierarchical Tiling Strategy
        self.hrc_Service = HRCService()
        
        # mp_st - Stands for Multi-Pass Simple Tiling Strategy
        self.mp_st_Service = MP_STService()
        # mp_rt - Stands for Multi-Pass Random Tiling Strategy
        self.mp_rt_Service = MP_RTService()
        # mp_apt - Stands for Multi-Pass Padding Tiling Strategy
        self.mp_apt_Service = MP_APTService()
        # mp_cpt - Stands for Multi-Pass Context-Padded Tiling Stretegy
        self.mp_cpt_Service = MP_CPTService()
        # mp_adp - Stands for Multi-Pass Adaptive Tiling Strategy
        self.mp_adp_Service = MP_ADPService()
        # mp_hrc - Stands for Multi-Pass Hierarchical Tiling Strategy
        self.mp_hrc_Service = MP_HRCService()

class LatentTiler:

    def __init__(self):
        self.tilingService = TilingService()
        self.conditionService = Condition_Service()
        self.conrolNetService = ControlNetService()
        self.t2iService = T2IService()
        self.gligenService = GligenService()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tiling_mode": (["single_pass", "multi_pass"],),
                "tiling_strategy": (["simple", "random", "padded", "adjacency_padded", "context_padded", "adaptive", "hierarchical"],),
                "padding_strategy": (["organic", "circular", "reflect", "replicate", "zero"],),
                "padding": ("INT", {"default": 16, "min": 0, "max": 128}),
                "disable_noise": ("BOOLEAN", {"default": False}),
            }
        }
    
    CATEGORY = "EUP - Ultimate Pack/latent"

    RETURN_TYPES = ("TILED_LATENT",)
    RETURN_NAMES = ("tiled_latent",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "tileLatent"

    def getLatentSize(self, image_size: int) -> int:
        return basic_utils.getLatentSize(image_size)
    
    def getPaddLatentSize(self, image_size: int) -> int:
        if image_size > 0:
            return basic_utils.getLatentSize(image_size)
        return 0
    
    def convertPXtoLatentSize(self, tile_height, tile_width, padding) -> int:   
        # Converts Size From Pixel to Latent
        tile_height = self.getLatentSize(tile_height)
        tile_width = self.getLatentSize(tile_width)
        padding = self.getPaddLatentSize(padding)

        return (tile_height, tile_width, padding)
    
    def tileLatent(self, seed, steps, latent_image, positive, negative, tile_width, tile_height, tiling_mode, tiling_strategy, padding_strategy, padding, disable_noise):
        samples = latent_image.get("samples")
        shape = samples.shape
        B, C, H, W = shape

        # Converts Size From Pixel to Latent
        tile_height, tile_width, padding = self.convertPXtoLatentSize(tile_height, tile_width, padding)

        print(f"[EUP - Latent Tiler]: Original Latent Size: {samples.shape}")
        print(f"[EUP - Latent Tiler]: Tile Size: {tile_height}x{tile_width}")

        noise = self.tilingService.generateRandomNoise(latent_image, seed, disable_noise)
        noise_mask = self.tilingService.generateNoiseMask(latent_image, noise)

        #### Chosing Tiling Strategy ####
        if tiling_mode == "single_pass":
            if tiling_strategy == "simple":
                tiled_positions = self.tilingService.st_Service.generatePos_forSTStrategy(W, H, tile_width, tile_height)
                tiled_tiles = self.tilingService.st_Service.getTilesforSTStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.st_Service.getNoiseTilesforSTStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.st_Service.getDenoiseMaskTilesforSTStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.st_Service.getBlendMaskTilesforSTStrategy(tiled_positions, B)
            elif tiling_strategy == "random":
                tiled_positions = self.tilingService.rt_Service.generatePos_forRTStrategy(W, H, tile_width, tile_height, seed)
                tiled_tiles = self.tilingService.rt_Service.getTilesforRTStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.rt_Service.getNoiseTilesforRTStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.rt_Service.getDenoiseMaskTilesforRTStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.rt_Service.getBlendMaskTilesforRTStrategy(tiled_positions, B)
            elif tiling_strategy == "adjacency_padded":
                tiled_positions = self.tilingService.apt_Service.generatePos_forAPTStrategy(W, H, tile_width, tile_height, padding)
                tiled_tiles = self.tilingService.apt_Service.getTilesforAPTStrategy(samples, tiled_positions)
                tiled_tiles = self.tilingService.apt_Service.getPaddedTilesforAPTStrategy(tiled_tiles, tiled_positions, padding_strategy, padding)
                tiled_noise_tiles = self.tilingService.apt_Service.getNoiseTilesforAPTStrategy(latent_image, tiled_tiles, seed, disable_noise)
                tiled_denoise_masks = self.tilingService.apt_Service.getDenoiseMaskTilesforAPTStrategy(noise_mask, tiled_positions, tiled_tiles)
                tiled_blend_masks = self.tilingService.apt_Service.getBlendMaskTilesforAPTStrategy(tiled_tiles, B)
            elif tiling_strategy == "context_padded":
                tiled_positions = self.tilingService.cpt_Service.generatePos_forCPTStrategy(W, H, tile_width, tile_height)
                tiled_denoise_masks = self.tilingService.cpt_Service.getDenoiseMaskTilesforCPTStrategy(noise_mask, tiled_positions, B, H, W)
                tiled_tiles = self.tilingService.cpt_Service.getTilesforCPTStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.cpt_Service.getNoiseTilesforCPTStrategy(noise, tiled_positions)
                tiled_blend_masks = self.tilingService.cpt_Service.getBlendeMaskTilesforCPTStrategy(tiled_positions, B)
            elif tiling_strategy == "adaptive":
                tiled_positions = self.tilingService.adp_Service.generatePos_forADPStrategy(samples, tile_width, tile_height)
                tiled_tiles = self.tilingService.adp_Service.getTilesforADPStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.adp_Service.getNoiseTilesforADPStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.adp_Service.getDenoiseMaskTilesforADPStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.adp_Service.getBlendeMaskTilesforADPStrategy(tiled_positions, B)
            elif tiling_strategy == "hierarchical":
                tiled_positions = self.tilingService.hrc_Service.generatePos_forHRCStrategy(samples)
                tiled_tiles = self.tilingService.hrc_Service.getTilesforHRCStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.hrc_Service.getNoiseTilesforHRCStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.hrc_Service.getDenoiseMaskTilesforHRCStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.hrc_Service.getBlendeMaskTilesforHRCStrategy(tiled_positions, B)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Strategy for Single-Pass Mode. Please select a valid Strategy.")
        elif tiling_mode == "multi_pass":
            if tiling_strategy == "simple":
                tiled_positions = self.tilingService.mp_st_Service.generatePos_forMP_STStrategy(W, H, tile_width, tile_height)
                tiled_tiles = self.tilingService.mp_st_Service.getTilesforMP_STStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_st_Service.getNoiseTilesforMP_STStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_st_Service.getDenoiseMaskTilesforMP_STStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_st_Service.getBlendMaskTilesforMP_STStrategy(tiled_positions, B)
            elif tiling_strategy == "random":
                tiled_positions = self.tilingService.mp_rt_Service.generatePos_forMP_RTStrategy(W, H, tile_width, tile_height, seed)
                tiled_tiles = self.tilingService.mp_rt_Service.getTilesforMP_RTStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_rt_Service.getNoiseTilesforMP_RTStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_rt_Service.getDenoiseMaskTilesforMP_RTStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_rt_Service.getBlendMaskTilesforMP_RTStrategy(tiled_positions, B)
            elif tiling_strategy == "adjacency_padded":
                tiled_positions = self.tilingService.mp_apt_Service.generatePos_forMP_APTStrategy(W, H, tile_width, tile_height, padding)
                tiled_tiles = self.tilingService.mp_apt_Service.getTilesforMP_APTStrategy(samples, tiled_positions)
                tiled_tiles = self.tilingService.mp_apt_Service.getPaddedTilesforMP_APTStrategy(tiled_tiles, tiled_positions, padding_strategy, padding)
                tiled_noise_tiles = self.tilingService.mp_apt_Service.getNoiseTilesforMP_APTStrategy(latent_image, tiled_tiles, seed, disable_noise)
                tiled_denoise_masks = self.tilingService.mp_apt_Service.getDenoiseMaskTilesforMP_APTStrategy(noise_mask, tiled_positions, tiled_tiles)
                tiled_blend_masks = self.tilingService.mp_apt_Service.getBlendMaskTilesforMP_APTStrategy(tiled_tiles, B)
            elif tiling_strategy == "context_padded":
                tiled_positions = self.tilingService.mp_cpt_Service.generatePos_forMP_CPTStrategy(W, H, tile_width, tile_height, steps)
                tiled_denoise_masks = self.tilingService.mp_cpt_Service.getDenoiseMaskTilesforMP_CPTStrategy(noise_mask, tiled_positions, B, H, W)
                tiled_tiles = self.tilingService.mp_cpt_Service.getTilesforMP_CPTStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_cpt_Service.getNoiseTilesforMP_CPTStrategy(noise, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_cpt_Service.getBlendeMaskTilesforMP_CPTStrategy(tiled_positions, B)
            elif tiling_strategy == "adaptive":
                tiled_positions = self.tilingService.mp_adp_Service.generatePos_forMP_ADPStrategy(samples, tile_width, tile_height)
                tiled_tiles = self.tilingService.mp_adp_Service.getTilesforMP_ADPStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_adp_Service.getNoiseTilesforMP_ADPStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_adp_Service.getDenoiseMaskTilesforMP_ADPStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_adp_Service.getBlendeMaskTilesforMP_ADPStrategy(tiled_positions, B)
            elif tiling_strategy == "hierarchical":
                tiled_positions = self.tilingService.mp_hrc_Service.generatePos_forMP_HRCStrategy(samples)
                tiled_tiles = self.tilingService.mp_hrc_Service.getTilesforMP_HRCStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_hrc_Service.getNoiseTilesforMP_HRCStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_hrc_Service.getDenoiseMaskTilesforMP_HRCStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_hrc_Service.getBlendeMaskTilesforMP_HRCStrategy(tiled_positions, B)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Strategy for Multi-Pass Mode. Please select a valid Strategy.")
        else:
            raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode. Please select a valid Tiling Mode.")

        #### Prepare the tiles for sampling ####
        return self.prepareTilesforSampling(positive, negative, shape, samples, tiled_positions, tiled_tiles, tiled_noise_tiles, tiled_denoise_masks, tiled_blend_masks) 
    
    def prepareTilesforSampling(self, positive, negative, shape, samples, tiled_positions, tiled_tiles, tiled_noise_tiles, tiled_denoise_masks, tiled_blend_masks):
        # Extract and prepare cnets, T2Is, gligen, and spatial conditions
        cnets = self.conrolNetService.extractCnet(positive, negative)
        cnet_imgs = self.conrolNetService.prepareCnet_imgs(cnets, shape)
        T2Is = self.t2iService.extract_T2I(positive, negative)
        T2I_imgs = self.t2iService.prepareT2I_imgs(T2Is, shape)
        gligen_pos, gligen_neg = self.gligenService.extractGligen(positive, negative)
        sptcond_pos, sptcond_neg = self.conditionService.sptcondService.extractSpatialConds(positive, negative, shape, samples.device)
        
        tiled_latent = []
        for t_pos_pass, tile_pass, noise_pass, denoise_mask_pass , blend_mask_pass in zip(tiled_positions, tiled_tiles, tiled_noise_tiles, tiled_denoise_masks, tiled_blend_masks):
            tile_pass_l = []
            for t_pos_group, tile_group, noise_group, denoise_mask_group, blend_mask_group in zip(t_pos_pass, tile_pass, noise_pass, denoise_mask_pass, blend_mask_pass):
                tile_group_l = []
                for (x, y, tile_w, tile_h), tile, noise_tile, denoise_mask, blend_mask in zip(t_pos_group, tile_group, noise_group, denoise_mask_group, blend_mask_group):
                    
                    # Copy positive and negative lists for modification
                    pos = [c.copy() for c in positive]
                    neg = [c.copy() for c in negative]

                    #### Slice cnets, T2Is, gligen, and spatial conditions for the current tile ####
                    ## Slice CNETS ##
                    pos, neg, self.conrolNetService.prepareSlicedCnets(pos, neg, cnets, cnet_imgs, y, tile_h, x, tile_w)
                    ## Slice T2Is ##
                    pos, neg = self.t2iService.prepareSlicedT2Is(pos, neg, T2Is, T2I_imgs, y, tile_h, x, tile_w)

                    ## Slice Spatial Conditions  ##
                    pos, neg = self.conditionService.sptcondService.prepareSlicedConds(pos, neg, sptcond_pos, sptcond_neg, y, tile_h, x, tile_w)

                    # Slice GLIGEN ##
                    pos, neg = self.gligenService.prepsreSlicedGligen(pos, neg, gligen_pos, gligen_neg, y, tile_h, x, tile_w)

                    tile_group_l.append((tile, (x, y, tile_w, tile_h), noise_tile, denoise_mask, blend_mask, pos, neg))
                tile_pass_l.append(tile_group_l)
            tiled_latent.append(tile_pass_l)
        return tiled_latent

class LatentMerger:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "proc_tiled_latent": ("PROC_LATENT_TILES",),
            }
        }

    INPUT_IS_LIST = (True,)

    CATEGORY = "EUP - Ultimate Pack/latent"
    
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The merged latent.",)
    FUNCTION = "mergeTiles"

    def mergeTiles(self, proc_tiled_latent):
        # List to hold all tiles from all passes
        all_tile_data = []

        # Collect all tiles, positions, and masks from each pass
        for pass_data in proc_tiled_latent:
            latent_tiles, tile_positions, noise_masks = self.unpackTiles(pass_data)

            # Combine the tiles, positions, and masks into one list of tuples
            tile_data = [(tile, pos, mask) for tile, pos, mask in zip(latent_tiles, tile_positions, noise_masks)]
            all_tile_data.extend(tile_data)  # Add all tiles to the combined list

        # Perform the merging for all passes at once
        merged_samples, weight_map = self.performMerging(all_tile_data)

        # Normalize the result
        weight_map = torch.clamp(weight_map, min=1e-6)  # Avoid division by zero
        merged_samples /= weight_map

        return ({"samples": merged_samples},)
        
    def performMerging(self, all_tile_data):
        """
        Perform merging of all tiles from all passes in one step.
        """
        device = comfy.model_management.get_torch_device()  # Get the current device from the model
        
        # Determine the max width and height needed to accommodate all tiles
        max_width = max(x + w for _, (x, y, w, h), _ in all_tile_data)
        max_height = max(y + h for _, (x, y, w, h), _ in all_tile_data)

        # Initialize merged samples and weight map with the required dimensions
        merged_samples = torch.zeros((1, 4, max_height, max_width), device=device)
        weight_map = torch.zeros_like(merged_samples)

        # Process each tile and apply it to the merged output
        for tile, (x, y, w, h), mask in all_tile_data:
            valid_tile, valid_mask = self.extractValidRegion(tile, mask, max_height, max_width, x, y)
            
            # Apply the tile to the merged samples and weight map
            merged_samples, weight_map = self.applyTiletoMerged(merged_samples, weight_map, valid_tile, valid_mask, x, y, w, h)

        return merged_samples, weight_map

    def applyTiletoMerged(self, merged_samples, weight_map, tile, mask, x, y, w, h):
        # Ensure all tensors are on the same device
        device = merged_samples.device  # Get the device of merged_samples
        tile, mask = self.processTileandMask(tile, mask, device)
        
        valid_tile_h, valid_tile_w = min(tile.shape[2], h), min(tile.shape[3], w)
        valid_mask_h, valid_mask_w = min(mask.shape[2], h), min(mask.shape[3], w)
        
        if valid_tile_h == 0 or valid_tile_w == 0:
            raise ValueError(f"Invalid tile dimensions: {valid_tile_h}, {valid_tile_w}")

        merged_samples[:, :, y:y + valid_tile_h, x:x + valid_tile_w] += tile[:, :, :valid_tile_h, :valid_tile_w] * mask[:, :, :valid_mask_h, :valid_mask_w]
        weight_map[:, :, y:y + valid_tile_h, x:x + valid_tile_w] += mask[:, :, :valid_mask_h, :valid_mask_w]

        return merged_samples, weight_map

    def extractValidRegion(self, tile, mask, max_height, max_width, x, y):
        valid_tile_h, valid_tile_w = min(tile.shape[2], max_height - y), min(tile.shape[3], max_width - x)
        valid_mask_h, valid_mask_w = min(mask.shape[2], max_height - y), min(mask.shape[3], max_width - x)
        if valid_tile_h == 0 or valid_tile_w == 0:
            raise ValueError(f"Invalid tile dimensions: {valid_tile_h}, {valid_tile_w}")
        return tile[:, :, :valid_tile_h, :valid_tile_w], mask[:, :, :valid_mask_h, :valid_mask_w]
    
    def processTileandMask(self, tile, mask, device):
        # Ensure the tensors are on the same device
        tile = tile.to(device)
        if tile.dim() == 3:
            tile = tile.unsqueeze(0)
        if tile.shape[1] == 3:
            tile = torch.cat([tile, torch.zeros((tile.shape[0], 1, tile.shape[2], tile.shape[3]), device=device)], dim=1)
        mask = mask.to(device)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        return tile, mask

    def unpackTiles(self, pass_data):
        latent_tiles = []
        tile_positions = []
        noise_masks = []

        for tile_group in pass_data:
            for tile, pos, mask in tile_group:
                latent_tiles.append(tile)
                tile_positions.append(pos)
                noise_masks.append(mask)

        return latent_tiles, tile_positions, noise_masks


            
class TKS_Base:
    def common_ksampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,  tile_width,tile_height, tiling_mode ,
                        tiling_strategy, padding_strategy, padding, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False
                        ):

        device = comfy.model_management.get_torch_device()
        
        # Step 2: Tile Latent **and Noise**
        tiler = LatentTiler()
        tiled_latent = tiler.tileLatent(
            seed=seed, steps=steps, latent_image=latent, positive=positive, negative=negative, tile_width=tile_width, tile_height=tile_height, 
            tiling_mode=tiling_mode, tiling_strategy=tiling_strategy, padding_strategy=padding_strategy, padding=padding, disable_noise=disable_noise
        )

        # Prepare and Progress Bar
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        if not disable_pbar:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)

        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
        
        total_tiles = sum(len(tile_group) for tile_pass in tiled_latent for tile_group in tile_pass)
        total_steps = steps * total_tiles
        
        with tqdm(total=total_steps) as pbar_tqdm:
            proc_tiled_latent = self.sampleTiles(sampler, tiled_latent, steps, total_steps, total_tiles, pbar_tqdm, disable_pbar, 
                                                    previewer, cfg, start_step, last_step, force_full_denoise, seed)
        # Step 4: Merge Tiles
        merger = LatentMerger()
        merged_latent = merger.mergeTiles(proc_tiled_latent)

        out = latent.copy()
        out["samples"] = merged_latent[0].get("samples")
        return (out, )

    def sampleTiles(self, sampler: comfy.samplers.KSampler, tiled_latent, steps, total_steps, total_tiles, 
                    pbar_tqdm, disable_pbar, previewer, cfg, start_step, last_step, force_full_denoise, seed):
        
        pbar = comfy.utils.ProgressBar(steps)
        proc_tiled_latent = []
        global_tile_index = 0  # Track tiles across all passes

        for tile_pass in tiled_latent:
            tile_pass_l = []
            for tile_group in tile_pass:
                tile_group_l = []
                for (tile, tile_position, noise_tile, denoise_mask, blend_mask, mp, mn) in tile_group:
                    tile_index = global_tile_index 

                    processed_tile = sampler.sample(
                        noise=noise_tile, 
                        positive=mp, 
                        negative=mn, 
                        cfg=cfg, 
                        latent_image=tile, 
                        start_step=start_step, 
                        last_step=last_step,
                        force_full_denoise=force_full_denoise,
                        denoise_mask=denoise_mask, 
                        callback=lambda step, x0, x, total_steps=total_steps, tile_index=tile_index: 
                            self.updateProgress(step, x0, x, total_steps, tile_index, total_tiles, pbar_tqdm, disable_pbar, previewer, pbar),
                        disable_pbar=disable_pbar, 
                        seed=seed
                    )

                    # Store processed tile
                    tile_group_l.append((processed_tile, tile_position, blend_mask))
                    global_tile_index += 1  # Increment index

                tile_pass_l.append(tile_group_l)
            proc_tiled_latent.append(tile_pass_l)

        return proc_tiled_latent
    
    def updateProgress(self, step, x0, x, total_steps, tile_index, total_tiles, pbar_tqdm, disable_pbar, previewer, pbar):
        # Update the overall progress bar
        pbar_tqdm.update(1)

        # Update tile-specific progress
        if not disable_pbar:
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)

            pbar_tqdm.set_description(f"[EUP - Tiled KSampler]")

            # Update the postfix with correct total and tile progress
            postfix = {
                "Tile Progress": f"Tile {tile_index + 1}/{total_tiles}" 
            }
            pbar_tqdm.set_postfix(postfix)

            # Update the progress bar with the preview image (optional)
            pbar.update_absolute(step, preview=preview_bytes)
        



class Tiled_KSampler (TKS_Base):      
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tiling_mode": (["single_pass", "multi_pass"],),
                "tiling_strategy": (["simple", "random", "padded", "adjacency_padded", "context_padded", "adaptive", "hierarchical"],),
                "padding_strategy": (["organic", "circular", "reflect", "replicate", "zero"],),
                "padding": ("INT", {"default": 16, "min": 0, "max": 128}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "EUP - Ultimate Pack/sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, 
               tile_width, tile_height, tiling_mode, tiling_strategy, padding_strategy, padding, denoise=1.0,
               ):

        return self.common_ksampler(model=model, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, 
                                    latent=latent_image, tile_width=tile_width, tile_height=tile_height, tiling_mode=tiling_mode, tiling_strategy=tiling_strategy, 
                                    padding_strategy=padding_strategy, padding=padding, denoise=denoise,
        )

class Tiled_KSamplerAdvanced (TKS_Base):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "add_noise": (["enable", "disable"], ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tiling_mode": (["single_pass", "multi_pass"],),
                "tiling_strategy": (["simple", "random", "padded", "context_padded", "adaptive", "hierarchical"],),
                "tiling_strategy": (["simple", "random", "padded", "adjacency_padded", "context_padded", "adaptive", "hierarchical"],),
                "padding": ("INT", {"default": 16, "min": 0, "max": 128}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"], ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "EUP - Ultimate Pack/sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, tile_width, tile_height, 
               tiling_mode, tiling_strategy, padding_strategy, padding, start_at_step, end_at_step,return_with_leftover_noise, denoise=1.0,
            ):
        
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        return self.common_ksampler(model=model, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, 
            latent=latent_image, tile_width=tile_width, tile_height=tile_height, tiling_mode=tiling_mode, tiling_strategy=tiling_strategy, padding_strategy=padding_strategy, 
            padding=padding, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise,
        )
    
NODE_CLASS_MAPPINGS = {
    "EUP - Latent Tiler": LatentTiler,
    "EUP - Latent Merger": LatentMerger,
    "EUP - Tiled KSampler" : Tiled_KSampler,
    "EUP - Tiled KSampler Advanced" : Tiled_KSamplerAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
}
