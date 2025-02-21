#### Build In Lib's #####
import sys
import os
import random
from typing import List


import nodes
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.controlnet

#### My Lib's #####
from EUP.utils import basic_utils

#### Third Party Lib's #####
import torch
import torch.nn.functional as F
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
        for m, img in zip(cnets, cnet_imgs):
            self.sliceCnet(tile_h, tile_h_len, tile_w, tile_w_len, m, img)
            pos.append((None, {'control': m}))
            neg.append((None, {'control': m}))


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
        for m, img in zip(T2Is, T2I_imgs):
            self.slices_T2I(tile_h, tile_h_len, tile_w, tile_w_len, m, img)
            pos.append((None, {'control': m}))
            neg.append((None, {'control': m}))

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
        for cond, gligen in zip(pos, gligen_pos):
            self.sliceGligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)
        for cond, gligen in zip(neg, gligen_neg):
            self.sliceGligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)

class Spatial_Condition_Service():

    def extractSpatialConds(self, positive, negative, shape, device):
        spatial_conds_pos = [
            (c[1]['area'] if 'area' in c[1] else None, 
                comfy.sample.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
            for c in positive
        ]
        spatial_conds_neg = [
            (c[1]['area'] if 'area' in c[1] else None, 
                comfy.sample.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
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

    def setPaddedSlice(self, tensor, tile, mask, h, h_len, w, w_len, padding=0):
        print(f"[DEBUG] Before Merging: tensor.shape={tensor.shape}, tile.shape={tile.shape}, mask.shape={mask.shape}")
        
        valid_h = max(1, min(h_len - 2 * padding, tensor.shape[2] - h))
        valid_w = max(1, min(w_len - 2 * padding, tensor.shape[3] - w))

        start_h = max(0, h + padding)
        start_w = max(0, w + padding)
        end_h = min(start_h + valid_h, tile.shape[2])
        end_w = min(start_w + valid_w, tile.shape[3])

        valid_tile = tile[:, :, start_h:end_h, start_w:end_w]
        valid_mask = mask[:, :, start_h:end_h, start_w:end_w]

        print(f"[DEBUG] Merging region (h:{h}, w:{w}): {valid_h}x{valid_w}")
        
        tensor[:, :, h:h + valid_h, w:w + valid_w] += valid_tile * valid_mask
        return tensor
    
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

class Padded_Tiling_Service():

    ## hp - Stands for Hard Blending
    ## smax - Stands for Softmax Blending

    def adjustTileSize_forPaddStrg(self, tile_size, latent_size):
        # Ensure the tile size is compatible with the latent size, adjusting to multiples of 4
        adjusted_size = min(max(4, (tile_size // 4) * 4), latent_size)
        return adjusted_size
    
    def generatePos_forPaddStrg(self, latent_size, tile_size, padding) -> List:
        # Calculate the positions in the latent space, ensuring the tiles fit with padding
        positions = list(range(0, latent_size - tile_size + 1, tile_size))  # Use full tile size as step

        # Ensure the last tile fits perfectly within the latent size
        if positions[-1] + tile_size > latent_size:
            positions[-1] = latent_size - tile_size

        return positions
    
    def generaterandNoise_forPaddStrg(self, latent, tile, seed, B, C, tile_w, tile_h, disable_noise: bool):
        samples = latent.get("samples")
        if disable_noise:
            return torch.zeros((B, C, tile_h, tile_w), dtype=samples.dtype, device="cpu")
        else:
            batch_inds = latent.get("batch_index")
            return comfy.sample.prepare_noise(tile, seed, batch_inds)

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

        print(f"Original tile shape: {tile.shape}")
        print(f"Padding values: left={left}, right={right}, top={top}, bottom={bottom}")

        # Create the padded tile
        padded_tile = torch.zeros((B, C, H + top + bottom, W + left + right), device=tile.device, dtype=tile.dtype)

        # Place the original tile in the center
        padded_tile[:, :, top:top + H, left:left + W] = tile

        def soft_blend(patch, expand_shape, flip_dims, blend_factor=0.7):
            """Mirrors a patch and blends it softly with adaptive randomness."""
            mirrored = patch
            for flip_dim in flip_dims:
                mirrored = torch.flip(mirrored, dims=[flip_dim])  # Mirror flip in both directions

            # Adaptive blending factor (less boxy)
            blend_factor = torch.tensor(blend_factor, device=tile.device).expand_as(patch)

            # Perlin-style noise (natural variation, no harsh edges)
            noise = (torch.rand_like(patch) - 0.5) * 0.08  # Less aggressive randomness
            blend_factor = torch.clamp(blend_factor + noise, 0.3, 0.85)  # Adaptive range

            return (patch * blend_factor + mirrored * (1 - blend_factor)).expand(expand_shape)

        # Left padding (soft blended from left edge)
        if left > 0:
            left_patch = soft_blend(tile[:, :, :, :left], (B, C, H, left), flip_dims=[3], blend_factor=0.75)
            padded_tile[:, :, top:top + H, :left] = left_patch

        # Right padding
        if right > 0:
            right_patch = soft_blend(tile[:, :, :, -right:], (B, C, H, right), flip_dims=[3], blend_factor=0.75)
            padded_tile[:, :, top:top + H, -right:] = right_patch

        # Top padding
        if top > 0:
            top_patch = soft_blend(tile[:, :, :top, :], (B, C, top, W), flip_dims=[2], blend_factor=0.75)
            padded_tile[:, :, :top, left:left + W] = top_patch

        # Bottom padding
        if bottom > 0:
            bottom_patch = soft_blend(tile[:, :, -bottom:, :], (B, C, bottom, W), flip_dims=[2], blend_factor=0.75)
            padded_tile[:, :, -bottom:, left:left + W] = bottom_patch

        # Corner regions (blend both X and Y for smooth edges)
        if left > 0 and top > 0:
            padded_tile[:, :, :top, :left] = soft_blend(tile[:, :, :top, :left], (B, C, top, left), flip_dims=[2, 3], blend_factor=0.65)

        if right > 0 and top > 0:
            padded_tile[:, :, :top, -right:] = soft_blend(tile[:, :, :top, -right:], (B, C, top, right), flip_dims=[2, 3], blend_factor=0.65)

        if left > 0 and bottom > 0:
            padded_tile[:, :, -bottom:, :left] = soft_blend(tile[:, :, -bottom:, :left], (B, C, bottom, left), flip_dims=[2, 3], blend_factor=0.65)

        if right > 0 and bottom > 0:
            padded_tile[:, :, -bottom:, -right:] = soft_blend(tile[:, :, -bottom:, -right:], (B, C, bottom, right), flip_dims=[2, 3], blend_factor=0.65)

        return padded_tile
            
    def hp_applyPadding(self, tile, padding, strategy, pos_x, pos_y, num_tiles_x, num_tiles_y):
        """Applies padding between tiles and scales tiles down if necessary."""
        if padding > 0:
            # Initialize padding tuple (no padding by default)
            pad = []

            # Add padding for each side based on the tile's position
            if pos_x != 0:  # Not the leftmost tile
                pad.append(padding)
            else:
                pad.append(0)  # No padding on the left side
            
            if pos_x != num_tiles_x - 1:  # Not the rightmost tile
                pad.append(padding)
            else:
                pad.append(0)  # No padding on the right side

            if pos_y != 0:  # Not the topmost tile
                pad.append(padding)
            else:
                pad.append(0)  # No padding on the top side
            
            if pos_y != num_tiles_y - 1:  # Not the bottommost tile
                pad.append(padding)
            else:
                pad.append(0)  # No padding on the bottom side

            # Apply padding based on the strategy
            if strategy in ["circular", "reflect", "replicate"]:
                return F.pad(tile, tuple(pad), mode=strategy)
            elif strategy == "organic_padding":
                return self.applyOrganicPadding(tile, tuple(pad))
            else:
                return F.pad(tile, tuple(pad), mode="constant", value=0)  # Default constant padding with 0 for normal use cases

        return tile

    def hp_createMask(self, batch_size, padded_h, padded_w):
        """Creates a smooth blending mask for latent tiles."""
        
        # Grid for radial blending
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1, 1, padded_h, device="cpu"),
            torch.linspace(-1, 1, padded_w, device="cpu"),
            indexing="ij",
        )

        # Compute radial distance
        radius = torch.sqrt(grid_x**2 + grid_y**2)
        
        # Apply a cosine-based fade (soft at edges, sharp in center)
        fade_radial = 0.5 * (1 - torch.cos(torch.clamp(radius * torch.pi, 0, torch.pi)))
        
        # Directional fades to prevent excessive center softening
        fade_x = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, padded_w, device="cpu"))).view(1, 1, 1, -1)
        fade_y = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, padded_h, device="cpu"))).view(1, 1, -1, 1)
        
        # Merge both blending styles
        fade = fade_radial * fade_x * fade_y
        
        # Adjust the mask to avoid too much softening at the center
        fade = torch.clamp(fade, min=0.01)  # Prevent total blackness
        
        # Ensure proper batch size
        mask = fade.expand(batch_size, 1, padded_h, padded_w)
        
        return mask
    
    def hp_applyHPStrategy(self, latent, tile, x, y, B, C, num_tiles_x, num_tiles_y, padding, padding_strategy, seed, disable_noise):
        tile = self.hp_applyPadding(tile, padding, padding_strategy, x, y, num_tiles_x, num_tiles_y)

        # Get final padded dimensions
        padded_h = tile.shape[-2]
        padded_w = tile.shape[-1]
        
        noise = self.generaterandNoise_forPaddStrg(latent, tile, seed, B, C, padded_w, padded_h, disable_noise)
        mask = self.hp_createMask(B, padded_h, padded_w)
        
        return (tile, noise, mask)

    
class TilingService():

    def __init__(self):

        # pts - Stands for Padded Tiling Service
        self.pts_Service = Padded_Tiling_Service()

class LatentTiler:

    def __init__(self):
        self.tensorService = TensorService()
        self.tilingService = TilingService()
        self.conditionService = Condition_Service()
        self.conrolNetService = ControlNetService()
        self.t2iService = T2IService()
        self.gligenService = GligenService()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "tile_height": ("INT", {"default": 64, "min": 1}),
                "tile_width": ("INT", {"default": 64, "min": 1}),
                "tiling_strategy": (["padded", "random", "random_strict", "simple"],),
                "padding_strategy": (["circular", "reflect", "replicate","random_padded", "organic_padding", "zero"],),
                "padding_blending": (["hard_padding", "softmax_padding"],),
                "padding": ("INT", {"default": 16, "min": 0, "max": 128}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "disable_noise": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST", "LIST", "LIST", "TUPLE")
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
    
    def applyTilePaddedtrategy(self, latent, tile, x, y, B, C, num_tiles_x, num_tiles_y, tiling_strategy, padding_strategy, padding_blending, padding, seed, disable_noise):
        if tiling_strategy == "padded" and padding_blending == "hard_padding":                                                              
            return self.tilingService.pts_Service.hp_applyHPStrategy(latent, tile, x, y, B, C, num_tiles_x, num_tiles_y, padding, padding_strategy, seed, disable_noise)
        elif tiling_strategy == "padded" and padding_blending == "softmax_padding":
            return self.tilingService.pts_Service.smax_applySMAXStrategy(latent, tile, x, y, B, C, num_tiles_x, num_tiles_y, padding, padding_strategy, seed, disable_noise)
    
    def tileLatent(self, latent, positive, negative, tile_height, tile_width, tiling_strategy, padding_strategy, padding_blending, padding, seed, disable_noise):
        samples = latent["samples"]
        shape = samples.shape
        B, C, H, W = shape

        # Converts Size From Pixel to Latent
        tile_height, tile_width, padding = self.convertPXtoLatentSize(tile_height, tile_width, padding)

        print(f"[Eddy's - Latent Tiler] Original Latent Size: {samples.shape}")
        print(f"[Eddy's - Latent Tiler] Tile Size: {tile_height}x{tile_width}")

        if tiling_strategy == "padded":
            # Adjust tile size for padding Strategy
            tile_h = self.tilingService.pts_Service.adjustTileSize_forPaddStrg(tile_height, H)
            tile_w = self.tilingService.pts_Service.adjustTileSize_forPaddStrg(tile_width, W)
            print("tile_h", tile_h)
            print("tile_w",tile_w)

            # Generate a list of positions to cover the entire image
            print(f"Adjusted tile_h: {tile_h}, tile_w: {tile_w}, Image H: {H}, W: {W}, Padding: {padding}")
            h_positions = self.tilingService.pts_Service.generatePos_forPaddStrg(H, tile_h, padding)
            w_positions = self.tilingService.pts_Service.generatePos_forPaddStrg(W, tile_w, padding)
            print(f"h_positions: {h_positions}")
            print(f"w_positions: {w_positions}")

        print(f"[Eddy's - Latent Tiler] Adjusted Tile Size: {tile_h}x{tile_w}")

        tiles, tile_positions, noise_tiles, denoise_masks, blending_masks, modified_positives, modified_negatives = [], [], [], [], [], [], []

        # Extract and prepare cnets, T2Is, gligen, and spatial conditions
        cnets = self.conrolNetService.extractCnet(positive, negative)
        cnet_imgs = self.conrolNetService.prepareCnet_imgs(cnets, shape)
        T2Is = self.t2iService.extract_T2I(positive, negative)
        T2I_imgs = self.t2iService.prepareT2I_imgs(T2Is, shape)
        gligen_pos, gligen_neg = self.gligenService.extractGligen(positive, negative)
        sptcond_pos, sptcond_neg = self.conditionService.sptcondService.extractSpatialConds(positive, negative, shape, samples.device)

        num_tiles_x = len(w_positions)
        num_tiles_y = len(h_positions)

        for y in h_positions:
            for x in w_positions:
                tile = self.tensorService.getSlice(samples, y, tile_h, x, tile_w)

                if tiling_strategy == "padded":
                    # Apply padding using updated function
                    tile, noise_tile, blending_mask = self.applyTilePaddedtrategy(
                        latent, 
                        tile, 
                        x, 
                        y, 
                        B, 
                        C,
                        num_tiles_x,
                        num_tiles_y, 
                        tiling_strategy, 
                        padding_strategy, 
                        padding_blending, 
                        padding, 
                        seed, 
                        disable_noise,
                    )

                denoise_mask = None  # self.createDenoiseMask() -- Im not sure what denoise mask is it the same as blending mask

                # Copy positive and negative lists for modification
                pos = [c.copy() for c in positive]
                neg = [c.copy() for c in negative]

                # Slice cnets, T2Is, gligen, and spatial conditions for the current tile
                self.conrolNetService.prepareSlicedCnets(pos, neg, cnets, cnet_imgs, y, tile_h, x, tile_w)
                self.t2iService.prepareSlicedT2Is(pos, neg, T2Is, T2I_imgs, y, tile_h, x, tile_w)

                # Slice spatial conditions and update pos and neg
                pos, neg = self.conditionService.sptcondService.prepareSlicedConds(pos, neg, sptcond_pos, sptcond_neg, y, tile_h, x, tile_w)

                # Slice gligen and update pos and neg
                self.gligenService.prepsreSlicedGligen(pos, neg, gligen_pos, gligen_neg, y, tile_h, x, tile_w)

                #### Appending Needed Resources ####
                # Append Sliced Tiles
                tiles.append({"samples": tile})
                # Append Tile Positions
                tile_positions.append((x, y))
                # Append Noise Tile
                noise_tiles.append({"noise_tile": noise_tile})
                # Append Sliced Denoise Mask
                denoise_masks.append({"denoise_mask": denoise_mask})
                # Append Sliced Blending Mask
                blending_masks.append({"blending_mask": blending_mask})
                # Append the modified pos and neg to the lists
                modified_positives.append(pos)
                modified_negatives.append(neg)

        return (tiles, tile_positions, noise_tiles, denoise_masks, blending_masks, modified_positives, modified_negatives, (B, C, H, W))

import torch
import torch.nn.functional as F

class LatentMerger:
    def __init__(self):
        self.tensorService = TensorService()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("LIST",),
                "blending_mask": ("LIST",),
                "tile_positions": ("LIST",),
                "original_size": ("TUPLE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "mergeTiles"

    def mergeTiles(self, tiles, blending_masks, tile_positions, original_size, blend_strength=0.01):
        device = comfy.model_management.get_torch_device()
        B, C, H, W = original_size

        merged_samples = torch.zeros((B, C, H, W), device=device)
        weight_map = torch.zeros((B, C, H, W), device=device)

        print(f"[DEBUG] Original Latent Size: {original_size}")

        for i, (tile_dict, mask_dict, (x, y)) in enumerate(zip(tiles, blending_masks, tile_positions)):
            tile = tile_dict["samples"].to(device)
            mask = mask_dict["blending_mask"].to(device)
            valid_tile, valid_mask = self.extractValidRegion(tile, mask, H, W, x, y, blend_strength)
            self.applyTiletoMerged(merged_samples, weight_map, valid_tile, valid_mask, x, y)

        merged_samples = self.normalizeOutput(merged_samples, weight_map)
        merged_samples = self.applyHP_strategyBlend(merged_samples, tile_positions, blend_strength)
        
        print(f"[DEBUG] Merged Output Shape: {merged_samples.shape}")
        return ({"samples": merged_samples},)

    # --- General Code ---
    def extractValidRegion(self, tile, mask, H, W, x, y, blend_strength):
        valid_h = min(tile.shape[2], H - y)
        valid_w = min(tile.shape[3], W - x)
        valid_tile = tile[:, :, :valid_h, :valid_w]
        valid_mask = mask[:, :, :valid_h, :valid_w] * blend_strength  # Apply blend strength
        return valid_tile, valid_mask

    def applyTiletoMerged(self, merged_samples, weight_map, tile, mask, x, y):
        merged_samples[:, :, y:y + tile.shape[2], x:x + tile.shape[3]] += tile * mask
        weight_map[:, :, y:y + tile.shape[2], x:x + tile.shape[3]] += mask

    def normalizeOutput(self, merged_samples, weight_map):
        weight_map = torch.clamp(weight_map, min=1e-6)  # Prevent division by zero
        return merged_samples / weight_map
    
    # --- Strategy-Specific Code ---
    def applyHP_strategyBlend(self, merged_samples, tile_positions, blend_strength):
        blend_width = 1
        B, C, H, W = merged_samples.shape

        for b in range(B):
            for c in range(C):
                for x, y in tile_positions:
                    if x > 0:
                        edge_diff_x = torch.abs(merged_samples[b, c, :, x] - merged_samples[b, c, :, x-1])
                        adaptive_blend_x = torch.clamp(1.0 - edge_diff_x.mean(), 0.02, 0.08) * blend_strength
                        merged_samples[b, c, :, x-blend_width:x+blend_width] = torch.lerp(
                            merged_samples[b, c, :, x-blend_width:x+blend_width],
                            merged_samples[b, c, :, x-1:x+1],
                            adaptive_blend_x
                        )
                    
                    if y > 0:
                        edge_diff_y = torch.abs(merged_samples[b, c, y, :] - merged_samples[b, c, y-1, :])
                        adaptive_blend_y = torch.clamp(1.0 - edge_diff_y.mean(), 0.02, 0.08) * blend_strength
                        merged_samples[b, c, y-blend_width:y+blend_width, :] = torch.lerp(
                            merged_samples[b, c, y-blend_width:y+blend_width, :],
                            merged_samples[b, c, y-1:y+1, :],
                            adaptive_blend_y
                        )
        if x > 0:
            edge_diff_x = torch.abs(merged_samples[b, c, :, x] - merged_samples[b, c, :, x-1])
            adaptive_blend_x = torch.clamp(1.0 - edge_diff_x.mean(), 0.02, 0.08) * blend_strength
            merged_samples[b, c, :, x-blend_width:x+blend_width] = torch.lerp(
                merged_samples[b, c, :, x-blend_width:x+blend_width],
                merged_samples[b, c, :, x-1:x+1],
                adaptive_blend_x
            )
        
        if y > 0:
            edge_diff_y = torch.abs(merged_samples[b, c, y, :] - merged_samples[b, c, y-1, :])
            adaptive_blend_y = torch.clamp(1.0 - edge_diff_y.mean(), 0.02, 0.08) * blend_strength
            merged_samples[b, c, y-blend_width:y+blend_width, :] = torch.lerp(
                merged_samples[b, c, y-blend_width:y+blend_width, :],
                merged_samples[b, c, y-1:y+1, :],
                adaptive_blend_y
            )

        return merged_samples
        
class TKS_Base:
    def common_ksampler(
            self,
            model, 
            seed, 
            steps, 
            cfg, 
            sampler_name, 
            scheduler, 
            positive, 
            negative, 
            latent,  
            tile_width,
            tile_height,
            tiling_strategy,
            padding_strategy,
            padding_blending,
            padding, 
            denoise=1.0,
            disable_noise=False, 
            start_step=None, 
            last_step=None, 
            force_full_denoise=False
        ):
        
        device = comfy.model_management.get_torch_device()
        
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]
        
        # Step 2: Tile Latent **and Noise**
        tiler = LatentTiler()
        tiles, tile_positions, noise_tiles, denoise_masks, blending_masks, modified_positives, modified_negatives, original_size = tiler.tileLatent(
            latent=latent,
            positive=positive, 
            negative=negative,
            tile_width=tile_width,
            tile_height=tile_height, 
            tiling_strategy=tiling_strategy,
            padding_strategy=padding_strategy,
            padding_blending=padding_blending,
            padding=padding,
            seed=seed,
            disable_noise=disable_noise
        )
        
        # Prepare and Progress Bar
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        if not disable_pbar:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)

        processed_tiles = []
        total_steps = steps * len(tiles)
        with tqdm(total=total_steps) as pbar_tqdm:
            pbar = comfy.utils.ProgressBar(steps)

            def update_progress(step, x0, x, total_steps, tile_index):
                # Update progress for the entire process
                pbar_tqdm.update(1)

                # Update tile-specific progress
                if not disable_pbar:
                    preview_bytes = None
                    if previewer:
                        preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)

                    # Set the postfix using the tqdm progress bar (not ProgressBar)
                    postfix = {
                        "Total Progress": f"{pbar_tqdm.n}/{total_steps} steps",  # Total progress
                        "Tile Progress": f"Tile {tile_index + 1}/{len(tiles)}"  # Current tile progress
                    }
                    pbar_tqdm.set_postfix(postfix)  # Use tqdm's set_postfix for updates

                    # Update preview image and step progress
                    pbar.update_absolute(step, preview=preview_bytes)
            # Process each tile
            for tile_index, (tile, noise_tile, denoise_mask, mp, mn) in enumerate(zip(tiles, noise_tiles, denoise_masks, modified_positives, modified_negatives)):

                noise_mask = None
                if "noise_mask" in tile:
                    noise_mask = tile["samples"]

                processed_tile = comfy.sample.sample(
                    model=model, 
                    noise=noise_tile["noise_tile"], 
                    steps=steps, 
                    cfg=cfg, 
                    sampler_name=sampler_name, 
                    scheduler=scheduler, 
                    positive=mp,
                    negative=mn, 
                    latent_image=tile["samples"],
                    denoise=denoise, 
                    disable_noise=disable_noise, 
                    start_step=start_step, 
                    last_step=last_step,
                    force_full_denoise=force_full_denoise, 
                    noise_mask=noise_mask, 
                    callback=lambda step, x0, x, total_steps: update_progress(step, x0, x, total_steps, tile_index),
                    disable_pbar=disable_pbar, 
                    seed=seed)
                
                processed_tiles.append({"samples": processed_tile})

        # Step 4: Merge Tiles
        merger = LatentMerger()
        merged_latent = merger.mergeTiles(processed_tiles, blending_masks, tile_positions, original_size, blend_strength=0.1)[0]

        return (merged_latent,)

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
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "tile_width": ("INT", {"default": 64, "min": 1}),
                "tile_height": ("INT", {"default": 64, "min": 1}),
                "tiling_strategy": (["padded", "random", "random_strict", "simple"],),
                "padding_strategy": (["circular", "reflect", "replicate", "random_padded", "organic_padding", "zero"],),
                "padding_blending": (["hard_padding", "softmax_padding"],),
                "padding": ("INT", {"default": 16, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, 
               model, 
               seed, 
               steps, 
               cfg, 
               sampler_name, 
               scheduler, 
               positive, 
               negative, 
               latent_image, 
               tile_width, 
               tile_height, 
               tiling_strategy, 
               padding_strategy, 
               padding_blending, 
               padding, 
               denoise=1.0
        ):
        return self.common_ksampler(
            model=model, 
            seed=seed, 
            steps=steps, 
            cfg=cfg, 
            sampler_name=sampler_name, 
            scheduler=scheduler, 
            positive=positive, 
            negative=negative, 
            latent=latent_image,  
            tile_width=tile_width,
            tile_height=tile_height,
            tiling_strategy=tiling_strategy, 
            padding=padding, 
            padding_strategy=padding_strategy,
            padding_blending=padding_blending,
            denoise=denoise,
        )

class LatentPipeline:
    @classmethod
    def INPUT_TYPES(cls):
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
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "tile_width": ("INT", {"default": 64, "min": 1}),
                "tile_height": ("INT", {"default": 64, "min": 1}),
                "padding": ("INT", {"default": 16, "min": 0, "max": 128}),
                "padding_strategy": (["circular", "reflect", "replicate", "zero"],),
                "padding_blending": (["hard_padding", "softmax_padding"],),
                "scale_factor": ("FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "number",
                    },),
                "upscale_version": (["SDXL", "SD 1.x"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process_latent"

    def process_latent(
            self, 
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            tile_width,
            tile_height,
            padding,
            padding_strategy,
            denoise,
            scale_factor, 
            upscale_version,
        ):

        # Step 1: Upscale Latent
        upscaler = LatentUpscaler()
        upscaled_latent = upscaler.upscale_latent(latent_image, scale_factor,upscale_version)[0]
        print("SampleSize after Scaling",upscaled_latent["samples"][0].shape)

        # Step 2: Merge Tiles
        tiledKSampler = Tiled_KSampler()
        merged_latent = tiledKSampler.sample(
            model=model, 
            seed=seed, 
            steps=steps, 
            cfg=cfg, 
            sampler_name=sampler_name, 
            scheduler=scheduler, 
            positive=positive, 
            negative=negative, 
            latent_image=upscaled_latent,  
            tile_width=tile_width,
            tile_height=tile_height, 
            padding=padding, 
            padding_strategy=padding_strategy,
            denoise=denoise,
        )[0]

        return (merged_latent,)

NODE_CLASS_MAPPINGS = {
    "EUP - Latent Upscaler": LatentUpscaler,
    "EUP - Latent Tiler": LatentTiler,
    "EUP - Latent Merger": LatentMerger,
    "EUP - Tiled KSampler" : Tiled_KSampler,
    "EUP - Latent Pipeline": LatentPipeline,
}
