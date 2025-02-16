#### Build In Lib's #####
import sys
import os
from typing import List


import nodes
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.controlnet as controlnet

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


def getSlice(tensor, h, h_len, w, w_len):
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

def setSlice(tensor, tile, mask, h, h_len, w, w_len, padding=0):
    print(tensor.dtype, tile.dtype, mask.dtype, tensor.device, tile.device, mask.device)
    print("tensor min/max:", tensor.min().item(), tensor.max().item())
    print("tile min/max:", tile.min().item(), tile.max().item())
    print("mask min/max:", mask.min().item(), mask.max().item())
    print("mask unique values:", mask.unique())

    valid_h = max(1, min(h_len - 2 * padding, tensor.shape[2] - h))
    valid_w = max(1, min(w_len - 2 * padding, tensor.shape[3] - w))

    start_h = padding
    start_w = padding
    end_h = min(start_h + valid_h, tile.shape[2])
    end_w = min(start_w + valid_w, tile.shape[3])

    valid_tile = tile[:, :, start_h:end_h, start_w:end_w]
    valid_mask = mask[:, :, start_h:end_h, start_w:end_w]

    valid_tile = valid_tile.to(tensor.dtype).to(tensor.device)
    valid_mask = valid_mask.to(tensor.dtype).to(tensor.device)

    valid_mask = torch.clamp(valid_mask, min=0, max=1)  # 🔥 Ensures mask values are valid

    tensor[:, :, h:h + valid_h, w:w + valid_w] += valid_tile * valid_mask

    return tensor


class LatentTiler:
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
                "padding_strategy": (["circular", "reflect", "replicate", "zero"],),
                "padding": ("INT", {"default": 16, "min": 0, "max": 128}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "disable_noise": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST", "LIST", "TUPLE")
    FUNCTION = "tileLatent"

    def getLatentSize(self, image_size: int) -> int:
        return basic_utils.getLatentSize(image_size)
    
    def getSlice(self,tensor, h, h_len, w, w_len):
        return getSlice(tensor, h, h_len, w, w_len)
    
    def adjustTileSize_forPadding(self,tile_size,latent_size):
        return min(max(4, (tile_size // 4) * 4), latent_size)
    
    def generatePositions (self,latent_size, tile_h, padding) -> List:
        return list(range(0, latent_size, tile_h - padding))
    
    def generateNoise(self,latent, tile, seed, B, C, tile_w, tile_h, disable_noise: bool):
        samples = latent.get("samples")
        if disable_noise:
            return torch.zeros((B, C, tile_h, tile_w), dtype=samples.dtype, device="cpu")
        else:
            batch_inds = latent.get("batch_index")
            return comfy.sample.prepare_noise(tile, seed, batch_inds)

    def applyPadding(self, tensor, padding, strategy):
        """Applies padding to the tensor."""
        if padding > 0:
            pad = (padding, padding, padding, padding)
            if strategy in ["circular", "reflect", "replicate"]:
                return F.pad(tensor, pad, mode=strategy)
            return F.pad(tensor, pad, mode="constant", value=0)
        return tensor

    def createMask_forPadding(self, latent, batch_size, tile_h, tile_w):
        """Creates a mask for blending tiles."""
        samples = latent.get("samples")
        mask = torch.ones((batch_size, 1, tile_h, tile_w), dtype=samples.dtype, device="cpu")
        fade_x = torch.linspace(0, 1, tile_w, device="cpu").view(1, 1, 1, -1)
        fade_y = torch.linspace(0, 1, tile_h, device="cpu").view(1, 1, -1, 1)
        mask *= fade_x * fade_y  # Smooth blending
        return mask
    
    def extractCnet(self, positive, negative):
        cnets = [c['control'] for (_, c) in positive + negative if 'control' in c]
        cnets = list(set([x for m in cnets for x in recursion_to_list(m, "previous_controlnet")]))  # Recursive extraction
        return [x for x in cnets if isinstance(x, controlnet.ControlNet)]
    
    def prepareCnet_imgs(self, cnets, shape):
        return [
            torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
            if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 else None
            for m in cnets
    ]

    def slice_cnet(self, h, h_len, w, w_len, model:comfy.controlnet.ControlBase, img):
        if img is None:
            img = model.cond_hint_original
        hint = self.getSlice(img, h*8, h_len*8, w*8, w_len*8)
        if isinstance(model, comfy.controlnet.ControlLora):
            model.cond_hint = hint.float().to(model.device)
        else:
            model.cond_hint = hint.to(model.control_model.dtype).to(model.control_model.device)

    def prepareSlicedCnets(self, pos, neg, cnets, cnet_imgs, tile_h, tile_h_len, tile_w, tile_w_len):
        for m, img in zip(cnets, cnet_imgs):
            self.slice_cnet(tile_h, tile_h_len, tile_w, tile_w_len, m, img)
            pos.append((None, {'control': m}))
            neg.append((None, {'control': m}))


    def extract_T2I(self, positive, negative):
        t2is = [c['control'] for (_, c) in positive + negative if 'control' in c]
        t2is = [x for m in t2is for x in recursion_to_list(m, "previous_controlnet")]  # Recursive extraction
        return [x for x in t2is if isinstance(x, controlnet.T2IAdapter)]
    
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
        model.cond_hint = self.getSlice(img, h*8, h_len*8, w*8, w_len*8).float().to(model.device)

    def prepareSlicedT2Is(self, pos, neg, T2Is, T2I_imgs, tile_h, tile_h_len, tile_w, tile_w_len):
        for m, img in zip(T2Is, T2I_imgs):
            self.slices_T2I(tile_h, tile_h_len, tile_w, tile_w_len, m, img)
            pos.append((None, {'control': m}))
            neg.append((None, {'control': m}))

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
            new_mask = self.getSlice(mask, tile_h, tile_h_len, tile_w, tile_w_len)
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
    
    def tileLatent(self, latent, positive, negative, tile_height, tile_width, tiling_strategy, padding_strategy, padding, seed, disable_noise):
        samples = latent["samples"]
        shape = samples.shape
        B, C, H, W = shape
        
        # Converts Size From Pixel to Latent
        tile_height = self.getLatentSize(tile_height)
        tile_width = self.getLatentSize(tile_width)
        padding = self.getLatentSize(padding)
        
        print(f"[Eddy's - Latent Tiler] Original Latent Size: {samples.shape}")

        # Adjust tile size for padding
        tile_h = self.adjustTileSize_forPadding(tile_height, H)
        tile_w = self.adjustTileSize_forPadding(tile_width, W)
        
        print(f"[Eddy's - Latent Tiler] Adjusted Tile Size: {tile_h}x{tile_w}")

        # Generate a list of positions to cover the entire image
        h_positions = self.generatePositions(H, tile_h, padding)
        w_positions = self.generatePositions(W, tile_h, padding)

        tiles, noise_tiles, denoise_masks, tile_positions, modified_positives, modified_negatives = [], [], [], [], [], []

        # Extract and prepare cnets, T2Is, gligen, and spatial conditions
        cnets = self.extractCnet(positive, negative)
        cnet_imgs = self.prepareCnet_imgs(cnets, shape)
        T2Is = self.extract_T2I(positive, negative)
        T2I_imgs = self.prepareT2I_imgs(T2Is, shape)
        gligen_pos, gligen_neg = self.extractGligen(positive, negative)
        spatial_conds_pos, spatial_conds_neg = self.extractSpatialConds(positive, negative, shape, samples.device)

        for y in h_positions:
            for x in w_positions:
                tile = self.getSlice(samples, y, tile_h, x, tile_w)

                if tiling_strategy == "padded":
                    tile = self.applyPadding(tile, padding, padding_strategy) if padding > 0 else tile
                    denoise_mask = self.createMask_forPadding(latent, B, tile.shape[2], tile.shape[3])

                noise_tile = self.generateNoise(latent, tile, seed, B, C, tile_w, tile_h, disable_noise)

                # Copy positive and negative lists for modification
                pos = [c.copy() for c in positive]
                neg = [c.copy() for c in negative]

                # Slice cnets, T2Is, gligen, and spatial conditions for the current tile
                self.prepareSlicedCnets(pos, neg, cnets, cnet_imgs, y, tile_h, x, tile_w)
                self.prepareSlicedT2Is(pos, neg, T2Is, T2I_imgs, y, tile_h, x, tile_w)
                
                # Slice spatial conditions and update pos and neg
                pos, neg = self.prepareSlicedConds(pos, neg, spatial_conds_pos, spatial_conds_neg, y, tile_h, x, tile_w)
                
                # Slice gligen and update pos and neg
                self.prepsreSlicedGligen(pos, neg, gligen_pos, gligen_neg, y, tile_h, x, tile_w)

                #### Appending Needed Tiles ####
                # Apennd Sliced Tiles
                tiles.append({"samples": tile})
                # Append Noise Tile
                noise_tiles.append({"noise_tile": noise_tile})
                # Append Sliced Denooise Mask
                denoise_masks.append({"denoise_mask": denoise_mask})
                # Append Tile Postions
                tile_positions.append((x, y))
                # Append the modified pos and neg to the lists
                modified_positives.append(pos)
                modified_negatives.append(neg)

        return tiles, noise_tiles, denoise_masks, tile_positions, modified_positives, modified_negatives, (B, C, H, W)

    
class LatentMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("LIST",),
                "denoise_masks": ("LIST",),
                "tile_positions": ("LIST",),
                "original_size": ("TUPLE",),
                "padding": ("INT", {"default": 16, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "merge_tiles"

    def getLatentSize(self, imageSize: int) -> int:
        return basic_utils.getLatentSize(imageSize)
    
    def setSlice(self,tensor, tile, mask, h, h_len, w, w_len, padding=0):
        return setSlice(tensor, tile, mask, h, h_len, w, w_len, padding)
    
    def getValid_regionAfterPadding(self, tileSize, padding, orig_latent_size, tile_pos) -> int:
        return min(tileSize - 2 * padding, orig_latent_size - tile_pos)

    def get_valid_tile(self, tile, start_h, end_h, start_w, end_w):
        """Extracts a valid tile slice based on given start and end indices."""
        return tile[:, :, start_h:end_h, start_w:end_w]

    def get_valid_mask(self, mask, start_h, end_h, start_w, end_w):
        """Extracts a valid mask slice based on given start and end indices."""
        return mask[:, :, start_h:end_h, start_w:end_w]

    def merge_tiles(self, tiles, denoise_masks, tile_positions, original_size, padding):
        device = comfy.model_management.get_torch_device()
        B, C, H, W = original_size

        padding = self.getLatentSize(padding)

        merged_samples = torch.zeros((B, C, H, W), device=device)  # 🔥 Fix NaN issue
        weight_map = torch.zeros((B, C, H, W), device=device)  # 🔥 Restore weight map

        for tile_dict, mask_dict, (x, y) in zip(tiles, denoise_masks, tile_positions):
            tile = tile_dict["samples"].to(device)
            mask = mask_dict["denoise_mask"].to(device)

            # Ensure the mask has the same channel count as the tile
            if mask.shape[1] == 1 and tile.shape[1] > 1:
                mask = mask.expand(-1, tile.shape[1], -1, -1)

            assert tile.shape == mask.shape, f"[ERROR] Tile and Mask Shape Mismatch: {tile.shape} vs {mask.shape}"

            valid_h = self.getValid_regionAfterPadding(tile.shape[2], padding, H, y)
            valid_w = self.getValid_regionAfterPadding(tile.shape[3], padding, W, x)

            valid_tile = self.get_valid_tile(tile, padding, padding + valid_h, padding, padding + valid_w)
            valid_mask = self.get_valid_mask(mask, padding, padding + valid_h, padding, padding + valid_w)

            assert valid_tile.shape == valid_mask.shape, f"[ERROR] Valid Tile & Mask Mismatch: {valid_tile.shape} vs {valid_mask.shape}"

            merged_samples = self.setSlice(merged_samples, valid_tile, valid_mask, y, valid_h, x, valid_w)

            # 🔥 Restore weight map accumulation
            weight_map[:, :, y:y + valid_h, x:x + valid_w] += valid_mask

        merged_samples /= torch.clamp(weight_map, min=1e-6)  # 🔥 Correct division
        return ({"samples": merged_samples},)

        merged_samples /= torch.clamp(weight_map, min=1e-6)
        print("After merging - merged_samples min/max:", merged_samples.min().item(), merged_samples.max().item())
        print("After merging - weight_map min/max:", weight_map.min().item(), weight_map.max().item())
        return ({"samples": merged_samples},)
    

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
        tiles, noise_tiles, denoise_masks, tile_positions, modified_positives, modified_negatives, original_size = tiler.tileLatent(
            latent=latent,
            positive=positive, 
            negative=negative,
            tile_width=tile_width,
            tile_height=tile_height, 
            tiling_strategy=tiling_strategy,
            padding_strategy=padding_strategy,
            padding=padding,
            seed=seed,
            disable_noise=disable_noise
        )
        
        # Prepare and Progress Bar
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        if not disable_pbar:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)

        # Step 3: Apply KSampler on Each Tile using Precomputed Noise
        sampler = comfy.samplers.KSampler(
            model=model, 
            steps=steps, 
            device=device, 
            sampler=sampler_name, 
            scheduler=scheduler, 
            denoise=denoise,
            model_options=model.model_options
        )
        
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
                processed_tile = sampler.sample(
                    noise=noise_tile["noise_tile"],
                    positive=mp,
                    negative=mn,
                    cfg=cfg,
                    latent_image=tile["samples"],
                    start_step=start_step,
                    last_step=last_step,
                    force_full_denoise=force_full_denoise,
                    denoise_mask=denoise_mask["denoise_mask"],
                    callback=lambda step, x0, x, total_steps: update_progress(step, x0, x, total_steps, tile_index),
                    disable_pbar=disable_pbar,
                    seed=seed
                )
                processed_tiles.append({"samples": processed_tile})

        # Step 4: Merge Tiles
        merger = LatentMerger()
        merged_latent = merger.merge_tiles(processed_tiles, denoise_masks, tile_positions, original_size, padding)[0]

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
                "padding_strategy": (["circular", "reflect", "replicate", "zero"],),
                "padding": ("INT", {"default": 16, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, tile_width, tile_height, tiling_strategy, padding_strategy, padding, denoise=1.0):
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
