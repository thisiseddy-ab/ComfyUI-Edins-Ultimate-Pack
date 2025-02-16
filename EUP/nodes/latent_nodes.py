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
    """
    Insert a tile into the larger tensor at the specified location, considering the mask.

    Args:
        tensor: The tensor to which the tile will be inserted (e.g., the merged tensor).
        tile: The tile to insert into the tensor.
        mask: The mask that determines where the tile will be placed.
        h: The starting height (y-coordinate) for the tile placement.
        h_len: The height length of the tile.
        w: The starting width (x-coordinate) for the tile placement.
        w_len: The width length of the tile.
        padding: The padding size to consider when placing the tile.

    Returns:
        The updated tensor with the tile inserted at the correct location.
    """
    valid_h = min(h_len - 2 * padding, tensor.shape[2] - h)
    valid_w = min(w_len - 2 * padding, tensor.shape[3] - w)

    # Adjusting the indices to handle padding and ensure the tile fits correctly
    start_h = padding
    start_w = padding
    end_h = start_h + valid_h
    end_w = start_w + valid_w

    # Ensure that the tile fits within the bounds
    end_h = min(end_h, tile.shape[2])
    end_w = min(end_w, tile.shape[3])

    valid_tile = tile[:, :, start_h:end_h, start_w:end_w]
    valid_mask = mask[:, :, start_h:end_h, start_w:end_w]

    # Perform the actual setting (similar to your merge_tiles function)
    tensor[:, :, h:h + valid_h, w:w + valid_w] += valid_tile * valid_mask

    return tensor


class LatentTiler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
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
    
    def generatePositions (latent_size, tile_h, padding) -> List:
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
    
    def tileLatent(self, latent, tile_height, tile_width, tiling_strategy, padding_strategy, padding, seed, disable_noise):
        samples = latent["samples"]
        B, C, H, W = samples.shape
        
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

        tiles, noise_tiles, masks, tile_positions = [], [], [], []

        for y in h_positions:
            for x in w_positions:
                tile = getSlice(samples, y, tile_h, x, tile_w)

                if(tiling_strategy == "padded"):
                    tile = self.applyPadding(tile, padding, padding_strategy) if padding > 0 else tile
                    # Ensure mask matches padded tile dimensions
                    denoise_mask = self.createMask_forPadding(latent, B, tile.shape[2], tile.shape[3])

                noise_tile = self.generateNoise(latent, tile, seed, B, C, tile_w, tile_h, disable_noise)

                tiles.append({"samples": tile})
                noise_tiles.append({"noise_tile": noise_tile})
                masks.append({"denoise_mask": denoise_mask})
                tile_positions.append((x, y))

        return tiles, noise_tiles, masks, tile_positions, (B, C, H, W)
    
class LatentMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("LIST",),
                "tile_positions": ("LIST",),
                "masks": ("LIST",),
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

    def merge_tiles(self, tiles, tile_positions, masks, original_size, padding):
        device = comfy.model_management.get_torch_device()
        B, C, H, W = original_size

        # Converts Size From Pixel to Latent
        padding = self.getLatentSize(padding)

        merged_samples = torch.zeros((B, C, H, W), device=device)
        weight_map = torch.zeros((B, C, H, W), device=device)

        for tile_dict, mask_dict, (x, y) in zip(tiles, masks, tile_positions):
            tile = tile_dict["samples"].to(device)
            mask = mask_dict["denoise_mask"].to(device)

            tile_height, tile_width = tile.shape[2], tile.shape[3]

            # Ensure the mask has the same channel count as the tile
            if mask.shape[1] == 1 and tile.shape[1] > 1:
                mask = mask.expand(-1, tile.shape[1], -1, -1)

            # Ensure tile and mask have the same shape
            assert tile.shape == mask.shape, f"[ERROR] Tile and Mask Shape Mismatch: {tile.shape} vs {mask.shape}"

            # Compute valid region after padding
            valid_h = self.getValid_regionAfterPadding(tileSize=tile_height, padding=padding, orig_latent_size=H, tile_pos=y)
            valid_w = self.getValid_regionAfterPadding(tileSize=tile_width, padding=padding, orig_latent_size=W, tile_pos=x)

            # Ensure edge tiles match expected size
            start_h, start_w = padding, padding
            end_h, end_w = start_h + valid_h, start_w + valid_w

            if end_h > tile_height:
                end_h = tile_height
                start_h = max(0, tile_height - valid_h)  # Adjust to keep size consistent
            if end_w > tile_width:
                end_w = tile_width
                start_w = max(0, tile_width - valid_w)  # Adjust to keep size consistent

            valid_tile = self.get_valid_tile(tile, start_h, end_h, start_w, end_w)
            valid_mask = self.get_valid_mask(mask, start_h, end_h, start_w, end_w)

            # Ensure shape consistency
            assert valid_tile.shape == valid_mask.shape, f"[ERROR] Valid Tile & Mask Mismatch: {valid_tile.shape} vs {valid_mask.shape}"

            merged_samples = setSlice(merged_samples, valid_tile, valid_mask, y, valid_h, x, valid_w)

        merged_samples /= torch.clamp(weight_map, min=1e-6)
        return ({"samples": merged_samples},)

class TKS_Base:
    def extract_cnets(self,positive, negative):
        cnets = [c['control'] for (_, c) in positive + negative if 'control' in c]
        cnets = list(set([x for m in cnets for x in recursion_to_list(m, "previous_controlnet")]))  # Recursive extraction
        return [x for x in cnets if isinstance(x, controlnet.ControlNet)]

    # Helper to extract T2I adapters (T2Is) from positive and negative conditions
    def extract_t2is(self,positive, negative):
        t2is = [c['control'] for (_, c) in positive + negative if 'control' in c]
        t2is = [x for m in t2is for x in recursion_to_list(m, "previous_controlnet")]  # Recursive extraction
        return [x for x in t2is if isinstance(x, controlnet.T2IAdapter)]

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
        tiles, noise_tiles, denoise_mask, tile_positions, original_size = tiler.tileLatent(
            latent=latent,
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
            denoise=denoise
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
            for tile_index, (tile, noise_tile, denoise_mask) in enumerate(zip(tiles, noise_tiles, denoise_mask)):
                processed_tile = sampler.sample(
                    noise=noise_tile["noise_tile"],
                    positive=positive,
                    negative=negative,
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
        merged_latent = merger.merge_tiles(
            processed_tiles, tile_positions, denoise_mask, original_size, padding
        )[0]

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
    
        device = comfy.model_management.get_torch_device()

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
