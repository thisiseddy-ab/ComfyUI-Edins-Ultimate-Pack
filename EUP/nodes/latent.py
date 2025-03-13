#### Build In Lib's #####
from typing import List, Tuple

#### Comfy Lib's ####
from nodes import MAX_RESOLUTION

#### Third Party Lib's #####
import torch


#### My Services's #####
from EUP.services.units import UnitsService
from EUP.services.tiling import TilingService
from EUP.services.latent import LatentTilerService, LatentMergerService
   
class LatentTiler:

    def __init__(self):
        self.untisService = UnitsService()
        self.tilingService = TilingService()
        self.latentTilerService = LatentTilerService()

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
                "tiling_mode": (["single-pass", "multi-pass"],),
                "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
                "tiling_strategy": (["simple", "random", "padded", "adjacency-padded", "context-padded", "overlaping", "adaptive", "hierarchical", "non-uniform"],),
                "padding_strategy": (["organic", "circular", "reflect", "replicate", "zero"],),
                "overalp_padding": ("INT", {"default": 16, "min": 0, "max": 128}),
                "disable_noise": ("BOOLEAN", {"default": False}),
            }
        }
    
    CATEGORY = "EUP - Ultimate Pack/latent"

    RETURN_TYPES = ("TILED_LATENT",)
    RETURN_NAMES = ("tiled_latent",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "tileLatent"
    
    def getPaddLatentSize(self, image_size: int) -> int:
        if image_size > 0:
            return self.untisService.getLatentSize(image_size)
        return 0
    
    def convertPXtoLatentSize(self, tile_height, tile_width, padding) -> int:   
        # Converts Size From Pixel to Latent
        tile_height = self.untisService.getLatentSize(tile_height)
        tile_width = self.untisService.getLatentSize(tile_width)
        padding = self.getPaddLatentSize(padding)

        return (tile_height, tile_width, padding)
    
    def tileLatent(self, seed, steps, latent_image, positive, negative, tile_width, tile_height, tiling_mode, passes, tiling_strategy, padding_strategy, overlap_padding, disable_noise):
        samples = latent_image.get("samples")
        shape = samples.shape
        B, C, H, W = shape

        # Converts Size From Pixel to Latent
        tile_height, tile_width, ov_pd = self.convertPXtoLatentSize(tile_height, tile_width, overlap_padding)

        print(f"[EUP - Latent Tiler]: Original Latent Size: {samples.shape}")
        print(f"[EUP - Latent Tiler]: Tile Size: {tile_height}x{tile_width}")

        noise = self.tilingService.generateRandomNoise(latent_image, seed, disable_noise)
        noise_mask = self.tilingService.generateNoiseMask(latent_image, noise)

        #### Chosing Tiling Strategy ####
        if tiling_mode == "single-pass":
            if tiling_strategy == "simple":
                tiled_positions = self.tilingService.st_Service.generatePosforSTStrategy(W, H, tile_width, tile_height)
                tiled_tiles = self.tilingService.st_Service.getTilesforSTStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.st_Service.getNoiseTilesforSTStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.st_Service.getDenoiseMaskTilesforSTStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.st_Service.getBlendMaskTilesforSTStrategy(tiled_positions, B)
            elif tiling_strategy == "random":
                tiled_positions = self.tilingService.rt_Service.generatePosforRTStrategy(W, H, tile_width, tile_height, seed)
                tiled_tiles = self.tilingService.rt_Service.getTilesforRTStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.rt_Service.getNoiseTilesforRTStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.rt_Service.getDenoiseMaskTilesforRTStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.rt_Service.getBlendMaskTilesforRTStrategy(tiled_positions, B)
            elif tiling_strategy == "padded":
                tiled_positions = self.tilingService.pt_Service.generatePosforPTStrategy(W, H, tile_width, tile_height, ov_pd)
                tiled_tiles = self.tilingService.pt_Service.getTilesforPTStrategy(samples, tiled_positions)
                tiled_tiles = self.tilingService.pt_Service.getPaddedTilesforPTStrategy(tiled_tiles, tiled_positions, padding_strategy, ov_pd)
                tiled_noise_tiles = self.tilingService.pt_Service.getNoiseTilesforPTStrategy(latent_image, tiled_tiles, seed, disable_noise)
                tiled_denoise_masks = self.tilingService.pt_Service.getDenoiseMaskTilesforPTStrategy(noise_mask, tiled_positions, tiled_tiles)
                tiled_blend_masks = self.tilingService.pt_Service.getBlendMaskTilesforPTStrategy(tiled_tiles, B)
            elif tiling_strategy == "adjacency-padded":
                tiled_positions = self.tilingService.apt_Service.generatePosforAPTStrategy(W, H, tile_width, tile_height, ov_pd)
                tiled_tiles = self.tilingService.apt_Service.getTilesforAPTStrategy(samples, tiled_positions)
                tiled_tiles = self.tilingService.apt_Service.getPaddedTilesforAPTStrategy(tiled_tiles, tiled_positions, padding_strategy, ov_pd)
                tiled_noise_tiles = self.tilingService.apt_Service.getNoiseTilesforAPTStrategy(latent_image, tiled_tiles, seed, disable_noise)
                tiled_denoise_masks = self.tilingService.apt_Service.getDenoiseMaskTilesforAPTStrategy(noise_mask, tiled_positions, tiled_tiles)
                tiled_blend_masks = self.tilingService.apt_Service.getBlendMaskTilesforAPTStrategy(tiled_tiles, B)
            elif tiling_strategy == "context-padded":
                tiled_positions = self.tilingService.cpt_Service.generatePosforCPTStrategy(W, H, tile_width, tile_height)
                tiled_denoise_masks = self.tilingService.cpt_Service.getDenoiseMaskTilesforCPTStrategy(noise_mask, tiled_positions, B, H, W)
                tiled_tiles = self.tilingService.cpt_Service.getTilesforCPTStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.cpt_Service.getNoiseTilesforCPTStrategy(noise, tiled_positions)
                tiled_blend_masks = self.tilingService.cpt_Service.getBlendeMaskTilesforCPTStrategy(tiled_positions, B)
            elif tiling_strategy == "overlaping":
                tiled_positions = self.tilingService.ovp_Service.generatePosforOVPStrategy(W, H, tile_width, tile_height, ov_pd)
                tiled_tiles = self.tilingService.ovp_Service.getTilesforOVPStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.ovp_Service.getNoiseTilesforOVPStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.ovp_Service.getDenoiseMaskTilesforOVPStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.ovp_Service.getBlendeMaskTilesforOVPStrategy(tiled_positions, B)
            elif tiling_strategy == "adaptive":
                tiled_positions = self.tilingService.adp_Service.generatePosforADPStrategy(samples, tile_width, tile_height)
                tiled_tiles = self.tilingService.adp_Service.getTilesforADPStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.adp_Service.getNoiseTilesforADPStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.adp_Service.getDenoiseMaskTilesforADPStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.adp_Service.getBlendeMaskTilesforADPStrategy(tiled_positions, B)
            elif tiling_strategy == "hierarchical":
                tiled_positions = self.tilingService.hrc_Service.generatePosforHRCStrategy(samples)
                tiled_tiles = self.tilingService.hrc_Service.getTilesforHRCStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.hrc_Service.getNoiseTilesforHRCStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.hrc_Service.getDenoiseMaskTilesforHRCStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.hrc_Service.getBlendMaskTilesforHRCStrategy(tiled_positions, B)
            elif tiling_strategy == "non-uniform":
                tiled_positions = self.tilingService.nu_Service.generatePosforNUStrategy(samples, tile_width, tile_height)
                tiled_tiles = self.tilingService.nu_Service.getTilesforNUStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.nu_Service.getNoiseTilesforNUStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.nu_Service.getDenoiseMaskTilesforNUStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.nu_Service.getBlendeMaskTilesforNUStrategy(tiled_positions, B)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Strategy for Single-Pass Mode. Please select a valid Strategy.")
        elif tiling_mode == "multi-pass":
            if tiling_strategy == "simple":
                tiled_positions = self.tilingService.mp_st_Service.generatePosforMP_STStrategy(W, H, tile_width, tile_height, passes)
                tiled_tiles = self.tilingService.mp_st_Service.getTilesforMP_STStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_st_Service.getNoiseTilesforMP_STStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_st_Service.getDenoiseMaskTilesforMP_STStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_st_Service.getBlendMaskTilesforMP_STStrategy(tiled_positions, B)
            elif tiling_strategy == "random":
                tiled_positions = self.tilingService.mp_rt_Service.generatePos_forMP_RTStrategy(W, H, tile_width, tile_height, seed, passes)
                tiled_tiles = self.tilingService.mp_rt_Service.getTilesforMP_RTStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_rt_Service.getNoiseTilesforMP_RTStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_rt_Service.getDenoiseMaskTilesforMP_RTStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_rt_Service.getBlendMaskTilesforMP_RTStrategy(tiled_positions, B)
            elif tiling_strategy == "padded":
                tiled_positions = self.tilingService.mp_apt_Service.generatePos_forMP_APTStrategy(W, H, tile_width, tile_height, ov_pd, passes)
                tiled_tiles = self.tilingService.mp_apt_Service.getTilesforMP_APTStrategy(samples, tiled_positions)
                tiled_tiles = self.tilingService.mp_apt_Service.getPaddedTilesforMP_APTStrategy(tiled_tiles, tiled_positions, padding_strategy, ov_pd)
                tiled_noise_tiles = self.tilingService.mp_apt_Service.getNoiseTilesforMP_APTStrategy(latent_image, tiled_tiles, seed, disable_noise)
                tiled_denoise_masks = self.tilingService.mp_apt_Service.getDenoiseMaskTilesforMP_APTStrategy(noise_mask, tiled_positions, tiled_tiles)
                tiled_blend_masks = self.tilingService.mp_apt_Service.getBlendMaskTilesforMP_APTStrategy(tiled_tiles, B)
            elif tiling_strategy == "adjacency-padded":
                tiled_positions = self.tilingService.mp_apt_Service.generatePos_forMP_APTStrategy(W, H, tile_width, tile_height, ov_pd, passes)
                tiled_tiles = self.tilingService.mp_apt_Service.getTilesforMP_APTStrategy(samples, tiled_positions)
                tiled_tiles = self.tilingService.mp_apt_Service.getPaddedTilesforMP_APTStrategy(tiled_tiles, tiled_positions, padding_strategy, ov_pd)
                tiled_noise_tiles = self.tilingService.mp_apt_Service.getNoiseTilesforMP_APTStrategy(latent_image, tiled_tiles, seed, disable_noise)
                tiled_denoise_masks = self.tilingService.mp_apt_Service.getDenoiseMaskTilesforMP_APTStrategy(noise_mask, tiled_positions, tiled_tiles)
                tiled_blend_masks = self.tilingService.mp_apt_Service.getBlendMaskTilesforMP_APTStrategy(tiled_tiles, B)
            elif tiling_strategy == "context-padded":
                print("[EUP - Latent Tiler]: Warning: In the Context-Padded Tiling Strategy, the passes, padding_strategy,padding parameters are not used.")
                tiled_positions = self.tilingService.mp_cpt_Service.generatePosforMP_CPTStrategy(W, H, tile_width, tile_height)
                tiled_denoise_masks = self.tilingService.mp_cpt_Service.getDenoiseMaskTilesforMP_CPTStrategy(noise_mask, tiled_positions, B, H, W)
                tiled_tiles = self.tilingService.mp_cpt_Service.getTilesforMP_CPTStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_cpt_Service.getNoiseTilesforMP_CPTStrategy(noise, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_cpt_Service.getBlendeMaskTilesforMP_CPTStrategy(tiled_positions, B)
            elif tiling_strategy == "overlaping":
                tiled_positions = self.tilingService.mp_ovp_Service.generatePosforMP_OVPStrategy(W, H, tile_width, tile_height, ov_pd, passes)
                tiled_tiles = self.tilingService.mp_ovp_Service.getTilesforMP_OVPStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_ovp_Service.getNoiseTilesforMP_OVPStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_ovp_Service.getDenoiseMaskTilesforMP_OVPStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_ovp_Service.getBlendeMaskTilesforMP_OVPStrategy(tiled_positions, B)
            elif tiling_strategy == "adaptive":
                tiled_positions = self.tilingService.mp_adp_Service.generatePos_forMP_ADPStrategy(samples, tile_width, tile_height, passes)
                tiled_tiles = self.tilingService.mp_adp_Service.getTilesforMP_ADPStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_adp_Service.getNoiseTilesforMP_ADPStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_adp_Service.getDenoiseMaskTilesforMP_ADPStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_adp_Service.getBlendeMaskTilesforMP_ADPStrategy(tiled_positions, B)
            elif tiling_strategy == "hierarchical":
                tiled_positions = self.tilingService.mp_hrc_Service.generatePos_forMP_HRCStrategy(samples, passes)
                tiled_tiles = self.tilingService.mp_hrc_Service.getTilesforMP_HRCStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_hrc_Service.getNoiseTilesforMP_HRCStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_hrc_Service.getDenoiseMaskTilesforMP_HRCStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_hrc_Service.getBlendMaskTilesforMP_HRCStrategy(tiled_positions, B)
            elif tiling_strategy == "non-uniform":
                tiled_positions = self.tilingService.mp_nu_Service.generatePosforMP_NUStrategy(samples, passes, tile_width, tile_height)
                tiled_tiles = self.tilingService.mp_nu_Service.getTilesforMP_NUStrategy(samples, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_nu_Service.getNoiseTilesforMP_NUStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_nu_Service.getDenoiseMaskTilesforMP_NUStrategy(noise_mask, tiled_positions)
                tiled_blend_masks = self.tilingService.mp_nu_Service.getBlendeMaskTilesforMP_NUStrategy(tiled_positions, B)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Strategy for Multi-Pass Mode. Please select a valid Strategy.")
        else:
            raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode. Please select a valid Tiling Mode.")

        #### Prepare the tiles for sampling ####
        return self.latentTilerService.prepareTilesforSampling(positive, negative, shape, samples, tiled_positions, tiled_tiles, tiled_noise_tiles, tiled_denoise_masks, tiled_blend_masks) 

class LatentMerger:
    def __init__(self):
        self.latentMergerService = LatentMergerService()

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
            latent_tiles, tile_positions, noise_masks = self.latentMergerService.unpackTiles(pass_data)

            # Combine the tiles, positions, and masks into one list of tuples
            tile_data = [(tile, pos, mask) for tile, pos, mask in zip(latent_tiles, tile_positions, noise_masks)]
            all_tile_data.extend(tile_data) 

        # Perform the merging for all passes at once
        merged_samples, weight_map = self.latentMergerService.performMerging(all_tile_data)

        # Normalize the result
        weight_map = torch.clamp(weight_map, min=1e-6) 
        merged_samples /= weight_map

        return ({"samples": merged_samples},)

NODE_CLASS_MAPPINGS = {
    "EUP - Latent Tiler": LatentTiler,
    "EUP - Latent Merger": LatentMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}
