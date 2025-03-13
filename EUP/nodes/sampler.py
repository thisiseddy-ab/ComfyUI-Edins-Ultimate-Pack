#### Comfy Lib's ####
from nodes import MAX_RESOLUTION
import comfy.samplers

#### My Services's #####
from EUP.services.sampler import (
    KSamplerService
)

class Tiled_KSampler():

    def __init__(self):
        self.ksamplerService = KSamplerService()

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
                "tiling_mode": (["single-pass", "multi-pass"],),
                "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
                "tiling_strategy": (["simple", "random", "padded", "adjacency-padded", "context-padded", "overlaping", "adaptive", "hierarchical", "non-uniform"],),
                "padding_strategy": (["organic", "circular", "reflect", "replicate", "zero"],),
                "overalp_padding": ("INT", {"default": 16, "min": 0, "max": 128}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "EUP - Ultimate Pack/sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, 
               tile_width, tile_height, tiling_mode, passes, tiling_strategy, padding_strategy, overalp_padding, denoise=1.0,
               ):

        return self.ksamplerService.commonKsampler(model=model, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, 
                                    latent=latent_image, tile_width=tile_width, tile_height=tile_height, tiling_mode=tiling_mode, passes=passes, tiling_strategy=tiling_strategy, 
                                    padding_strategy=padding_strategy, ov_pd=overalp_padding, denoise=denoise,
        )

class Tiled_KSamplerAdvanced():
    def __init__(self):
        self.ksamplerService = KSamplerService()
        
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
                "tiling_mode": (["single-pass", "multi-pass"],),
                "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
                "tiling_strategy": (["simple", "random", "padded", "adjacency-padded", "context-padded", "overlaping", "adaptive", "hierarchical", "non-uniform"],),
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
               tiling_mode, passes, tiling_strategy, padding_strategy, overalp_padding, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0,
            ):
        
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        return self.ksamplerService.commonKsampler(model=model, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, 
            latent=latent_image, tile_width=tile_width, tile_height=tile_height, tiling_mode=tiling_mode, passes=passes, tiling_strategy=tiling_strategy, padding_strategy=padding_strategy, 
            ov_pd=overalp_padding, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise,
        )
    
NODE_CLASS_MAPPINGS = {
    "EUP - Tiled KSampler" : Tiled_KSampler,
    "EUP - Tiled KSampler Advanced" : Tiled_KSamplerAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
}