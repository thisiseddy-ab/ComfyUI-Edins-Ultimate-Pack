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
                "sampler_advanced_pars": ("SAMPLER_ADVANCED_PARS",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "EUP - Ultimate Pack/sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, 
               sampler_advanced_pars, denoise=1.0,
               ):
        
        return_with_leftover_noise = "enable"
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        
        add_noise = "enable"
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        steps_total = int(steps / denoise)

        return self.ksamplerService.commonKsampler(model=model, seed=seed, steps=steps_total, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, 
                                    latent=latent_image, sampler_advanced_pars=sampler_advanced_pars, denoise=denoise, disable_noise=disable_noise, start_step=steps_total-steps, 
                                    last_step=steps_total, force_full_denoise=force_full_denoise, actual_steps=steps,
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
                "sampler_advanced_pars": ("SAMPLER_ADVANCED_PARS",),
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

    def sample(self, model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, sampler_advanced_pars, start_at_step, end_at_step, 
               return_with_leftover_noise, denoise=1.0,
            ):
        
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        return self.ksamplerService.commonKsampler(model=model, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, 
            latent=latent_image, sampler_advanced_pars=sampler_advanced_pars, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, 
            force_full_denoise=force_full_denoise, actual_steps=steps
        )

class SamplersAdvancedParameters():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ksampler_type": (["Eddy's", "BlenderNeko's"],),
                "tiling_strategy": (["simple", "random", "padded", "adjacency-padded", "context-padded", "overlaping", "adaptive", "hierarchical", "random-hierarchical", "non-uniform"],),
                "tiling_strategy_pars": ("TILING_STRG_PARS",),
                "noise_strategy": (["random"],),
                "noise_mask_strategy": (["standart"],),
                "denoise_strategy": (["none", "smooth", "aspect-adaption"],),
                "blend_strategy": (["none", "soft-focus"],),
            },
            "optional": {
                "noise_strategy_pars": ("NOISE_STRG_PARS",),
                "noise_mask_strategy_pars": ("NOISE_MASK_STRG_PARS",),
                "denoise_strategy_pars": ("DENOISE_STRG_PARS",),
                "blend_strategy_pars": ("BLEND_STRG_PARS",),
            }
        }

    RETURN_TYPES = ("SAMPLER_ADVANCED_PARS",)
    RETURN_NAMES = ("sampler_advanced_pars",)

    OUTPUT_TOOLTIPS = ("The Sampler Advanced Parameters",)
    FUNCTION = "passParameters"

    CATEGORY = "EUP - Ultimate Pack/sampling"
    DESCRIPTION = "Provides the Advanced parameters for the KSampler"

    def passParameters(self, ksampler_type, tiling_strategy, tiling_strategy_pars, noise_strategy, noise_strategy_pars, noise_mask_strategy, noise_mask_strategy_pars, 
                       denoise_strategy, denoise_strategy_pars, blend_strategy, blend_strategy_pars
                    ):
        return ((ksampler_type, tiling_strategy, tiling_strategy_pars, noise_strategy, noise_strategy_pars, noise_mask_strategy, noise_mask_strategy_pars, 
                       denoise_strategy, denoise_strategy_pars, blend_strategy, blend_strategy_pars),)   

NODE_CLASS_MAPPINGS = {
    "EUP - Tiled KSampler" : Tiled_KSampler,
    "EUP - Tiled KSampler Advanced" : Tiled_KSamplerAdvanced,
    "EUP - Sampler's Advanced Parameters" : SamplersAdvancedParameters,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}