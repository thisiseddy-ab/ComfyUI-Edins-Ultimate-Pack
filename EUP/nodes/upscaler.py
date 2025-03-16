
#### Comfy UI Lib's ####
from nodes import MAX_RESOLUTION
import comfy.samplers
import comfy.sample

#### Services ####
from EUP.services.status import StatusService
from EUP.services.upscaler import PixelTiledKSampleUpscalerService

SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]']

class IterativeLatentUpscale:

    def __init__(self):
        self.statusService = StatusService()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "samples": ("LATENT", ),
                     "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1, "max": 10000, "step": 0.1}),
                     "steps": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                     "temp_prefix": ("STRING", {"default": ""}),
                     "upscaler": ("UPSCALER",),
                     "step_mode": (["simple", "geometric"], {"default": "simple"})
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("LATENT", "VAE")
    RETURN_NAMES = ("latent", "vae")
    FUNCTION = "doit"

    CATEGORY = "EUP - Ultimate Pack/Upscale"

    def doit(self, samples, upscale_factor, steps, temp_prefix, upscaler : PixelTiledKSampleUpscalerService, step_mode="simple", unique_id=None):
        w = samples['samples'].shape[3]*8  # image width
        h = samples['samples'].shape[2]*8  # image height

        if temp_prefix == "":
            temp_prefix = None

        if step_mode == "geometric":
            upscale_factor_unit = pow(upscale_factor, 1.0/steps)
        else:  # simple
            upscale_factor_unit = max(0, (upscale_factor - 1.0) / steps)

        current_latent = samples
        noise_mask = current_latent.get('noise_mask')
        scale = 1

        for i in range(steps-1):
            if step_mode == "geometric":
                scale *= upscale_factor_unit
            else:  # simple
                scale += upscale_factor_unit

            new_w = w*scale
            new_h = h*scale
            self.statusService.updateNodeStatus(unique_id, f"{i+1}/{steps} steps | x{scale:.2f}", (i+1)/steps)
            print(f"IterativeLatentUpscale[{i+1}/{steps}]: {new_w:.1f}x{new_h:.1f} (scale:{scale:.2f}) ")
            step_info = i, steps
            current_latent = upscaler.upscaleShape(step_info, current_latent, new_w, new_h, temp_prefix)
            if noise_mask is not None:
                current_latent['noise_mask'] = noise_mask

        if scale < upscale_factor:
            new_w = w*upscale_factor
            new_h = h*upscale_factor
            self.statusService.updateNodeStatus(unique_id, f"Final step | x{upscale_factor:.2f}", 1.0)
            print(f"IterativeLatentUpscale[Final]: {new_w:.1f}x{new_h:.1f} (scale:{upscale_factor:.2f}) ")
            step_info = steps-1, steps
            current_latent = upscaler.upscaleShape(step_info, current_latent, new_w, new_h, temp_prefix)

        self.statusService.updateNodeStatus(unique_id, "", None)

        return current_latent, upscaler.vae

class PixelTiledKSampleUpscalerProvider:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_mode": (["single-pass", "multi-pass"],),
                    "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
                    "tiling_strategy": (["simple", "random", "padded", "adjacency-padded", "context-padded", "overlaping", "adaptive", "hierarchical", "non-uniform"],),
                    "padding_strategy": (["organic", "circular", "reflect", "replicate", "zero"],),
                    "overalp_padding": ("INT", {"default": 16, "min": 0, "max": 128}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                        },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "tile_cnet_opt": ("CONTROL_NET", ),
                        "tile_cnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "EUP - Ultimate Pack/Upscale"

    def doit(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy, 
             tiling_mode, passes, padding_strategy, overalp_padding, upscale_model_opt=None,pk_hook_opt=None, tile_cnet_opt=None, tile_cnet_strength=1.0, overlap=64):
        
        upscaler = PixelTiledKSampleUpscalerService(
            scale_method=scale_method, model=model, vae=vae, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, 
            denoise=denoise, tile_width=tile_width, tile_height=tile_height, tiling_mode=tiling_mode, passes=passes, tiling_strategy=tiling_strategy, padding_strategy=padding_strategy, 
            overalp_padding=overalp_padding, upscale_model_opt=upscale_model_opt, hook_opt=pk_hook_opt, tile_cnet_opt=tile_cnet_opt, tile_size=max(tile_width, tile_height), 
            tile_cnet_strength=tile_cnet_strength, overlap=overlap 
        )
        
        return (upscaler, )


class PixelTiledKSampleUpscalerProviderPipe:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_mode": (["single-pass", "multi-pass"],),
                    "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
                    "tiling_strategy": (["simple", "random", "padded", "adjacency-padded", "context-padded", "overlaping", "adaptive", "hierarchical", "non-uniform"],),
                    "padding_strategy": (["organic", "circular", "reflect", "replicate", "zero"],),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "basic_pipe": ("BASIC_PIPE",)
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "tile_cnet_opt": ("CONTROL_NET", ),
                        "tile_cnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "EUP - Ultimate Pack/Upscale"

    def doit(self, scale_method, seed, steps, cfg, sampler_name, scheduler, denoise, tile_width, tile_height, tiling_mode, passes, tiling_strategy, padding_strategy, overalp_padding, 
             basic_pipe, upscale_model_opt=None, pk_hook_opt=None, tile_cnet_opt=None, tile_cnet_strength=1.0):
        
        model, _, vae, positive, negative = basic_pipe
        upscaler = upscaler = PixelTiledKSampleUpscalerService(
            scale_method=scale_method, model=model, vae=vae, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, 
            denoise=denoise, tile_width=tile_width, tile_height=tile_height, tiling_mode=tiling_mode, passes=passes, tiling_strategy=tiling_strategy, padding_strategy=padding_strategy, 
            overalp_padding=overalp_padding, upscale_model_opt=upscale_model_opt, hook_opt=pk_hook_opt, tile_cnet_opt=tile_cnet_opt, tile_size=max(tile_width, tile_height), 
            tile_cnet_strength=tile_cnet_strength)
        
        return (upscaler, )
    

NODE_CLASS_MAPPINGS = {
    "EUP - Iterative Latent Upscaler": IterativeLatentUpscale,
    "EUP - Pixel TiledKSample Upscaler Provider": PixelTiledKSampleUpscalerProvider,
    "EUP - Pixel TiledKSample Upscaler Provider Pipe": PixelTiledKSampleUpscalerProviderPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}

