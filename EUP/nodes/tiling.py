
#### Comfy Lib's ####
from nodes import MAX_RESOLUTION

class SimpleTilingStrategyParameters():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tiling_mode": (["single-pass", "multi-pass"],),
                "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
            }
        }

    RETURN_TYPES = ("TILING_STRG_PARS",)
    RETURN_NAMES = ("tiling_strategy_pars",)

    OUTPUT_TOOLTIPS = ("The Strategy Parameters",)
    FUNCTION = "passParameters"

    CATEGORY = "EUP - Ultimate Pack/tiling"
    DESCRIPTION = "Provides the parameters for the simple tiling strategy."

    def passParameters(self, tile_width, tile_height, tiling_mode, passes):
        return ((tile_width, tile_height, tiling_mode, passes),)
    
class RandomTilingStrategyParameters(SimpleTilingStrategyParameters):
    pass
    
class PaddedTilingStrategyParameters():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tiling_mode": (["single-pass", "multi-pass"],),
                "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
                "padding_strategy": (["organic", "circular", "reflect", "replicate", "zero"],),
                "padding": ("INT", {"default": 16, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("TILING_STRG_PARS",)
    RETURN_NAMES = ("tiling_strategy_pars",)

    OUTPUT_TOOLTIPS = ("The Strategy Parameters",)
    FUNCTION = "passParameters"

    CATEGORY = "EUP - Ultimate Pack/tiling"
    DESCRIPTION = "Provides the parameters for the padded tiling strategy."

    def passParameters(self, tile_width, tile_height, tiling_mode, passes, padding_strategy, padding):
        return ((tile_width, tile_height, tiling_mode, passes, padding_strategy, padding),)
    
class AdjacencyPaddedTilingStrategyParameters(PaddedTilingStrategyParameters):
    pass
    
class ContextPaddedTilingStrategyParameters():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tiling_mode": (["single-pass", "multi-pass"],),
                "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
            }
        }

    RETURN_TYPES = ("TILING_STRG_PARS",)
    RETURN_NAMES = ("tiling_strategy_pars",)

    OUTPUT_TOOLTIPS = ("The Strategy Parameters",)
    FUNCTION = "passParameters"

    CATEGORY = "EUP - Ultimate Pack/sampling"
    DESCRIPTION = "Provides the parameters for the context-padded tiling strategy."

    def passParameters(self, tile_width, tile_height, tiling_mode, passes):
        return ((tile_width, tile_height, tiling_mode, passes),)
    
class OverlapingTilingStrategyParameters():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tiling_mode": (["single-pass", "multi-pass"],),
                "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
                "overalp": ("INT", {"default": 16, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("TILING_STRG_PARS",)
    RETURN_NAMES = ("tiling_strategy_pars",)

    OUTPUT_TOOLTIPS = ("The Strategy Parameters",)
    FUNCTION = "passParameters"

    CATEGORY = "EUP - Ultimate Pack/tiling"
    DESCRIPTION = "Provides the parameters for the overlaping tiling strategy."

    def passParameters(self, tile_width, tile_height, tiling_mode, passes, overalp):
        return ((tile_width, tile_height, tiling_mode, passes, overalp),)
    
class AdaptiveTilingStrategyParameters():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                "tiling_mode": (["single-pass", "multi-pass"],),
                "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
                "base_model": (["SD 1.4 & SD 1.5", "SDXL 1.0", "SD 2.0", "Flux", "SD 2.1", "SD 3.0", "SD 3.5", "SD 3.5 Large Turbo"],),
                "VRAM": (["8GB", "6GB", "8GB", "12GB", "16GB", "24GB", "32GB", "40GB"],),
                "precision": (["FP16", "FP32"],),
                "tile_growth_exponent": ("FLOAT", {"default": 0.65, "min": 0.1, "max": 5.0, "step":0.1, "round": 0.01,}),
            }
        }

    RETURN_TYPES = ("TILING_STRG_PARS",)
    RETURN_NAMES = ("tiling_strategy_pars",)

    OUTPUT_TOOLTIPS = ("The Strategy Parameters",)
    FUNCTION = "passParameters"

    CATEGORY = "EUP - Ultimate Pack/tiling"
    DESCRIPTION = "Provides the parameters for the overlaping tiling strategy."

    def passParameters(self, tile_width, tile_height, tiling_mode, passes, base_model, VRAM, precision, tile_growth_exponent):
        return ((tile_width, tile_height, tiling_mode, passes, base_model, VRAM, precision, tile_growth_exponent),)
    
class HierarchicalTilingStrategyParameters():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiling_mode": (["single-pass", "multi-pass"],),
                "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
                "base_model": (["SD 1.4 & SD 1.5", "SDXL 1.0", "SD 2.0", "Flux", "SD 2.1", "SD 3.0", "SD 3.5", "SD 3.5 Large Turbo"],),
                "VRAM": (["8GB", "6GB", "8GB", "12GB", "16GB", "24GB", "32GB", "40GB"],),
                "precision": (["FP16", "FP32"],),
                "tile_growth_exponent": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 3.0, "step": 0.01,}),
            }
        }

    RETURN_TYPES = ("TILING_STRG_PARS",)
    RETURN_NAMES = ("tiling_strategy_pars",)

    OUTPUT_TOOLTIPS = ("The Strategy Parameters",)
    FUNCTION = "passParameters"

    CATEGORY = "EUP - Ultimate Pack/tiling"
    DESCRIPTION = "Provides the parameters for the overlaping tiling strategy."

    def passParameters(self, tiling_mode, passes, base_model, VRAM, precision, tile_growth_exponent):
        return ((tiling_mode, passes, base_model, VRAM, precision, tile_growth_exponent),)
    
class RandomHierarchicalTilingStrategyParameters(HierarchicalTilingStrategyParameters):
    pass

    
class NonUniformTilingStrategyParameters():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiling_mode": (["single-pass", "multi-pass"],),
                "passes": ("INT", {"default": 2, "min": 2, "max": 8}),
                "base_model": (["SD 1.4 & SD 1.5", "SDXL 1.0", "SD 2.0", "Flux", "SD 2.1", "SD 3.0", "SD 3.5", "SD 3.5 Large Turbo"],),
                "VRAM": (["8GB", "6GB", "8GB", "12GB", "16GB", "24GB", "32GB", "40GB"],),
                "precision": (["FP16", "FP32"],),
                "tile_growth_exponent": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 3.0, "step": 0.01,}),
                "scale_min": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,}),
                "scale_max": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 2.0, "step": 0.01,}),
            }
        }

    RETURN_TYPES = ("TILING_STRG_PARS",)
    RETURN_NAMES = ("tiling_strategy_pars",)

    OUTPUT_TOOLTIPS = ("The Strategy Parameters",)
    FUNCTION = "passParameters"

    CATEGORY = "EUP - Ultimate Pack/tiling"
    DESCRIPTION = "Provides the parameters for the overlaping tiling strategy."

    def passParameters(self, tiling_mode, passes, base_model, VRAM, precision, tile_growth_exponent, scale_min, scale_max):
        return ((tiling_mode, passes, base_model, VRAM, precision, tile_growth_exponent, scale_min, scale_max),)
    

NODE_CLASS_MAPPINGS = {
    "EUP - Simple Tiling Strategy Parameters" : SimpleTilingStrategyParameters,
    "EUP - Random Tiling Strategy Parameters" : SimpleTilingStrategyParameters,
    "EUP - Padded Tiling Strategy Parameters" : PaddedTilingStrategyParameters,
    "EUP - Adjacency Padded Tiling Strategy Parameters" : AdjacencyPaddedTilingStrategyParameters,
    "EUP - Context Padded Tiling Strategy Parameters" : ContextPaddedTilingStrategyParameters,
    "EUP - Overlaping Tiling Strategy Parameters" : OverlapingTilingStrategyParameters,
    "EUP - Adaptive Tiling Strategy Parameters" : AdaptiveTilingStrategyParameters,
    "EUP - Hierarchical Tiling Strategy Parameters" : HierarchicalTilingStrategyParameters,
    "EUP - Random-Hierarchical Tiling Strategy Parameters" : RandomHierarchicalTilingStrategyParameters,
    "EUP - Non-Uniform Tiling Strategy Parameters" : NonUniformTilingStrategyParameters,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}