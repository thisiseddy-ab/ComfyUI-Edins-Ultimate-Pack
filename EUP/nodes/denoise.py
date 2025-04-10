
class SmoothDenoisingStrategyParameters():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_value": ("FLOAT", {"default": 0.95, "min": 0.50, "max": 1.0, "step": 0.01,}),
                "max_value": ("FLOAT", {"default": 1.0, "min": 0.50, "max": 1.0, "step": 0.01,}),
                "falloff": ("FLOAT", {"default": 0.30, "min": 0, "max": 1.0, "step": 0.01,}),
            }
        }

    RETURN_TYPES = ("DENOISE_STRG_PARS",)
    RETURN_NAMES = ("denoise_strategy_pars",)

    OUTPUT_TOOLTIPS = ("The Denoise Strategy Parameters",)
    FUNCTION = "passParameters"

    CATEGORY = "EUP - Ultimate Pack/denoise"
    DESCRIPTION = "Provides the parameters for the smooth denoising strategy."

    def passParameters(self, min_value, max_value, falloff):
        return ((min_value, max_value, falloff),)
    
class AspectAdaptionDenoisingStrategyParameters():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_value": ("FLOAT", {"default": 0.95, "min": 0.50, "max": 1.0, "step": 0.01,}),
                "max_value": ("FLOAT", {"default": 1.0, "min": 0.50, "max": 1.0, "step": 0.01,}),
                "base_falloff": ("FLOAT", {"default": 0.30, "min": 0, "max": 1.0, "step": 0.01,}),
            }
        }

    RETURN_TYPES = ("DENOISE_STRG_PARS",)
    RETURN_NAMES = ("denoise_strategy_pars",)

    OUTPUT_TOOLTIPS = ("The Denoise Strategy Parameters",)
    FUNCTION = "passParameters"

    CATEGORY = "EUP - Ultimate Pack/denoise"
    DESCRIPTION = "Provides the parameters for the smooth denoising strategy."

    def passParameters(self, min_value, max_value, falloff):
        return ((min_value, max_value, falloff),)
    

NODE_CLASS_MAPPINGS = {
    "EUP - Smmooth Denoising Strategy Parameters" : SmoothDenoisingStrategyParameters,
    "EUP - Aspect Adaption Denoising Strategy Parameters" : AspectAdaptionDenoisingStrategyParameters,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}