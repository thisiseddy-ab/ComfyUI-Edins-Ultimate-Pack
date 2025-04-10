
class SmoothDenoisingStrategyParameters():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "radial_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,}),
                "directional_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,}),
                "min_fade": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.02, "step": 0.01,}),
                "blend_power": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01,}),
                "fade_offset": ("FLOAT", {"default": 0.5, "min": 0.00, "max": 1.0, "step": 0.01,}),
                "fade_scale": ("FLOAT", {"default": 0.5, "min": 0.00, "max": 1.0, "step": 0.01,}),
            }
        }

    RETURN_TYPES = ("DENOISE_STRG_PARS",)
    RETURN_NAMES = ("denoise_strategy_pars",)

    OUTPUT_TOOLTIPS = ("The Denoise Strategy Parameters",)
    FUNCTION = "passParameters"

    CATEGORY = "EUP - Ultimate Pack/denoise"
    DESCRIPTION = "Provides the parameters for the smooth denoising strategy."

    def passParameters(self, radial_strength, directional_strength, min_fade, blend_power, fade_offset, fade_scale):
        return ((radial_strength, directional_strength, min_fade, blend_power, fade_offset, fade_scale),)