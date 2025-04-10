#### Third Party Lib's #####
import torch

#### Services #####
from EUP.services.tensor import TensorService

class BlendService():

    def __init__(self):
        self.tensorService = TensorService()
    
    # SF - Soft-Focus Blending Strategy
    def generateBlendMaskforSFStrategy(
        self, latentImage, tileSize, radial_strength: float = 1.0, directional_strength: float = 1.0, min_fade: float = 0.01,
        blend_power: float = 1.0, fade_offset: float = 0.5, fade_scale: float = 0.5, device: str = 'cpu'
    ):
        """
        Creates a soft-focus blending mask for latent tiles with customizable radial and directional strengths.

        Parameters:
            latentImage: The latent image tensor wrapper.
            tileSize (tuple): (tile_w, tile_h)
            device (str): Device to place the tensor on.
            radial_strength (float): Strength of the radial fade component.
            directional_strength (float): Strength of the directional fade component.
            min_fade (float): Minimum fade value to avoid black center.
            blend_power (float): Power factor to sharpen/soften blend.
            fade_offset (float): Offset applied to cosine fade (default 0.5).
            fade_scale (float): Scale of cosine fade range (default 0.5).

        Returns:
            torch.Tensor: Blend mask of shape [B, C, tile_h, tile_w]
        """

        #### Latent Shape ####
        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)

        #### Tile Size ####
        tile_w, tile_h = tileSize

        # --- Radial Grid ---
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1, 1, tile_h, device=device),
            torch.linspace(-1, 1, tile_w, device=device),
            indexing="ij",
        )

        radius = torch.sqrt(grid_x**2 + grid_y**2)
        fade_radial = fade_offset + fade_scale * (1 - torch.cos(torch.clamp(radius * torch.pi, 0, torch.pi)))

        # --- Directional Fades ---
        fade_x = fade_offset + fade_scale * (1 - torch.cos(torch.linspace(0, torch.pi, tile_w, device=device)))
        fade_y = fade_offset + fade_scale * (1 - torch.cos(torch.linspace(0, torch.pi, tile_h, device=device)))
        
        fade_x = fade_x.view(1, 1, 1, -1)
        fade_y = fade_y.view(1, 1, -1, 1)

        # --- Combine ---
        fade = 1.0  # Start neutral
        if radial_strength > 0:
            fade *= fade_radial ** radial_strength
        if directional_strength > 0:
            fade *= (fade_x * fade_y) ** directional_strength

        # --- Clamp & Power Adjust ---
        fade = torch.clamp(fade, min=min_fade)
        fade = fade ** blend_power

        # --- Expand to match latent shape ---
        mask = fade.expand(B, C, tile_h, tile_w)

        return mask