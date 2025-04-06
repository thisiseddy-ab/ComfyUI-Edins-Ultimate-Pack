#### Build In Lib's #####
from typing import List, Tuple

#### Comfy Lib's ####
from nodes import MAX_RESOLUTION

#### Third Party Lib's #####
import torch


#### My Services's #####
from EUP.services.latent import LatentMergerService
   
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
    "EUP - Latent Merger": LatentMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}
