#### Comfy Lib's ####
import comfy.model_management

#### Third Party Lib's #####
import torch

#### My Services's #####
from EUP.services.cnet import ControlNetService
from EUP.services.t2i import T2IService
from EUP.services.gligen import GligenService
from EUP.services.condition import ConditionService

class LatentTilerService():

    def __init__(self):
        self.conditionService = ConditionService()
        self.conrolNetService = ControlNetService()
        self.t2iService = T2IService()
        self.gligenService = GligenService()

    def prepareTilesforSampling(self, positive, negative, shape, samples, tiled_positions, tiled_tiles, tiled_noise_tiles, tiled_denoise_masks, tiled_blend_masks):
        # Extract and prepare cnets, T2Is, gligen, and spatial conditions
        cnets = self.conrolNetService.extractCnet(positive, negative)
        cnet_imgs = self.conrolNetService.prepareCnet_imgs(cnets, shape)
        T2Is = self.t2iService.extract_T2I(positive, negative)
        T2I_imgs = self.t2iService.prepareT2I_imgs(T2Is, shape)
        gligen_pos, gligen_neg = self.gligenService.extractGligen(positive, negative)
        sptcond_pos, sptcond_neg = self.conditionService.sptcondService.extractSpatialConds(positive, negative, shape, samples.device)
        
        tiled_latent = []
        for t_pos_pass, tile_pass, noise_pass, denoise_mask_pass , blend_mask_pass in zip(tiled_positions, tiled_tiles, tiled_noise_tiles, tiled_denoise_masks, tiled_blend_masks):
            tile_pass_l = []
            for t_pos_group, tile_group, noise_group, denoise_mask_group, blend_mask_group in zip(t_pos_pass, tile_pass, noise_pass, denoise_mask_pass, blend_mask_pass):
                tile_group_l = []
                for (x, y, tile_w, tile_h), tile, noise_tile, denoise_mask, blend_mask in zip(t_pos_group, tile_group, noise_group, denoise_mask_group, blend_mask_group):
                    
                    # Copy positive and negative lists for modification
                    pos = [c.copy() for c in positive]
                    neg = [c.copy() for c in negative]

                    #### Slice cnets, T2Is, gligen, and spatial conditions for the current tile ####
                    ## Slice CNETS ##
                    pos, neg, self.conrolNetService.prepareSlicedCnets(pos, neg, cnets, cnet_imgs, y, tile_h, x, tile_w)
                    ## Slice T2Is ##
                    pos, neg = self.t2iService.prepareSlicedT2Is(pos, neg, T2Is, T2I_imgs, y, tile_h, x, tile_w)

                    ## Slice Spatial Conditions  ##
                    pos, neg = self.conditionService.sptcondService.prepareSlicedConds(pos, neg, sptcond_pos, sptcond_neg, y, tile_h, x, tile_w)

                    # Slice GLIGEN ##
                    pos, neg = self.gligenService.prepsreSlicedGligen(pos, neg, gligen_pos, gligen_neg, y, tile_h, x, tile_w)

                    tile_group_l.append((tile, (x, y, tile_w, tile_h), noise_tile, denoise_mask, blend_mask, pos, neg))
                tile_pass_l.append(tile_group_l)
            tiled_latent.append(tile_pass_l)
        return tiled_latent

class LatentMergerService():
    
    def performMerging(self, all_tile_data):
        """
        Perform merging of all tiles from all passes in one step.
        """
        device = comfy.model_management.get_torch_device() 
        
        # Determine the max width and height needed to accommodate all tiles
        max_width = max(x + w for _, (x, y, w, h), _ in all_tile_data)
        max_height = max(y + h for _, (x, y, w, h), _ in all_tile_data)

        # Initialize merged samples and weight map with the required dimensions
        merged_samples = torch.zeros((1, 4, max_height, max_width), device=device)
        weight_map = torch.zeros_like(merged_samples)

        # Process each tile and apply it to the merged output
        for tile, (x, y, w, h), mask in all_tile_data:
            valid_tile, valid_mask = self.extractValidRegion(tile, mask, max_height, max_width, x, y)
            
            # Apply the tile to the merged samples and weight map
            merged_samples, weight_map = self.applyTiletoMerged(merged_samples, weight_map, valid_tile, valid_mask, x, y, w, h)

        return merged_samples, weight_map

    def applyTiletoMerged(self, merged_samples, weight_map, tile, mask, x, y, w, h):
        # Ensure all tensors are on the same device
        device = merged_samples.device  # Get the device of merged_samples
        tile, mask = self.processTileandMask(tile, mask, device)
        
        valid_tile_h, valid_tile_w = min(tile.shape[2], h), min(tile.shape[3], w)
        valid_mask_h, valid_mask_w = min(mask.shape[2], h), min(mask.shape[3], w)
        
        if valid_tile_h == 0 or valid_tile_w == 0:
            raise ValueError(f"Invalid tile dimensions: {valid_tile_h}, {valid_tile_w}")

        merged_samples[:, :, y:y + valid_tile_h, x:x + valid_tile_w] += tile[:, :, :valid_tile_h, :valid_tile_w] * mask[:, :, :valid_mask_h, :valid_mask_w]
        weight_map[:, :, y:y + valid_tile_h, x:x + valid_tile_w] += mask[:, :, :valid_mask_h, :valid_mask_w]

        return merged_samples, weight_map

    def extractValidRegion(self, tile, mask, max_height, max_width, x, y):
        valid_tile_h, valid_tile_w = min(tile.shape[2], max_height - y), min(tile.shape[3], max_width - x)
        valid_mask_h, valid_mask_w = min(mask.shape[2], max_height - y), min(mask.shape[3], max_width - x)
        if valid_tile_h == 0 or valid_tile_w == 0:
            raise ValueError(f"Invalid tile dimensions: {valid_tile_h}, {valid_tile_w}")
        return tile[:, :, :valid_tile_h, :valid_tile_w], mask[:, :, :valid_mask_h, :valid_mask_w]
    
    def processTileandMask(self, tile, mask, device):
        # Ensure the tensors are on the same device
        tile = tile.to(device)
        if tile.dim() == 3:
            tile = tile.unsqueeze(0)
        if tile.shape[1] == 3:
            tile = torch.cat([tile, torch.zeros((tile.shape[0], 1, tile.shape[2], tile.shape[3]), device=device)], dim=1)
        mask = mask.to(device)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        return tile, mask

    def unpackTiles(self, pass_data):
        latent_tiles = []
        tile_positions = []
        noise_masks = []

        for tile_group in pass_data:
            for tile, pos, mask in tile_group:
                latent_tiles.append(tile)
                tile_positions.append(pos)
                noise_masks.append(mask)

        return latent_tiles, tile_positions, noise_masks
