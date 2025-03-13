#### Comfy Lib's #####
import comfy.controlnet
import comfy.sampler_helpers

#### Third Party Lib's #####

#### Services #####
from EUP.services.tensor import TensorService

class Spatial_Condition_Service():
    def __init__(self):
        self.tensorService = TensorService()

    def extractSpatialConds(self, positive, negative, shape, device):
        spatial_conds_pos = [
            (c[1]['area'] if 'area' in c[1] else None, 
                comfy.sampler_helpers.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
            for c in positive
        ]
        spatial_conds_neg = [
            (c[1]['area'] if 'area' in c[1] else None, 
                comfy.sampler_helpers.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
            for c in negative
        ]
        return (spatial_conds_pos, spatial_conds_neg)
    
    def sliceSpatialConds(self, tile_h, tile_h_len, tile_w, tile_w_len, cond, area):
        tile_h_end = tile_h + tile_h_len
        tile_w_end = tile_w + tile_w_len
        coords = area[0] #h_len, w_len, h, w,
        mask = area[1]
        if coords is not None:
            h_len, w_len, h, w = coords
            h_end = h + h_len
            w_end = w + w_len
            if h < tile_h_end and h_end > tile_h and w < tile_w_end and w_end > tile_w:
                new_h = max(0, h - tile_h)
                new_w = max(0, w - tile_w)
                new_h_end = min(tile_h_end, h_end - tile_h)
                new_w_end = min(tile_w_end, w_end - tile_w)
                cond[1]['area'] = (new_h_end - new_h, new_w_end - new_w, new_h, new_w)
            else:
                return (cond, True)
        if mask is not None:
            new_mask = self.tensorService.getSlice(mask, tile_h, tile_h_len, tile_w, tile_w_len)
            if new_mask.sum().cpu() == 0.0 and 'mask' in cond[1]:
                return (cond, True)
            else:
                cond[1]['mask'] = new_mask
        return (cond, False)
    
    def prepareSlicedConds(self, pos, neg, spatial_conds_pos, spatial_conds_neg, tile_h, tile_h_len, tile_w, tile_w_len):
        pos = [self.sliceSpatialConds(tile_h, tile_h_len, tile_w, tile_w_len, c, area) for c, area in zip(pos, spatial_conds_pos)]
        pos = [c for c, ignore in pos if not ignore]
        neg = [self.sliceSpatialConds(tile_h, tile_h_len, tile_w, tile_w_len, c, area) for c, area in zip(neg, spatial_conds_neg)]
        neg = [c for c, ignore in neg if not ignore]
        return (pos, neg)
    
class ConditionService():
    # sptcond - Stands for Spatial Condition
    def __init__(self):
        self.sptcondService = Spatial_Condition_Service()