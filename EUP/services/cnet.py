#### Comfy Lib's #####
import comfy.controlnet

#### Third Party Lib's #####
import torch

#### Services #####
from EUP.services.tensor import TensorService
from EUP.services.list import ListService

class ControlNetService():

    def __init__(self):
        self.tensorService = TensorService()
        self.listService = ListService()
    
    def extractCnet(self, positive, negative):
        cnets = [c['control'] for (_, c) in positive + negative if 'control' in c]
        cnets = list(set([x for m in cnets for x in self.listService.recursion_toList(m, "previous_controlnet")]))  # Recursive extraction
        return [x for x in cnets if isinstance(x, comfy.controlnet.ControlNet)]
    
    def prepareCnet_imgs(self, cnets, shape):
        return [
            torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
            if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 else None
            for m in cnets
    ]

    def sliceCnet(self, h, h_len, w, w_len, model:comfy.controlnet.ControlBase, img):
        if img is None:
            img = model.cond_hint_original
        hint = self.tensorService.getSlice(img, h*8, h_len*8, w*8, w_len*8)
        if isinstance(model, comfy.controlnet.ControlLora):
            model.cond_hint = hint.float().to(model.device)
        else:
            model.cond_hint = hint.to(model.control_model.dtype).to(model.control_model.device)
    
    def prepareSlicedCnets(self, pos, neg, cnets, cnet_imgs, tile_h, tile_h_len, tile_w, tile_w_len):
        for idx, (m, img) in enumerate(zip(cnets, cnet_imgs)):
            self.sliceCnet(tile_h, tile_h_len, tile_w, tile_w_len, m, img)
            
            # Replace the original 'control' key in the corresponding positions
            if idx < len(pos):
                pos[idx] = (None, {'control': m})
            else:
                pos.append((None, {'control': m}))
            
            if idx < len(neg):
                neg[idx] = (None, {'control': m})
            else:
                neg.append((None, {'control': m}))
        
        # Return the modified pos and neg lists
        return pos, neg