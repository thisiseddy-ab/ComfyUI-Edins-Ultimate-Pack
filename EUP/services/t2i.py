#### Comfy Lib's #####
import comfy.controlnet

#### Third Party Lib's #####
import torch

#### Services #####
from EUP.services.tensor import TensorService
from EUP.services.list import ListService

class T2IService():

    def __init__(self):
        self.tensorService = TensorService()
        self.listService = ListService()

    def extract_T2I(self, positive, negative):
        t2is = [c['control'] for (_, c) in positive + negative if 'control' in c]
        t2is = [x for m in t2is for x in self.listService.recursion_toList(m, "previous_controlnet")]  # Recursive extraction
        return [x for x in t2is if isinstance(x, comfy.controlnet.T2IAdapter)]
    
    def prepareT2I_imgs(self, T2Is, shape):
        return [
            torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
            if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 or (m.channels_in == 1 and m.cond_hint_original.shape[1] != 1) else None
            for m in T2Is
        ]
    
    def slices_T2I(self, h, h_len, w, w_len, model:comfy.controlnet.ControlBase, img):
        model.control_input = None
        if img is None:
            img = model.cond_hint_original
        model.cond_hint = self.tensorService.getSlice(img, h*8, h_len*8, w*8, w_len*8).float().to(model.device)

    def prepareSlicedT2Is(self, pos, neg, T2Is, T2I_imgs, tile_h, tile_h_len, tile_w, tile_w_len):
        for idx, (m, img) in enumerate(zip(T2Is, T2I_imgs)):
            self.slices_T2I(tile_h, tile_h_len, tile_w, tile_w_len, m, img)
            
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