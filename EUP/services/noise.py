#### Comfy Lib's #####
import comfy.sample
import comfy.sampler_helpers

#### Third Party Lib's #####
import torch

#### Services #####
from EUP.services.tensor import TensorService

class NoiseService():

    def __init__(self):
        self.tensorService = TensorService()
    
    # rdm - Random Noise Generation Strategy
    def generateNoiseforRDMStrategy(self, latentImage, seed, disable_noise: bool, device='cpu'):
        
        latentTensor = self.tensorService.getTensorfromLatentImage(latentImage)
        B, C, H, W = self.tensorService.getShapefromLatentImage(latentImage)
        
        if disable_noise:
            return torch.zeros((B, C, H, W), dtype=latentTensor.dtype, layout=latentTensor.layout, device=device)
        else:
            batchIndex = self.tensorService.getBatchIndexfromLatentTensor(latentTensor)
            return comfy.sample.prepare_noise(latentTensor, seed, batchIndex)

    # sda - Standart Noise Mask Generation Strategy
    def generateNoiseMaskforSDAStrategy(self, latentImage, noiseTensor, device='cpu'):
        noise_mask = self.tensorService.getNoiseMaskfromLatentImage(latentImage)
        if noise_mask is not None:
            noise_mask = comfy.sampler_helpers.prepare_mask(noise_mask=noise_mask, shape=noiseTensor.shape, device=device)
        return noise_mask