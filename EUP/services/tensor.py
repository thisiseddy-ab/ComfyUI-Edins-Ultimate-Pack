class TensorService(): 

    def getSlice(self, tensor, tileSize):
        """
        Extract a slice from the input tensor at the specified location and size.
        
        Args:
            tensor: The input tensor (e.g., a latent tensor).
            h: The starting height (y-coordinate) of the slice.
            h_len: The height length of the slice.
            w: The starting width (x-coordinate) of the slice.
            w_len: The width length of the slice.
        
        Returns:
            A slice of the input tensor.
        """
        x, y, tile_w, tile_h = tileSize
        # (B, C, H, W)
        return tensor[:, :, y:y + tile_h, x:x + tile_w]
    
    def getTensorfromLatentImage(self, latentImage):
        """
        Extract the tensor from the latent image.
        
        Args:
            latentImage: The input latent image.
        
        Returns:
            The tensor extracted from the latent image.
        """
        #### Tile Shape ####
        return latentImage.get("samples")
    
    def setTensorInLatentImage(latentImage, newSamples):
        """
        Replace the 'samples' tensor in the latent image dictionary.

        Args:
            latentImage (dict): The latent image dictionary.
            new_samples (torch.Tensor): The new tensor to set as 'samples'.

        Returns:
            dict: A modified copy of the latent image with updated 'samples'.
        """

        latentImage.update({"samples": newSamples})
        return latentImage
    
    def getShapefromLatentImage(self, latentImage):
        latentTensor = self.getTensorfromLatentImage(latentImage)
        return latentTensor.shape
    
    def getNoiseMaskfromLatentImage(self, latentImage):
        return latentImage.get("noise_mask")
    
    def getBatchIndexfromLatentTensor(self, latentTesor):
        return latentTesor.get("noise_mask")
