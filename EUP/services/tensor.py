class TensorService(): 

    def getSlice(self, tensor, h, h_len, w, w_len):
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
        return tensor[:, :, h:h + h_len, w:w + w_len]