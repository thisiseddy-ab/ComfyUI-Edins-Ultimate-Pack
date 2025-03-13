

class UnitsService():

    def getLatentSize(self, pixel_size : int) -> int:
        factor = 8  # Fixed factor
        
        if pixel_size % factor != 0:
            adjusted_size = (pixel_size // factor) * factor  # Round down to nearest multiple of 8
            print(f"Warning: Adjusting {pixel_size} to {adjusted_size} to be divisible by {factor}")
            return max(1, adjusted_size // factor)
        
        return max(1, pixel_size // factor)


