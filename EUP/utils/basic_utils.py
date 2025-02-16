
def getLatentSize(image_size : int) -> int:
    factor = 8  # Fixed factor
    
    if image_size % factor != 0:
        adjusted_size = (image_size // factor) * factor  # Round down to nearest multiple of 8
        print(f"Warning: Adjusting {image_size} to {adjusted_size} to be divisible by {factor}")
        return max(1, adjusted_size // factor)
    
    return max(1, image_size // factor)


