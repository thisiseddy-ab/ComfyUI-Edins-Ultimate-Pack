import torch

class CustomAspectRatio():
    ASPECT_RATIO = ["custom",
                        ## Square ##
                        "1:1 - Square",
                        ## Portrait ##
                        "1:1.43 - Portrait",
                        "1:1.618 - Portrait",
                        "1:1.91 - Portrait",
                        "1:2 - Portrait",
                        "1:2.2 - Portrait",
                        "1:2.4 - Portrait (a.k.a 5:12)",
                        "1:2.55 - Landscape (a.k.a 20:21)",
                        "1:3 - Portrait",
                        "1:3.2 - Landscape (a.k.a 5:16 a.k.a 10:32)", 
                        "1:3.56 - Landscape (a.k.a 9:32)",
                        "1:4 - Portrait",
                        "1:5 - Portrait",
                        "1:6 - Portrait",
                        "2:3 - Portrait",
                        "3:4 - Portrait",
                        "3:5 - Portrait",
                        "3:7 - Portrait",
                        "4:5 - Portrait",
                        "4:7 - Portrait",
                        "5:6 - Portrait",
                        "5:7 - Portrait",
                        "5:8 - Portrait",
                        "5:18 - Landscape (a.k.a 10:36 a.k.a 1:3.6)",
                        "6:7 - Portrait",
                        "6:13 - Portrait",
                        "7:8 - Portrait",
                        "7:9 - Portrait",
                        "7:10 - Portrait",
                        "7:16 - Portrait",
                        "8:11 - Portrait",
                        "8:15 - Portrait",
                        "8:19 - Portrait (a.k.a 40:59)",
                        "9:11 - Portrait",
                        "9:14 - Portrait",
                        "9:16 - Portrait",
                        "9:19 - Portrait",
                        "9:21 - Portrait (a.k.a 18:43)",
                        "11:14 - Portrait",
                        "14:17 - Portrait",
                        "16:19 - Portrait",
                        "16:21 - Portrait",
                        "16:25 - Portrait",
                        "17:22 - Portrait",
                        "23:33 - Portrait",
                        "20:37 - Portrait",
                        "27:64 - Portrait",
                        "44:105 - Landscape (a.k.a 429:1024)",
                        "135:256 - Portrait",
                        "147:190 - Portrait",
                        "715:1678 - portrait",
                        ## Landscape ##
                        "1.43:1 - Landscape",
                        "1.618:1 - Landscape",
                        "1.91:1 - Landscape",
                        "2:1 - Landscape",
                        "2.2:1 - Landscape",
                        "2.4:1 - Landscape (a.k.a 12:5)",
                        "2.55:1 - Landscape (a.k.a 51:20)", 
                        "3:1 - Landscape",
                        "3:2 - Landscape",
                        "3.2:1 - Landscape (a.ka 16:5 a.k.a 32:10)",
                        "3.56:1 - Landscape (a.k.a 32:9)",
                        "4:1 - Landscape",
                        "4:3 - Landscape",
                        "5:1 - Landscape",
                        "5:3 - Landscape",
                        "5:4 - Landscape",
                        "6:1 - Landscape",
                        "6:5 - Landscape",
                        "7:3 - Landscape",
                        "7:4 - Landscape",
                        "7:5 - Landscape",
                        "7:6 - Landscape",
                        "8:5 - Landscape",
                        "8:7 - Landscape",
                        "9:7 - Landscape",
                        "10:7 - Landscape",
                        "11:8 - Landscape",
                        "11:9 - Landscape",
                        "13:6 - Landscape",
                        "14:9 - Landscape",
                        "14:11 - Landscape",
                        "15:8 - Landscape",
                        "16:7 - Landscape",
                        "16:9 - Landscape",
                        "17:14 - Landscape",
                        "18:5 - Landscape (a.ka 36:10 a.k.a 3.6:1)",
                        "19:9 - Landscape",
                        "19:8 - Landscape (a.k.a 59:40)",
                        "19:16 - Landscape",
                        "21:9 - Landscape (a.k.a 43:18)",
                        "21:16 - Landscape",
                        "22:17 - Landscape",
                        "25:16 - Landscape",
                        "33:23 - Landscape",
                        "37:20 - Landscape",
                        "64:27 - Landscape",
                        "105:44 - Landscape (a.k.a 1024:429)",
                        "256:135 - Landscape",
                        "190:147 - Landscape",
                        "1678:715 - Landscape",
                        ]
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "biggest_size": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "aspect_ratio": (s.ASPECT_RATIO,),
                "upscale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step":0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
            }
        }
    RETURN_TYPES = ("INT", "INT", "STRING", "FLOAT", "INT", "LATENT")
    RETURN_NAMES = ("width", "height", "aspect_ratio", "upscale_factor", "batch_size", "empty_latent")
    FUNCTION = "Aspect_Ratio"
    CATEGORY = "EUP - Ultimate Pack/Essential"

    def Aspect_Ratio(self, biggest_size, aspect_ratio, upscale_factor, batch_size):
         # Extract numerical aspect ratio
        real_aspect_ratio = aspect_ratio.split(" - ")[0]
        aspect_ratio_width, aspect_ratio_height = map(int, real_aspect_ratio.split(":"))
        
        # Adjust the biggest size to ensure perfect aspect ratio fit
        if aspect_ratio_width > aspect_ratio_height:  # Landscape
            width = (biggest_size // aspect_ratio_width) * aspect_ratio_width
            height = (width * aspect_ratio_height) // aspect_ratio_width
        else:  # Portrait or Square
            height = (biggest_size // aspect_ratio_height) * aspect_ratio_height
            width = (height * aspect_ratio_width) // aspect_ratio_height

             
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
           
        return(width, height, real_aspect_ratio, upscale_factor, batch_size, {"samples":latent})
    
NODE_CLASS_MAPPINGS = {
    "EUP - Custom Aspect Ratio": CustomAspectRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}
