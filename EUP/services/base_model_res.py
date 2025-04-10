from typing import Dict, Optional, Any

BASE_MODEL_RES = {
    "SD 1.4 & SD 1.5": {
        "FP16": {
            "4GB": {
                "Practical Max - WxH": "512×512",
                "Practical Max - Average": 512,
                "Slow Max - WxH": "576×576",
                "Slow Max - Average": 576
            },
            "6GB": {
                "Practical Max - WxH": "768×768",
                "Practical Max - Average": 768,
                "Slow Max - WxH": "896×896",
                "Slow Max - Average": 896
            },
            "8GB": {
                "Practical Max - WxH": "1024×1024",
                "Practical Max - Average": 1024,
                "Slow Max - WxH": "1152×1152",
                "Slow Max - Average": 1152
            },
            "10GB": {
                "Practical Max - WxH": "1152×1152",
                "Practical Max - Average": 1152,
                "Slow Max - WxH": "1280×1280",
                "Slow Max - Average": 1280
            },
            "12GB": {
                "Practical Max - WxH": "1280×1280",
                "Practical Max - Average": 1280,
                "Slow Max - WxH": "1536×1536",
                "Slow Max - Average": 1536
            },
            "16GB": {
                "Practical Max - WxH": "1536×1536",
                "Practical Max - Average": 1536,
                "Slow Max - WxH": "1792×1792",
                "Slow Max - Average": 1792
            },
            "24GB": {
                "Practical Max - WxH": "1792×1792",
                "Practical Max - Average": 1792,
                "Slow Max - WxH": "2048×2048",
                "Slow Max - Average": 2048
            },
            "32GB": {
                "Practical Max - WxH": "2048×2048",
                "Practical Max - Average": 2048,
                "Slow Max - WxH": "2560×2560",
                "Slow Max - Average": 2560
            },
            "40GB": {
                "Practical Max - WxH": "2560×2560",
                "Practical Max - Average": 2560,
                "Slow Max - WxH": "3072×3072",
                "Slow Max - Average": 3072
            }
        },
        "FP32": {
            "4GB": {
                "Practical Max - WxH": "512×512",
                "Practical Max - Average": 512,
                "Slow Max - WxH": "576×576",
                "Slow Max - Average": 576
            },
            "6GB": {
                "Practical Max - WxH": "768×768",
                "Practical Max - Average": 768,
                "Slow Max - WxH": "896×896",
                "Slow Max - Average": 896
            },
            "8GB": {
                "Practical Max - WxH": "1024×1024",
                "Practical Max - Average": 1024,
                "Slow Max - WxH": "1152×1152",
                "Slow Max - Average": 1152
            },
            "10GB": {
                "Practical Max - WxH": "1152×1152",
                "Practical Max - Average": 1152,
                "Slow Max - WxH": "1280×1280",
                "Slow Max - Average": 1280
            },
            "12GB": {
                "Practical Max - WxH": "1280×1280",
                "Practical Max - Average": 1280,
                "Slow Max - WxH": "1536×1536",
                "Slow Max - Average": 1536
            },
            "16GB": {
                "Practical Max - WxH": "1536×1536",
                "Practical Max - Average": 1536,
                "Slow Max - WxH": "1792×1792",
                "Slow Max - Average": 1792
            },
            "24GB": {
                "Practical Max - WxH": "1792×1792",
                "Practical Max - Average": 1792,
                "Slow Max - WxH": "2048×2048",
                "Slow Max - Average": 2048
            },
            "32GB": {
                "Practical Max - WxH": "2048×2048",
                "Practical Max - Average": 2048,
                "Slow Max - WxH": "2560×2560",
                "Slow Max - Average": 2560
            },
            "40GB": {
                "Practical Max - WxH": "2560×2560",
                "Practical Max - Average": 2560,
                "Slow Max - WxH": "3072×3072",
                "Slow Max - Average": 3072
            },
        },
    },
    "SDXL 1.0" : {
        "FP16" : {
            "8GB" : {
                "Practical Max - WxH" : "1024×1024",
                "Practical Max - Average" : 1024,
                "Slow Max - WxH" : "1152×1152",
                "Slow Max - Average" : 1152,
            },
            "10GB": {
                "Practical Max - WxH": "1152×1152",
                "Practical Max - Average": 1152,
                "Slow Max - WxH": "1280×1280",
                "Slow Max - Average": 1280
            },
            "12GB" : {
                "Practical Max - WxH" : "1280×1280",
                "Practical Max - Average" : 1280,
                "Slow Max - WxH" : "1536×1536",
                "Slow Max - Average" : 1536,
            },
            "16GB" : {
                "Practical Max - WxH" : "1536×1536",
                "Practical Max - Average" : 1536,
                "Slow Max - WxH" : "1792×1792",
                "Slow Max - Average" : 1792,
            },
            "24GB" : {
                "Practical Max - WxH" : "1792×1792",
                "Practical Max - Average" : 1792,
                "Slow Max - WxH" : "2048×2048",
                "Slow Max - Average" : 2048,
            },
            "32GB" : {
                "Practical Max - WxH" : "2048×2048",
                "Practical Max - Average" : 2048,
                "Slow Max - WxH" : "2560×2560",
                "Slow Max - Average" : 2560,
            },
            "40GB" : {
                "Practical Max - WxH" : "2560×2560",
                "Practical Max - Average" : 2560,
                "Slow Max - WxH" : "3072×3072",
                "Slow Max - Average" : 3072,
            },
        },
        "FP32" : {
            "8GB" : {
                "Practical Max - WxH" : "1024×1024",
                "Practical Max - Average" : 1024,
                "Slow Max - WxH" : "1152×1152",
                "Slow Max - Average" : 1152,
            },
            "10GB": {
                "Practical Max - WxH": "1152×1152",
                "Practical Max - Average": 1152,
                "Slow Max - WxH": "1280×1280",
                "Slow Max - Average": 1280
            },
            "12GB" : {
                "Practical Max - WxH" : "1152×1152",
                "Practical Max - Average" : 1152,
                "Slow Max - WxH" : "1280×1280",
                "Slow Max - Average" : 1280,
            },
            "16GB" : {
                "Practical Max - WxH" : "1280×1280",
                "Practical Max - Average" : 1280,
                "Slow Max - WxH" : "1536×1536",
                "Slow Max - Average" : 1536,
            },
            "24GB" : {
                "Practical Max - WxH" : "1536×1536",
                "Practical Max - Average" : 1536,
                "Slow Max - WxH" : "1792×1792",
                "Slow Max - Average" : 1792,
            },
            "32GB" : {
                "Practical Max - WxH" : "1792×1792",
                "Practical Max - Average" : 1792,
                "Slow Max - WxH" : "2048×2048",
                "Slow Max - Average" : 2048,
            },
            "40GB" : {
                "Practical Max - WxH" : "2048×2048",
                "Practical Max - Average" : 2048,
                "Slow Max - WxH" : "2560×2560",
                "Slow Max - Average" : 2560,
            },
        },
    },
    "SD 2.0": {
        "FP16": {
            "8GB": {
                "Practical Max - WxH": "896×896",
                "Practical Max - Average": 896,
                "Slow Max - WxH": "1024×1024",
                "Slow Max - Average": 1024
            },
            "10GB": {
                "Practical Max - WxH": "1024×1024",
                "Practical Max - Average": 1024,
                "Slow Max - WxH": "1152×1152",
                "Slow Max - Average": 1152
            },
            "12GB": {
                "Practical Max - WxH": "1024×1024",
                "Practical Max - Average": 1024,
                "Slow Max - WxH": "1280×1280",
                "Slow Max - Average": 1280
            },
            "16GB": {
                "Practical Max - WxH": "1280×1280",
                "Practical Max - Average": 1280,
                "Slow Max - WxH": "1536×1536",
                "Slow Max - Average": 1536
            },
            "24GB": {
                "Practical Max - WxH": "1536×1536",
                "Practical Max - Average": 1536,
                "Slow Max - WxH": "1792×1792",
                "Slow Max - Average": 1792
            },
            "32GB": {
                "Practical Max - WxH": "1792×1792",
                "Practical Max - Average": 1792,
                "Slow Max - WxH": "2048×2048",
                "Slow Max - Average": 2048
            },
            "40GB": {
                "Practical Max - WxH": "2048×2048",
                "Practical Max - Average": 2048,
                "Slow Max - WxH": "2560×2560",
                "Slow Max - Average": 2560
            }
        },
        "FP32": {
            "8GB": {
                "Practical Max - WxH": "768×768",
                "Practical Max - Average": 768,
                "Slow Max - WxH": "896×896",
                "Slow Max - Average": 896
            },
            "10GB": {
                "Practical Max - WxH": "896×896",
                "Practical Max - Average": 896,
                "Slow Max - WxH": "1024×1024",
                "Slow Max - Average": 1024
            },
            "12GB": {
                "Practical Max - WxH": "896×896",
                "Practical Max - Average": 896,
                "Slow Max - WxH": "1024×1024",
                "Slow Max - Average": 1024
            },
            "16GB": {
                "Practical Max - WxH": "1024×1024",
                "Practical Max - Average": 1024,
                "Slow Max - WxH": "1280×1280",
                "Slow Max - Average": 1280
            },
            "24GB": {
                "Practical Max - WxH": "1280×1280",
                "Practical Max - Average": 1280,
                "Slow Max - WxH": "1536×1536",
                "Slow Max - Average": 1536
            },
            "32GB": {
                "Practical Max - WxH": "1536×1536",
                "Practical Max - Average": 1536,
                "Slow Max - WxH": "1792×1792",
                "Slow Max - Average": 1792
            },
            "40GB": {
                "Practical Max - WxH": "1792×1792",
                "Practical Max - Average": 1792,
                "Slow Max - WxH": "2048×2048",
                "Slow Max - Average": 2048
            },
        },
    },
    "SD 2.1": {
        "FP16": {
            "8GB": {
                "Practical Max - WxH": "1024×1024",
                "Practical Max - Average": 1024,
                "Slow Max - WxH": "1152×1152",
                "Slow Max - Average": 1152
            },
            "10GB": {
                "Practical Max - WxH": "1152×1152",
                "Practical Max - Average": 1152,
                "Slow Max - WxH": "1280×1280",
                "Slow Max - Average": 1280
            },
            "12GB": {
                "Practical Max - WxH": "1280×1280",
                "Practical Max - Average": 1280,
                "Slow Max - WxH": "1536×1536",
                "Slow Max - Average": 1536
            },
            "16GB": {
                "Practical Max - WxH": "1536×1536",
                "Practical Max - Average": 1536,
                "Slow Max - WxH": "1792×1792",
                "Slow Max - Average": 1792
            },
            "24GB": {
                "Practical Max - WxH": "1792×1792",
                "Practical Max - Average": 1792,
                "Slow Max - WxH": "2048×2048",
                "Slow Max - Average": 2048
            },
            "32GB": {
                "Practical Max - WxH": "2048×2048",
                "Practical Max - Average": 2048,
                "Slow Max - WxH": "2304×2304",
                "Slow Max - Average": 2304
            },
            "40GB": {
                "Practical Max - WxH": "2304×2304",
                "Practical Max - Average": 2304,
                "Slow Max - WxH": "2560×2560",
                "Slow Max - Average": 2560
            }
        },
        "FP32": {
            "8GB": {
                "Practical Max - WxH": "896×896",
                "Practical Max - Average": 896,
                "Slow Max - WxH": "1024×1024",
                "Slow Max - Average": 1024
            },
            "10GB": {
                "Practical Max - WxH": "1024×1024",
                "Practical Max - Average": 1024,
                "Slow Max - WxH": "1152×1152",
                "Slow Max - Average": 1152
            },
            "12GB": {
                "Practical Max - WxH": "1152×1152",
                "Practical Max - Average": 1152,
                "Slow Max - WxH": "1280×1280",
                "Slow Max - Average": 1280
            },
            "16GB": {
                "Practical Max - WxH": "1280×1280",
                "Practical Max - Average": 1280,
                "Slow Max - WxH": "1536×1536",
                "Slow Max - Average": 1536
            },
            "24GB": {
                "Practical Max - WxH": "1536×1536",
                "Practical Max - Average": 1536,
                "Slow Max - WxH": "1792×1792",
                "Slow Max - Average": 1792
            },
            "32GB": {
                "Practical Max - WxH": "1792×1792",
                "Practical Max - Average": 1792,
                "Slow Max - WxH": "2048×2048",
                "Slow Max - Average": 2048
            },
            "40GB": {
                "Practical Max - WxH": "2048×2048",
                "Practical Max - Average": 2048,
                "Slow Max - WxH": "2304×2304",
                "Slow Max - Average": 2304
            },
        },
    },
    "SD 3.0" : {
        "FP16" : {
            "8GB" : {
                "Practical Max - WxH" : "640×640",
                "Practical Max - Average" : 640,
                "Slow Max - WxH" : "768×768",
                "Slow Max - Average" : 768
            },
            "10GB" : {
                "Practical Max - WxH" : "768×768",
                "Practical Max - Average" : 768,
                "Slow Max - WxH" : "896×896",
                "Slow Max - Average" : 896
            },
            "12GB" : {
                "Practical Max - WxH" : "768×768",
                "Practical Max - Average" : 768,
                "Slow Max - WxH" : "896×896",
                "Slow Max - Average" : 896
            },
            "16GB" : {
                "Practical Max - WxH" : "896×896",
                "Practical Max - Average" : 896,
                "Slow Max - WxH" : "1024×1024",
                "Slow Max - Average" : 1024
            },
            "24GB" : {
                "Practical Max - WxH" : "1024×1024",
                "Practical Max - Average" : 1024,
                "Slow Max - WxH" : "1280×1280",
                "Slow Max - Average" : 1280
            },
            "32GB" : {
                "Practical Max - WxH" : "1280×1280",
                "Practical Max - Average" : 1280,
                "Slow Max - WxH" : "1536×1536",
                "Slow Max - Average" : 1536
            },
            "40GB" : {
                "Practical Max - WxH" : "1536×1536",
                "Practical Max - Average" : 1536,
                "Slow Max - WxH" : "1792×1792",
                "Slow Max - Average" : 1792
            }
        },
        "FP32" : {
            "8GB" : {
                "Practical Max - WxH" : "512×512",
                "Practical Max - Average" : 512,
                "Slow Max - WxH" : "576×576",
                "Slow Max - Average" : 576
            },
            "10GB" : {
                "Practical Max - WxH" : "640×640",
                "Practical Max - Average" : 640,
                "Slow Max - WxH" : "768×768",
                "Slow Max - Average" : 768
            },
            "12GB" : {
                "Practical Max - WxH" : "640×640",
                "Practical Max - Average" : 640,
                "Slow Max - WxH" : "768×768",
                "Slow Max - Average" : 768
            },
            "16GB" : {
                "Practical Max - WxH" : "768×768",
                "Practical Max - Average" : 768,
                "Slow Max - WxH" : "896×896",
                "Slow Max - Average" : 896
            },
            "24GB" : {
                "Practical Max - WxH" : "896×896",
                "Practical Max - Average" : 896,
                "Slow Max - WxH" : "1024×1024",
                "Slow Max - Average" : 1024
            },
            "32GB" : {
                "Practical Max - WxH" : "1024×1024",
                "Practical Max - Average" : 1024,
                "Slow Max - WxH" : "1280×1280",
                "Slow Max - Average" : 1280
            },
            "40GB" : {
                "Practical Max - WxH" : "1280×1280",
                "Practical Max - Average" : 1280,
                "Slow Max - WxH" : "1536×1536",
                "Slow Max - Average" : 1536
            },
        },
    },
    "SD 3.5" : {
        "FP16" : {
            "8GB" : {
                "Practical Max - WxH" : "704×704",
                "Practical Max - Average" : 704,
                "Slow Max - WxH" : "768×768",
                "Slow Max - Average" : 768,
            },
            "10GB" : {
                "Practical Max - WxH" : "768×768",
                "Practical Max - Average" : 768,
                "Slow Max - WxH" : "896×896",
                "Slow Max - Average" : 896,
            },
            "12GB" : {
                "Practical Max - WxH" : "896×896",
                "Practical Max - Average" : 896,
                "Slow Max - WxH" : "1024×1024",
                "Slow Max - Average" : 1024,
            },
            "16GB" : {
                "Practical Max - WxH" : "1024×1024",
                "Practical Max - Average" : 1024,
                "Slow Max - WxH" : "1280×1280",
                "Slow Max - Average" : 1280,
            },
            "24GB" : {
                "Practical Max - WxH" : "1280×1280",
                "Practical Max - Average" : 1280,
                "Slow Max - WxH" : "1536×1536",
                "Slow Max - Average" : 1536,
            },
            "32GB" : {
                "Practical Max - WxH" : "1536×1536",
                "Practical Max - Average" : 1536,
                "Slow Max - WxH" : "1792×1792",
                "Slow Max - Average" : 1792,
            },
            "40GB" : {
                "Practical Max - WxH" : "1792×1792",
                "Practical Max - Average" : 1792,
                "Slow Max - WxH" : "2048×2048",
                "Slow Max - Average" : 2048,
            },
        },
        "FP32" : {
            "8GB" : {
                "Practical Max - WxH" : "576×576",
                "Practical Max - Average" : 576,
                "Slow Max - WxH" : "640×640",
                "Slow Max - Average" : 640,
            },
            "10GB" : {
                "Practical Max - WxH" : "640×640",
                "Practical Max - Average" : 640,
                "Slow Max - WxH" : "704×704",
                "Slow Max - Average" : 704,
            },
            "12GB" : {
                "Practical Max - WxH" : "704×704",
                "Practical Max - Average" : 704,
                "Slow Max - WxH" : "768×768",
                "Slow Max - Average" : 768,
            },
            "16GB" : {
                "Practical Max - WxH" : "768×768",
                "Practical Max - Average" : 768,
                "Slow Max - WxH" : "896×896",
                "Slow Max - Average" : 896,
            },
            "24GB" : {
                "Practical Max - WxH" : "896×896",
                "Practical Max - Average" : 896,
                "Slow Max - WxH" : "1024×1024",
                "Slow Max - Average" : 1024,
            },
            "32GB" : {
                "Practical Max - WxH" : "1024×1024",
                "Practical Max - Average" : 1024,
                "Slow Max - WxH" : "1280×1280",
                "Slow Max - Average" : 1280,
            },
            "40GB" : {
                "Practical Max - WxH" : "1280×1280",
                "Practical Max - Average" : 1280,
                "Slow Max - WxH" : "1536×1536",
                "Slow Max - Average" : 1536,
            },
        },
    },
    "SD 3.5 Large Turbo" : {
        "FP16" : {
            "10GB" : {
                "Practical Max - WxH" : "640×640",
                "Practical Max - Average" : 640,
                "Slow Max - WxH" : "768×768",
                "Slow Max - Average" : 768,
            },
            "12GB" : {
                "Practical Max - WxH" : "704×704",
                "Practical Max - Average" : 704,
                "Slow Max - WxH" : "768×768",
                "Slow Max - Average" : 768,
            },
            "16GB" : {
                "Practical Max - WxH" : "768×768",
                "Practical Max - Average" : 768,
                "Slow Max - WxH" : "896×896",
                "Slow Max - Average" : 896,
            },
            "24GB" : {
                "Practical Max - WxH" : "1024×1024",
                "Practical Max - Average" : 1024,
                "Slow Max - WxH" : "1280×1280",
                "Slow Max - Average" : 1280,
            },
            "32GB" : {
                "Practical Max - WxH" : "1280×1280",
                "Practical Max - Average" : 1280,
                "Slow Max - WxH" : "1536×1536",
                "Slow Max - Average" : 1536,
            },
            "40GB" : {
                "Practical Max - WxH" : "1536×1536",
                "Practical Max - Average" : 1536,
                "Slow Max - WxH" : "1792×1792",
                "Slow Max - Average" : 1792,
            },
        },
        "FP32" : {
            "10GB" : {
                "Practical Max - WxH" : "576×576",
                "Practical Max - Average" : 576,
                "Slow Max - WxH" : "640×640",
                "Slow Max - Average" : 640,
            },
            "12GB" : {
                "Practical Max - WxH" : "640×640",
                "Practical Max - Average" : 640,
                "Slow Max - WxH" : "704×704",
                "Slow Max - Average" : 704,
            },
            "16GB" : {
                "Practical Max - WxH" : "704×704",
                "Practical Max - Average" : 704,
                "Slow Max - WxH" : "768×768",
                "Slow Max - Average" : 768,
            },
            "24GB" : {
                "Practical Max - WxH" : "768×768",
                "Practical Max - Average" : 768,
                "Slow Max - WxH" : "896×896",
                "Slow Max - Average" : 896,
            },
            "32GB" : {
                "Practical Max - WxH" : "896×896",
                "Practical Max - Average" : 896,
                "Slow Max - WxH" : "1024×1024",
                "Slow Max - Average" : 1024,
            },
            "40GB" : {
                "Practical Max - WxH" : "1024×1024",
                "Practical Max - Average" : 1024,
                "Slow Max - WxH" : "1280×1280",
                "Slow Max - Average" : 1280,
            },
        },
    },
}

class BaseModelResolutionService:
    def getModel(self, model: str) -> Dict[str, Any]:
        model_data = BASE_MODEL_RES.get(model)
        if model_data is None:
            raise ValueError(f"Model '{model}' not found in base model resolutions")
        return model_data

    def getPrecision(self, model: str, precision: str) -> Dict[str, Any]:
        model_data = self.getModel(model)
        precision_data = model_data.get(precision)
        if precision_data is None:
            raise ValueError(f"Precision '{precision}' not found for model '{model}'")
        return precision_data

    def getVRAM(self, model: str, precision: str, vram: str) -> Dict[str, Any]:
        precision_data = self.getPrecision(model, precision)
        vram_option = precision_data.get(vram)
        if vram_option is None:
            raise ValueError(f"VRAM option '{vram}' not found for model '{model}' and precision '{precision}'")
        return vram_option

    def getPracticalMaxWxH(self, model: str, precision: str, vram: str) -> Optional[str]:
        return self.getVRAM(model, precision, vram).get("Practical Max - WxH")

    def getPracticalMaxAverage(self, model: str, precision: str, vram: str) -> Optional[int]:
        return self.getVRAM(model, precision, vram).get("Practical Max - Average")

    def getSlowMaxWxH(self, model: str, precision: str, vram: str) -> Optional[str]:
        return self.getVRAM(model, precision, vram).get("Slow Max - WxH")

    def getSlowMaxAverage(self, model: str, precision: str, vram: str) -> Optional[int]:
        return self.getVRAM(model, precision, vram).get("Slow Max - Average")