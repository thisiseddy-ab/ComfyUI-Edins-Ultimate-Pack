#### Comfy Lib's ####
import comfy.model_management

#### Third Party Lib's #####
import torch

#### My Services's #####
from EUP.services.units import UnitsService
from EUP.services.tensor import TensorService
from EUP.services.tiling import TilingService
from EUP.services.cnet import ControlNetService
from EUP.services.t2i import T2IService
from EUP.services.gligen import GligenService
from EUP.services.condition import ConditionService

class LatentTilerService():

    def __init__(self):
        self.tensorService = TensorService()
        self.untisService = UnitsService()
        self.tilingService = TilingService()
        self.conditionService = ConditionService()
        self.conrolNetService = ControlNetService()
        self.t2iService = T2IService()
        self.gligenService = GligenService()

    def getPaddLatentSize(self, image_size: int) -> int:
        if image_size > 0:
            return self.untisService.getLatentSize(image_size)
        return 0
    
    def convertPXtoLatentSize(self, tile_height, tile_width, padding) -> int:   
        # Converts Size From Pixel to Latent
        tile_height = self.untisService.getLatentSize(tile_height)
        tile_width = self.untisService.getLatentSize(tile_width)
        padding = self.getPaddLatentSize(padding)

        return (tile_height, tile_width, padding)
    
    def tileLatent(self, model, sampler, seed, actual_steps, latent_image, positive, negative, sampler_advanced_pars, disable_noise, start_step):
        
        ksampler_type, tiling_strategy, tiling_strategy_pars, noise_strategy, noise_strategy_pars, noise_mask_strategy, noise_mask_strategy_pars, denoise_strategy, denoise_strategy_pars, blend_strategy, blend_strategy_pars = sampler_advanced_pars

        samples = self.tensorService.getTensorfromLatentImage(latent_image)
        print(f"[EUP - Latent Tiler]: Original Latent Size: {samples.shape}")

        noise = self.tilingService.choseNoiseStrategy(latent_image, seed, disable_noise, noise_strategy, noise_strategy_pars)
        noise_mask = self.tilingService.choseNoiseMaskStrategy(latent_image, noise, noise_mask_strategy, noise_mask_strategy_pars)

        if ksampler_type == "BlenderNeko's":
            if noise_mask is not None:
                samples += sampler.sigmas[start_step].cpu() * noise_mask * model.model.process_latent_out(noise)
            else:
                samples += sampler.sigmas[start_step].cpu() * model.model.process_latent_out(noise)
            self.tensorService.setTensorInLatentImage(latent_image, samples)

        #### Chosing Tiling Strategy ####
        if tiling_strategy == "simple":
            tile_width, tile_height, tiling_mode, passes = tiling_strategy_pars
            tile_height, tile_width, _ = self.convertPXtoLatentSize(tile_height, tile_width, 0)
            if tiling_mode == "single-pass":
                tiled_positions = self.tilingService.st_Service.generatePosforSTStrategy(latent_image, (tile_width, tile_height))
                tiled_tiles = self.tilingService.st_Service.getTilesforSTStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.st_Service.getNoiseTilesforSTStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.st_Service.getDenoiseMaskTilesforSTStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.st_Service.getBlendMaskTilesforSTStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.st_Service.getStepsforSTStrategy(actual_steps, tiled_positions)
            elif tiling_mode == "multi-pass":
                tiled_positions = self.tilingService.mp_st_Service.generatePosforMP_STStrategy(latent_image, (tile_width, tile_height), passes)
                tiled_tiles = self.tilingService.mp_st_Service.getTilesforMP_STStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_st_Service.getNoiseTilesforMP_STStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_st_Service.getDenoiseMaskTilesforMP_STStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.mp_st_Service.getBlendMaskTilesforMP_STStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.mp_st_Service.getStepsforMP_STStrategy(actual_steps, tiled_positions, passes)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode for Simple Tiling. Please select a valid Tiling Mode.")
        elif tiling_strategy == "random":
            tile_width, tile_height, tiling_mode, passes = tiling_strategy_pars
            tile_height, tile_width, _ = self.convertPXtoLatentSize(tile_height, tile_width, 0)
            if tiling_mode == "single-pass":
                tiled_positions = self.tilingService.rt_Service.generatePosforRTStrategy(latent_image, (tile_width, tile_height), seed)
                tiled_tiles = self.tilingService.rt_Service.getTilesforRTStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.rt_Service.getNoiseTilesforRTStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.rt_Service.getDenoiseMaskTilesforRTStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.rt_Service.getBlendMaskTilesforRTStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.rt_Service.getStepsforRTStrategy(actual_steps, tiled_positions)
            elif tiling_mode == "multi-pass":
                tiled_positions = self.tilingService.mp_rt_Service.generatePos_forMP_RTStrategy(latent_image, (tile_width, tile_height), seed, passes)
                tiled_tiles = self.tilingService.mp_rt_Service.getTilesforMP_RTStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_rt_Service.getNoiseTilesforMP_RTStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_rt_Service.getDenoiseMaskTilesforMP_RTStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.mp_rt_Service.getBlendMaskTilesforMP_RTStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.mp_rt_Service.getStepsforMP_RTStrategy(actual_steps, tiled_positions, passes)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode for Random Tiling. Please select a valid Tiling Mode.")
        elif tiling_strategy == "padded":
            tile_width, tile_height, tiling_mode, passes, padding_strategy, padding = tiling_strategy_pars
            tile_height, tile_width, padding = self.convertPXtoLatentSize(tile_height, tile_width, padding)
            if tiling_mode == "single-pass":
                tiled_positions = self.tilingService.pt_Service.generatePosforPTStrategy(latent_image, (tile_width, tile_height), padding)
                tiled_tiles = self.tilingService.pt_Service.getTilesforPTStrategy(latent_image, tiled_positions)
                tiled_tiles = self.tilingService.pt_Service.getPaddedTilesforPTStrategy(tiled_tiles, tiled_positions, padding_strategy, padding)
                tiled_noise_tiles = self.tilingService.pt_Service.getNoiseTilesforPTStrategy(latent_image, tiled_tiles, seed, disable_noise)
                tiled_denoise_masks = self.tilingService.pt_Service.getDenoiseMaskTilesforPTStrategy(noise_mask, tiled_positions, tiled_tiles, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.pt_Service.getBlendMaskTilesforPTStrategy(latent_image, tiled_tiles, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.pt_Service.getStepsforPTStrategy(actual_steps, tiled_positions)
            elif tiling_mode == "multi-pass":
                tiled_positions = self.tilingService.mp_pt_Service.generatePos_forMP_PTStrategy(latent_image, (tile_width, tile_height), padding, passes)
                tiled_tiles = self.tilingService.mp_pt_Service.getTilesforMP_PTStrategy(latent_image, tiled_positions)
                tiled_tiles = self.tilingService.mp_pt_Service.getPaddedTilesforMP_PTStrategy(tiled_tiles, tiled_positions, padding_strategy, padding)
                tiled_noise_tiles = self.tilingService.mp_pt_Service.getNoiseTilesforMP_PTStrategy(latent_image, tiled_tiles, seed, disable_noise)
                tiled_denoise_masks = self.tilingService.mp_pt_Service.getDenoiseMaskTilesforMP_PTStrategy(noise_mask, tiled_positions, tiled_tiles, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.mp_pt_Service.getBlendMaskTilesforMP_PTStrategy(latent_image, tiled_tiles, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.mp_pt_Service.getStepsforMP_PTStrategy(actual_steps, tiled_positions, passes)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode for Padded Tiling. Please select a valid Tiling Mode.")
        elif tiling_strategy == "adjacency-padded":
            tile_width, tile_height, tiling_mode, passes, padding_strategy, padding = tiling_strategy_pars
            tile_height, tile_width, padding = self.convertPXtoLatentSize(tile_height, tile_width, padding)
            if tiling_mode == "single-pass":
                tiled_positions = self.tilingService.apt_Service.generatePosforAPTStrategy(latent_image, (tile_width, tile_height), padding)
                tiled_tiles = self.tilingService.apt_Service.getTilesforAPTStrategy(latent_image, tiled_positions)
                tiled_tiles = self.tilingService.apt_Service.getPaddedTilesforAPTStrategy(tiled_tiles, tiled_positions, padding_strategy, padding)
                tiled_noise_tiles = self.tilingService.apt_Service.getNoiseTilesforAPTStrategy(latent_image, tiled_tiles, seed, disable_noise)
                tiled_denoise_masks = self.tilingService.apt_Service.getDenoiseMaskTilesforAPTStrategy(noise_mask, tiled_positions, tiled_tiles, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.apt_Service.getBlendMaskTilesforAPTStrategy(latent_image, tiled_tiles, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.apt_Service.getStepsforAPTStrategy(actual_steps, tiled_positions)
            elif tiling_mode == "multi-pass":
                tiled_positions = self.tilingService.mp_apt_Service.generatePos_forMP_APTStrategy(latent_image, (tile_width, tile_height), padding, passes)
                tiled_tiles = self.tilingService.mp_apt_Service.getTilesforMP_APTStrategy(latent_image, tiled_positions)
                tiled_tiles = self.tilingService.mp_apt_Service.getPaddedTilesforMP_APTStrategy(tiled_tiles, tiled_positions, padding_strategy, padding)
                tiled_noise_tiles = self.tilingService.mp_apt_Service.getNoiseTilesforMP_APTStrategy(latent_image, tiled_tiles, seed, disable_noise)
                tiled_denoise_masks = self.tilingService.mp_apt_Service.getDenoiseMaskTilesforMP_APTStrategy(noise_mask, tiled_positions, tiled_tiles, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.mp_apt_Service.getBlendMaskTilesforMP_APTStrategy(latent_image, tiled_tiles, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.mp_apt_Service.getStepsforMP_APTStrategy(actual_steps, tiled_positions, passes)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode for Adjacency Padded Tiling. Please select a valid Tiling Mode.")
        elif tiling_strategy == "context-padded":
            tile_width, tile_height, tiling_mode, passes = tiling_strategy_pars
            tile_height, tile_width, _ = self.convertPXtoLatentSize(tile_height, tile_width, 0)
            if tiling_mode == "single-pass":
                tiled_positions = self.tilingService.cpt_Service.generatePosforCPTStrategy(latent_image, (tile_width, tile_height))
                tiled_tiles = self.tilingService.cpt_Service.getTilesforCPTStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.cpt_Service.getNoiseTilesforCPTStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.cpt_Service.getDenoiseMaskTilesforCPTStrategy(latent_image, noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.cpt_Service.getBlendeMaskTilesforCPTStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.cpt_Service.getStepsforCPTStrategy(actual_steps, tiled_positions)
            elif tiling_mode == "multi-pass":
                tiled_positions = self.tilingService.mp_cpt_Service.generatePosforMP_CPTStrategy(latent_image, (tile_width, tile_height))
                tiled_tiles = self.tilingService.mp_cpt_Service.getTilesforMP_CPTStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_cpt_Service.getNoiseTilesforMP_CPTStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_cpt_Service.getDenoiseMaskTilesforMP_CPTStrategy(latent_image, noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.mp_cpt_Service.getBlendeMaskTilesforMP_CPTStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.mp_cpt_Service.getStepsforMP_CPTStrategy(actual_steps, tiled_positions, passes)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode for Contextual Padded Tiling. Please select a valid Tiling Mode.")
        elif tiling_strategy == "overlaping":
            tile_width, tile_height, tiling_mode, passes, overalp = tiling_strategy_pars
            tile_height, tile_width, overalp = self.convertPXtoLatentSize(tile_height, tile_width, overalp)
            if tiling_mode == "single-pass":
                tiled_positions = self.tilingService.ovp_Service.generatePosforOVPStrategy(latent_image, (tile_width, tile_height))
                tiled_tiles = self.tilingService.ovp_Service.getTilesforOVPStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.ovp_Service.getNoiseTilesforOVPStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.ovp_Service.getDenoiseMaskTilesforOVPStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.ovp_Service.getBlendeMaskTilesforOVPStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.ovp_Service.getStepsforOVPStrategy(actual_steps, tiled_positions)
            elif tiling_mode == "multi-pass":
                tiled_positions = self.tilingService.mp_ovp_Service.generatePosforMP_OVPStrategy(latent_image, (tile_width, tile_height), overalp, passes)
                tiled_tiles = self.tilingService.mp_ovp_Service.getTilesforMP_OVPStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_ovp_Service.getNoiseTilesforMP_OVPStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_ovp_Service.getDenoiseMaskTilesforMP_OVPStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.mp_ovp_Service.getBlendeMaskTilesforMP_OVPStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.mp_ovp_Service.getStepsforMP_OVPStrategy(actual_steps, tiled_positions, passes)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode for Overlapping Tiling. Please select a valid Tiling Mode.")
        elif tiling_strategy == "adaptive":
            tile_width, tile_height, tiling_mode, passes, base_model, VRAM, precision, tile_growth_exponent = tiling_strategy_pars
            tile_height, tile_width, _ = self.convertPXtoLatentSize(tile_height, tile_width, 0)
            if tiling_mode == "single-pass":
                tiled_positions = self.tilingService.adp_Service.generatePosforADPStrategy(latent_image, (tile_width, tile_height), base_model, VRAM, precision, tile_growth_exponent)
                tiled_tiles = self.tilingService.adp_Service.getTilesforADPStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.adp_Service.getNoiseTilesforADPStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.adp_Service.getDenoiseMaskTilesforADPStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.adp_Service.getBlendeMaskTilesforADPStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.adp_Service.getStepsforADPStrategy(actual_steps, tiled_positions)
            elif tiling_mode == "multi-pass":
                tiled_positions = self.tilingService.mp_adp_Service.generatePos_forMP_ADPStrategy(latent_image, (tile_width, tile_height), passes, base_model, VRAM, precision, tile_growth_exponent)
                tiled_tiles = self.tilingService.mp_adp_Service.getTilesforMP_ADPStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_adp_Service.getNoiseTilesforMP_ADPStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_adp_Service.getDenoiseMaskTilesforMP_ADPStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.mp_adp_Service.getBlendeMaskTilesforMP_ADPStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.mp_adp_Service.getStepsforMP_ADPStrategy(actual_steps, tiled_positions, passes)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode for Adaptive Tiling. Please select a valid Tiling Mode.")
        elif tiling_strategy == "hierarchical":
            tiling_mode, passes, base_model, VRAM, precision, tile_growth_exponent = tiling_strategy_pars
            if tiling_mode == "single-pass":
                tiled_positions = self.tilingService.hrc_Service.generatePosforHRCStrategy(latent_image, base_model, VRAM, precision, tile_growth_exponent)
                tiled_tiles = self.tilingService.hrc_Service.getTilesforHRCStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.hrc_Service.getNoiseTilesforHRCStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.hrc_Service.getDenoiseMaskTilesforHRCStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.hrc_Service.getBlendMaskTilesforHRCStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.hrc_Service.getStepsforHRCStrategy(actual_steps, tiled_positions)
            elif tiling_mode == "multi-pass":
                tiled_positions = self.tilingService.mp_hrc_Service.generatePos_forMP_HRCStrategy(latent_image, passes, base_model, VRAM, precision, tile_growth_exponent)
                tiled_tiles = self.tilingService.mp_hrc_Service.getTilesforMP_HRCStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_hrc_Service.getNoiseTilesforMP_HRCStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_hrc_Service.getDenoiseMaskTilesforMP_HRCStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.mp_hrc_Service.getBlendMaskTilesforMP_HRCStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.mp_hrc_Service.getStepsforMP_HRCStrategy(actual_steps, tiled_positions, passes)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode for Hierarchical Tiling. Please select a valid Tiling Mode.")
        elif tiling_strategy == "random-hierarchical":
            tiling_mode, passes, base_model, VRAM, precision, tile_growth_exponent = tiling_strategy_pars
            if tiling_mode == "single-pass":
                tiled_positions = self.tilingService.rthrc_Service.generatePosforRTHRCStrategy(latent_image, seed, base_model, VRAM, precision, tile_growth_exponent)
                tiled_tiles = self.tilingService.rthrc_Service.getTilesforRTHRCStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.rthrc_Service.getNoiseTilesforRTHRCStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.rthrc_Service.getDenoiseMaskTilesforRTHRCStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.rthrc_Service.getBlendMaskTilesforRTHRCStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.rthrc_Service.getStepsforHRCStrategy(actual_steps, tiled_positions)
            elif tiling_mode == "multi-pass":
                tiled_positions = self.tilingService.mp_rthrc_Service.generatePos_forMP_RTHRCStrategy(latent_image, seed, passes, base_model, VRAM, precision, tile_growth_exponent)
                tiled_tiles = self.tilingService.mp_rthrc_Service.getTilesforMP_RTHRCStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_rthrc_Service.getNoiseTilesforMP_RTHRCStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_rthrc_Service.getDenoiseMaskTilesforMP_RTHRCStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.mp_rthrc_Service.getBlendMaskTilesforMP_RTHRCStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.mp_rthrc_Service.getStepsforMP_RTHRCStrategy(actual_steps, tiled_positions, passes)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode for Random-Hierarchical Tiling. Please select a valid Tiling Mode.")
        elif tiling_strategy == "non-uniform":
            tiling_mode, passes, base_model, VRAM, precision, tile_growth_exponent, scale_min, scale_max = tiling_strategy_pars
            if tiling_mode == "single-pass":
                tiled_positions = self.tilingService.nu_Service.generatePosforNUStrategy(latent_image, base_model, VRAM, precision, tile_growth_exponent, scale_min, scale_max)
                tiled_tiles = self.tilingService.nu_Service.getTilesforNUStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.nu_Service.getNoiseTilesforNUStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.nu_Service.getDenoiseMaskTilesforNUStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.nu_Service.getBlendeMaskTilesforNUStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.nu_Service.getStepsforNUStrategy(actual_steps, tiled_positions)
            elif tiling_mode == "multi-pass":
                tiled_positions = self.tilingService.mp_nu_Service.generatePosforMP_NUStrategy(latent_image, passes, base_model, VRAM, precision, tile_growth_exponent, scale_min, scale_max)
                tiled_tiles = self.tilingService.mp_nu_Service.getTilesforMP_NUStrategy(latent_image, tiled_positions)
                tiled_noise_tiles = self.tilingService.mp_nu_Service.getNoiseTilesforMP_NUStrategy(noise, tiled_positions)
                tiled_denoise_masks = self.tilingService.mp_nu_Service.getDenoiseMaskTilesforMP_NUStrategy(noise_mask, tiled_positions, denoise_strategy, denoise_strategy_pars)
                tiled_blend_masks = self.tilingService.mp_nu_Service.getBlendeMaskTilesforMP_NUStrategy(latent_image, tiled_positions, blend_strategy, blend_strategy_pars)
                tiled_steps = self.tilingService.mp_nu_Service.getStepsforMP_NUStrategy(actual_steps, tiled_positions, passes)
            else:
                raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode for Non-Uniform Tiling. Please select a valid Tiling Mode.")
        else:
            raise ValueError("[EUP - Latent Tiler]: Warning: Invalid Tiling Mode. Please select a valid Tiling Mode.")

        #### Prepare the tiles for sampling ####
        return self.prepareTilesforSampling(positive, negative, latent_image, tiled_positions, tiled_tiles, tiled_noise_tiles, tiled_denoise_masks, tiled_blend_masks) 

    def prepareTilesforSampling(self, positive, negative, latentImage, tiled_positions, tiled_tiles, tiled_noise_tiles, tiled_denoise_masks, tiled_blend_masks):
        
        latentTesor = self.tensorService.getTensorfromLatentImage(latentImage)
        latentShape = self.tensorService.getShapefromLatentImage(latentImage)
        
        # Extract and prepare cnets, T2Is, gligen, and spatial conditions
        cnets = self.conrolNetService.extractCnet(positive, negative)
        cnet_imgs = self.conrolNetService.prepareCnet_imgs(cnets, latentShape)
        T2Is = self.t2iService.extract_T2I(positive, negative)
        T2I_imgs = self.t2iService.prepareT2I_imgs(T2Is, latentShape)
        gligen_pos, gligen_neg = self.gligenService.extractGligen(positive, negative)
        sptcond_pos, sptcond_neg = self.conditionService.sptcondService.extractSpatialConds(positive, negative, latentShape, latentTesor.device)
        
        tiled_latent = []
        for t_pos_pass, tile_pass, noise_pass, denoise_mask_pass , blend_mask_pass in zip(tiled_positions, tiled_tiles, tiled_noise_tiles, tiled_denoise_masks, tiled_blend_masks):
            tile_pass_l = []
            for t_pos_group, tile_group, noise_group, denoise_mask_group, blend_mask_group in zip(t_pos_pass, tile_pass, noise_pass, denoise_mask_pass, blend_mask_pass):
                tile_group_l = []
                for (x, y, tile_w, tile_h), tile, noise_tile, denoise_mask, blend_mask in zip(t_pos_group, tile_group, noise_group, denoise_mask_group, blend_mask_group):
                    
                    # Copy positive and negative lists for modification
                    pos = [c.copy() for c in positive]
                    neg = [c.copy() for c in negative]

                    #### Slice cnets, T2Is, gligen, and spatial conditions for the current tile ####
                    ## Slice CNETS ##
                    pos, neg, self.conrolNetService.prepareSlicedCnets(pos, neg, cnets, cnet_imgs, y, tile_h, x, tile_w)
                    ## Slice T2Is ##
                    pos, neg = self.t2iService.prepareSlicedT2Is(pos, neg, T2Is, T2I_imgs, y, tile_h, x, tile_w)

                    ## Slice Spatial Conditions  ##
                    pos, neg = self.conditionService.sptcondService.prepareSlicedConds(pos, neg, sptcond_pos, sptcond_neg, y, tile_h, x, tile_w)

                    # Slice GLIGEN ##
                    pos, neg = self.gligenService.prepsreSlicedGligen(pos, neg, gligen_pos, gligen_neg, y, tile_h, x, tile_w)

                    tile_group_l.append((tile, (x, y, tile_w, tile_h), noise_tile, denoise_mask, blend_mask, pos, neg))
                tile_pass_l.append(tile_group_l)
            tiled_latent.append(tile_pass_l)
        return tiled_latent

class LatentMergerService():
    
    def performMerging(self, all_tile_data):
        """
        Perform merging of all tiles from all passes in one step.
        """
        device = comfy.model_management.get_torch_device() 
        
        # Determine the max width and height needed to accommodate all tiles
        max_width = max(x + w for _, (x, y, w, h), _ in all_tile_data)
        max_height = max(y + h for _, (x, y, w, h), _ in all_tile_data)

        # Initialize merged samples and weight map with the required dimensions
        merged_samples = torch.zeros((1, 4, max_height, max_width), device=device)
        weight_map = torch.zeros_like(merged_samples)

        # Process each tile and apply it to the merged output
        for tile, (x, y, w, h), mask in all_tile_data:
            valid_tile, valid_mask = self.extractValidRegion(tile, mask, max_height, max_width, x, y)
            
            # Apply the tile to the merged samples and weight map
            merged_samples, weight_map = self.applyTiletoMerged(merged_samples, weight_map, valid_tile, valid_mask, x, y, w, h)

        return merged_samples, weight_map

    def applyTiletoMerged(self, merged_samples, weight_map, tile, mask, x, y, w, h):
        # Ensure all tensors are on the same device
        device = merged_samples.device  # Get the device of merged_samples
        tile, mask = self.processTileandMask(tile, mask, device)
        
        valid_tile_h, valid_tile_w = min(tile.shape[2], h), min(tile.shape[3], w)
        valid_mask_h, valid_mask_w = min(mask.shape[2], h), min(mask.shape[3], w)
        
        if valid_tile_h == 0 or valid_tile_w == 0:
            raise ValueError(f"Invalid tile dimensions: {valid_tile_h}, {valid_tile_w}")

        merged_samples[:, :, y:y + valid_tile_h, x:x + valid_tile_w] += tile[:, :, :valid_tile_h, :valid_tile_w] * mask[:, :, :valid_mask_h, :valid_mask_w]
        weight_map[:, :, y:y + valid_tile_h, x:x + valid_tile_w] += mask[:, :, :valid_mask_h, :valid_mask_w]

        return merged_samples, weight_map

    def extractValidRegion(self, tile, mask, max_height, max_width, x, y):
        valid_tile_h, valid_tile_w = min(tile.shape[2], max_height - y), min(tile.shape[3], max_width - x)
        valid_mask_h, valid_mask_w = min(mask.shape[2], max_height - y), min(mask.shape[3], max_width - x)
        if valid_tile_h == 0 or valid_tile_w == 0:
            raise ValueError(f"Invalid tile dimensions: {valid_tile_h}, {valid_tile_w}")
        return tile[:, :, :valid_tile_h, :valid_tile_w], mask[:, :, :valid_mask_h, :valid_mask_w]
    
    def processTileandMask(self, tile, mask, device):
        # Ensure the tensors are on the same device
        tile = tile.to(device)
        if tile.dim() == 3:
            tile = tile.unsqueeze(0)
        if tile.shape[1] == 3:
            tile = torch.cat([tile, torch.zeros((tile.shape[0], 1, tile.shape[2], tile.shape[3]), device=device)], dim=1)
        mask = mask.to(device)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        return tile, mask

    def unpackTiles(self, pass_data):
        latent_tiles = []
        tile_positions = []
        noise_masks = []

        for tile_group in pass_data:
            for tile, pos, mask in tile_group:
                latent_tiles.append(tile)
                tile_positions.append(pos)
                noise_masks.append(mask)

        return latent_tiles, tile_positions, noise_masks
