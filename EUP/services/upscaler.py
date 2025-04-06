import nodes
import warnings

#### My Nodes #####
from EUP.services.lat_upsc_pixel import LatentUpscalerPixelSpaceService
from EUP.nodes.sampler import Tiled_KSampler, Tiled_KSamplerAdvanced



class PixelTiledKSampleUpscalerService():
    def __init__(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                 denoise, tiling_strategy, tiling_strategy_pars, upscale_model_opt=None, 
                 hook_opt=None, tile_cnet_opt=None, tile_size=512, tile_cnet_strength=1.0, overlap=64
                 ):
        
        #### Services ####
        self.lupsService = LatentUpscalerPixelSpaceService()
        
        #### Parameters ####
        self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.scale_method = scale_method
        self.vae = vae
        self.tile_params = tiling_strategy, tiling_strategy_pars,
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.tile_cnet = tile_cnet_opt
        self.tile_size = tile_size
        self.is_tiled = True
        self.tile_cnet_strength = tile_cnet_strength
        self.overlap = overlap

    def tiledKsample(self, latent, images):

        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params
        tiling_strategy, tiling_strategy_pars = self.tile_params

        if self.tile_cnet is not None:
            image_batch, image_w, image_h, _ = images.shape
            if image_batch > 1:
                warnings.warn('Multiple latents in batch, Tile ControlNet being ignored')
            else:
                if 'TilePreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
                    raise RuntimeError("'TilePreprocessor' node (from comfyui_controlnet_aux) isn't installed.")
                preprocessor = nodes.NODE_CLASS_MAPPINGS['TilePreprocessor']()
                # might add capacity to set pyrUp_iters later, not needed for now though
                preprocessed = preprocessor.execute(images, pyrUp_iters=3, resolution=min(image_w, image_h))[0]

                positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive=positive,
                                                                                      negative=negative,
                                                                                      control_net=self.tile_cnet,
                                                                                      image=preprocessed,
                                                                                      strength=self.tile_cnet_strength,
                                                                                      start_percent=0, end_percent=1.0,
                                                                                      vae=self.vae)

        return Tiled_KSampler().sample(
            model=model, 
            seed=seed,  
            steps=steps, 
            cfg=cfg, 
            sampler_name=sampler_name,
            scheduler=scheduler, 
            positive=positive, 
            negative=negative, 
            latent_image=latent,
            tiling_strategy=tiling_strategy,
            tiling_strategy_pars=tiling_strategy_pars,
            denoise=denoise,
            )[0]

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space2(samples, self.scale_method, upscale_factor, self.vae,
                                               use_tile=True, save_temp_prefix=save_temp_prefix,
                                               hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_with_model2(samples, self.scale_method, self.upscale_model,
                                                          upscale_factor, self.vae, use_tile=True,
                                                          save_temp_prefix=save_temp_prefix,
                                                          hook=self.hook, tile_size=self.tile_size)

        refined_latent = self.tiledKsample(upscaled_latent, upscaled_images)

        return refined_latent

    def upscaleShape(self, step_info, samples, w, h, save_temp_prefix=None):

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_shape2(samples, self.scale_method, w, h, self.vae,
                                                     use_tile=True, save_temp_prefix=save_temp_prefix,
                                                     hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_with_model_shape2(samples, self.scale_method,
                                                                self.upscale_model, w, h, self.vae,
                                                                use_tile=True,
                                                                save_temp_prefix=save_temp_prefix,
                                                                hook=self.hook,
                                                                tile_size=self.tile_size)

        refined_latent = self.tiledKsample(upscaled_latent, upscaled_images)

        return refined_latent
    

class AdvancedPixelTiledKSampleUpscalerService():
    def __init__(self, scale_method, model, vae, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, tiling_strategy, tiling_strategy_pars, 
                 start_at_step, end_at_step,return_with_leftover_noise, upscale_model_opt=None, hook_opt=None, tile_cnet_opt=None, tile_size=512, 
                 tile_cnet_strength=1.0, overlap=64
                 ):
        
        #### Services ####
        self.lupsService = LatentUpscalerPixelSpaceService()
        
        #### Parameters ####
        self.params = model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, start_at_step, end_at_step, return_with_leftover_noise
        self.scale_method = scale_method
        self.vae = vae
        self.tile_params = tiling_strategy, tiling_strategy_pars,
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.tile_cnet = tile_cnet_opt
        self.tile_size = tile_size
        self.is_tiled = True
        self.tile_cnet_strength = tile_cnet_strength
        self.overlap = overlap

    def tiledKsample(self, latent, images):

        model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, start_at_step, end_at_step, return_with_leftover_noise = self.params
        tiling_strategy, tiling_strategy_pars = self.tile_params

        if self.tile_cnet is not None:
            image_batch, image_w, image_h, _ = images.shape
            if image_batch > 1:
                warnings.warn('Multiple latents in batch, Tile ControlNet being ignored')
            else:
                if 'TilePreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
                    raise RuntimeError("'TilePreprocessor' node (from comfyui_controlnet_aux) isn't installed.")
                preprocessor = nodes.NODE_CLASS_MAPPINGS['TilePreprocessor']()
                # might add capacity to set pyrUp_iters later, not needed for now though
                preprocessed = preprocessor.execute(images, pyrUp_iters=3, resolution=min(image_w, image_h))[0]

                positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive=positive,
                                                                                      negative=negative,
                                                                                      control_net=self.tile_cnet,
                                                                                      image=preprocessed,
                                                                                      strength=self.tile_cnet_strength,
                                                                                      start_percent=0, end_percent=1.0,
                                                                                      vae=self.vae)

        return Tiled_KSamplerAdvanced().sample(
            model=model, 
            add_noise=add_noise,
            seed=seed,  
            steps=steps, 
            cfg=cfg, 
            sampler_name=sampler_name,
            scheduler=scheduler, 
            positive=positive, 
            negative=negative, 
            latent_image=latent,
            tiling_strategy=tiling_strategy,
            tiling_strategy_pars=tiling_strategy_pars,
            start_at_step=start_at_step,
            end_at_step=end_at_step,
            return_with_leftover_noise=return_with_leftover_noise,
            )[0]

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space2(samples, self.scale_method, upscale_factor, self.vae,
                                               use_tile=True, save_temp_prefix=save_temp_prefix,
                                               hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_with_model2(samples, self.scale_method, self.upscale_model,
                                                          upscale_factor, self.vae, use_tile=True,
                                                          save_temp_prefix=save_temp_prefix,
                                                          hook=self.hook, tile_size=self.tile_size)

        refined_latent = self.tiledKsample(upscaled_latent, upscaled_images)

        return refined_latent

    def upscaleShape(self, step_info, samples, w, h, save_temp_prefix=None):

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_shape2(samples, self.scale_method, w, h, self.vae,
                                                     use_tile=True, save_temp_prefix=save_temp_prefix,
                                                     hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_with_model_shape2(samples, self.scale_method,
                                                                self.upscale_model, w, h, self.vae,
                                                                use_tile=True,
                                                                save_temp_prefix=save_temp_prefix,
                                                                hook=self.hook,
                                                                tile_size=self.tile_size)

        refined_latent = self.tiledKsample(upscaled_latent, upscaled_images)

        return refined_latent
    