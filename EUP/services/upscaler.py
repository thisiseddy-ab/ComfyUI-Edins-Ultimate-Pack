import nodes
import warnings
from server import PromptServer

import comfy_extras.nodes_upscale_model as model_upscale
from EUP.services.vae import vae_decode, vae_encode

#### Nodes #####
from EUP.nodes.sampler import Tiled_KSampler


def update_node_status(node, text, progress=None):
    if PromptServer.instance.client_id is None:
        return

    PromptServer.instance.send_sync("impact/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, PromptServer.instance.client_id)


def latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


def latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


def latent_upscale_on_pixel_space(samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


def latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2] * scale_factor
    h = pixels.shape[1] * scale_factor
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


def latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


def latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]

    # upscale by model upscaler
    current_w = w
    while current_w < new_w:
        pixels = model_upscale.ImageUpscaleWithModel().upscale(upscale_model, pixels)[0]
        current_w = pixels.shape[2]
        if current_w == w:
            print(f"[latent_upscale_on_pixel_space_with_model] x1 upscale model selected")
            break

    # downscale to target scale
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


def latent_upscale_on_pixel_space_with_model(samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
                                             tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model, scale_factor, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]

def latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
                                              tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]
    h = pixels.shape[1]

    new_w = w * scale_factor
    new_h = h * scale_factor

    # upscale by model upscaler
    current_w = w
    while current_w < new_w:
        pixels = model_upscale.ImageUpscaleWithModel().upscale(upscale_model, pixels)[0]
        current_w = pixels.shape[2]
        if current_w == w:
            print(f"[latent_upscale_on_pixel_space_with_model] x1 upscale model selected")
            break

    # downscale to target scale
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels

class PixelTiledKSampleUpscaler:
    def __init__(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                 denoise, tile_width, tile_height, tiling_strategy, padding_strategy, padding,upscale_model_opt=None, 
                 hook_opt=None, tile_cnet_opt=None, tile_size=512, tile_cnet_strength=1.0, overlap=64
                 ):
        
        self.params = scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.vae = vae
        self.tile_params = tile_width, tile_height, tiling_strategy
        self.padding_params = padding_strategy, padding,
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.tile_cnet = tile_cnet_opt
        self.tile_size = tile_size
        self.is_tiled = True
        self.tile_cnet_strength = tile_cnet_strength
        self.overlap = overlap

    def tiled_ksample(self, latent, images):

        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params
        tile_width, tile_height, tiling_strategy = self.tile_params
        padding_strategy, padding = self.padding_params

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
            tile_width=tile_width, 
            tile_height=tile_height, 
            tiling_strategy=tiling_strategy, 
            steps=steps, 
            cfg=cfg, 
            sampler_name=sampler_name,
            scheduler=scheduler, 
            positive=positive, 
            negative=negative, 
            latent_image=latent,
            padding_strategy=padding_strategy,
            padding=padding,
            denoise=denoise,
            )[0]

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space2(samples, scale_method, upscale_factor, vae,
                                               use_tile=True, save_temp_prefix=save_temp_prefix,
                                               hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_with_model2(samples, scale_method, self.upscale_model,
                                                          upscale_factor, vae, use_tile=True,
                                                          save_temp_prefix=save_temp_prefix,
                                                          hook=self.hook, tile_size=self.tile_size)

        refined_latent = self.tiled_ksample(upscaled_latent, upscaled_images)

        return refined_latent

    def upscale_shape(self, step_info, samples, w, h, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae,
                                                     use_tile=True, save_temp_prefix=save_temp_prefix,
                                                     hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method,
                                                                self.upscale_model, w, h, vae,
                                                                use_tile=True,
                                                                save_temp_prefix=save_temp_prefix,
                                                                hook=self.hook,
                                                                tile_size=self.tile_size)

        refined_latent = self.tiled_ksample(upscaled_latent, upscaled_images)

        return refined_latent