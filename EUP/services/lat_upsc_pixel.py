#### Services ####
from EUP.services.vae import VaeService

#### ComfyUI Nodes ####
import nodes
import comfy_extras.nodes_upscale_model as model_upscale


class LatentUpscalerPixelSpaceService():

    def __init__(self):
        self.vae_service = VaeService()

    def latent_upscale_on_pixel_space_shape(self, samples, scale_method, w, h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
        return self.latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


    def latent_upscale_on_pixel_space_shape2(self, samples, scale_method, w, h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
        pixels = self.vae_service.vaeDecode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

        if save_temp_prefix is not None:
            nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

        pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

        old_pixels = pixels
        if hook is not None:
            pixels = hook.post_upscale(pixels)

        return self.vae_service.vaeEncode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


    def latent_upscale_on_pixel_space(self, samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
        return self.latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


    def latent_upscale_on_pixel_space2(self, samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
        pixels = self.vae_service.vaeDecode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

        if save_temp_prefix is not None:
            nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

        w = pixels.shape[2] * scale_factor
        h = pixels.shape[1] * scale_factor
        pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

        old_pixels = pixels
        if hook is not None:
            pixels = hook.post_upscale(pixels)

        return self.vae_service.vaeEncode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


    def latent_upscale_on_pixel_space_with_model_shape(self, samples, scale_method, upscale_model, new_w, new_h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
        return self.latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


    def latent_upscale_on_pixel_space_with_model_shape2(self, samples, scale_method, upscale_model, new_w, new_h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
        pixels = self.vae_service.vaeDecode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

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

        return self.vae_service.vaeEncode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


    def latent_upscale_on_pixel_space_with_model(self, samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
                                                tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
        return self.latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model, scale_factor, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]

    def latent_upscale_on_pixel_space_with_model2(self, samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
                                                tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
        pixels = self.vae_service.vaeDecode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

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

        return self.vae_service.vaeEncode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels
