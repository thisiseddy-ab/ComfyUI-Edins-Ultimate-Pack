import nodes
import inspect

class VaeService():

    def vaeDecode(self, vae, samples, use_tile, hook, tile_size=512, overlap=64):
        if use_tile:
            decoder = nodes.VAEDecodeTiled()
            if 'overlap' in inspect.signature(decoder.decode).parameters:
                pixels = decoder.decode(vae, samples, tile_size, overlap=overlap)[0]
            else:
                print(f"[Impact Pack] Your ComfyUI is outdated.")
                pixels = decoder.decode(vae, samples, tile_size)[0]
        else:
            pixels = nodes.VAEDecode().decode(vae, samples)[0]

        if hook is not None:
            pixels = hook.post_decode(pixels)

        return pixels

    def vaeEncode(self, vae, pixels, use_tile, hook, tile_size=512, overlap=64):
        if use_tile:
            encoder = nodes.VAEEncodeTiled()
            if 'overlap' in inspect.signature(encoder.encode).parameters:
                samples = encoder.encode(vae, pixels, tile_size, overlap=overlap)[0]
            else:
                print(f"[Impact Pack] Your ComfyUI is outdated.")
                samples = encoder.encode(vae, pixels, tile_size)[0]
        else:
            samples = nodes.VAEEncode().encode(vae, pixels)[0]

        if hook is not None:
            samples = hook.post_encode(samples)

        return samples