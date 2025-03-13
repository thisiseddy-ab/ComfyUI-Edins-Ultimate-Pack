#### Comfy Lib's ####
import comfy.model_management
import comfy.samplers
import comfy.sampler_helpers
import comfy.utils

#### Third Party Lib's #####
import latent_preview
from tqdm.auto import tqdm

#### Custom Nodes ####
from EUP.nodes.latent import LatentTiler, LatentMerger


class KSamplerService:
    def commonKsampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,  tile_width,tile_height, tiling_mode , passes,
                        tiling_strategy, padding_strategy, ov_pd, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False
                        ):

        device = comfy.model_management.get_torch_device()

        # Step 1: Load Additional Models
        conds0 = {
        "positive": comfy.sampler_helpers.convert_cond(positive),
        "negative": comfy.sampler_helpers.convert_cond(negative)
        }

        conds = {}
        for k in conds0:
            conds[k] = list(map(lambda a: a.copy(), conds0[k]))

        modelPatches, inference_memory = comfy.sampler_helpers.get_additional_models(conds, model.model_dtype())
        comfy.model_management.load_models_gpu([model] + modelPatches, model.memory_required(latent.get("samples").shape) + inference_memory)
        
        # Step 2: Tile Latent **and Noise**
        tiler = LatentTiler()
        tiled_latent = tiler.tileLatent(
            seed=seed, steps=steps, latent_image=latent, positive=positive, negative=negative, tile_width=tile_width, tile_height=tile_height, passes=passes,
            tiling_mode=tiling_mode, tiling_strategy=tiling_strategy, padding_strategy=padding_strategy, overlap_padding=ov_pd, disable_noise=disable_noise
        )

        # Step 3: Prepare Needet Resources
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        if not disable_pbar:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)

        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
        
        total_tiles = sum(len(tile_group) for tile_pass in tiled_latent for tile_group in tile_pass)
        total_steps = steps * total_tiles
        
        # Step 4: Sample Tiles
        with tqdm(total=total_steps) as pbar_tqdm:
            proc_tiled_latent = self.sampleTiles(sampler, tiled_latent, steps, total_steps, total_tiles, pbar_tqdm, disable_pbar, 
                                                    previewer, cfg, start_step, last_step, force_full_denoise, seed)
        # Step 5: Merge Tiles
        merger = LatentMerger()
        merged_latent = merger.mergeTiles(proc_tiled_latent)
        
        comfy.sampler_helpers.cleanup_additional_models(modelPatches)
        
        # Step 5: Return Merge Tiles
        out = latent.copy()
        out["samples"] = merged_latent[0].get("samples")
        return (out, )

    def sampleTiles(self, sampler: comfy.samplers.KSampler, tiled_latent, steps, total_steps, total_tiles, 
                    pbar_tqdm, disable_pbar, previewer, cfg, start_step, last_step, force_full_denoise, seed):
        
        pbar = comfy.utils.ProgressBar(steps)
        proc_tiled_latent = []
        global_tile_index = 0  # Track tiles across all passes

        for tile_pass in tiled_latent:
            tile_pass_l = []
            for tile_group in tile_pass:
                tile_group_l = []
                for (tile, tile_position, noise_tile, denoise_mask, blend_mask, mp, mn) in tile_group:
                    tile_index = global_tile_index 

                    processed_tile = sampler.sample(
                        noise=noise_tile, 
                        positive=mp, 
                        negative=mn, 
                        cfg=cfg, 
                        latent_image=tile, 
                        start_step=start_step, 
                        last_step=last_step,
                        force_full_denoise=force_full_denoise,
                        denoise_mask=denoise_mask, 
                        callback=lambda step, x0, x, total_steps=total_steps, tile_index=tile_index: 
                            self.updateProgress(step, x0, x, total_steps, tile_index, total_tiles, pbar_tqdm, disable_pbar, previewer, pbar),
                        disable_pbar=disable_pbar, 
                        seed=seed
                    )

                    # Store processed tile
                    tile_group_l.append((processed_tile, tile_position, blend_mask))
                    global_tile_index += 1  # Increment index

                tile_pass_l.append(tile_group_l)
            proc_tiled_latent.append(tile_pass_l)

        return proc_tiled_latent
    
    def updateProgress(self, step, x0, x, total_steps, tile_index, total_tiles, pbar_tqdm, disable_pbar, previewer, pbar):
        # Update the overall progress bar
        pbar_tqdm.update(1)

        # Update tile-specific progress
        if not disable_pbar:
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)

            pbar_tqdm.set_description(f"[EUP - Tiled KSampler]")

            # Update the postfix with correct total and tile progress
            postfix = {
                "Tile Progress": f"Tile {tile_index + 1}/{total_tiles}" 
            }
            pbar_tqdm.set_postfix(postfix)

            # Update the progress bar with the preview image (optional)
            pbar.update_absolute(step, preview=preview_bytes)  