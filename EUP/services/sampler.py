#### Comfy Lib's ####
import comfy.model_management
import comfy.samplers
import comfy.sampler_helpers
import comfy.utils

#### Third Party Lib's #####
import latent_preview
from tqdm.auto import tqdm

#### My Services's #####
from EUP.services.latent import LatentTilerService, LatentMergerService

#### Custom Nodes ####
from EUP.nodes.latent import LatentMerger

import torch

class KSamplerService:
    def __init__(self):
        self.latentService = LatentTilerService()
    
    def commonKsampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, sampler_advanced_pars, denoise=1.0, disable_noise=False, 
                       start_step=None, last_step=None, force_full_denoise=False,actual_steps=30
                       ):
        
        #print(dir(model.model.latent_format))
        #print("latent_channels:", model.model.latent_format.latent_channels)
        #print("latent_dimensions:", model.model.latent_format.latent_dimensions)
        
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
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

        # Step 2: Tile Latent **and Noise**
        tiled_latent = self.latentService.tileLatent(
            model=model, sampler=sampler, seed=seed, latent_image=latent, positive=positive, negative=negative, sampler_advanced_pars=sampler_advanced_pars, 
            disable_noise=disable_noise, start_step=start_step,
        )

        # Step 3: Prepare Needet Resources
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        if not disable_pbar:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)
        
        total_tiles = sum(len(tile_group) for tile_pass in tiled_latent for tile_group in tile_pass)
        total_steps = steps * total_tiles
        
        # Step 4: Sample Tiles
        with tqdm(total=total_steps) as pbar_tqdm:
            proc_tiled_latent = self.sampleTiles(model, sampler, tiled_latent, steps, total_steps, total_tiles, pbar_tqdm, disable_pbar, 
                                                    previewer, cfg, start_step, last_step, force_full_denoise, seed, actual_steps)
        # Step 5: Merge Tiles
        merger = LatentMerger()
        merged_latent = merger.mergeTiles(proc_tiled_latent)
        
        comfy.sampler_helpers.cleanup_additional_models(modelPatches)
        
        # Step 5: Return Merge Tiles
        out = latent.copy()
        out["samples"] = merged_latent[0].get("samples").to("cpu")
        return (out, )

    def sampleTiles(self, model, sampler: comfy.samplers.KSampler, tiled_latent, steps, total_steps, total_tiles, 
                    pbar_tqdm, disable_pbar, previewer, cfg, start_step, last_step, force_full_denoise, seed, actual_steps):
        
        pbar = comfy.utils.ProgressBar(steps)
        proc_tiled_latent = []
        global_tile_index = 0  # Track tiles across all passes

        for tile_pass in tiled_latent:
            tile_pass_l = []
            for i, tile_group in enumerate(tile_pass):
                tile_group_l = []
                for (tile, tile_position, noise_tile, denoise_mask, blend_mask, mp, mn) in tile_group:
                    tile_index = global_tile_index 

                    #tile = tile.to(device)
                    #denoise_mask = denoise_mask.to(device)
                    #noise_tile = noise_tile.to(device)

                    #denoise_mask = denoise_mask.to("cpu")
                    #noise_tile = noise_tile.to("cpu")
                    
                    processed_tile = sampler.sample(
                        noise=noise_tile,  
                        positive=mp, 
                        negative=mn, 
                        cfg=cfg, 
                        latent_image=tile,  
                        start_step=start_step + i * actual_steps,
                        last_step=start_step + i*actual_steps + actual_steps,
                        force_full_denoise=force_full_denoise and i+1 == last_step - start_step,
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