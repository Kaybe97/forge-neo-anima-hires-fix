# SPDX-License-Identifier: AGPL-3.0-only
"""Anima Hires Fix second-pass extension for Forge Neo.

This implementation adapts upstream Hires Fix processing; see NOTICE.md.
Forge Neo's core files are not changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import inspect
import logging

import numpy as np
import torch
from PIL import Image

from modules import devices, extra_networks, images, processing, rng, sd_samplers, shared
from modules.sd_models import apply_token_merging
from modules.sd_samplers_common import approximation_indexes, decode_first_stage, images_tensor_to_samples
from .tiling import (
    accumulate, crop_conditioning, grid_regions, region_weight,
    seam_blend_weights, seam_regions, slice_roi,
)

LOG = logging.getLogger("anima_hires_fix")
_TAG = "[Anima Hires Fix]"


@dataclass(frozen=True)
class Options:
    enabled: bool = False
    selection: str = "Auto (Anima filename)"
    tile_core: int = 96
    tile_pad: int = 32
    repair_seams: bool = False
    seam_half: int = 12
    repair_strength: float = 0.5
    dynamic_normalize: bool = False
    mode: str = "Synchronized (Mixture of Diffusers)"
    shared_initial_noise: bool = False


def say(message):
    print(f"{_TAG} {message}", flush=True)


def _is_anima(p):
    # Hires can load a different checkpoint. Prefer the model actually in use;
    # avoid automatically tiling a non-Anima Hires checkpoint just because the
    # first-pass model was Anima.
    info = getattr(getattr(p, "sd_model", None), "sd_checkpoint_info", None)
    loaded_names = (getattr(info, "filename", ""), getattr(info, "name", ""))
    present = [str(x) for x in loaded_names if x]
    if present:
        return any("anima" in name.lower() for name in present)
    hr_name = getattr(p, "hr_checkpoint_name", None)
    if hr_name and hr_name != "Use same checkpoint":
        return "anima" in str(hr_name).lower()
    return "anima" in str(getattr(p, "sd_model_name", "")).lower()


def _should_apply(p, opts):
    return bool(opts and opts.enabled and getattr(p, "enable_hr", False)
                and (opts.selection == "Force (current model)" or _is_anima(p)))


def _normalize_decoded(decoded, dynamic=False):
    if not isinstance(decoded, torch.Tensor):
        raise TypeError("Expected Forge's decoded_samples tensor; got " + str(type(decoded)))
    if decoded.ndim == 5:
        # Decode commonly produces N,C,1,H,W or N,1,C,H,W.
        if decoded.shape[2] == 1:
            decoded = decoded[:, :, 0]
        elif decoded.shape[1] == 1:
            decoded = decoded[:, 0]
        else:
            raise ValueError("Pixel Hires Fix does not support multi-frame VAE output")
    if decoded.ndim != 4 or decoded.shape[1] not in (3, 4):
        raise ValueError(f"Unexpected decoded pixel tensor shape: {tuple(decoded.shape)}")
    decoded = decoded.float().clamp(0, 1)
    if dynamic:
        # Upstream has already remapped [-1,+1] to [0,1]. This is an optional
        # second stretch, not a substitute for the original VAE normalization.
        lo = decoded.amin(dim=(1, 2, 3), keepdim=True)
        hi = decoded.amax(dim=(1, 2, 3), keepdim=True)
        decoded = ((decoded - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
    return decoded


def _crop_spatial(latents, truncate_y, truncate_x):
    if latents.ndim not in (4, 5):
        raise ValueError(f"Hires Fix expected 4D/5D latents, got {tuple(latents.shape)}")
    h, w = latents.shape[-2:]
    ty, tx = int(truncate_y), int(truncate_x)
    if ty < 0 or tx < 0 or ty >= h or tx >= w:
        raise ValueError(f"Invalid truncation {ty}x{tx} for {h}x{w}")
    return latents[..., ty // 2 : h - (ty + 1) // 2,
                   tx // 2 : w - (tx + 1) // 2]


def _check_interrupt():
    return shared.state.interrupted or shared.state.skipped


def _sample_region(p, region_latent, image_cond, seeds, subseeds, sequence_id, *,
                   denoising=None, initial_noise=None):
    """Sample a tile with valid N-sample noise. Propagate failures; don't hide bad tiles."""
    # Deliberately don't use seed_resize_from on partial tiles: it assumes an
    # NCHW/full-image field and can produce malformed NCTHW noise.
    offset = (int(sequence_id) * 0x9E3779B1) & 0xFFFFFFFF
    tile_seeds = [(int(seed) + offset) & 0xFFFFFFFF for seed in seeds]
    tile_subseeds = ([(int(seed) + offset) & 0xFFFFFFFF for seed in subseeds]
                     if subseeds is not None else None)
    tile_rng = rng.ImageRNG(region_latent.shape[1:], tile_seeds,
                            subseeds=tile_subseeds,
                            subseed_strength=p.subseed_strength)
    # Forge Neo's K-diffusion Sampler.initialize() replaces the module-level
    # torch object with TorchHijack(p), which captures p.rng at initialization.
    # Some samplers (e.g. ER SDE, Euler ancestral) ask torch.randn_like(x) for
    # *additional* noise during sampling.  Leaving p.rng at full-image size
    # produces an oversized tensor even though the initial `noise` is tiled.
    # Bind the tile RNG before the sampler AND before extension sampling hooks;
    # restore the original RNG even when the sampler raises or is interrupted.
    original_rng = getattr(p, "rng", None)
    original_denoise = p.denoising_strength
    p.rng = tile_rng
    try:
        noise = (tile_rng.next() if initial_noise is None
                 else initial_noise.to(device=region_latent.device, dtype=region_latent.dtype))
        if tuple(noise.shape) != tuple(region_latent.shape):
            raise RuntimeError(
                f"Tile noise shape {tuple(noise.shape)} != {tuple(region_latent.shape)}"
            )

        if p.scripts is not None:
            p.scripts.process_before_every_sampling(
                p=p, x=region_latent, noise=noise, c=p.hr_c, uc=p.hr_uc
            )
        if p.modified_noise is not None:
            noise = p.modified_noise
            p.modified_noise = None
            if tuple(noise.shape) != tuple(region_latent.shape):
                raise ValueError("Another extension supplied modified_noise with wrong tile shape")

        if denoising is not None:
            p.denoising_strength = float(denoising)
        result = p.sampler.sample_img2img(
            p, region_latent, noise, p.hr_c, p.hr_uc,
            steps=p.hr_second_pass_steps or p.steps,
            image_conditioning=image_cond,
        )
        if tuple(result.shape) != tuple(region_latent.shape):
            raise RuntimeError(f"Sampler output shape {tuple(result.shape)} != tile shape {tuple(region_latent.shape)}")
        if not torch.isfinite(result).all().item():
            raise RuntimeError("Non-finite values in tile output")
        return result
    finally:
        p.rng = original_rng
        p.denoising_strength = original_denoise


def tiled_second_pass(p, latents, image_conditioning, seeds, subseeds, config):
    if latents.ndim not in (4, 5):
        raise ValueError(f"Invalid latent dimensions: {tuple(latents.shape)}")
    batch, *_, h, w = latents.shape
    if len(seeds) != batch:
        raise ValueError(f"Seed count {len(seeds)} != batch {batch}")
    if config.tile_pad >= config.tile_core:
        raise ValueError("Tile padding must be smaller than core")

    regions = list(grid_regions(h, w, config.tile_core, config.tile_pad))
    say(f"Tiled pass: {h}x{w} latent, {len(regions)} tiles, "
        f"core={config.tile_core}, pad={config.tile_pad}, shape={tuple(latents.shape)}")
    p.extra_generation_params["Anima Hires Fix Tile Core"] = config.tile_core
    p.extra_generation_params["Anima Hires Fix Tile Padding"] = config.tile_pad
    p.extra_generation_params["Anima Hires Fix Tiles"] = len(regions)

    # Preserve original latent on CPU for interruption and avoid an additional
    # full-sized copy on the GPU. Work in fp32 CPU accumulators.
    original = latents.to(device=devices.cpu)
    out = torch.zeros(original.shape, dtype=torch.float32, device=devices.cpu)
    weights = torch.zeros((1,) * (original.ndim - 2) + (h, w),
                          dtype=torch.float32, device=devices.cpu)

    # Optional shared initial noise field. Spatial overlap receives identical
    # noise across tiles, though later stochastic steps remain per-tile.
    shared_noise = None
    if config.shared_initial_noise:
        field_rng = rng.ImageRNG(latents.shape[1:], list(seeds),
                                 subseeds=list(subseeds) if subseeds is not None else None,
                                 subseed_strength=p.subseed_strength)
        shared_noise = field_rng.next().to(device=devices.cpu)
        if tuple(shared_noise.shape) != tuple(original.shape):
            raise RuntimeError("Shared initial noise does not match full latent shape")
        p.extra_generation_params["Anima Hires Fix Shared Noise"] = True

    for i, region in enumerate(regions, 1):
        if _check_interrupt():
            say("Generation interrupted during tiled pass")
            return original
        say(f"Tile {i}/{len(regions)}: [{region.y0}:{region.y1}, {region.x0}:{region.x1}]")
        tile_lat = slice_roi(original, region).to(device=shared.device)
        tile_cond = crop_conditioning(image_conditioning, region, h, w)
        tile_noise = (slice_roi(shared_noise, region).contiguous()
                      if shared_noise is not None else None)
        tile_out = _sample_region(p, tile_lat, tile_cond, seeds, subseeds, i,
                                  initial_noise=tile_noise)
        tile_out = tile_out.to(device=devices.cpu)
        accumulate(out, weights, tile_out, region, region_weight(region, h, w))
        del tile_lat, tile_out
        devices.torch_gc()

    result = out / weights.clamp_min(1e-8)
    del out, weights
    if config.repair_seams and len(regions) > 1 and not _check_interrupt():
        result = _repair_seams(p, result, image_conditioning,
                               seeds, subseeds, config)
    return result.to(dtype=original.dtype)


def synchronized_second_pass(p, latents, image_conditioning, seeds, subseeds, config):
    """One global img2img sampler with native per-denoiser-step tile fusion.

    This deliberately reuses Forge Neo's bundled MultiDiffusion implementation.
    Its UNet-function wrapper tiles the model evaluation *inside* the sampler
    loop; it does not run independently completed img2img processes.
    """
    if latents.ndim not in (4, 5) or (latents.ndim == 5 and latents.shape[2] != 1):
        raise ValueError("Synchronized mode supports NCHW and singleton-frame NCTHW only")
    if len(seeds) != latents.shape[0]:
        raise ValueError("Seed count does not match latent batch")
    if config.tile_pad >= config.tile_core:
        raise ValueError("Padding must be less than the core size")
    if not hasattr(p.sampler, "model_wrap_cfg"):
        raise RuntimeError("Synchronized mode requires a Forge K-diffusion-compatible sampler")

    # One full-frame noise field + one sampler invocation means ER SDE, Euler a,
    # etc. continue requesting *full-sized* stochastic noise from p.rng.
    noise = p.rng.next()
    if tuple(noise.shape) != tuple(latents.shape):
        raise RuntimeError(f"Full latent noise {tuple(noise.shape)} != {tuple(latents.shape)}")
    if p.scripts is not None:
        p.scripts.process_before_every_sampling(
            p=p, x=latents, noise=noise, c=p.hr_c, uc=p.hr_uc
        )
    if p.modified_noise is not None:
        noise = p.modified_noise
        p.modified_noise = None
        if tuple(noise.shape) != tuple(latents.shape):
            raise RuntimeError("Another extension supplied noise with incorrect full-image dimensions")

    # Tile width includes the context on both sides; adjacent windows overlap.
    # Native MultiDiffusion receives *pixel*, not latent dimensions.
    try:
        from lib_multidiffusion.tiled_diffusion import TiledDiffusion
    except ImportError as exc:
        raise RuntimeError(
            "Forge Neo's bundled MultiDiffusion library is missing. "
            "Use Independent tiles mode or update the Forge Neo installation."
        ) from exc
    from modules import processing
    factor = int(processing.opt_f)
    tile_width = (config.tile_core + 2 * config.tile_pad) * factor
    overlap = 2 * config.tile_pad * factor
    method = ("Mixture of Diffusers" if config.mode == "Synchronized (Mixture of Diffusers)"
              else "MultiDiffusion")

    unet = p.sd_model.forge_objects.unet
    if getattr(unet, "model_options", {}).get("tiled_diffusion"):
        raise RuntimeError("Another tiled diffusion wrapper is already active; disable it first")

    p.extra_generation_params["Anima Hires Fix Synced Method"] = method
    p.extra_generation_params["Anima Hires Fix Synced Tile Size"] = tile_width
    p.extra_generation_params["Anima Hires Fix Synced Overlap"] = overlap
    say(f"Synchronized: {method}; {tuple(latents.shape)}; "
        f"tile={tile_width}px overlap={overlap}px, single full-frame sampler")
    try:
        # Importantly, this happens AFTER process_before_every_sampling hooks,
        # so standard ControlNet/LLLite patches can be applied first.
        p.sd_model.forge_objects.unet = TiledDiffusion.apply(
            p.sd_model.forge_objects.unet,
            method, tile_width, tile_width, overlap, 1
        )
        result = p.sampler.sample_img2img(
            p, latents.to(device=shared.device), noise.to(device=shared.device),
            p.hr_c, p.hr_uc,
            steps=p.hr_second_pass_steps or p.steps,
            image_conditioning=image_conditioning,
        )
        if tuple(result.shape) != tuple(latents.shape):
            raise RuntimeError("Synchronized sampler returned an unexpected latent shape")
        if not torch.isfinite(result).all().item():
            raise RuntimeError("Non-finite values in synchronized sampler result")
        return result
    finally:
        # Even when the sampler aborts, don't leave an active tiled model patch.
        p.sd_model.forge_objects.unet = unet


def _repair_seams(p, samples, image_conditioning, seeds, subseeds, config):
    h, w = samples.shape[-2:]
    strips = list(seam_regions(h, w, config.tile_core,
                               config.seam_half, config.tile_pad))
    if not strips:
        return samples
    strength = max(0.05, min(1.0, float(p.denoising_strength) * config.repair_strength))
    say(f"Seam repair: {len(strips)} bounded patches, denoise={strength:.3f}")
    p.extra_generation_params["Anima Hires Fix Seam Repair"] = strength
    for index, (region, orientation, boundary) in enumerate(strips, 1):
        if _check_interrupt():
            say("Seam repair interrupted")
            break
        patch = slice_roi(samples, region).to(device=shared.device)
        ic = crop_conditioning(image_conditioning, region, h, w)
        repaired = _sample_region(p, patch, ic, seeds, subseeds, 100000 + index,
                                  denoising=strength).to(device=devices.cpu,
                                                         dtype=torch.float32)
        blend = seam_blend_weights(region, orientation, boundary,
                                   config.seam_half)
        blend = blend.reshape((1,) * (samples.ndim - 2) + tuple(blend.shape))
        source = slice_roi(samples, region)
        source.mul_(1 - blend).add_(repaired * blend)
        del patch, repaired
        devices.torch_gc()
    return samples


def replacement_sample_hr_pass(p, samples, decoded_samples,
                               seeds, subseeds, subseed_strength, prompts, config):
    """Full pixel/latent preprocessing + tiled Hires Fix + standard Forge decode."""
    if _check_interrupt():
        return samples

    p.is_hr_pass = True
    try:
        devices.torch_gc()
        p.sd_model.set_shift(shift=p.hr_distilled_cfg,
                             width=p.hr_upscale_to_x, height=p.hr_upscale_to_y)
        if p.sd_model.use_distilled_cfg_scale:
            p.extra_generation_params["Hires Distilled CFG Scale"] = p.hr_distilled_cfg
        if p.sd_model.use_shift:
            p.extra_generation_params["Hires Shift"] = p.hr_distilled_cfg
        if p.hr_sampler_name is not None and p.hr_sampler_name != p.sampler_name:
            p.extra_generation_params["Hires sampler"] = p.hr_sampler_name
        if p.hr_scheduler is not None and p.hr_scheduler != p.scheduler:
            p.extra_generation_params["Hires schedule type"] = p.hr_scheduler
        if p.hr_upscaler is not None:
            p.extra_generation_params["Hires upscaler"] = p.hr_upscaler
        if p.hr_second_pass_steps:
            p.extra_generation_params["Hires steps"] = p.hr_second_pass_steps
        p.extra_generation_params["Denoising strength"] = p.denoising_strength
        p.extra_generation_params["Hires CFG Scale"] = p.hr_cfg
        p.extra_generation_params["Hires prompt"] = p.get_hr_prompt
        p.extra_generation_params["Hires negative prompt"] = p.get_hr_negative_prompt
        p.extra_generation_params["Anima Hires Fix"] = "v0.2.0"
        p.extra_generation_params["Anima Hires Fix Mode"] = config.mode

        target_w, target_h = int(p.hr_upscale_to_x), int(p.hr_upscale_to_y)
        p.sampler = sd_samplers.create_sampler(p.hr_sampler_name or p.sampler_name, p.sd_model)

        def save_intermediate(item, index):
            if not p.save_samples() or not shared.opts.save_images_before_highres_fix:
                return
            if not isinstance(item, Image.Image):
                item = sd_samplers.sample_to_image(item, index, approximation=0)
            info = processing.create_infotext(p, p.all_prompts, p.all_seeds,
                                              p.all_subseeds, [],
                                              iteration=p.iteration,
                                              position_in_batch=index)
            images.save_image(item, p.outpath_samples, "", seeds[index],
                              prompts[index], shared.opts.samples_format,
                              info=info, p=p, suffix="-before-highres-fix")

        if p.latent_scale_mode is not None:
            for i in range(samples.shape[0]):
                save_intermediate(samples, i)
            # Interpolate only H,W. Preserve N,C,T for singleton-frame 5D.
            if samples.ndim == 5:
                n, c, t, h, w = samples.shape
                up = samples.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
            else:
                up = samples
            up = torch.nn.functional.interpolate(
                up, size=(target_h // processing.opt_f, target_w // processing.opt_f),
                mode=p.latent_scale_mode["mode"],
                antialias=p.latent_scale_mode["antialias"])
            if samples.ndim == 5:
                up = up.reshape(n, t, c, up.shape[-2], up.shape[-1]).permute(0, 2, 1, 3, 4)
            samples = up
            if getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                image_conditioning = p.img2img_image_conditioning(
                    decode_first_stage(p.sd_model, samples), samples)
            else:
                image_conditioning = p.txt2img_image_conditioning(samples)
        else:
            # The original sample() already provided first-pass pixels in [0,1].
            pixels = _normalize_decoded(decoded_samples, config.dynamic_normalize)
            batch_images = []
            for i, x_sample in enumerate(pixels):
                arr = (x_sample.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(np.moveaxis(arr, 0, 2))
                save_intermediate(image, i)
                image = images.resize_image(0, image, target_w, target_h,
                                            upscaler_name=p.hr_upscaler)
                arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                batch_images.append(np.moveaxis(arr, 2, 0))
            pixels = torch.from_numpy(np.stack(batch_images)).to(
                shared.device, dtype=torch.float32)
            if shared.opts.sd_vae_encode_method != "Full":
                p.extra_generation_params["VAE Encoder"] = shared.opts.sd_vae_encode_method
            samples = images_tensor_to_samples(
                pixels, approximation_indexes.get(shared.opts.sd_vae_encode_method))
            p.sd_model.ini_latent = None
            devices.torch_gc()
            image_conditioning = p.img2img_image_conditioning(pixels, samples)
            del pixels

        shared.state.nextjob()
        samples = _crop_spatial(samples, p.truncate_y, p.truncate_x)
        # Standard per-request RNG remains available to other scripts. Tiled
        # noise itself uses separate, batch-aware seeded generators.
        p.rng = rng.ImageRNG(samples.shape[1:], p.seeds,
                             subseeds=p.subseeds,
                             subseed_strength=p.subseed_strength,
                             seed_resize_from_h=0, seed_resize_from_w=0)
        p.outpath_samples = shared.opts.outdir_hires_samples or p.outpath_samples
        devices.torch_gc()

        if not p.disable_extra_networks:
            with devices.autocast():
                extra_networks.activate(p, p.hr_extra_network_data)
        with devices.autocast():
            p.calculate_hr_conds()
        if p.scripts is not None:
            p.scripts.before_hr(p)
        p.sd_model.forge_objects = p.sd_model.forge_objects_after_applying_lora.shallow_copy()
        apply_token_merging(p.sd_model, p.get_token_merging_ratio(for_hr=True))

        if config.mode.startswith("Synchronized ("):
            if config.repair_seams:
                say("Seam repair is only applied in Independent mode; skipping for synchronized mode")
            samples = synchronized_second_pass(
                p, samples, image_conditioning, seeds, subseeds, config)
        elif config.mode == "Independent tiles":
            samples = tiled_second_pass(p, samples, image_conditioning,
                                        seeds, subseeds, config)
        else:
            raise ValueError(f"Unsupported Anima Hires Fix mode: {config.mode}")
        devices.torch_gc()
        return processing.decode_latent_batch(
            p.sd_model, samples, target_device=devices.cpu, check_for_nans=True)
    finally:
        # Restore state even after a failed tile or an interrupt. Do not
        # silently fall back: a mid-pass failure must be reported to the user.
        p.sampler = None
        p.is_hr_pass = False
        devices.torch_gc()


def install_patch():
    cls = processing.StableDiffusionProcessingTxt2Img
    current = cls.sample_hr_pass
    if getattr(current, "_anima_hires_wrapped", False):
        return
    expected = ["self", "samples", "decoded_samples", "seeds", "subseeds",
                "subseed_strength", "prompts"]
    if list(inspect.signature(current).parameters) != expected:
        say("Incompatible sample_hr_pass signature; extension disabled")
        return

    @wraps(current)
    def wrapper(self, samples, decoded_samples, seeds, subseeds, subseed_strength, prompts):
        config = getattr(self, "_anima_hires_fix_options", None)
        if not _should_apply(self, config):
            return current(self, samples, decoded_samples,
                           seeds, subseeds, subseed_strength, prompts)
        if (getattr(self.sd_model, "is_wan", False) and
                getattr(samples, "ndim", 4) == 5 and samples.shape[2] != 1):
            raise RuntimeError("Anima Hires Fix does not support multi-frame Wan/video latent")
        say(f"Activated for {getattr(self, 'sd_model_name', 'unknown model')}; "
            f"upscaler={self.hr_upscaler}")
        return replacement_sample_hr_pass(self, samples, decoded_samples,
                                          seeds, subseeds, subseed_strength, prompts,
                                          config)

    wrapper._anima_hires_wrapped = True
    wrapper._anima_hires_original = current
    cls.sample_hr_pass = wrapper
    say("Installed Hires Fix interception (disabled unless enabled in txt2img)")
