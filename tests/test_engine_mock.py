# SPDX-License-Identifier: AGPL-3.0-only
"""CPU smoke test for Forge interfaces. No actual Forge Neo installation required."""
import contextlib
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def test_pixel_second_pass_mock(monkeypatch):
    base = types.ModuleType("modules")
    base.__path__ = []
    monkeypatch.setitem(sys.modules, "modules", base)
    names = ["devices", "extra_networks", "images", "processing", "rng",
             "sd_samplers", "shared", "sd_models", "sd_samplers_common"]
    fake = {}
    for name in names:
        m = types.ModuleType("modules." + name)
        monkeypatch.setitem(sys.modules, "modules." + name, m)
        setattr(base, name, m)
        fake[name] = m

    fake["devices"].cpu = torch.device("cpu")
    fake["devices"].torch_gc = lambda: None
    fake["devices"].autocast = contextlib.nullcontext
    fake["shared"].device = torch.device("cpu")
    fake["shared"].state = types.SimpleNamespace(interrupted=False, skipped=False, nextjob=lambda: None)
    fake["shared"].opts = types.SimpleNamespace(
        sd_vae_encode_method="Full", outdir_hires_samples=None,
        inpainting_mask_weight=1.0, save_images_before_highres_fix=False,
        samples_format="png")
    fake["extra_networks"].activate = lambda *a, **k: None
    fake["images"].resize_image = lambda mode, img, w, h, upscaler_name=None: img.resize((w, h))
    fake["sd_models"].apply_token_merging = lambda *a, **k: None
    fake["processing"].opt_f = 8
    fake["processing"].create_infotext = lambda *a, **k: "test"
    fake["processing"].decode_latent_batch = lambda model, batch, **kw: [batch[0]]
    fake["sd_samplers_common"].approximation_indexes = {"Full": 0}
    fake["sd_samplers_common"].decode_first_stage = lambda model, latents: latents
    fake["sd_samplers_common"].images_tensor_to_samples = (
        lambda pixels, method: torch.nn.functional.avg_pool2d(pixels[:, :2], 8))

    class FakeRNG:
        def __init__(self, shape, seeds, **kwargs):
            self.shape = tuple(shape)
            self.seeds = seeds

        def next(self):
            shape = (len(self.seeds),) + self.shape
            return torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)

    fake["rng"].ImageRNG = FakeRNG
    calls = []
    recorded_noise = []

    class FakeSampler:
        def sample_img2img(self, p, x, noise, c, uc, *, steps, image_conditioning):
            # Reproduce ER SDE's secondary stochastic-noise path via Forge's
            # TorchHijack(p).randn_like(x), which pulls noise from `p.rng`.
            from_p_rng = p.rng.next()
            assert tuple(from_p_rng.shape) == tuple(x.shape), (
                f"Sampler stochastic noise {tuple(from_p_rng.shape)} "
                f"does not fit tile {tuple(x.shape)}"
            )
            calls.append((tuple(x.shape), tuple(noise.shape), steps))
            recorded_noise.append(noise.detach().cpu().clone())
            if getattr(p, "_raise_on_sampling", False):
                raise RuntimeError("simulated sampler error")
            return x + from_p_rng * 0.0 + 0.1

    fake["sd_samplers"].create_sampler = lambda *a: FakeSampler()
    fake["sd_samplers"].sample_to_image = lambda a, b, approximation: Image.new("RGB", (16, 16))
    fake["images"].save_image = lambda *a, **kw: None

    class FakeTxt:
        def sample_hr_pass(self, samples, decoded_samples, seeds, subseeds,
                           subseed_strength, prompts):
            return "stock"

    fake["processing"].StableDiffusionProcessingTxt2Img = FakeTxt
    sys.modules.pop("anima_hires_fix.engine", None)
    engine = importlib.import_module("anima_hires_fix.engine")
    engine.install_patch()

    p = FakeTxt()
    p._anima_hires_fix_options = engine.Options(True, "Auto (Anima filename)", 4, 1,
                                               mode="Independent tiles")
    p.enable_hr = True
    p.sd_model_name = "Anima"
    class FakeModel:
        sd_checkpoint_info = types.SimpleNamespace(filename="anima.safetensors", name="Anima")
        use_distilled_cfg_scale = False
        use_shift = False
        ini_latent = None
        forge_objects_after_applying_lora = types.SimpleNamespace(shallow_copy=lambda: object())
        forge_objects = None
        def set_shift(self, **kwargs):
            pass
    p.sd_model = FakeModel()
    p.hr_distilled_cfg = 3.5
    p.hr_upscale_to_x = 48
    p.hr_upscale_to_y = 48
    p.hr_sampler_name = None
    p.sampler_name = "Euler"
    p.hr_scheduler = None
    p.scheduler = "Normal"
    p.hr_upscaler = "Lanczos"
    p.hr_second_pass_steps = 5
    p.steps = 10
    p.denoising_strength = 0.7
    p.hr_cfg = 1.0
    p.get_hr_prompt = None
    p.get_hr_negative_prompt = None
    p.latent_scale_mode = None
    p.extra_generation_params = {}
    p.sampler = None
    p.save_samples = lambda: False
    p.truncate_y = p.truncate_x = 0
    p.seed_resize_from_h = p.seed_resize_from_w = 0
    p.seeds = [42]
    p.subseeds = [17]
    p.subseed_strength = 0.0
    p.scripts = None
    p.modified_noise = None
    p.disable_extra_networks = True
    p.hr_extra_network_data = []
    p.outpath_samples = "/tmp"
    p.get_token_merging_ratio = lambda for_hr=False: 0
    p.calculate_hr_conds = lambda: (setattr(p, "hr_c", "cond"), setattr(p, "hr_uc", None))
    p.img2img_image_conditioning = lambda pixels, latents: torch.zeros(latents.shape)
    p.is_hr_pass = False

    decoded = torch.rand(1, 3, 16, 16)
    result = p.sample_hr_pass(None, decoded, [42], [17], 0.0, ["test"])
    assert len(result) == 1
    assert result[0].shape == (2, 6, 6)
    assert len(calls) == 4  # 6x6 latent, 4x4 tile cores
    assert all(shp == noise_shape for shp, noise_shape, _ in calls)
    assert all(steps == 5 for _, _, steps in calls)
    assert p.sampler is None and not p.is_hr_pass
    assert p.extra_generation_params["Anima Hires Fix Tiles"] == 4
    # The full-image RNG must be reinstalled after each tile, not left pointing
    # at the final tile (whose size would cause other extensions to misbehave).
    assert p.rng.shape == (2, 6, 6)

    # Shared initial noise must be spatially aligned in overlapping tiles.
    # It is intentionally an opt-in so legacy independent behavior is intact.
    recorded_noise.clear()
    p._anima_hires_fix_options = engine.Options(
        True, "Auto (Anima filename)", 4, 1,
        mode="Independent tiles", shared_initial_noise=True)
    p.sample_hr_pass(None, decoded, [42], [17], 0, ["test"])
    full_noise = torch.arange(1 * 2 * 6 * 6, dtype=torch.float32).reshape(1, 2, 6, 6)
    assert len(recorded_noise) == 4
    assert torch.equal(recorded_noise[0], full_noise[..., 0:5, 0:5])
    assert torch.equal(recorded_noise[1], full_noise[..., 0:5, 3:6])

    # Regression: 5D singleton-frame latents need correctly shaped stochastic
    # sampler noise too.  Ensure the previous full-pass RNG survives sampling.
    full_pass_rng = p.rng
    p.sampler = FakeSampler()
    p._raise_on_sampling = False
    five_d = torch.zeros(1, 2, 1, 3, 4)
    assert engine._sample_region(p, five_d, None, [42], [17], 77).shape == five_d.shape
    assert p.rng is full_pass_rng

    # Regression: tile RNG and denoising strength must be restored on failure.
    p._raise_on_sampling = True
    with pytest.raises(RuntimeError, match="simulated sampler error"):
        engine._sample_region(p, five_d, None, [42], [17], 78, denoising=0.15)
    assert p.rng is full_pass_rng
    assert p.denoising_strength == 0.7

    # v0.2 regression: in synchronized mode the native Forge MultiDiffusion
    # UNet wrapper must be installed for exactly one global img2img call;
    # the RNG and stochastic sampler noise remain full-size.
    seen = []
    unet = types.SimpleNamespace(model_options={})
    fake_copy = lambda: types.SimpleNamespace(unet=unet)
    p.sd_model.forge_objects_after_applying_lora = types.SimpleNamespace(
        shallow_copy=fake_copy)
    fake_tiled = types.ModuleType("lib_multidiffusion.tiled_diffusion")
    fake_pkg = types.ModuleType("lib_multidiffusion")
    fake_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "lib_multidiffusion", fake_pkg)
    monkeypatch.setitem(sys.modules, "lib_multidiffusion.tiled_diffusion", fake_tiled)

    class FakeTiledDiffusion:
        @staticmethod
        def apply(model, method, width, height, overlap, batch):
            seen.append(("apply", method, width, height, overlap, batch))
            assert model is unet
            return types.SimpleNamespace(model_options={"tiled_diffusion": True})

    fake_tiled.TiledDiffusion = FakeTiledDiffusion
    class FakeSamplerSynced(FakeSampler):
        model_wrap_cfg = object()

        def sample_img2img(self, p, x, noise, c, uc, *, steps, image_conditioning):
            assert p.sd_model.forge_objects.unet.model_options["tiled_diffusion"]
            assert tuple(noise.shape) == tuple(x.shape)
            assert tuple(p.rng.next().shape) == tuple(x.shape)
            seen.append(("sample", tuple(x.shape)))
            if getattr(p, "_raise_on_synced", False):
                raise RuntimeError("synchronized sampler failure")
            return x + 0.2

    fake["sd_samplers"].create_sampler = lambda *args: FakeSamplerSynced()

    class ScriptHooks:
        def __init__(self):
            self.before = 0
            self.sampling = 0
        def before_hr(self, proc):
            self.before += 1
        def process_before_every_sampling(self, **kwargs):
            assert kwargs["x"].shape == kwargs["noise"].shape
            self.sampling += 1

    p.scripts = ScriptHooks()
    p._anima_hires_fix_options = engine.Options(
        True, "Auto (Anima filename)", 4, 1,
        mode="Synchronized (Mixture of Diffusers)")
    p._raise_on_sampling = False
    p._raise_on_synced = False
    out = p.sample_hr_pass(None, decoded, [42], [17], 0, ["test"])
    assert len(out) == 1 and out[0].shape == (2, 6, 6)
    assert seen == [("apply", "Mixture of Diffusers", 48, 48, 16, 1),
                    ("sample", (1, 2, 6, 6))]
    assert p.scripts.before == 1 and p.scripts.sampling == 1
    assert p.sd_model.forge_objects.unet is unet
    assert not p.is_hr_pass and p.sampler is None

    seen.clear()
    p._anima_hires_fix_options = engine.Options(
        True, "Auto (Anima filename)", 4, 1,
        mode="Synchronized (MultiDiffusion)")
    p.sample_hr_pass(None, decoded, [42], [17], 0, ["test"])
    assert seen[0][1] == "MultiDiffusion"

    # Failure cleanup leaves no active model patch or Hires flags behind.
    p._raise_on_synced = True
    with pytest.raises(RuntimeError, match="synchronized sampler failure"):
        p.sample_hr_pass(None, decoded, [42], [17], 0, ["test"])
    assert p.sd_model.forge_objects.unet is unet
    assert p.sampler is None and not p.is_hr_pass

    p._anima_hires_fix_options = engine.Options(False)
    assert p.sample_hr_pass(None, decoded, [42], [17], 0, ["test"]) == "stock"
    del sys.modules["anima_hires_fix.engine"]
