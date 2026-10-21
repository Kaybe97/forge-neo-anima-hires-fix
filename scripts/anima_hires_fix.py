# SPDX-License-Identifier: AGPL-3.0-only
"""Forge Neo always-visible txt2img extension UI."""
import gradio as gr
from modules import scripts
from anima_hires_fix.engine import Options, install_patch

install_patch()


class Script(scripts.Script):
    sorting_priority = 500

    def title(self):
        return "Anima Hires Fix"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    def ui(self, is_img2img):
        with gr.Accordion("Anima Hires Fix", open=False):
            enable = gr.Checkbox(label="Enable second-pass replacement", value=False)
            selection = gr.Dropdown(label="Which checkpoint?",
                                    choices=["Auto (Anima filename)", "Force (current model)"],
                                    value="Auto (Anima filename)")
            mode = gr.Dropdown(
                label="Second-pass sampling mode",
                choices=["Synchronized (Mixture of Diffusers)",
                         "Synchronized (MultiDiffusion)",
                         "Independent tiles"],
                value="Synchronized (Mixture of Diffusers)",
            )
            gr.Markdown("**Synchronized:** tiles are combined throughout sampling. **Independent:** each tile is sampled separately, then blended into the final image.")
            core = gr.Slider(label="Tile core (latent px; 96 = 768 image px)",
                             minimum=48, maximum=128, step=8, value=96)
            pad = gr.Slider(label="Context padding per side (latent px; 32 = 256 image px)",
                            minimum=0, maximum=48, step=8, value=32)
            shared_noise = gr.Checkbox(label="Independent mode: share the initial noise across overlapping tiles", value=False)
            seams = gr.Checkbox(label="Independent mode: repair tile seams (slower)", value=False)
            half = gr.Slider(label="Seam half-width (latent px)",
                             minimum=4, maximum=32, step=4, value=12)
            strength = gr.Slider(label="Seam denoise multiplier", value=0.5,
                                 minimum=0.1, maximum=1.0, step=0.05)
            dynamic = gr.Checkbox(label="Additional decoded-pixel min/max stretch (normally leave off)",
                                  value=False)
        return [enable, selection, mode, core, pad, shared_noise,
                seams, half, strength, dynamic]

    def process(self, p, enable, selection, mode, core, pad, shared_noise,
                seams, half, strength, dynamic):
        p._anima_hires_fix_options = Options(
            enabled=bool(enable), selection=str(selection),
            tile_core=int(core), tile_pad=int(pad), repair_seams=bool(seams),
            seam_half=int(half), repair_strength=float(strength),
            dynamic_normalize=bool(dynamic), mode=str(mode),
            shared_initial_noise=bool(shared_noise),
        )
