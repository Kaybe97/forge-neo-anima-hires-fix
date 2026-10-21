# Anima Hires Fix

An extension for **Forge Neo** that improves Hires Fix with Anima by processing the enlarged image in overlapping tiles. This helps prevent the severe noise and image breakdown that can happen when the entire high-resolution image is denoised at once. It uses Forge Neo's normal **txt2img → Hires Fix** controls; no core files need to be changed.

## Installation

1. Remove older copies of this extension from `<Forge Neo>/extensions/` before installing; older releases used a different folder name.
2. Clone this repository into `<Forge Neo>/extensions/` using the URL under **Code**, or extract a repository ZIP so that `extensions/forge-neo-anima-hires-fix/scripts/` exists.
3. Restart Forge Neo.


## Features

- **Synchronized — Mixture of Diffusers (default):** Tiles are blended throughout denoising instead of being generated independently. Forge Neo's built-in Mixture of Diffusers combines overlapping model predictions at each sampling step while the sampler operates on a shared, full-image latent.
- **Synchronized — MultiDiffusion:** Another shared-image mode, using averaging to combine predictions from overlapping tiles.
- **Independent tiles:** Each tile completes its own denoising pass before the results are blended together. Optional **seam repair** resamples tile borders to reduce visible joins, at the cost of additional processing time. **Shared initial noise** aligns noise in overlapping regions.
- **Tile core and context padding:** Control how much of the image each tile processes and how much surrounding context it receives. Values are in *latent pixels* (typically 1 latent pixel ≈ 8 image pixels). Larger regions may improve consistency but require more memory.

Synchronized modes reduce tile seams, but do not provide full-image attention to each tile. Very high denoising strengths can still introduce unexpected details. Seam repair applies to independent mode only.

## Usage

Enable **Anima Hires Fix** in **txt2img**, then use the regular **Hires Fix** settings. Checkpoints with `anima` in their name are selected automatically; **Force (current model)** bypasses name detection. Turning off the extension uses Forge Neo's normal Hires Fix path.

## License and credits

Licensed under the **[GNU Affero General Public License v3.0](LICENSE)** (AGPL-3.0-only).

**All code newly written for this extension was AI-generated with OpenAI ChatGPT; code adapted from other sources is excluded from that statement.** See [NOTICE.md](NOTICE.md) for attribution to [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) and [GJL's Anima Hires Fix article](https://note.com/198619891990/n/na8f429e18440?hl=en).
