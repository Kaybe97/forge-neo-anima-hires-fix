# Source attribution and AI disclosure

This extension runs on [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).

**AI-generated code:** All newly written extension code was AI-generated using OpenAI ChatGPT under user direction. Code adapted or reproduced from external sources is excluded from that statement and retains its original attribution and applicable rights.

**Upstream sources and dependencies:**

- [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) (AGPLv3) — its `modules/processing.py` Hires Fix control flow was adapted for the replacement second pass. The synchronized modes invoke Forge Neo's existing [MultiDiffusion](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo/extensions-builtin/sd_forge_multidiffusion) implementation; that module is not redistributed here. Forge Neo incorporates work from AUTOMATIC1111, lllyasviel, Haoming02, and others; their rights remain with their owners.
- [GJL's Anima Hires Fix technical article](https://note.com/198619891990/n/na8f429e18440?hl=en) — technical basis for tiled sampling, 5D latent handling, overlap blending, and seam repair. Credit does not imply the author's endorsement or grant additional permissions.

No model weights, VAE files, or text encoders are distributed. Third-party software continues to be governed by its original licenses.

**License:** The extension is distributed under the [GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE) to retain the obligations applicable to Forge Neo-derived processing code. AGPLv3 is not interchangeable with GPLv3-only for upstream-derived code. See `LICENSE` for the full terms and warranty disclaimer.
