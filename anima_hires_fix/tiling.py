# SPDX-License-Identifier: AGPL-3.0-only
"""Pure tiled-latent helpers; independent of Forge Neo so they can be tested."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Region:
    y0: int
    y1: int
    x0: int
    x1: int
    cy0: int
    cy1: int
    cx0: int
    cx1: int


def grid_regions(height: int, width: int, core: int, padding: int):
    """Return regular cores and context-inclusive outer regions, in latent pixels."""
    if min(height, width, core) < 1 or padding < 0:
        raise ValueError("invalid dimensions, core or padding")
    for y in range(0, height, core):
        for x in range(0, width, core):
            yield Region(max(0, y - padding), min(height, y + core + padding),
                         max(0, x - padding), min(width, x + core + padding),
                         y, min(height, y + core), x, min(width, x + core))


def slice_roi(tensor, region):
    return tensor[..., region.y0:region.y1, region.x0:region.x1]


def crop_conditioning(conditioning, region, latent_h, latent_w):
    """Crop spatial tensors proportionally; retain 1x1 conditioning intact.

    A Forge img2img image_conditioning is generally a tensor; nested/unknown
    conditioning is intentionally forwarded rather than guessed at.
    """
    if not isinstance(conditioning, torch.Tensor) or conditioning.ndim < 4:
        return conditioning
    ch, cw = conditioning.shape[-2:]
    if ch <= 1 or cw <= 1:
        return conditioning
    y0 = (region.y0 * ch) // latent_h
    y1 = math.ceil(region.y1 * ch / latent_h)
    x0 = (region.x0 * cw) // latent_w
    x1 = math.ceil(region.x1 * cw / latent_w)
    return conditioning[..., y0:y1, x0:x1]


def _axis_weight(start, end, core_start, core_end, full, *, device="cpu"):
    """Unit in the core and Gaussian falloff only outside the core."""
    coordinates = torch.arange(start, end, dtype=torch.float32, device=device)
    delta = torch.maximum(torch.as_tensor(core_start, device=device) - coordinates,
                          coordinates - (core_end - 1)).clamp_min_(0)
    sigma = max(1.0, max(core_start - start, end - core_end) / 2.5)
    result = torch.exp(-0.5 * (delta / sigma) ** 2)
    return result


def region_weight(region: Region, height: int, width: int):
    wy = _axis_weight(region.y0, region.y1, region.cy0, region.cy1, height)
    wx = _axis_weight(region.x0, region.x1, region.cx0, region.cx1, width)
    return wy[:, None] * wx[None, :]


def accumulate(out, weights, tile, region, weight_2d):
    """Accumulate in CPU fp32, supports NCHW and NCTHW."""
    expected_shape = (region.y1 - region.y0, region.x1 - region.x0)
    if tile.ndim not in (4, 5) or tuple(tile.shape[-2:]) != expected_shape:
        raise ValueError(f"Unexpected tile shape: {tuple(tile.shape)} for {expected_shape}")
    bweight = weight_2d.reshape((1,) * (tile.ndim - 2) + expected_shape)
    out[..., region.y0:region.y1, region.x0:region.x1] += tile.to(device="cpu", dtype=torch.float32) * bweight
    weights[..., region.y0:region.y1, region.x0:region.x1] += bweight


def seam_regions(height, width, core, half_width, context):
    """Small strips split along the opposite axis; no full-height/width strips.

    Returns (sampling_region, boundary_orientation, boundary_position).
    """
    if not (0 < half_width < core):
        raise ValueError("seam half width must be in (0, core)")
    # H seams: thin on Y, segmented on X
    for boundary in range(core, height, core):
        for x in range(0, width, core):
            x1 = min(width, x + core)
            yield Region(max(0, boundary - half_width - context),
                         min(height, boundary + half_width + context),
                         max(0, x - context), min(width, x1 + context),
                         boundary - half_width, boundary + half_width, x, x1), "H", boundary
    # V seams: thin on X, segmented on Y
    for boundary in range(core, width, core):
        for y in range(0, height, core):
            y1 = min(height, y + core)
            yield Region(max(0, y - context), min(height, y1 + context),
                         max(0, boundary - half_width - context),
                         min(width, boundary + half_width + context),
                         y, y1, boundary - half_width, boundary + half_width), "V", boundary


def seam_blend_weights(region, orientation, boundary, half_width):
    """Blend only the target strip, feather toward its boundary center."""
    wy = torch.arange(region.y0, region.y1, dtype=torch.float32)
    wx = torch.arange(region.x0, region.x1, dtype=torch.float32)
    if orientation == "H":
        cross = torch.exp(-0.5 * ((wy - boundary + 0.5) / max(1.0, half_width / 1.7)) ** 2)
        cross *= ((wy >= region.cy0) & (wy < region.cy1))
        # Segment ends are lightly feathered to avoid another seam.
        along = _axis_weight(region.x0, region.x1, region.cx0, region.cx1, region.x1)
        return (cross[:, None] * along[None, :]).clamp(0, 1)
    cross = torch.exp(-0.5 * ((wx - boundary + 0.5) / max(1.0, half_width / 1.7)) ** 2)
    cross *= ((wx >= region.cx0) & (wx < region.cx1))
    along = _axis_weight(region.y0, region.y1, region.cy0, region.cy1, region.y1)
    return (along[:, None] * cross[None, :]).clamp(0, 1)
