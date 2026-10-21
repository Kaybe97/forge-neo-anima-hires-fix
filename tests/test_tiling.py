# SPDX-License-Identifier: AGPL-3.0-only
import torch
from anima_hires_fix.tiling import (
    accumulate, crop_conditioning, grid_regions, region_weight,
    seam_regions, seam_blend_weights, slice_roi,
)


def test_blend_identity_4d():
    x = torch.randn(2, 3, 272, 304)
    regions = list(grid_regions(272, 304, 96, 16))
    out = torch.zeros_like(x)
    weight = torch.zeros(1, 1, 272, 304)
    for r in regions:
        accumulate(out, weight, slice_roi(x, r), r, region_weight(r, 272, 304))
    assert (weight > 0).all()
    assert torch.allclose(out / weight, x, rtol=1e-5, atol=1e-5)


def test_blend_identity_5d():
    x = torch.randn(1, 16, 1, 161, 149)
    out = torch.zeros_like(x)
    weight = torch.zeros(1, 1, 1, 161, 149)
    for r in grid_regions(161, 149, 64, 12):
        accumulate(out, weight, slice_roi(x, r), r, region_weight(r, 161, 149))
    assert torch.allclose(out / weight, x, rtol=1e-5, atol=1e-5)


def test_conditioning_crop():
    r = next(iter(grid_regions(256, 256, 64, 16)))
    full = torch.randn(1, 4, 256, 256)
    assert crop_conditioning(full, r, 256, 256).shape[-2:] == (80, 80)
    vector = torch.randn(1, 5, 1, 1)
    assert crop_conditioning(vector, r, 256, 256) is vector


def test_seam_patches_are_bounded():
    regions = list(seam_regions(354, 256, 96, 12, 16))
    assert regions
    for r, orient, boundary in regions:
        assert r.y1 - r.y0 <= 96 + 2 * 16
        assert r.x1 - r.x0 <= 96 + 2 * 16
        weight = seam_blend_weights(r, orient, boundary, 12)
        assert tuple(weight.shape) == (r.y1 - r.y0, r.x1 - r.x0)
        assert weight.min() >= 0 and weight.max() <= 1


def test_5d_spatial_roi_does_not_crop_time():
    x = torch.zeros(1, 16, 3, 150, 170)
    r = next(iter(grid_regions(150, 170, 64, 8)))
    tile = slice_roi(x, r)
    assert tile.shape[:3] == (1, 16, 3)
