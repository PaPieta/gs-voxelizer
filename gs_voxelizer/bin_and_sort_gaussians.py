
"""Python bindings for binning and sorting gaussians
adapted from image-gs https://github.com/NYU-ICL/image-gs"""

from typing import Tuple

from jaxtyping import Float, Int
from torch import Tensor
import torch

import gs_voxelizer.cuda as _C


def bin_and_sort_gaussians(
    num_gaussians: int,
    num_intersects: int,
    tile_min: Int[Tensor, "batch 3"],
    tile_max: Int[Tensor, "batch 3"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    vol_size: Tuple[int, int, int],
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    """Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return only gaussian IDs sorted by tile ID, as well as tile bins giving the range of gaussian IDs per tile.

    Note:
        This function is not differentiable to any input.

    Args:
        num_gaussians (int): number of gaussians.
        num_intersects (int): cumulative number of total gaussian intersections
        tile_min (Tensor): minimum tile indices hit per gaussian.
        tile_max (Tensor): maximum tile indices hit per gaussian.
        cum_tiles_hit (Tensor): Cumulative number of tiles hit.
        vol_size (Tuple): Dimensions of the volume (depth, height, width).

    Returns:
        A tuple of {Tensor, Tensor}:
        - **gaussian_ids_sorted** (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (Tensor): range of gaussians hit per tile.
    """
    gaussian_ids_sorted, tile_bins = _C.bin_and_sort_gaussians(
        num_gaussians,
        num_intersects,
        tile_min.contiguous(),
        tile_max.contiguous(),
        cum_tiles_hit.contiguous(),
        vol_size,
    )

    return gaussian_ids_sorted.contiguous(), tile_bins.contiguous()
