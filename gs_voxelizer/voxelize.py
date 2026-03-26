"""Voxelizer python-cuda bindings with Python-based Gaussian sorting"""

import torch
from jaxtyping import Float, Int
from typing import Tuple
from torch import Tensor
from torch.autograd import Function

import gs_voxelizer.cuda as _C

from .bin_and_sort_gaussians import bin_and_sort_gaussians


def voxelize_gaussians(
    pos3d_radii: Float[Tensor, "*batch 4"],
    conics: Float[Tensor, "*batch 6"],
    intensities: Float[Tensor, "*batch channels"],
    vol_size_voxel: Tuple[int, int, int],
    tile_min: Int[Tensor, "*batch 3"],
    tile_max: Int[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    use_per_gaussian_backward: bool = True,
) -> Tensor:
    """Voxelizes 3D Gaussians into a 3D voxel grid.

    Args:
        pos3d_radii: (*batch, 4) Tensor of 3D positions of Gaussians (xyz), with radii in the w component.
        conics: (*batch, 6) Tensor of conic parameters of Gaussians.
        intensities: (*batch, *channels) Tensor of intensity values for Gaussians. Max 4 channels supported.
        vol_size_voxel: Tuple[int, int, int] representing depth (z), height (y), and width (x) of the output volume.
        tile_min: Tensor of minimum tile indices per gaussian (from optim_to_render).
        tile_max: Tensor of maximum tile indices per gaussian (from optim_to_render).
        num_tiles_hit: Tensor storing number of tiles hit per gaussian (from optim_to_render).
        use_per_gaussian_backward: If True use the per-Gaussian backward CUDA kernel. Per-pixel otherwise.

    Returns:
        Voxel grid of shape (vol_d, vol_h, vol_w, channels) with accumulated Gaussian values.
    """

    return _VoxelizeGaussians.apply(
        pos3d_radii.contiguous(),
        conics.contiguous(),
        intensities.contiguous(),
        vol_size_voxel,
        tile_min,
        tile_max,
        num_tiles_hit,
        use_per_gaussian_backward,
    )


class _VoxelizeGaussians(Function):

    @staticmethod
    def forward(
        ctx,
        pos3d_radii: Float[Tensor, "*batch 4"],
        conics: Float[Tensor, "*batch 6"],
        intensities: Float[Tensor, "*batch channels"],
        vol_size_voxel: Tuple[int, int, int],
        tile_min: Int[Tensor, "*batch 3"],
        tile_max: Int[Tensor, "*batch 3"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        use_per_gaussian_backward: bool = False,
    ) -> Tensor:
        # Check if intensities has channel dimension
        if intensities.dim() == 1:
            intensities = intensities.unsqueeze(-1)

        num_gaussians = pos3d_radii.size(-2)
        ctx.vol_size_voxel = vol_size_voxel
        ctx.use_per_gaussian_backward = use_per_gaussian_backward

        if num_gaussians == 0:
            out_img = torch.zeros(
                vol_size_voxel[0],
                vol_size_voxel[1],
                vol_size_voxel[2],
                intensities.shape[-1],
                device=pos3d_radii.device,
                dtype=pos3d_radii.dtype,
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=pos3d_radii.device, dtype=torch.int32)
            tile_bins = torch.zeros(0, 3, device=pos3d_radii.device, dtype=torch.int32)
            ctx.num_intersects = 0
        else:
            cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
            num_intersects = cum_tiles_hit[-1].item()
            ctx.num_intersects = num_intersects

            if num_intersects < 1:
                out_img = torch.zeros(
                    vol_size_voxel[0],
                    vol_size_voxel[1],
                    vol_size_voxel[2],
                    intensities.shape[-1],
                    device=pos3d_radii.device,
                )
                gaussian_ids_sorted = torch.zeros(0, 1, device=pos3d_radii.device, dtype=torch.int32)
                tile_bins = torch.zeros(0, 3, device=pos3d_radii.device, dtype=torch.int32)
            else:

                gaussian_ids_sorted, tile_bins = bin_and_sort_gaussians(
                    num_gaussians,
                    num_intersects,
                    tile_min,
                    tile_max,
                    cum_tiles_hit,
                    vol_size_voxel,
                )

                out_img = _C.voxelize_forward(
                    vol_size_voxel,
                    gaussian_ids_sorted,
                    tile_bins,
                    pos3d_radii,
                    conics,
                    intensities)

        ctx.save_for_backward(
            pos3d_radii,
            conics,
            intensities,
            gaussian_ids_sorted,
            tile_bins,
        )

        return out_img

    @staticmethod
    def backward(ctx, vol_grad):
      
        vol_size_voxel = ctx.vol_size_voxel
        
        num_intersects = ctx.num_intersects
        pos3d_radii, conics, intensities, gaussian_ids_sorted, tile_bins = ctx.saved_tensors

        if num_intersects < 1:
            pos3d_grad = torch.zeros_like(pos3d_radii)
            conics_grad = torch.zeros_like(conics)
            intensities_grad = torch.zeros_like(intensities)
        else:
            if ctx.use_per_gaussian_backward:
                pos3d_grad, conics_grad, intensities_grad = _C.voxelize_backward_per_gaussian(
                    vol_size_voxel,
                    gaussian_ids_sorted,
                    tile_bins,
                    pos3d_radii,
                    conics,
                    intensities,
                    vol_grad.contiguous(),
                )
            else:
                pos3d_grad, conics_grad, intensities_grad = _C.voxelize_backward(
                    vol_size_voxel,
                    gaussian_ids_sorted,
                    tile_bins,
                    pos3d_radii,
                    conics,
                    intensities,
                    vol_grad.contiguous(),
                )

        return (
            pos3d_grad,
            conics_grad,
            intensities_grad,
            None,
            None,
            None,
            None,
            None,
        )
