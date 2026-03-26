"""Voxelizer python-cuda bindings with Python-based Gaussian sorting"""


from typing import Optional, Tuple

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gs_voxelizer.cuda as _C

def optim_to_render(
    pos3d: Float[Tensor, "*batch 3"],
    scale3d: Float[Tensor, "*batch 3"],
    quat: Float[Tensor, "*batch 4"],
    intensities: Float[Tensor, "*batch"],
    vol_size_voxel: Tuple[int, int, int],
    vol_size_world: Tuple[float, float, float],
    vol_center_pos: Tuple[float, float, float],
    pos3d_radii_buffer: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """ Converts Gaussians from optimization-oriented parameters to rendering-oriented.
    Args:
        pos3d: (*batch, 3) Tensor of 3D positions of Gaussians (0-1 range, will be multiplied by volume dimensions).
        scale3d: (*batch, 3) Tensor of scale parameters of Gaussians.
        quat: (*batch, 4) Tensor of quaternion orientations of Gaussians.
        intensities: (*batch, *channels) Tensor of Gaussian intensities. Multi-channel values must be preprocessed before calling.
        vol_size_voxel: Tuple[int, int, int] representing depth (z), height (y), and width (x) of the processed volume in voxels.
        vol_size_world: Tuple[float, float, float] representing the physical size of the volume in world units (z, y, x).
        vol_center_pos: Tuple[float, float, float] representing the center position of the volume in world coordinates (z, y, x).
        pos3d_radii_buffer: Optional preallocated tensor that will be overwritten with the returned positions/radii.
            Must be contiguous, of shape (*batch, 4) and the same dtype/device as the inputs. If None, a new tensor will be allocated.
    Returns:
        (*batch, 4) Tensor of Gaussian 3D positions adjusted for image dimensions and radii (w-component).
        (*batch, 6) Tensor of Conics (upper triangular matrix representation).
        (*batch, 3) Tensor of minimum tile indices (x, y, z) intersected by each gaussian.
        (*batch, 3) Tensor of maximum tile indices (x, y, z) intersected by each gaussian.
        (*batch,) Tensor with number of tiles hit by each gaussian.
    """


    return _OptimToRender.apply(
        pos3d.contiguous(),
        scale3d.contiguous(),
        quat.contiguous(),
        intensities.contiguous(),
        vol_size_voxel,
        vol_size_world,
        vol_center_pos,
        pos3d_radii_buffer,
    )

class _OptimToRender(Function):

    @staticmethod
    def forward(
        ctx,
        pos3d: Float[Tensor, "*batch 3"],
        scale3d: Float[Tensor, "*batch 3"],
        quat: Float[Tensor, "*batch 4"],
        intensities: Float[Tensor, "*batch"],
        vol_size_voxel: Tuple[int, int, int],
        vol_size_world: Tuple[float, float, float],
        vol_center_pos: Tuple[float, float, float],
        pos3d_radii_buffer: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        num_gaussians = pos3d.size(-2)   
        if num_gaussians < 1 or pos3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for pos3d: {pos3d.shape}")
        (
            pos3d_radii_out,
            conics_out,
            tile_min_out,
            tile_max_out,
            num_tiles_hit_out,
        ) = _C.optim_to_render_forward(
            num_gaussians,
            pos3d,
            scale3d,
            quat,
            intensities,
            vol_size_voxel,
            vol_size_world,
            vol_center_pos,
        )

        ctx.num_gaussians = num_gaussians
        ctx.vol_size_voxel = vol_size_voxel
        ctx.vol_size_world = vol_size_world
        ctx.vol_center_pos = vol_center_pos

        ctx.save_for_backward(
            pos3d,
            scale3d,
            quat,
            pos3d_radii_out,
        )
        if pos3d_radii_buffer is not None:
            if not pos3d_radii_buffer.is_contiguous():
                raise ValueError("pos3d_radii_buffer must be contiguous")
            if pos3d_radii_buffer.shape != pos3d_radii_out.shape:
                raise ValueError(
                    f"pos3d_radii_buffer shape mismatch: expected {pos3d_radii_out.shape}, "
                    f"got {pos3d_radii_buffer.shape}"
                )
            if pos3d_radii_buffer.device != pos3d_radii_out.device:
                raise ValueError("pos3d_radii_buffer must be on the same device as inputs")
            if pos3d_radii_buffer.dtype != pos3d_radii_out.dtype:
                raise ValueError("pos3d_radii_buffer must share dtype with inputs")
            pos3d_radii_buffer.copy_(pos3d_radii_out)
            pos3d_radii_return = pos3d_radii_buffer
        else:
            pos3d_radii_return = pos3d_radii_out

        return (
            pos3d_radii_return,
            conics_out,
            tile_min_out,
            tile_max_out,
            num_tiles_hit_out,
        )

    @staticmethod
    def backward(
        ctx,
        pos3d_radii_grad_in: Float[Tensor, "*batch 4"],
        conics_grad_in: Float[Tensor, "*batch 6"],
        tile_min_grad_in,
        tile_max_grad_in,
        num_tiles_grad_in,
    ) -> Tuple[Tensor, Tensor, Tensor, None, None, None, None, None]:
        _ = (tile_min_grad_in, tile_max_grad_in, num_tiles_grad_in)
        pos3d, scale3d, quat, pos3d_radii_out = ctx.saved_tensors

        grad_pos3d, grad_scale3d, grad_quat = _C.optim_to_render_backward(
            ctx.num_gaussians,
            pos3d,
            scale3d,
            quat,
            ctx.vol_size_voxel,
            ctx.vol_size_world,
            ctx.vol_center_pos,
            pos3d_radii_grad_in,
            conics_grad_in,
            pos3d_radii_out,
        )

        # Gradients for vol_h, vol_w, vol_d are None
        return grad_pos3d, grad_scale3d, grad_quat, None, None, None, None, None
