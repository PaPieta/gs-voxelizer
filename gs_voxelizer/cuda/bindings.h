#pragma once

#include <tuple>
#include <torch/extension.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)


/**
 * optim_to_render_forward kernel binding. Kernel description:
 * Converts gaussian parameters from optimization format to rendering format.
 * Position from [0,1] to pixel-based, scale and quat to conics
 * Prepares radii for assigning gaussians to volume tiles.
 * Each thread processes one gaussian.
 * 
 * Inputs:
 * @param num_gaussians Number of gaussians.
 * @param pos3d 3D positions in optimization format [0-1] range.
 * @param scale3d 3D scale parameters.
 * @param quat Quaternions representing rotation (w, x, y, z order).
 * @param intensities Intensity values for each gaussian.
 * @param vol_size_voxel Dimensions of the volume in voxels (depth, height, width).
 * @param vol_size_world Physical size of the volume in world units (x, y, z).
 * @param vol_center_pos Center position of the volume in world coordinates (x, y, z).
 *
 * @return Outputs:
 * 4D vectors storing rendering positions (xyz) and in w component:
 *      Minimum enclosing radii of each gaussian for volume tile assignment.
 * Conic parameters a, b, c, d, e, f for each gaussian.
 * Minimum...
 * and maximum tile indices,
 * Number of tiles hit per gaussian.
 */
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
optim_to_render_forward_torch(
    const int num_gaussians,
    const torch::Tensor &pos3d,
    const torch::Tensor &scale3d,
    const torch::Tensor &quat,
    const torch::Tensor &intensities,
    const std::tuple<int, int, int> vol_size_voxel,
    const std::tuple<float, float, float> vol_size_world,
    const std::tuple<float, float, float> vol_center_pos
);


/**
 * optim_to_render_backward kernel binding. Kernel description:
 * Backward pass for optim_to_render_forward. 
 * Computes gradients w.r.t. input parameters (pos3d, scale3d, quat).
 * 
 * Inputs:
 * @param num_gaussians Number of gaussians.
 * @param pos3d 3D positions in optimization format [0-1] range.
 * @param scale3d 3D scale parameters.
 * @param quat Quaternions representing rotation (w, x, y, z order).
 * @param vol_size_voxel Dimensions of the volume in voxels (depth, height, width).
 * @param vol_size_world Physical size of the volume in world units (x, y, z).
 * @param vol_center_pos Center position of the volume in world coordinates (x, y, z).
 * @param pos3d_radii_grad_in Gradient w.r.t. output positions (4D vector with unused w component).
 * @param conics_grad_in Gradient w.r.t. output conics a, b, c, d, e, f.
 * @param pos3d_radii_out 4D vectors storing rendering positions (xyz) and radii in w component.
 * 
 * @return Outputs:
 * Gradient w.r.t. input positions.
 * Gradient w.r.t. input scales.
 * Gradient w.r.t. input quaternions.
 */
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
optim_to_render_backward_torch(
    const int num_gaussians,
    const torch::Tensor &pos3d,
    const torch::Tensor &scale3d,
    const torch::Tensor &quat,
    const std::tuple<int, int, int> vol_size_voxel,
    const std::tuple<float, float, float> vol_size_world,
    const std::tuple<float, float, float> vol_center_pos,
    const torch::Tensor &pos3d_radii_grad_in,
    const torch::Tensor &conics_grad_in,
    const torch::Tensor &pos3d_radii_out
);

/** 
 * Bin and sort gaussians based on their tile intersections.
 * Outputs sorted gaussian IDs and per-tile bins (start/end offsets).
 * 
 * Inputs:
 * @param num_gaussians Number of gaussians.
 * @param num_intersects Total number of gaussian-tile intersections.
 * @param tile_min Minimum tile indices (x,y,z) for each gaussian.
 * @param tile_max Maximum tile indices (x,y,z) for each gaussian.
 * @param cum_tiles_hit Cumulative count of tiles hit by gaussians.
 * @param vol_size Dimensions of the volume (depth, height, width).
 * 
 * @return Outputs:
 * Sorted gaussian IDs per intersection.
 * Tile bins storing start/end offsets of gaussians intersecting each tile.
 */
std::tuple<
    torch::Tensor,
    torch::Tensor> bin_and_sort_gaussians_torch(
    const int num_gaussians,
    const int num_intersects,
    const torch::Tensor &tile_min,
    const torch::Tensor &tile_max,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> vol_size
);

/**
 * voxelize_forward kernel binding. Kernel description:
 * Forward voxelization pass of a set of 3D Gaussians into a volumetric grid.
 * Each thread processes one voxel in the volume. Block size defined in config.h.
 * 
 * Inputs:
 * @param vol_size Dimensions of the volume (depth, height, width).
 * @param gaussian_ids_sorted Sorted gaussian IDs corresponding to intersections.
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * @param pos3d 4D vectors storing gaussian positions (xyz) and unused w component.
 * @param conics Conic parameters a, b, c, d, e, f for each gaussian.
 * @param intensities Intensity values for each gaussian.
 * 
 * @return Outputs:
 * Voxelized output volume (Z, Y, X)/(D, H, W).
 */

torch::Tensor voxelize_forward_torch(
    const std::tuple<int, int, int> vol_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &pos3d,
    const torch::Tensor &conics,
    const torch::Tensor &intensities
);


/**
 * voxelize_backward kernel binding. Kernel description:
 * Backward voxelization pass computing gradients w.r.t. input parameters.
 * Each thread processes one voxel in the volume. Block size defined in config.h.
 * 
 * Inputs:
 * @param vol_size Dimensions of the volume (depth, height, width).
 * @param gaussian_ids_sorted Sorted gaussian IDs corresponding to intersections.
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * @param pos3d 4D vectors storing gaussian positions (xyz) and unused w component.
 * @param conics Conic parameters a, b, c, d, e, f for each gaussian.
 * @param intensities Intensity values for each gaussian.
 * @param vol_grad_in Gradient w.r.t. output volume.
 * 
 * @return Outputs:
 * Gradient w.r.t. input positions (4D vector with unused w component (legacy from minimum enclosing radii)).
 * Gradient w.r.t. input conics a, b, c, d, e, f.
 * Gradient w.r.t. input intensities.
 */
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor> voxelize_backward_torch(
    const std::tuple<int, int, int> vol_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &pos3d,
    const torch::Tensor &conics,
    const torch::Tensor &intensities,
    const torch::Tensor &vol_grad_in
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor> voxelize_backward_per_gaussian_torch(
    const std::tuple<int, int, int> vol_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &pos3d,
    const torch::Tensor &conics,
    const torch::Tensor &intensities,
    const torch::Tensor &vol_grad_in
);
