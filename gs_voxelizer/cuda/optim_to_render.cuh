#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ALPHA_THRESHOLD (1/0.000001f)

/**
 * Converts gaussian parameters from optimization format to rendering format.
 * Position from [0,1] to pixel-based, scale and quat to conics
 * Prepares radii for assigning gaussians to volume tiles.
 * Each thread processes one gaussian.
 * 
 * Template parameter:
 * @tparam CHANNELS Number of channels in the input intensities.
 * 
 * Inputs:
 * @param num_gaussians Number of gaussians.
 * @param pos3d 3D positions in optimization format [0-1] range.
 * @param scale3d 3D scale parameters.
 * @param quat Quaternions representing rotation (w, x, y, z order).
 * @param intensities Intensity values per gaussian (single channel expected).
 * @param vol_size_voxel Volume dimensions as dim3 (width, height, depth) in voxels.
 * @param tile_bounds Volume dimensions in tiles (x,y,z).
 * @param vol_size_world Volume dimensions as float3 (width, height, depth) in world units.
 * @param vol_center_pos Volume center position in world coordinates.
 * 
 * @return Outputs:
 * @param pos3d_radii_out 4D vector storing positions (xyz) and radius in the w component.
 * @param conics_out Conic parameters in rendering format (float2 triples).
 * @param tile_min_out Minimum tile indices (x,y,z) intersected by each gaussian.
 * @param tile_max_out Maximum tile indices (x,y,z) intersected by each gaussian.
 * @param num_tiles_hit_out Number of tiles touched by each gaussian.
 */
template <int CHANNELS> __global__ void optim_to_render_forward(
    const int num_gaussians,
    const float3* __restrict__ pos3d,
    const float3* __restrict__ scale3d,
    const float4* __restrict__ quat,
    const float* __restrict__ intensities,
    const dim3 vol_size_voxel,
    const dim3 tile_bounds,
    const float3 vol_size_world,
    const float3 vol_center_pos,
    float4* __restrict__ pos3d_radii_out,
    float* __restrict__ conics_out,
    int3* __restrict__ tile_min_out,
    int3* __restrict__ tile_max_out,
    int32_t* __restrict__ num_tiles_hit_out
);

/**
 * Backward pass for optim_to_render_forward. 
 * Computes gradients w.r.t. input parameters (pos3d, scale3d, quat).
 * 
 * Inputs:
 * @param num_gaussians Number of gaussians.
 * @param pos3d 3D positions in optimization format [0-1] range.
 * @param scale3d 3D scale parameters.
 * @param quat Quaternions representing rotation (w, x, y, z order).
 * @param vol_size_voxel Volume dimensions as dim3 (width, height, depth) in voxels.
 * @param vol_size_world Volume dimensions as float3 (width, height, depth) in world units.
 * @param vol_center_pos Volume center position in world coordinates.
 * @param pos3d_radii_grad_in Gradient w.r.t. output positions.
 * @param conics_grad_in Gradient w.r.t. output conics (float2 triples).
 * @param pos3d_radii_out 4D vector storing positions in rendering format (xyz) and radius in the w component.
 * 
 * @return Outputs:
 * @param pos3d_grad_out Gradient w.r.t. input positions.
 * @param scale3d_grad_out Gradient w.r.t. input scales.
 * @param quat_grad_out Gradient w.r.t. input quaternions.
 */
__global__ void optim_to_render_backward(
    const int num_gaussians,
    const float3* __restrict__ pos3d,
    const float3* __restrict__ scale3d,
    const float4* __restrict__ quat,
    const dim3 vol_size_voxel,
    const float3 vol_size_world,
    const float3 vol_center_pos,
    const float4* __restrict__ pos3d_radii_grad_in,
    const float* __restrict__ conics_grad_in,
    const float4* __restrict__ pos3d_radii_out,
    float3* __restrict__ pos3d_grad_out,
    float3* __restrict__ scale3d_grad_out,
    float4* __restrict__ quat_grad_out
);
