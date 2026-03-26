#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>


/**
 * Forward Voxelization kernel.
 * 
 * Forward voxelization pass of a set of 3D Gaussians into a volumetric grid.
 * Each thread processes one voxel in the volume. Block size defined in config.h.
 * 
 * Template parameter:
 * @tparam CHANNELS Number of channels in the output volume.
 * Inputs:
 * @param tile_bounds Dimensions of the volume in tiles (width, height, depth).
 * @param vol_size Dimensions of the volume in voxels (width, height, depth).
 * @param gaussian_ids_sorted Sorted gaussian IDs corresponding to intersections
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * @param pos3d 3D positions of gaussians (xyz), with minimum enclosing radii inw (here unused).
 * @param conics  Conic parameters a, b, c, d, e, f for each gaussian (float2 pairs).
 * @param intensities Intensity values for each gaussian and channel.
 * 
 * @return Outputs:
 * @param out_vol Voxelized output volume.
 */
template<int CHANNELS> __global__ void voxelize_forward(
    const dim3 tile_bounds,
    const dim3 vol_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float4* __restrict__ pos3d,
    const float* __restrict__ conics,
    const float* __restrict__ intensities,
    float* __restrict__ out_vol
);

/**
 * Backward Voxelization kernel.
 * 
 * Backward voxelization pass computing gradients w.r.t. input parameters.
 * Each thread processes one voxel in the volume. Block size defined in config.h.
 * 
 * Template parameter:
 * @tparam CHANNELS Number of channels in the output volume.
 * 
 * Inputs:
 * @param tile_bounds Dimensions of the volume in tiles (width, height, depth).
 * @param vol_size Dimensions of the volume in voxels (width, height, depth).
 * @param gaussian_ids_sorted Sorted gaussian IDs corresponding to intersections
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * @param pos3d 3D positions of gaussians (xyz), with minimum enclosing radii in w (here unused).
 * @param conics  Conic parameters a, b, c, d, e, f for each gaussian (float2 pairs).
 * @param intensities Intensity values for each gaussian and channel.
 * @param vol_grad_in Gradient w.r.t. output volume.
 * 
 * @return Outputs:
 * @param pos3d_grad_out Gradient w.r.t. input positions.
 * @param conics_grad_out Gradient w.r.t. input conics a, b, c, d, e, f.
 * @param intensities_grad_out Gradient w.r.t. input intensities.
 */
template<int CHANNELS> __global__ void voxelize_backward(
    const dim3 tile_bounds,
    const dim3 vol_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float4* __restrict__ pos3d,
    const float* __restrict__ conics,
    const float* __restrict__ intensities,
    const float* __restrict__ vol_grad_in,
    float4* __restrict__ pos3d_grad_out,
    float* __restrict__ conics_grad_out,
    float* __restrict__ intensities_grad_out
);

/**
 * Alternative backward voxelization kernel where work is distributed per-gaussian.
 * Each warp in a block processes one gaussian-tile pair and iterates over voxels.
 * 
 * Template parameter:
 * @tparam CHANNELS Number of channels in the output volume.
 * 
 * Inputs:
 * @param tile_bounds Dimensions of the volume in tiles (width, height, depth).
 * @param vol_size Dimensions of the volume in voxels (width, height, depth).
 * @param gaussian_ids_sorted Sorted gaussian IDs corresponding to intersections
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * @param pos3d 3D positions of gaussians (xyz), with minimum enclosing radii in w (here unused).
 * @param conics  Conic parameters a, b, c, d, e, f for each gaussian (float2 pairs).
 * @param intensities Intensity values for each gaussian and channel.
 * @param vol_grad_in Gradient w.r.t. output volume.
 * @param per_tile_bucket_offset Precomputed offsets to access tile bins from bucket_to_tile.
 * @param bucket_to_tile Mapping from bucket index to tile index for non-empty tiles.
 * @param total_buckets Total number of non-empty buckets (tiles).
 * 
 * @return Outputs:
 * @param pos3d_grad_out Gradient w.r.t. input positions.
 * @param conics_grad_out Gradient w.r.t. input conics a, b, c, d, e, f.
 * @param intensities_grad_out Gradient w.r.t. input intensities.
 */
template<int CHANNELS> __global__ void voxelize_backward_per_gaussian(
    const dim3 tile_bounds,
    const dim3 vol_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float4* __restrict__ pos3d,
    const float* __restrict__ conics,
    const float* __restrict__ intensities,
    const float* __restrict__ vol_grad_in,
    const uint32_t* __restrict__ per_tile_bucket_offset,
    const uint32_t* __restrict__ bucket_to_tile,
    const uint32_t total_buckets,
    float4* __restrict__ pos3d_grad_out,
    float* __restrict__ conics_grad_out,
    float* __restrict__ intensities_grad_out
);

/** Kernel to compute the number of gaussians in each bucket. 
 * 
 * Inputs:
 * @param num_tiles Number of tiles.
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * 
 * Output:
 * @param bucket_counts Number of gaussians in each bucket.
 */
__global__ void compute_bucket_counts_kernel(
    const int num_tiles,
    const int2* __restrict__ tile_bins,
    uint32_t* __restrict__ bucket_counts
);

/** Kernel to build a mapping from buckets to tiles based on the bucket counts and tile_bins.
 * 
 * Inputs:
 * @param num_tiles Number of tiles.
 * @param bucket_counts Number of gaussians in each bucket.
 * @param bucket_offsets Offsets for each bucket to access the output array.
 * 
 * Output:
 * @param bucket_to_tile Mapping from bucket index to tile index for non-empty buckets.
 */
__global__ void build_bucket_to_tile_kernel(
    const int num_tiles,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t* __restrict__ bucket_to_tile
);
