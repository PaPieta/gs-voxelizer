#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/**
 * Map each intersection from tile ID to a gaussian ID. 
 * Outputs two arrays of size cum_tiles_hit[-1] containing tile-gaussian pairs.
 * Gaussian dimension is slower.
 * 
 * Inputs:
 * @param num_gaussians Number of gaussians.
 * @param tile_min Minimum tile indices (x,y,z) intersected by each gaussian.
 * @param tile_max Maximum tile indices (x,y,z) intersected by each gaussian.
 * @param cum_tiles_hit Cumulative sum of number of tiles hit by each gaussian.
 * @param tile_bounds Dimensions of the volume in tiles (width, height, depth).
 * 
 * @return Outputs:
 * @param isect_ids Tile IDs for each intersection.
 * @param gaussian_ids Corresponding Gaussian IDs.
 */
__global__ void map_gaussian_to_intersects(
    const int num_gaussians,
    int3* __restrict__ tile_min,
    int3* __restrict__ tile_max,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    int32_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
);

/**
 * Given sorted intersection IDs, find its bins (how many gaussians in each tile).
 * 
 * Inputs:
 * @param num_intersects Number of intersections.
 * @param isect_ids_sorted !Sorted! tile IDs for each intersection.
 * 
 * @return Outputs:
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 */
__global__ void get_tile_bin_edges(
    const int num_intersects, 
    const int32_t* __restrict__ isect_ids_sorted, 
    int2* __restrict__ tile_bins
);
