#include "voxelize.cuh"
#include "config.h"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>


namespace cg = cooperative_groups;

inline __device__ void warpSum4(float4& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
    val.w = cg::reduce(tile, val.w, cg::plus<float>());
}

inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile){
    val = cg::reduce(tile, val, cg::plus<float>());
}



/**
 * Forward voxelization pass of a set of 3D Gaussians into a volumetric grid.
 * Each thread processes one voxel in the volume. Block size defined in config.h.
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
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().z * tile_bounds.y * tile_bounds.x + 
            block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().z * block.group_dim().z  + block.thread_index().z;
    unsigned j =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned k =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)k +0.5f; // center of voxel
    float py = (float)j +0.5f;
    float pz = (float)i +0.5f;
    int32_t pix_id = i * vol_size.y * vol_size.x + j * vol_size.x + k;
     // Get the start and end indices of the gaussians in this tile
    bool inside = (i < vol_size.z && j < vol_size.y && k < vol_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches

    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float4 xyz_batch[BLOCK_SIZE];
    __shared__ float2 conic_batch_ab[BLOCK_SIZE];
    __shared__ float2 conic_batch_cd[BLOCK_SIZE];
    __shared__ float2 conic_batch_ef[BLOCK_SIZE];
    __shared__ float intensity_batch[BLOCK_SIZE][CHANNELS];

    const float2* conics_ptr = (const float2*)conics;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();

    // **** max 4 channels for speed ***
    float pix_out[CHANNELS] = {0.f};

    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xyz_batch[tr] = pos3d[g_id];
            conic_batch_ab[tr] = conics_ptr[3 * g_id + 0];
            conic_batch_cd[tr] = conics_ptr[3 * g_id + 1];
            conic_batch_ef[tr] = conics_ptr[3 * g_id + 2];
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) 
                intensity_batch[tr][c] = intensities[CHANNELS * g_id + c];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float2 conic_ab = conic_batch_ab[t];
            const float2 conic_cd = conic_batch_cd[t];
            const float2 conic_ef = conic_batch_ef[t];
            const float4 xyz = xyz_batch[t];
            const float3 delta = {xyz.x - px, xyz.y - py, xyz.z - pz};
            const float sigma = -0.5f * (conic_ab.x * delta.x * delta.x +
                                        conic_cd.y * delta.y * delta.y +
                                        conic_ef.y * delta.z * delta.z) - 
                                        (conic_ab.y * delta.x * delta.y +
                                        conic_cd.x * delta.x * delta.z +
                                        conic_ef.x * delta.y * delta.z);
            
            if (sigma > 0.f || isnan(sigma) || isinf(sigma)) {
                continue;
            }
            
            const float alpha = exp(sigma);
            float mean_intensity = 0.f;
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                mean_intensity += intensity_batch[t][c];
            } 
            mean_intensity /= CHANNELS;
            // int32_t g = id_batch[t];
            // const float vis = alpha;
            if (alpha * mean_intensity < 0.000001f) {
                continue;
            }
            
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                pix_out[c] += intensity_batch[t][c] * alpha;
            }
        }
    }

    if (inside) {
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            out_vol[pix_id * CHANNELS + c] = pix_out[c];
        }
    }

}

/**
 * Backward voxelization pass computing gradients w.r.t. input parameters.
 * Each thread processes one voxel in the volume. Block size defined in config.h.
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
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().z * tile_bounds.y * tile_bounds.x + 
            block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().z * block.group_dim().z  + block.thread_index().z;
    unsigned j =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned k =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)k +0.5f; // center of voxel
    float py = (float)j +0.5f;
    float pz = (float)i +0.5f;
   

    // Get the start and end indices of the gaussians in this tile
    bool inside = (i < vol_size.z && j < vol_size.y && k < vol_size.x);
    bool done = !inside;
    // if (!inside) return;

     // Clamp this value to the last pixel
    int32_t pix_id = i * vol_size.y * vol_size.x + j * vol_size.x + k;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches

    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float4 xyz_batch[BLOCK_SIZE];
    __shared__ float2 conic_batch_ab[BLOCK_SIZE];
    __shared__ float2 conic_batch_cd[BLOCK_SIZE];
    __shared__ float2 conic_batch_ef[BLOCK_SIZE];
    __shared__ float intensity_batch[BLOCK_SIZE][CHANNELS];

    const float2* conics_ptr = (const float2*)conics;

    // df/d_out for this pixel
    const float *vol_grad = inside ? &(vol_grad_in[CHANNELS * pix_id]) : nullptr;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing

    
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // copy vol_grad to local memory to avoid repeated global memory access
    float vol_grad_local[CHANNELS];
    if (inside) {
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            vol_grad_local[c] = vol_grad[c];
        }
    }

    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xyz_batch[tr] = pos3d[g_id];
            conic_batch_ab[tr] = conics_ptr[3 * g_id + 0];
            conic_batch_cd[tr] = conics_ptr[3 * g_id + 1];
            conic_batch_ef[tr] = conics_ptr[3 * g_id + 2];
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) 
                intensity_batch[tr][c] = intensities[CHANNELS * g_id + c];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        
        for (int t = 0; t < batch_size; ++t) {

            // Moved from previous for loop to ensure reset after one gaussian is wrong
            bool valid = inside;

            const float2 conic_ab = conic_batch_ab[t];
            const float2 conic_cd = conic_batch_cd[t];
            const float2 conic_ef = conic_batch_ef[t];
            const float4 xyz = xyz_batch[t];
            const float3 delta = {xyz.x - px, xyz.y - py, xyz.z - pz};
            const float sigma = -0.5f * (conic_ab.x * delta.x * delta.x +
                                        conic_cd.y * delta.y * delta.y +
                                        conic_ef.y * delta.z * delta.z) - 
                                        (conic_ab.y * delta.x * delta.y +
                                        conic_cd.x * delta.x * delta.z +
                                        conic_ef.x * delta.y * delta.z);
            
            if (sigma > 0.f || isnan(sigma) || isinf(sigma)) {
                valid = false;
            }

            float  intensities_grad_local[CHANNELS] = {0.f};
            float2 conics_grad_ab_local = {0.f, 0.f};
            float2 conics_grad_cd_local = {0.f, 0.f};
            float2 conics_grad_ef_local = {0.f, 0.f};
            float3 pos3d_grad_local = {0.f, 0.f, 0.f};

            float mean_intensity = 0.f;
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                mean_intensity += intensity_batch[t][c];
            } 
            mean_intensity /= CHANNELS;

            float d = exp(sigma);
            if (d * mean_intensity < 0.000001f) 
                valid = false;
            
            if (valid) {

                // update intensities_grad for this gaussian
                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c)
                    intensities_grad_local[c] = d * vol_grad_local[c];

                const float* intensities_temp = intensity_batch[t];
                // update sigma_grad for this gaussian
                float sigma_grad = 0.f;
                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c)
                    sigma_grad += intensities_temp[c] * vol_grad_local[c];
                sigma_grad *= d;
                
                // update v_conic for this gaussian
                conics_grad_ab_local = {-0.5f * sigma_grad * delta.x * delta.x,
                                          -sigma_grad * delta.x * delta.y};
                conics_grad_cd_local = {-sigma_grad * delta.x * delta.z,
                                          -0.5f * sigma_grad * delta.y * delta.y};
                conics_grad_ef_local = {-sigma_grad * delta.y * delta.z,
                                        -0.5f * sigma_grad * delta.z * delta.z};
                // update v_xy for this gaussian
                pos3d_grad_local = { -sigma_grad * (conic_ab.x * delta.x + conic_ab.y * delta.y + conic_cd.x * delta.z),
                                    -sigma_grad * (conic_ab.y * delta.x + conic_cd.y * delta.y + conic_ef.x * delta.z),
                                    -sigma_grad * (conic_cd.x * delta.x + conic_ef.x * delta.y + conic_ef.y * delta.z) };
            }
            
            // sum across the warp
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c)
                warpSum(intensities_grad_local[c], warp);
            warpSum2(conics_grad_ab_local, warp);
            warpSum2(conics_grad_cd_local, warp);
            warpSum2(conics_grad_ef_local, warp);
            warpSum3(pos3d_grad_local, warp);

            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* intensities_grad_ptr = (float*)(intensities_grad_out);
                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c)
                    atomicAdd(intensities_grad_ptr + CHANNELS * g + c, intensities_grad_local[c]);
                
                float* conics_grad_ptr = (float*)(conics_grad_out);
                atomicAdd(conics_grad_ptr + 6*g + 0, conics_grad_ab_local.x);
                atomicAdd(conics_grad_ptr + 6*g + 1, conics_grad_ab_local.y);
                atomicAdd(conics_grad_ptr + 6*g + 2, conics_grad_cd_local.x);
                atomicAdd(conics_grad_ptr + 6*g + 3, conics_grad_cd_local.y);
                atomicAdd(conics_grad_ptr + 6*g + 4, conics_grad_ef_local.x);
                atomicAdd(conics_grad_ptr + 6*g + 5, conics_grad_ef_local.y);

                float* pos3d_grad_ptr = (float*)(pos3d_grad_out);
                atomicAdd(pos3d_grad_ptr + 4*g + 0, pos3d_grad_local.x);
                atomicAdd(pos3d_grad_ptr + 4*g + 1, pos3d_grad_local.y);
                atomicAdd(pos3d_grad_ptr + 4*g + 2, pos3d_grad_local.z);
            }
        }
    }
}

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
) {
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warps_per_block = block.size() / warp.size();
    // Each warp processes one bucket (32 gaussians) assigned to a tile instead
    // of the voxel-centric work distribution used in voxelize_backward.
    const uint32_t global_bucket_idx = block.group_index().x * warps_per_block + warp.meta_group_rank();
    if (global_bucket_idx >= total_buckets)
        return;

    const uint32_t tile_id = bucket_to_tile[global_bucket_idx];
    // Tiles without contributors are filtered by range.y == range.x.
    const int2 range = tile_bins[tile_id];
    const int num_splats = range.y - range.x;
    if (num_splats <= 0)
        return;

    const uint32_t tile_bucket_base = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
    const uint32_t bucket_idx_in_tile = global_bucket_idx - tile_bucket_base;
    const uint32_t splat_idx_in_tile = bucket_idx_in_tile * warp.size() + warp.thread_rank();
    const bool valid_splat = splat_idx_in_tile < (uint32_t)num_splats;
    const uint32_t splat_idx_global = range.x + splat_idx_in_tile;

    const float2* conics_ptr = (const float2*)conics;
    // Map tile id back to spatial coordinates so we can iterate over voxels.
    const int tiles_per_layer = tile_bounds.x * tile_bounds.y;
    const int tile_z = tile_id / tiles_per_layer;
    const int tile_xy = tile_id - tile_z * tiles_per_layer;
    const int tile_y = tile_xy / tile_bounds.x;
    const int tile_x = tile_xy - tile_y * tile_bounds.x;
    const int tile_origin_x = tile_x * BLOCK_X;
    const int tile_origin_y = tile_y * BLOCK_Y;
    const int tile_origin_z = tile_z * BLOCK_Z;
    const int xy_plane = BLOCK_X * BLOCK_Y;

    int32_t g_id = 0;
    float4 xyz = {0.f, 0.f, 0.f, 0.f};
    float2 conic_ab = {0.f, 0.f};
    float2 conic_cd = {0.f, 0.f};
    float2 conic_ef = {0.f, 0.f};
    float gaussian_intensity[CHANNELS] = {0.f};
    float mean_intensity = 0.f;

    // Each warp processes one gaussian-tile pair, so we only need to load one
    if (valid_splat) {
        g_id = gaussian_ids_sorted[splat_idx_global];
        xyz = pos3d[g_id];
        conic_ab = conics_ptr[3 * g_id + 0];
        conic_cd = conics_ptr[3 * g_id + 1];
        conic_ef = conics_ptr[3 * g_id + 2];
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            gaussian_intensity[c] = intensities[CHANNELS * g_id + c];
            mean_intensity += gaussian_intensity[c];
        }
        mean_intensity /= CHANNELS;
    }

    float intensities_grad_local[CHANNELS] = {0.f};
    float2 conics_grad_ab_local = {0.f, 0.f};
    float2 conics_grad_cd_local = {0.f, 0.f};
    float2 conics_grad_ef_local = {0.f, 0.f};
    float3 pos3d_grad_local = {0.f, 0.f, 0.f};

    float px = 0.f;
    float py = 0.f;
    float pz = 0.f;
    int voxel_active = 0;
    float voxel_grad[CHANNELS] = {0.f};

    // Simulate the shuffle-based pixel traversal from the renderer: each pass
    // shifts the voxel state down the warp while lane 0 brings in the next
    // voxel for this tile. This lets us reuse the same dataflow as PerGaussian
    // rasterization without storing per-voxel state in shared memory.
    const int pipeline_iters = BLOCK_SIZE + warp.size() - 1;
    for (int iter = 0; iter < pipeline_iters; ++iter) {
        voxel_active = warp.shfl_up(voxel_active, 1);
        px = warp.shfl_up(px, 1);
        py = warp.shfl_up(py, 1);
        pz = warp.shfl_up(pz, 1);
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c)
            voxel_grad[c] = warp.shfl_up(voxel_grad[c], 1);

        const int idx = iter - warp.thread_rank();
        const bool idx_in_range = (idx >= 0) && (idx < BLOCK_SIZE);

        if (idx_in_range && warp.thread_rank() == 0) {
            // Lane 0 computes global voxel coordinates and pulls gradients.
            const int local_z = idx / xy_plane;
            const int rem = idx - local_z * xy_plane;
            const int local_y = rem / BLOCK_X;
            const int local_x = rem - local_y * BLOCK_X;

            const int global_x = tile_origin_x + local_x;
            const int global_y = tile_origin_y + local_y;
            const int global_z = tile_origin_z + local_z;

            const bool inside = (global_x < vol_size.x) &&
                                (global_y < vol_size.y) &&
                                (global_z < vol_size.z);
            voxel_active = inside ? 1 : 0;
            if (inside) {
                px = (float)global_x + 0.5f;
                py = (float)global_y + 0.5f;
                pz = (float)global_z + 0.5f;
                const int pix_id = global_z * vol_size.y * vol_size.x +
                                   global_y * vol_size.x + global_x;
                const float* grad_ptr = vol_grad_in + CHANNELS * pix_id;
                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c)
                    voxel_grad[c] = grad_ptr[c];
            } else {
                px = py = pz = 0.f;
                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c)
                    voxel_grad[c] = 0.f;
            }
        }

        const bool process = valid_splat && idx_in_range && voxel_active;
        // The original voxel-wise kernel accumulated into shared memory per
        // pixel; here we only accumulate when both a voxel and gaussian are
        // valid for this warp.
        if (!process)
            continue;

        const float3 delta = {xyz.x - px, xyz.y - py, xyz.z - pz};
        const float sigma = -0.5f * (conic_ab.x * delta.x * delta.x +
                                     conic_cd.y * delta.y * delta.y +
                                     conic_ef.y * delta.z * delta.z) -
                            (conic_ab.y * delta.x * delta.y +
                             conic_cd.x * delta.x * delta.z +
                             conic_ef.x * delta.y * delta.z);

        if (sigma > 0.f || isnan(sigma) || isinf(sigma))
            continue;

        const float d = exp(sigma);
        if (d * mean_intensity < 0.000001f)
            continue;

        float sigma_grad = 0.f;
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            const float grad = voxel_grad[c];
            intensities_grad_local[c] += d * grad;
            sigma_grad += gaussian_intensity[c] * grad;
        }
        sigma_grad *= d;

        conics_grad_ab_local.x += -0.5f * sigma_grad * delta.x * delta.x;
        conics_grad_ab_local.y += -sigma_grad * delta.x * delta.y;
        conics_grad_cd_local.x += -sigma_grad * delta.x * delta.z;
        conics_grad_cd_local.y += -0.5f * sigma_grad * delta.y * delta.y;
        conics_grad_ef_local.x += -sigma_grad * delta.y * delta.z;
        conics_grad_ef_local.y += -0.5f * sigma_grad * delta.z * delta.z;

        pos3d_grad_local.x += -sigma_grad * (conic_ab.x * delta.x +
                                             conic_ab.y * delta.y +
                                             conic_cd.x * delta.z);
        pos3d_grad_local.y += -sigma_grad * (conic_ab.y * delta.x +
                                             conic_cd.y * delta.y +
                                             conic_ef.x * delta.z);
        pos3d_grad_local.z += -sigma_grad * (conic_cd.x * delta.x +
                                             conic_ef.x * delta.y +
                                             conic_ef.y * delta.z);
    }

    if (valid_splat) {
        // Unlike voxelize_backward (which atomically adds per pixel), each warp
        // accumulates a gaussian's entire contribution before touching global
        // memory, so we only emit one set of atomics per gaussian.
        float* intensities_grad_ptr = intensities_grad_out + CHANNELS * g_id;
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c)
            atomicAdd(intensities_grad_ptr + c, intensities_grad_local[c]);

        float* conics_grad_ptr = conics_grad_out + 6 * g_id;
        atomicAdd(conics_grad_ptr + 0, conics_grad_ab_local.x);
        atomicAdd(conics_grad_ptr + 1, conics_grad_ab_local.y);
        atomicAdd(conics_grad_ptr + 2, conics_grad_cd_local.x);
        atomicAdd(conics_grad_ptr + 3, conics_grad_cd_local.y);
        atomicAdd(conics_grad_ptr + 4, conics_grad_ef_local.x);
        atomicAdd(conics_grad_ptr + 5, conics_grad_ef_local.y);

        float* pos3d_grad_ptr = (float*)(pos3d_grad_out);
        atomicAdd(pos3d_grad_ptr + 4 * g_id + 0, pos3d_grad_local.x);
        atomicAdd(pos3d_grad_ptr + 4 * g_id + 1, pos3d_grad_local.y);
        atomicAdd(pos3d_grad_ptr + 4 * g_id + 2, pos3d_grad_local.z);
    }
}

// Kernel to compute the number of gaussians in each bucket. 
__global__ void compute_bucket_counts_kernel(
    const int num_tiles,
    const int2* __restrict__ tile_bins,
    uint32_t* __restrict__ bucket_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tiles)
        return;
    const int2 range = tile_bins[idx];
    int splats = range.y - range.x;
    splats = splats < 0 ? 0 : splats;
    // Each bucket can hold up to 32 gaussians, mirroring the rendering path.
    bucket_counts[idx] = (uint32_t)((splats + 31) / 32);
}

// Kernel to build a mapping from buckets to tiles based on the bucket counts and tile_bins
__global__ void build_bucket_to_tile_kernel(
    const int num_tiles,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t* __restrict__ bucket_to_tile
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tiles)
        return;
    const uint32_t count = bucket_counts[idx];
    if (count == 0 || bucket_to_tile == nullptr)
        return;
    const uint32_t base = idx == 0 ? 0 : bucket_offsets[idx - 1];
    // Fill the mapping so warp i directly knows which tile to sample from.
    for (uint32_t b = 0; b < count; ++b) {
        bucket_to_tile[base + b] = (uint32_t)idx;
    }
}

template __global__ void voxelize_forward<1>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,float*);

template __global__ void voxelize_forward<2>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,float*);

template __global__ void voxelize_forward<3>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,float*);

template __global__ void voxelize_forward<4>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,float*);

template __global__ void voxelize_backward<1>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,const float*,float4*,float*,float*);

template __global__ void voxelize_backward<2>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,const float*,float4*,float*,float*);

template __global__ void voxelize_backward<3>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,const float*,float4*,float*,float*);

template __global__ void voxelize_backward<4>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,const float*,float4*,float*,float*);

template __global__ void voxelize_backward_per_gaussian<1>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,const float*,const uint32_t*,const uint32_t*,const uint32_t,float4*,float*,float*);

template __global__ void voxelize_backward_per_gaussian<2>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,const float*,const uint32_t*,const uint32_t*,const uint32_t,float4*,float*,float*);

template __global__ void voxelize_backward_per_gaussian<3>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,const float*,const uint32_t*,const uint32_t*,const uint32_t,float4*,float*,float*);

template __global__ void voxelize_backward_per_gaussian<4>(const dim3,const dim3,const int32_t*,
    const int2*,const float4*,const float*,const float*,const float*,const uint32_t*,const uint32_t*,const uint32_t,float4*,float*,float*);
