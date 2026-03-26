#include "bindings.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cstdint>
#include "config.h"

#include "bin_and_sort_gaussians.cuh"
#include "optim_to_render.cuh"
#include "voxelize.cuh"

// Zero out an existing tensor in-place, respecting its device and dtype.
inline void zero_tensor(torch::Tensor &tensor) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return;
    }
    if (tensor.is_cuda()) {
        auto stream = at::cuda::getCurrentCUDAStream();
        CUDA_CALL(cudaMemsetAsync(
            tensor.data_ptr(),
            0,
            tensor.numel() * tensor.element_size(),
            stream.stream()));
    } else {
        tensor.zero_();
    }
}


/**
 * Kernel description:
 * Converts gaussian parameters from optimization format to rendering format.
 * Position from [0,1] to pixel-based, scale and quat to conics
 * Prepares radii for assigning gaussians to volume tiles.
 * Each thread processes one gaussian.
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
) {
    CHECK_INPUT(pos3d);
    CHECK_INPUT(scale3d);
    CHECK_INPUT(quat);
    CHECK_INPUT(intensities);

    const int channels = intensities.size(1);

    if (channels > MAX_CHANNELS) {
        throw std::runtime_error("Number of channels exceeds MAX_CHANNELS");
    }
    
    dim3 vol_size_dim3 = dim3(
        std::get<0>(vol_size_voxel),
        std::get<1>(vol_size_voxel),
        std::get<2>(vol_size_voxel)
    );
    dim3 tile_bounds_dim3 = dim3(
        (std::get<2>(vol_size_voxel) + BLOCK_X - 1) / BLOCK_X,
        (std::get<1>(vol_size_voxel) + BLOCK_Y - 1) / BLOCK_Y,
        (std::get<0>(vol_size_voxel) + BLOCK_Z - 1) / BLOCK_Z
    );
    float3 vol_size_world_float3 = make_float3(std::get<0>(vol_size_world), std::get<1>(vol_size_world), std::get<2>(vol_size_world));
    float3 vol_center_pos_float3 = make_float3(std::get<0>(vol_center_pos), std::get<1>(vol_center_pos), std::get<2>(vol_center_pos));

    torch::Tensor pos3d_radii_out = 
        torch::empty({num_gaussians, 4}, pos3d.options().dtype(torch::kFloat32));
    torch::Tensor conics_out = 
        torch::empty(
            {num_gaussians, 6}, 
            pos3d.options().dtype(torch::kFloat32)
        );
    torch::Tensor tile_min_out =
        torch::empty({num_gaussians, 3}, pos3d.options().dtype(torch::kInt32));
    torch::Tensor tile_max_out =
        torch::empty({num_gaussians, 3}, pos3d.options().dtype(torch::kInt32));
    torch::Tensor num_tiles_hit_out =
        torch::empty({num_gaussians}, pos3d.options().dtype(torch::kInt32));

    zero_tensor(conics_out);
    zero_tensor(pos3d_radii_out);
    zero_tensor(tile_min_out);
    zero_tensor(tile_max_out);
    zero_tensor(num_tiles_hit_out);

    switch(channels) {
        case 1:
            optim_to_render_forward<1><<<
                (num_gaussians + N_THREADS - 1) / N_THREADS,
                N_THREADS>>>(
                num_gaussians,
                (float3*)pos3d.contiguous().data_ptr<float>(),
                (float3*)scale3d.contiguous().data_ptr<float>(),
                (float4*)quat.contiguous().data_ptr<float>(),
                (float*)intensities.contiguous().data_ptr<float>(),
                vol_size_dim3,
                tile_bounds_dim3,
                vol_size_world_float3,
                vol_center_pos_float3,
                (float4*)pos3d_radii_out.contiguous().data_ptr<float>(),
                (float*)conics_out.contiguous().data_ptr<float>(),
                (int3*)tile_min_out.contiguous().data_ptr<int>(),
                (int3*)tile_max_out.contiguous().data_ptr<int>(),
                (int32_t*)num_tiles_hit_out.contiguous().data_ptr<int32_t>()
            );
            break;
        case 2:
            optim_to_render_forward<2><<<
                (num_gaussians + N_THREADS - 1) / N_THREADS,
                N_THREADS>>>(
                num_gaussians,
                (float3*)pos3d.contiguous().data_ptr<float>(),
                (float3*)scale3d.contiguous().data_ptr<float>(),
                (float4*)quat.contiguous().data_ptr<float>(),
                (float*)intensities.contiguous().data_ptr<float>(),
                vol_size_dim3,
                tile_bounds_dim3,
                vol_size_world_float3,
                vol_center_pos_float3,
                (float4*)pos3d_radii_out.contiguous().data_ptr<float>(),
                (float*)conics_out.contiguous().data_ptr<float>(),
                (int3*)tile_min_out.contiguous().data_ptr<int>(),
                (int3*)tile_max_out.contiguous().data_ptr<int>(),
                (int32_t*)num_tiles_hit_out.contiguous().data_ptr<int32_t>()
            );
            break;
        case 3:
            optim_to_render_forward<3><<<
                (num_gaussians + N_THREADS - 1) / N_THREADS,
                N_THREADS>>>(
                num_gaussians,
                (float3*)pos3d.contiguous().data_ptr<float>(),
                (float3*)scale3d.contiguous().data_ptr<float>(),
                (float4*)quat.contiguous().data_ptr<float>(),
                (float*)intensities.contiguous().data_ptr<float>(),
                vol_size_dim3,
                tile_bounds_dim3,
                vol_size_world_float3,
                vol_center_pos_float3,
                (float4*)pos3d_radii_out.contiguous().data_ptr<float>(),
                (float*)conics_out.contiguous().data_ptr<float>(),
                (int3*)tile_min_out.contiguous().data_ptr<int>(),
                (int3*)tile_max_out.contiguous().data_ptr<int>(),
                (int32_t*)num_tiles_hit_out.contiguous().data_ptr<int32_t>()
            );
            break;
        case 4:
            optim_to_render_forward<4><<<
                (num_gaussians + N_THREADS - 1) / N_THREADS,
                N_THREADS>>>(
                num_gaussians,
                (float3*)pos3d.contiguous().data_ptr<float>(),
                (float3*)scale3d.contiguous().data_ptr<float>(),
                (float4*)quat.contiguous().data_ptr<float>(),
                (float*)intensities.contiguous().data_ptr<float>(),
                vol_size_dim3,
                tile_bounds_dim3,
                vol_size_world_float3,
                vol_center_pos_float3,
                (float4*)pos3d_radii_out.contiguous().data_ptr<float>(),
                (float*)conics_out.contiguous().data_ptr<float>(),
                (int3*)tile_min_out.contiguous().data_ptr<int>(),
                (int3*)tile_max_out.contiguous().data_ptr<int>(),
                (int32_t*)num_tiles_hit_out.contiguous().data_ptr<int32_t>()
            );
            break;
        default:
            throw std::runtime_error("Unsupported number of channels");
    }
    return std::make_tuple(
        pos3d_radii_out,
        conics_out,
        tile_min_out,
        tile_max_out,
        num_tiles_hit_out
    );
}


/**
 * Kernel description:
 * Backward pass for optim_to_render_forward. 
 * Computes gradients w.r.t. input parameters (pos3d, scale3d, quat).
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
) {
    dim3 vol_size_dim3 = dim3(std::get<0>(vol_size_voxel), std::get<1>(vol_size_voxel), std::get<2>(vol_size_voxel));
    float3 vol_size_world_float3 = make_float3(std::get<0>(vol_size_world), std::get<1>(vol_size_world), std::get<2>(vol_size_world));
    float3 vol_center_pos_float3 = make_float3(std::get<0>(vol_center_pos), std::get<1>(vol_center_pos), std::get<2>(vol_center_pos));

    torch::Tensor pos3d_grad_out = 
        torch::empty({num_gaussians, 3}, pos3d.options().dtype(torch::kFloat32));
    zero_tensor(pos3d_grad_out);
    torch::Tensor scale3d_grad_out = 
        torch::empty({num_gaussians, 3}, pos3d.options().dtype(torch::kFloat32));
    zero_tensor(scale3d_grad_out);
    torch::Tensor quat_grad_out = 
        torch::empty({num_gaussians, 4}, pos3d.options().dtype(torch::kFloat32));
    zero_tensor(quat_grad_out);

    optim_to_render_backward<<<
        (num_gaussians + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_gaussians,
        (float3*)pos3d.contiguous().data_ptr<float>(),
        (float3*)scale3d.contiguous().data_ptr<float>(),
        (float4*)quat.contiguous().data_ptr<float>(),
        vol_size_dim3,
        vol_size_world_float3,
        vol_center_pos_float3,
        (float4*)pos3d_radii_grad_in.contiguous().data_ptr<float>(),
        (float*)conics_grad_in.contiguous().data_ptr<float>(),
        (float4*)pos3d_radii_out.contiguous().data_ptr<float>(),
        (float3*)pos3d_grad_out.contiguous().data_ptr<float>(),
        (float3*)scale3d_grad_out.contiguous().data_ptr<float>(),
        (float4*)quat_grad_out.contiguous().data_ptr<float>()
    );
    
    return std::make_tuple(
        pos3d_grad_out,
        scale3d_grad_out,
        quat_grad_out
    );

}

/**
 * Kernel description:
 * Map each intersection from tile ID to a gaussian ID. 
 * Outputs two arrays of size cum_tiles_hit[-1] containing tile-gaussian pairs.
 * Gaussian dimension is slower.
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
) {
    CHECK_INPUT(tile_min);
    CHECK_INPUT(tile_max);
    CHECK_INPUT(cum_tiles_hit);

    dim3 tile_bounds_dim3 = dim3(
        (std::get<2>(vol_size) + BLOCK_X - 1) / BLOCK_X,
        (std::get<1>(vol_size) + BLOCK_Y - 1) / BLOCK_Y,
        (std::get<0>(vol_size) + BLOCK_Z - 1) / BLOCK_Z
    );

    int num_tiles = 
        tile_bounds_dim3.x * tile_bounds_dim3.y * tile_bounds_dim3.z;

    auto options_i32 = tile_min.options().dtype(torch::kInt32);

    if (num_intersects <= 0) {
        torch::Tensor gaussian_ids_sorted = torch::empty({0}, options_i32);
        torch::Tensor tile_bins = torch::empty({num_tiles, 2}, options_i32);
        zero_tensor(tile_bins);
        return std::make_tuple(gaussian_ids_sorted, tile_bins);
    }

    auto tile_min_c = tile_min.contiguous();
    auto tile_max_c = tile_max.contiguous();
    auto cum_tiles_hit_c = cum_tiles_hit.contiguous();

    torch::Tensor isect_ids = 
        torch::empty({num_intersects}, options_i32);
    torch::Tensor gaussian_ids = 
        torch::empty({num_intersects}, options_i32);
    map_gaussian_to_intersects<<<
        (num_gaussians + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_gaussians,
        (int3*)tile_min_c.data_ptr<int>(),
        (int3*)tile_max_c.data_ptr<int>(),
        (int32_t*)cum_tiles_hit_c.data_ptr<int32_t>(),
        tile_bounds_dim3,
        (int32_t*)isect_ids.contiguous().data_ptr<int32_t>(),
        (int32_t*)gaussian_ids.contiguous().data_ptr<int32_t>()
    );
    torch::Tensor isect_ids_sorted = torch::empty({num_intersects}, options_i32);
    torch::Tensor gaussian_ids_sorted = torch::empty({num_intersects}, options_i32);

    size_t temp_storage_bytes = 0;
    auto stream = at::cuda::getCurrentCUDAStream();

    cub::DeviceRadixSort::SortPairs(
        nullptr,
        temp_storage_bytes,
        (int32_t*)isect_ids.data_ptr<int32_t>(),
        (int32_t*)isect_ids_sorted.data_ptr<int32_t>(),
        (int32_t*)gaussian_ids.data_ptr<int32_t>(),
        (int32_t*)gaussian_ids_sorted.data_ptr<int32_t>(),
        num_intersects,
        0,
        sizeof(int32_t) * 8,
        stream.stream());

    torch::Tensor temp_storage;
    uint8_t* temp_ptr = nullptr;
    if (temp_storage_bytes > 0) {
        temp_storage = torch::empty({static_cast<int64_t>(temp_storage_bytes)}, tile_min.options().dtype(torch::kUInt8));
        temp_ptr = temp_storage.data_ptr<uint8_t>();
    }

    cub::DeviceRadixSort::SortPairs(
        temp_ptr,
        temp_storage_bytes,
        (int32_t*)isect_ids.data_ptr<int32_t>(),
        (int32_t*)isect_ids_sorted.data_ptr<int32_t>(),
        (int32_t*)gaussian_ids.data_ptr<int32_t>(),
        (int32_t*)gaussian_ids_sorted.data_ptr<int32_t>(),
        num_intersects,
        0,
        sizeof(int32_t) * 8,
        stream.stream());

    torch::Tensor tile_bins =
        torch::empty({num_tiles, 2}, options_i32);
    zero_tensor(tile_bins);

    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_intersects,
        (int32_t*)isect_ids_sorted.data_ptr<int32_t>(),
        (int2*)tile_bins.data_ptr<int>()
    );    

    return std::make_tuple(gaussian_ids_sorted, tile_bins);
}

torch::Tensor voxelize_forward_torch(
    const std::tuple<int, int, int> vol_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &pos3d,
    const torch::Tensor &conics,
    const torch::Tensor &intensities
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(pos3d);
    CHECK_INPUT(conics);
    CHECK_INPUT(intensities);

    dim3 block_dim3 = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);

    dim3 vol_size_dim3 = dim3(
        std::get<2>(vol_size),
        std::get<1>(vol_size),
        std::get<0>(vol_size)
    );

        dim3 tile_bounds_dim3 = dim3(
        (vol_size_dim3.x + BLOCK_X - 1) / BLOCK_X,
        (vol_size_dim3.y + BLOCK_Y - 1) / BLOCK_Y,
        (vol_size_dim3.z + BLOCK_Z - 1) / BLOCK_Z
    );

    const int channels = intensities.size(1);

    if (channels > MAX_CHANNELS) {
        throw std::runtime_error("Number of channels exceeds MAX_CHANNELS");
    }

    torch::Tensor out_vol = 
        torch::empty(
            {vol_size_dim3.z, vol_size_dim3.y, vol_size_dim3.x, channels}, 
            pos3d.options().dtype(torch::kFloat32)
        );

    switch(channels) {
        case 1:
            voxelize_forward<1><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                vol_size_dim3,

                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float4*)pos3d.contiguous().data_ptr<float>(),
                (float*)conics.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                out_vol.contiguous().data_ptr<float>()
            );
            break;
        case 2:
            voxelize_forward<2><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                vol_size_dim3,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float4*)pos3d.contiguous().data_ptr<float>(),
                (float*)conics.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                out_vol.contiguous().data_ptr<float>()
            );
            break;
        case 3:
            voxelize_forward<3><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                vol_size_dim3,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float4*)pos3d.contiguous().data_ptr<float>(),
                (float*)conics.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                out_vol.contiguous().data_ptr<float>()
            );
            break;
        case 4:
            voxelize_forward<4><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                vol_size_dim3,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float4*)pos3d.contiguous().data_ptr<float>(),
                (float*)conics.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                out_vol.contiguous().data_ptr<float>()
            );
            break;
        default:
            throw std::runtime_error("Unsupported number of channels");
    }

    return out_vol;
}

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
) {
    CHECK_INPUT(pos3d);
    CHECK_INPUT(intensities);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(conics);
    CHECK_INPUT(vol_grad_in);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(conics);
    CHECK_INPUT(vol_grad_in);

    const int channels = intensities.size(1);
    const int num_gaussians = pos3d.size(0);

    if (channels > MAX_CHANNELS) {
        throw std::runtime_error("Number of channels exceeds MAX_CHANNELS");
    }

    const dim3 vol_size_dim3 = dim3(
        std::get<2>(vol_size),
        std::get<1>(vol_size),
        std::get<0>(vol_size)
    );

    dim3 tile_bounds_dim3 = dim3(
        (vol_size_dim3.x + BLOCK_X - 1) / BLOCK_X,
        (vol_size_dim3.y + BLOCK_Y - 1) / BLOCK_Y,
        (vol_size_dim3.z + BLOCK_Z - 1) / BLOCK_Z
    );

    dim3 block_dim3 = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);

    torch::Tensor pos3d_grad_out = 
        torch::empty({num_gaussians, 4}, pos3d.options());
    torch::Tensor conics_grad_out = 
        torch::empty({num_gaussians, 6}, pos3d.options());
    torch::Tensor intensities_grad_out = 
        torch::empty({num_gaussians, channels}, pos3d.options());

    zero_tensor(pos3d_grad_out);
    zero_tensor(conics_grad_out);
    zero_tensor(intensities_grad_out);

    switch (channels) {
        case 1:
            voxelize_backward<1><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                vol_size_dim3,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float4*)pos3d.contiguous().data_ptr<float>(),
                (float*)conics.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                vol_grad_in.contiguous().data_ptr<float>(),
                (float4*)pos3d_grad_out.contiguous().data_ptr<float>(),
                (float*)conics_grad_out.contiguous().data_ptr<float>(),
                intensities_grad_out.contiguous().data_ptr<float>()
            );
            break;
        case 2:
            voxelize_backward<2><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                vol_size_dim3,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float4*)pos3d.contiguous().data_ptr<float>(),
                (float*)conics.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                vol_grad_in.contiguous().data_ptr<float>(),
                (float4*)pos3d_grad_out.contiguous().data_ptr<float>(),
                (float*)conics_grad_out.contiguous().data_ptr<float>(),
                intensities_grad_out.contiguous().data_ptr<float>()
            );
            break;
        case 3:
            voxelize_backward<3><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                vol_size_dim3,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float4*)pos3d.contiguous().data_ptr<float>(),
                (float*)conics.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                vol_grad_in.contiguous().data_ptr<float>(),
                (float4*)pos3d_grad_out.contiguous().data_ptr<float>(),
                (float*)conics_grad_out.contiguous().data_ptr<float>(),
                intensities_grad_out.contiguous().data_ptr<float>()
            );
            break;
        case 4:
            voxelize_backward<4><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                vol_size_dim3,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float4*)pos3d.contiguous().data_ptr<float>(),
                (float*)conics.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                vol_grad_in.contiguous().data_ptr<float>(),
                (float4*)pos3d_grad_out.contiguous().data_ptr<float>(),
                (float*)conics_grad_out.contiguous().data_ptr<float>(),
                intensities_grad_out.contiguous().data_ptr<float>()
            );
            break;
        default:
            throw std::runtime_error("Unsupported number of channels");
    }

    return std::make_tuple(
        pos3d_grad_out,
        conics_grad_out,
        intensities_grad_out
    );
}

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
) {
    CHECK_INPUT(pos3d);
    CHECK_INPUT(intensities);

    const int channels = intensities.size(1);
    const int num_gaussians = pos3d.size(0);

    if (channels > MAX_CHANNELS) {
        throw std::runtime_error("Number of channels exceeds MAX_CHANNELS");
    }

    const dim3 vol_size_dim3 = dim3(
        std::get<2>(vol_size),
        std::get<1>(vol_size),
        std::get<0>(vol_size)
    );

    dim3 tile_bounds_dim3 = dim3(
        (vol_size_dim3.x + BLOCK_X - 1) / BLOCK_X,
        (vol_size_dim3.y + BLOCK_Y - 1) / BLOCK_Y,
        (vol_size_dim3.z + BLOCK_Z - 1) / BLOCK_Z
    );

    const int threads = N_THREADS;
    const int warps_per_block = threads / 32;

    torch::Tensor pos3d_grad_out =
        torch::empty({num_gaussians, 4}, pos3d.options());
    torch::Tensor conics_grad_out =
        torch::empty({num_gaussians, 6}, pos3d.options());
    torch::Tensor intensities_grad_out =
        torch::empty({num_gaussians, channels}, pos3d.options());

    zero_tensor(pos3d_grad_out);
    zero_tensor(conics_grad_out);
    zero_tensor(intensities_grad_out);

    const int num_tiles = tile_bounds_dim3.x * tile_bounds_dim3.y * tile_bounds_dim3.z;

    torch::Tensor bucket_counts;
    torch::Tensor bucket_offsets;
    int total_buckets = 0;

    if (num_tiles > 0) {
        bucket_counts = torch::empty({num_tiles}, tile_bins.options().dtype(torch::kInt32));
        bucket_offsets = torch::empty({num_tiles}, tile_bins.options().dtype(torch::kInt32));
        auto tile_bins_contig = tile_bins.contiguous();
        compute_bucket_counts_kernel<<<(num_tiles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
            num_tiles,
            (int2*)tile_bins_contig.data_ptr<int>(),
            (uint32_t*)bucket_counts.data_ptr<int>()
        );

        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_storage_bytes,
            (uint32_t*)bucket_counts.data_ptr<int>(),
            (uint32_t*)bucket_offsets.data_ptr<int>(),
            num_tiles
        );
        auto temp_storage = torch::empty(
            {static_cast<long>(temp_storage_bytes)},
            torch::TensorOptions().dtype(torch::kUInt8).device(bucket_counts.device())
        );
        cub::DeviceScan::InclusiveSum(
            temp_storage.data_ptr<uint8_t>(),
            temp_storage_bytes,
            (uint32_t*)bucket_counts.data_ptr<int>(),
            (uint32_t*)bucket_offsets.data_ptr<int>(),
            num_tiles
        );

        CUDA_CALL(cudaMemcpy(
            &total_buckets,
            ((uint32_t*)bucket_offsets.data_ptr<int>()) + num_tiles - 1,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost
        ));

        if (total_buckets > 0) {
            torch::Tensor bucket_to_tile = torch::empty({total_buckets}, tile_bins.options().dtype(torch::kInt32));
            build_bucket_to_tile_kernel<<<(num_tiles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
                num_tiles,
                (uint32_t*)bucket_counts.data_ptr<int>(),
                (uint32_t*)bucket_offsets.data_ptr<int>(),
                (uint32_t*)bucket_to_tile.data_ptr<int>()
            );

            auto gaussian_ids_sorted_contig = gaussian_ids_sorted.contiguous();
            auto pos3d_contig = pos3d.contiguous();
            auto conics_contig = conics.contiguous();
            auto intensities_contig = intensities.contiguous();
            auto vol_grad_contig = vol_grad_in.contiguous();

            const int blocks = (total_buckets + warps_per_block - 1) / warps_per_block;

            switch (channels) {
                case 1:
                    voxelize_backward_per_gaussian<1><<<blocks,threads>>>(
                        tile_bounds_dim3,
                        vol_size_dim3,
                        (int32_t*)gaussian_ids_sorted_contig.data_ptr<int32_t>(),
                        (int2*)tile_bins_contig.data_ptr<int>(),
                        (float4*)pos3d_contig.data_ptr<float>(),
                        (float*)conics_contig.data_ptr<float>(),
                        intensities_contig.data_ptr<float>(),
                        vol_grad_contig.data_ptr<float>(),
                        (uint32_t*)bucket_offsets.data_ptr<int>(),
                        (uint32_t*)bucket_to_tile.data_ptr<int>(),
                        total_buckets,
                        (float4*)pos3d_grad_out.contiguous().data_ptr<float>(),
                        (float*)conics_grad_out.contiguous().data_ptr<float>(),
                        intensities_grad_out.contiguous().data_ptr<float>()
                    );
                    break;
                case 2:
                    voxelize_backward_per_gaussian<2><<<blocks,threads>>>(
                        tile_bounds_dim3,
                        vol_size_dim3,
                        (int32_t*)gaussian_ids_sorted_contig.data_ptr<int32_t>(),
                        (int2*)tile_bins_contig.data_ptr<int>(),
                        (float4*)pos3d_contig.data_ptr<float>(),
                        (float*)conics_contig.data_ptr<float>(),
                        intensities_contig.data_ptr<float>(),
                        vol_grad_contig.data_ptr<float>(),
                        (uint32_t*)bucket_offsets.data_ptr<int>(),
                        (uint32_t*)bucket_to_tile.data_ptr<int>(),
                        total_buckets,
                        (float4*)pos3d_grad_out.contiguous().data_ptr<float>(),
                        (float*)conics_grad_out.contiguous().data_ptr<float>(),
                        intensities_grad_out.contiguous().data_ptr<float>()
                    );
                    break;
                case 3:
                    voxelize_backward_per_gaussian<3><<<blocks,threads>>>(
                        tile_bounds_dim3,
                        vol_size_dim3,
                        (int32_t*)gaussian_ids_sorted_contig.data_ptr<int32_t>(),
                        (int2*)tile_bins_contig.data_ptr<int>(),
                        (float4*)pos3d_contig.data_ptr<float>(),
                        (float*)conics_contig.data_ptr<float>(),
                        intensities_contig.data_ptr<float>(),
                        vol_grad_contig.data_ptr<float>(),
                        (uint32_t*)bucket_offsets.data_ptr<int>(),
                        (uint32_t*)bucket_to_tile.data_ptr<int>(),
                        total_buckets,
                        (float4*)pos3d_grad_out.contiguous().data_ptr<float>(),
                        (float*)conics_grad_out.contiguous().data_ptr<float>(),
                        intensities_grad_out.contiguous().data_ptr<float>()
                    );
                    break;
                case 4:
                    voxelize_backward_per_gaussian<4><<<blocks,threads>>>(
                        tile_bounds_dim3,
                        vol_size_dim3,
                        (int32_t*)gaussian_ids_sorted_contig.data_ptr<int32_t>(),
                        (int2*)tile_bins_contig.data_ptr<int>(),
                        (float4*)pos3d_contig.data_ptr<float>(),
                        (float*)conics_contig.data_ptr<float>(),
                        intensities_contig.data_ptr<float>(),
                        vol_grad_contig.data_ptr<float>(),
                        (uint32_t*)bucket_offsets.data_ptr<int>(),
                        (uint32_t*)bucket_to_tile.data_ptr<int>(),
                        total_buckets,
                        (float4*)pos3d_grad_out.contiguous().data_ptr<float>(),
                        (float*)conics_grad_out.contiguous().data_ptr<float>(),
                        intensities_grad_out.contiguous().data_ptr<float>()
                    );
                    break;
                default:
                    throw std::runtime_error("Unsupported number of channels");
            }
        }
    } else {
        bucket_counts = torch::empty({0}, tile_bins.options().dtype(torch::kInt32));
        bucket_offsets = torch::empty({0}, tile_bins.options().dtype(torch::kInt32));
    }

    return std::make_tuple(
        pos3d_grad_out,
        conics_grad_out,
        intensities_grad_out
    );
}
