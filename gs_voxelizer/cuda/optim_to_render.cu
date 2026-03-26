#include "optim_to_render.cuh"
#include "optim_to_render_helpers.cuh"
#include "config.h"
#include <algorithm>
#include <cstdint>
#include <cooperative_groups.h>
#include <iostream>
#include "../third_party/glm/glm/glm.hpp"
#include "../third_party/glm/glm/gtc/type_ptr.hpp"

namespace cg = cooperative_groups;

/**
 * Given a gaussian's pixel center and radius, compute the min and max tile
 * indices it intersects.
 * 
 * Inputs:
 * @param pix_center 3D Gaussian center in pixel coordinates (x,y,z).
 * @param pix_radius 3D Gaussian radius in pixels (x,y,z).
 * @param tile_bounds Volume dimensions in tiles (x,y,z).
 * 
 * @return Outputs:
 * @param tile_min Minimum tile indices (x,y,z) intersected by the gaussian.
 * @param tile_max Maximum tile indices (x,y,z) intersected by the gaussian.
 */
inline __device__  void get_tile_bbox(
    const float3 pix_center,
    const float3 pix_radius,
    const dim3 tile_bounds,
    int3 &tile_min,
    int3 &tile_max
) {
    tile_min.x = min((int)tile_bounds.x, max(0, (int)((pix_center.x - pix_radius.x) / (float)BLOCK_X)));
    tile_min.y = min((int)tile_bounds.y, max(0, (int)((pix_center.y - pix_radius.y) / (float)BLOCK_Y)));
    tile_min.z = min((int)tile_bounds.z, max(0, (int)((pix_center.z - pix_radius.z) / (float)BLOCK_Z)));

    tile_max.x = min((int)tile_bounds.x, max(0, (int)((pix_center.x + pix_radius.x + (float)(BLOCK_X - 1)) / (float)BLOCK_X)));
    tile_max.y = min((int)tile_bounds.y, max(0, (int)((pix_center.y + pix_radius.y + (float)(BLOCK_Y - 1)) / (float)BLOCK_Y)));
    tile_max.z = min((int)tile_bounds.z, max(0, (int)((pix_center.z + pix_radius.z + (float)(BLOCK_Z - 1)) / (float)BLOCK_Z)));
}

/**
 * Converts gaussian parameters from optimization format to rendering format.
 * Position from [0,1] to pixel-based, scale and quat to conics
 * Prepares radii for assigning gaussians to volume tiles.
 * Each thread processes one gaussian.
 */
template<int CHANNELS> __global__ void optim_to_render_forward(
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
    int32_t* __restrict__ num_tiles_hit_out) 
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    tile_min_out[idx] = make_int3(0, 0, 0);
    tile_max_out[idx] = make_int3(0, 0, 0);
    num_tiles_hit_out[idx] = 0;

    float3 voxel_size_world = {
        vol_size_world.x / (float)vol_size_voxel.x,
        vol_size_world.y / (float)vol_size_voxel.y,
        vol_size_world.z / (float)vol_size_voxel.z
    };

    // Compute pixel pos in pixel coordinates
    float3 pos_pixel = make_float3(
        (pos3d[idx].x - vol_center_pos.z + (vol_size_world.z * 0.5f)) / voxel_size_world.z,
        (pos3d[idx].y - vol_center_pos.y + (vol_size_world.y * 0.5f)) / voxel_size_world.y,
        (pos3d[idx].z - vol_center_pos.x + (vol_size_world.x * 0.5f)) / voxel_size_world.x
    );

    //Scale matrix
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = scale3d[idx].x;
    S[1][1] = scale3d[idx].y;
    S[2][2] = scale3d[idx].z;

    // Rotation matrix from quaternion
    glm::mat3 R = quat_to_rotmat(quat[idx]);

    // Covariance matrix
    glm::mat3 M = S * R;
    glm::mat3 Mt = glm::transpose(M);
    glm::mat3 cov = Mt * M;

    // Convert to voxel space
    glm::mat3 M2 =  glm::mat3(
		1.f / voxel_size_world.z, 0.0f, 0.0f,
		0.0f, 1.f  / voxel_size_world.y, 0.0f,
		0.0f, 0.0f, 1.f / voxel_size_world.x);
    cov = glm::transpose(M2) * cov * M2;

    // Determinant 
    float hata = cov[0][0];
	float hatb = cov[0][1];
	float hatc = cov[0][2];
	float hatd = cov[1][1];
	float hate = cov[1][2];
	float hatf = cov[2][2];
	float det = hata * hatd * hatf + 2 * hatb * hatc * hate - hata * hate * hate - hatf * hatb * hatb - hatd * hatc * hatc;
	// Sanity check
    if (det == 0.0f)
		return;
    float inv_det = 1.0f / det;

    // conic elemennts
    float inva = inv_det * (hatd * hatf - hate * hate);
    float invb = inv_det * (hatc * hate - hatb * hatf);
    float invc = inv_det * (hatb * hate - hatc * hatd);
    float invd = inv_det * (hata * hatf - hatc * hatc);
    float inve = inv_det * (hatb * hatc - hata * hate);
    float invf = inv_det * (hata * hatd - hatb * hatb);

    // Radius
    // Two approaches exist in other implementations: 
    // A) using the square root of largest covariance matrix eigenvalue 
    // B) using the largest scale
    // We use B to avoid computing eigenvalues, plus it produces equivalent results.
    float radius = ceil((3.f * max(scale3d[idx].x, max(scale3d[idx].y, scale3d[idx].z)))/min(voxel_size_world.z, min(voxel_size_world.y, voxel_size_world.x)));

    float mean_intensity = 0.f;
    #pragma unroll
    for (int i = 0; i < CHANNELS; ++i) {
        mean_intensity += intensities[idx * CHANNELS + i];
    }
    mean_intensity /= CHANNELS;
    const float opacity_power_threshold = logf(mean_intensity * ALPHA_THRESHOLD); 
	float extent =  min(3.f,sqrtf(2.0f * opacity_power_threshold));
	float3 radius3 = make_float3(max(0.0f, (sqrtf(cov[0][0])*extent)-0.5f), max(0.0f, (sqrtf(cov[1][1])*extent)-0.5f), max(0.0f, (sqrtf(cov[2][2])*extent)-0.5f)); 



    if (radius <= 0.f ||
        pos_pixel.x + radius < 0.f || pos_pixel.x - radius > vol_size_voxel.z ||
        pos_pixel.y + radius < 0.f || pos_pixel.y - radius > vol_size_voxel.y ||
        pos_pixel.z + radius < 0.f || pos_pixel.z - radius > vol_size_voxel.x) {
        // Outside volume or invalid radius, skip
        return;
    }

    int3 tile_min = make_int3(0, 0, 0);
    int3 tile_max = make_int3(0, 0, 0);
    get_tile_bbox(pos_pixel, radius3, tile_bounds, tile_min, tile_max);
    int32_t tile_vol = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y) * (tile_max.z - tile_min.z);
    if (tile_vol <= 0) {
        return;
    }

    // Write remaining outputs
    // Radius is appended to pos3d output. We want float4 anyway for memory alignment, so we just use the extra space.
    pos3d_radii_out[idx] = make_float4(pos_pixel.x, pos_pixel.y, pos_pixel.z, radius);

    tile_min_out[idx] = tile_min;
    tile_max_out[idx] = tile_max;
    num_tiles_hit_out[idx] = tile_vol;

    reinterpret_cast<float2*>(conics_out + 6 * idx)[0] = make_float2(inva, invb);
    reinterpret_cast<float2*>(conics_out + 6 * idx)[1] = make_float2(invc, invd);
    reinterpret_cast<float2*>(conics_out + 6 * idx)[2] = make_float2(inve, invf);
}

/** Returns the gradient of the covariance matrix with respect to the conic elements.
 * 
 * Inputs:
 * @param cov 3x3 covariance matrix.
 * @param M2 3x3 matrix for converting between world and voxel space scaling.
 * @param conics_grad_ab Gradient w.r.t. conic elements a and b (float2).
 * @param conics_grad_cd Gradient w.r.t. conic elements c and d (float2).
 * @param conics_grad_ef Gradient w.r.t. conic elements e and f (float2).
 * 
 * @return cov_grad 3x3 gradient of the covariance matrix.
 */

inline __device__ glm::mat3 get_cov_grad(
    glm::mat3& cov,
    glm::mat3& M2,
    const float2& conics_grad_ab,
    const float2& conics_grad_cd,
    const float2& conics_grad_ef
) {
    float hata = cov[0][0];
	float hatb = cov[0][1];
	float hatc = cov[0][2];
	float hatd = cov[1][1];
	float hate = cov[1][2];
	float hatf = cov[2][2];
	float denom = hata * hatd * hatf + 2 * hatb * hatc * hate - hata * hate * hate - hatf * hatb * hatb - hatd * hatc * hatc;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

    float dL_dhata = 0, dL_dhatb = 0, dL_dhatc = 0, dL_dhatd = 0, dL_dhate = 0, dL_dhatf = 0;

    glm::mat3 cov_grad = glm::mat3(0.f);

	if (denom2inv != 0)
	{   
        // Precompute common subexpressions for efficiency
		float denom_da = hatd * hatf - hate * hate;
		float denom_db = 2 * hatc * hate - 2 * hatf * hatb;
		float denom_dc = 2 * hatb * hate - 2 * hatd * hatc;
		float denom_dd = hata * hatf - hatc * hatc;
		float denom_de = 2 * hatb * hatc - 2 * hata * hate;
		float denom_df = hata * hatd - hatb * hatb;
		
		float ce_bf = hatc * hate - hatb * hatf; //b
		float be_cd = hatb * hate - hatc * hatd; //c
		float bc_ae = hatb * hatc - hata * hate; //e

        // Compute gradients w.r.t. conic elements
		dL_dhata = denom2inv * (-denom_da*denom_da*conics_grad_ab.x - ce_bf*denom_da*conics_grad_ab.y - be_cd*denom_da*conics_grad_cd.x + (hatf*denom-denom_dd*denom_da)*conics_grad_cd.y + (-hate*denom-bc_ae*denom_da)*conics_grad_ef.x + (hatd*denom-denom_df*denom_da)*conics_grad_ef.y);
		dL_dhatb = denom2inv * (-denom_da*denom_db*conics_grad_ab.x + (-hatf*denom-ce_bf*denom_db)*conics_grad_ab.y + (hate*denom-be_cd*denom_db)*conics_grad_cd.x - denom_dd*denom_db*conics_grad_cd.y + (hatc*denom-bc_ae*denom_db)*conics_grad_ef.x + (-2*hatb*denom-denom_df*denom_db)*conics_grad_ef.y);
		dL_dhatc = denom2inv * (-denom_da*denom_dc*conics_grad_ab.x + (hate*denom-ce_bf*denom_dc)*conics_grad_ab.y + (-hatd*denom-be_cd*denom_dc)*conics_grad_cd.x + (-2*hatc*denom-denom_dd*denom_dc)*conics_grad_cd.y + (hatb*denom-bc_ae*denom_dc)*conics_grad_ef.x - denom_df*denom_dc*conics_grad_ef.y);
		dL_dhatd = denom2inv * ((hatf*denom-denom_da*denom_dd)*conics_grad_ab.x - ce_bf*denom_dd*conics_grad_ab.y +(-hatc*denom-be_cd*denom_dd)*conics_grad_cd.x - denom_dd*denom_dd*conics_grad_cd.y - bc_ae*denom_dd*conics_grad_ef.x + (hata*denom-denom_df*denom_dd)*conics_grad_ef.y);
		dL_dhate = denom2inv * ((-2*hate*denom-denom_da*denom_de)*conics_grad_ab.x + (hatc*denom-ce_bf*denom_de)*conics_grad_ab.y + (hatb*denom-be_cd*denom_de)*conics_grad_cd.x - denom_dd*denom_de*conics_grad_cd.y + (-hata*denom-bc_ae*denom_de)*conics_grad_ef.x + -denom_df*denom_de*conics_grad_ef.y);
		dL_dhatf = denom2inv * ((hatd*denom-denom_da*denom_df)*conics_grad_ab.x + (-hatb*denom-ce_bf*denom_df)*conics_grad_ab.y - be_cd*denom_df*conics_grad_cd.x + (hata*denom-denom_dd*denom_df)*conics_grad_cd.y - bc_ae*denom_df*conics_grad_ef.x - denom_df*denom_df*conics_grad_ef.y);
        // Populate into a symmetric matrix
        const glm::mat3 dL_dhat(
            dL_dhata, dL_dhatb, dL_dhatc,
            dL_dhatb, dL_dhatd, dL_dhate,
            dL_dhatc, dL_dhate, dL_dhatf);

        cov_grad = M2 * dL_dhat * glm::transpose(M2);
        cov_grad = 0.5f * (cov_grad + glm::transpose(cov_grad)); // ensure symmetry

    }//Else zero gradient

    return cov_grad;
}


/**
 * Backward pass for optim_to_render_forward. 
 * Computes gradients w.r.t. input parameters (pos3d, scale3d, quat).
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
)

{

    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians || pos3d_radii_out[idx].w <= 0) return;

    const float2* conics_grad_ptr = (const float2*)conics_grad_in;

    const float2 conics_grad_ab = conics_grad_ptr[idx * 3 + 0];
    const float2 conics_grad_cd = conics_grad_ptr[idx * 3 + 1];
    const float2 conics_grad_ef = conics_grad_ptr[idx * 3 + 2];

    float3 voxel_size_world = {
    vol_size_world.x / vol_size_voxel.x,
    vol_size_world.y / vol_size_voxel.y,
    vol_size_world.z / vol_size_voxel.z
    };

    //Scale matrix
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = scale3d[idx].x;
    S[1][1] = scale3d[idx].y;
    S[2][2] = scale3d[idx].z;

    // Rotation matrix from quaternion
    glm::mat3 R = quat_to_rotmat(quat[idx]);

    // Covariance matrix
    glm::mat3 M = S * R;
    glm::mat3 Mt = glm::transpose(M);
    glm::mat3 cov = Mt * M;

    glm::mat3 M2 = glm::mat3(
        1.f / voxel_size_world.z, 0.f, 0.f,
        0.f, 1.f / voxel_size_world.y, 0.f,
        0.f, 0.f, 1.f / voxel_size_world.x
    );

    cov = glm::transpose(M2) * cov * M2;

    glm::mat3 cov_grad = get_cov_grad(
        cov,
        M2,
        conics_grad_ab,
        conics_grad_cd,
        conics_grad_ef
    );
    glm::mat3 dL_dSigma = glm::mat3(
		cov_grad[0][0], 0.5f * cov_grad[0][1], 0.5f * cov_grad[0][2],
		0.5f * cov_grad[1][0], cov_grad[1][1], 0.5f * cov_grad[1][2],
		0.5f * cov_grad[2][0], 0.5f * cov_grad[2][1], cov_grad[2][2]
	);

    // Rotation matrix gradient
    glm::mat3 R_grad = 2.0f * S * R *  dL_dSigma; 
    glm::mat3 R_grad_t = glm::transpose(R_grad);

    glm::mat3 R_t = glm::transpose(R);
    scale3d_grad_out[idx].x = glm::dot(R_t[0], R_grad_t[0]);
    scale3d_grad_out[idx].y = glm::dot(R_t[1], R_grad_t[1]);
    scale3d_grad_out[idx].z = glm::dot(R_t[2], R_grad_t[2]);

    R_grad_t[0] *= scale3d[idx].x;
    R_grad_t[1] *= scale3d[idx].y;
    R_grad_t[2] *= scale3d[idx].z;


    // Position gradient
    pos3d_grad_out[idx].x = pos3d_radii_grad_in[idx].x * voxel_size_world.z;
    pos3d_grad_out[idx].y = pos3d_radii_grad_in[idx].y * voxel_size_world.y;
    pos3d_grad_out[idx].z = pos3d_radii_grad_in[idx].z * voxel_size_world.x;


    // Quaternion gradients
    quat_grad_out[idx] = rotmat_grad_to_quat_grad(quat[idx], R_grad_t);
}

template __global__ void optim_to_render_forward<1>(const int, const float3*, const float3*, const float4*, const float*, 
    const dim3, const dim3, const float3, const float3, float4*, float*, int3*, int3*, int32_t*);
template __global__ void optim_to_render_forward<2>(const int, const float3*, const float3*, const float4*, const float*, 
    const dim3, const dim3, const float3, const float3, float4*, float*, int3*, int3*, int32_t*);
template __global__ void optim_to_render_forward<3>(const int, const float3*, const float3*, const float4*, const float*, 
    const dim3, const dim3, const float3, const float3, float4*, float*, int3*, int3*, int32_t*);
template __global__ void optim_to_render_forward<4>(const int, const float3*, const float3*, const float4*, const float*, 
    const dim3, const dim3, const float3, const float3, float4*, float*, int3*, int3*, int32_t*);

