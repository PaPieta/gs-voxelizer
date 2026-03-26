#include <iostream>
#include <cuda_runtime.h>
#include "../third_party/glm/glm/glm.hpp"
#include "../third_party/glm/glm/gtc/type_ptr.hpp"
#include <iostream>

/**
 * Convert quaternion to rotation matrix.
 * 
 * Input: 
 * @param quat quaternion (w,x,y,z)
 * 
 * @return Output: rotation matrix (column-major)
 */
inline __device__ glm::mat3 quat_to_rotmat(const float4 quat) {
    
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

	// Baseline doesn't normalize quaternions, use below for complete equivalence
	// float w = quat.x;
    // float x = quat.y;
    // float y = quat.z;
    // float z = quat.w;

    // glm matrices are column-major
    return glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z), 2.f * (x * z + w * y), 
		2.f * (x * y + w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - w * x),
        2.f * (x * z - w * y), 2.f * (y * z + w * x), 1.f - 2.f * (x * x + y * y)
    );
}



/**
 * Converts rotation matrix gradient to quaternion gradient.
 * 
 * Inputs:
 * @param quat Quaternion (w,x,y,z).
 * @param v_R Rotation matrix gradient (column-major).
 * 
 * @return Output:
 * @param v_quat Quaternion gradient (w,x,y,z).
 */
inline __device__ float4 rotmat_grad_to_quat_grad(
    const float4 quat, const glm::mat3 v_R_t
) {
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    // Baseline doesn't normalize quaternions, use below for complete equivalence
	// float w = quat.x;
    // float x = quat.y;
    // float y = quat.z;
    // float z = quat.w;

    float4 v_quat;
    v_quat.x = 2 * z * (v_R_t[0][1] - v_R_t[1][0]) + 2 * y * (v_R_t[2][0] - v_R_t[0][2]) + 2 * x * (v_R_t[1][2] - v_R_t[2][1]);
	v_quat.y = 2 * y * (v_R_t[1][0] + v_R_t[0][1]) + 2 * z * (v_R_t[2][0] + v_R_t[0][2]) + 2 * w * (v_R_t[1][2] - v_R_t[2][1]) - 4 * x * (v_R_t[2][2] + v_R_t[1][1]);
	v_quat.z = 2 * x * (v_R_t[1][0] + v_R_t[0][1]) + 2 * w * (v_R_t[2][0] - v_R_t[0][2]) + 2 * z * (v_R_t[1][2] + v_R_t[2][1]) - 4 * y * (v_R_t[2][2] + v_R_t[0][0]);
	v_quat.w = 2 * w * (v_R_t[0][1] - v_R_t[1][0]) + 2 * x * (v_R_t[2][0] + v_R_t[0][2]) + 2 * y * (v_R_t[1][2] + v_R_t[2][1]) - 4 * z * (v_R_t[1][1] + v_R_t[0][0]);

    return v_quat;
}
