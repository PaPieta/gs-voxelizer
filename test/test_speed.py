from gs_voxelizer import voxelize, optim_to_render
import utils
import torch

from xray_gaussian_rasterization_voxelization import (
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)
from fused_ssim import fused_ssim3d

from time import time

def test_speed():

    # Test parameters
    vol_size = 190
    vol_size_voxel = (vol_size, vol_size, vol_size)
    vol_size_world = (1.0, 1.0, 1.0)
    vol_center_pos = (0.5, 0.5, 0.5)
    num_gaussians = 5000
    repetitions = 30
    warmup = 10
    # Init test volume
    vol = utils.generate_test_volume([vol_size, vol_size, vol_size])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Test xray_gaussian_voxelization speed

    voxel_settings = GaussianVoxelizationSettings(    
            scale_modifier=1,
            nVoxel_x=vol_size,
            nVoxel_y=vol_size,
            nVoxel_z=vol_size,
            sVoxel_x=1.0,
            sVoxel_y=1.0,
            sVoxel_z=1.0,
            center_x=0.5,
            center_y=0.5,
            center_z=0.5,
            prefiltered=False,
            debug=False,
        )
    voxelizer = GaussianVoxelizer(voxel_settings)

    speed_xray = 0
    for i in range(repetitions + warmup):
        pos3d, scale3d, quat, intensity = utils.random_gauss_init(
            num_gaussians, vol, device=device, anisotropicScale=True
        )
        device = pos3d.device
        # scale_norm = torch.tensor(
        #     [vol_size, vol_size, vol_size], dtype=scale3d.dtype, device=device
        # )
        # scale3d = scale3d / scale_norm

        torch.cuda.synchronize()
        start_time = time()
        
        out_image, radii = voxelizer(
            means3D=pos3d,
            opacities=intensity,
            scales=scale3d,
            rotations=quat,
            cov3D_precomp=None,
        )
        torch.cuda.synchronize()
        end_time = time()
        if i >= warmup:
            speed_xray += end_time - start_time

    speed_xray /= (repetitions)
    print(f"XRay Gaussian Voxelizer forward pass average time over {repetitions} runs: {speed_xray*1000:.2f} ms")


    speed_gs = 0
    speed_optim = 0
    speed_voxelize = 0
    # Test gs_voxelizer speed
    for i in range(repetitions + warmup):
        
        pos3d, scale3d, quat, intensity = utils.random_gauss_init(
            num_gaussians, vol, device=device, anisotropicScale=True
        )
        torch.cuda.synchronize()
        start_time = time()
        pos3d_viz_radii, conics, tile_min, tile_max, num_tiles_hit = optim_to_render.optim_to_render(
            pos3d,
            scale3d,
            quat,
            intensity,
            vol_size_voxel,
            vol_size_world,
            vol_center_pos,
        )

        torch.cuda.synchronize()
        mid_time = time()

        # Voxelize gaussians
        voxelized_vol = voxelize.voxelize_gaussians(
            pos3d_viz_radii,
            conics,
            intensity,
            vol_size_voxel,
            tile_min,
            tile_max,
            num_tiles_hit,
            use_per_gaussian_backward=True,
        )
        torch.cuda.synchronize()
        end_time = time()
        if i >= warmup:
            speed_gs += end_time - start_time
            speed_optim += mid_time - start_time
            speed_voxelize += end_time - mid_time

    speed_gs /= (repetitions)
    speed_optim /= (repetitions)
    speed_voxelize /= (repetitions)
    print(f"GS Voxelizer forward pass average time over {repetitions} runs: {speed_gs*1000:.2f} ms")
    print(f"  -  of which optim_to_render: {speed_optim*1000:.2f} ms")
    print(f"  -  of which voxelize_gaussians: {speed_voxelize*1000:.2f} ms")

    
def test_speed_backwards():

    # Test parameters
    vol_size = 190
    vol_size_voxel = (vol_size, vol_size, vol_size)
    vol_size_world = (1.0, 1.0, 1.0)
    vol_center_pos = (0.5, 0.5, 0.5)
    num_gaussians = 5000
    repetitions = 30
    warmup = 10
    # Init test volume
    vol = utils.generate_test_volume([vol_size, vol_size, vol_size])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_volume = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)


    # Test xray_gaussian_voxelization speed

    voxel_settings = GaussianVoxelizationSettings(    
            scale_modifier=1,
            nVoxel_x=vol_size,
            nVoxel_y=vol_size,
            nVoxel_z=vol_size,
            sVoxel_x=1.0,
            sVoxel_y=1.0,
            sVoxel_z=1.0,
            center_x=0.5,
            center_y=0.5,
            center_z=0.5,
            prefiltered=False,
            debug=False,
        )
    voxelizer = GaussianVoxelizer(voxel_settings)

    speed_xray = 0
    for i in range(repetitions + warmup):
        pos3d, scale3d, quat, intensity = utils.random_gauss_init(
            num_gaussians, vol, device=device, anisotropicScale=True
        )
        device = pos3d.device

        pos3d = pos3d.requires_grad_()
        scale3d = (scale3d).requires_grad_()
        quat = quat.requires_grad_()
        intensity = intensity.requires_grad_()

        out_image, radii = voxelizer(
            means3D=pos3d,
            opacities=intensity,
            scales=scale3d,
            rotations=quat,
            cov3D_precomp=None,
        )

        out_image = out_image.squeeze().unsqueeze(0).unsqueeze(0)
        loss = fused_ssim3d(out_image, target_volume)
    
        torch.cuda.synchronize()
        start_time = time()
        
        loss.backward()
        torch.cuda.synchronize()
        end_time = time()
        if i >= warmup:
            speed_xray += end_time - start_time

    speed_xray /= (repetitions)
    print(f"XRay Gaussian Voxelizer backward pass average time over {repetitions} runs: {speed_xray*1000:.2f} ms")


    speed_gs = 0
    # Test gs_voxelizer speed
    for i in range(repetitions + warmup):
        
        pos3d, scale3d, quat, intensity = utils.random_gauss_init(
            num_gaussians, vol, device=device, anisotropicScale=True
        )
        pos3d = pos3d.requires_grad_()
        scale3d = scale3d.requires_grad_()
        quat = quat.requires_grad_()
        intensity = intensity.requires_grad_()

        
        # Convert to rendering parameters
        pos3d_viz_radii, conics, tile_min, tile_max, num_tiles_hit = optim_to_render.optim_to_render(
            pos3d,
            scale3d,
            quat,
            intensity,
            vol_size_voxel,
            vol_size_world,
            vol_center_pos,
        )

        # Voxelize gaussians
        voxelized_vol = voxelize.voxelize_gaussians(
            pos3d_viz_radii,
            conics,
            intensity,
            vol_size_voxel,
            tile_min,
            tile_max,
            num_tiles_hit,
            use_per_gaussian_backward=True,
        )

        voxelized_vol = voxelized_vol.squeeze().unsqueeze(0).unsqueeze(0)
        loss = fused_ssim3d(voxelized_vol, target_volume)
        
        torch.cuda.synchronize()
        start_time = time()
        loss.backward()
        torch.cuda.synchronize()
        end_time = time()
        if i >= warmup:
            speed_gs += end_time - start_time

    speed_gs /= (repetitions)
    print(f"GS Voxelizer backward pass average time over {repetitions} runs: {speed_gs*1000:.2f} ms")

if __name__ == "__main__":
    test_speed()
    test_speed_backwards()
    torch.cuda.empty_cache()
