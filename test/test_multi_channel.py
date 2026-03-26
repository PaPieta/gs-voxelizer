from gs_voxelizer import voxelize, optim_to_render
import utils
import torch
import numpy as np
from fused_ssim import fused_ssim3d

from xray_gaussian_rasterization_voxelization import (
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

def test_voxelization():

    # Test parameters
    vol_size_world = (1.0, 1.5, 0.8)
    vol_size = [100, 150, 80]
    vol_size_voxel = (vol_size[0], vol_size[1], vol_size[2])
    vol_center_pos = (0.5, 0.7, 0.5)
    scenario_name = "voxelization"
    num_gaussians = 5000
    # Init test volume
    vol = utils.generate_test_volume(vol_size)
    # Initialize gaussians within the volume
    pos3d, scale3d, quat, intensity = utils.random_gauss_init(num_gaussians, vol, anisotropicScale=True)
    intensity = torch.cat([intensity, intensity/2], dim=1)  # Create 2 channels with the same intensity for testing
    pos3d = pos3d.requires_grad_()
    scale3d = scale3d.requires_grad_()
    quat = quat.requires_grad_()
    intensity = intensity.requires_grad_()

    print("Initialized test setup. Volume shape:", vol.shape)

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
    print("Conversion to rendering parameters done.")

    # Voxelize gaussians
    our_vol = voxelize.voxelize_gaussians(
        pos3d_viz_radii,
        conics,
        intensity,
        vol_size_voxel,
        tile_min,
        tile_max,
        num_tiles_hit,
        use_per_gaussian_backward=True,
    )
    our_vol = our_vol.squeeze()
    print("Voxelization done. Volume shape:", our_vol.shape)


    # Backward pass test
    loss = fused_ssim3d(our_vol.unsqueeze(0).unsqueeze(0), torch.from_numpy(vol).unsqueeze(0).unsqueeze(0))
    print(f"Computed loss: {loss.item()}. Starting backward pass...")
    loss.backward()

    assert torch.isfinite(pos3d.grad).all()
    assert torch.isfinite(scale3d.grad).all()
    assert torch.isfinite(quat.grad).all()
    assert torch.isfinite(intensity.grad).all()


    

if __name__ == "__main__":
    test_voxelization()
    torch.cuda.empty_cache()
