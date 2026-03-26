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
    vol_size_world = (1.0, 1.0, 1.0)
    vol_size = [100, 150, 80]
    vol_size_voxel = (vol_size[0], vol_size[1], vol_size[2])
    vol_center_pos = (0.5, 0.5, 0.5)
    scenario_name = "voxelization"
    num_gaussians = 100000
    # Init test volume
    vol = utils.generate_test_volume(vol_size)
    utils.save_slices_as_images(vol, "test_out/generated_vol")
    # Initialize gaussians within the volume
    pos3d, scale3d, quat, intensity = utils.random_gauss_init(num_gaussians, vol, anisotropicScale=True, rotation=True)
    intensity = intensity/10
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
    

    params = (pos3d, scale3d, quat, intensity)
    param_names = ("pos3d", "scale3d", "quat", "intensity")
    for tensor in params:
        tensor.grad = None
    loss.backward()
    ours_grads = []
    for tensor in params:
        grad = tensor.grad
        if grad is None:
            raise AssertionError(f"Missing ours gradient for {scenario_name}")
        ours_grads.append(grad.detach().clone())


    assert torch.isfinite(pos3d.grad).all()
    assert torch.isfinite(scale3d.grad).all()
    assert torch.isfinite(quat.grad).all()
    assert torch.isfinite(intensity.grad).all()

    # Reference x_ray_gaussian_voxelization
    print("Starting reference voxelization with xray_gaussian_voxelization...")

    voxel_settings = GaussianVoxelizationSettings(    
            scale_modifier=1,
            nVoxel_x=vol_size[2],
            nVoxel_y=vol_size[1],
            nVoxel_z=vol_size[0],
            sVoxel_x=vol_size_world[2],
            sVoxel_y=vol_size_world[1],
            sVoxel_z=vol_size_world[0],
            center_x=vol_center_pos[2],
            center_y=vol_center_pos[1],
            center_z=vol_center_pos[0],
            prefiltered=False,
            debug=False,
        )
    voxelizer = GaussianVoxelizer(voxel_settings)



    ref_vol, radii = voxelizer(
            means3D=pos3d,
            opacities=intensity,
            scales=scale3d,
            rotations=quat,
            cov3D_precomp=None,
        )
    print("Reference voxelization done.")
    ref_vol = torch.permute(ref_vol, (2, 1, 0))  # Reorder to z,y,x

    ref_loss = fused_ssim3d(ref_vol.unsqueeze(0).unsqueeze(0), torch.from_numpy(vol).unsqueeze(0).unsqueeze(0))
    print(f"Computed reference loss: {ref_loss.item()}. Starting backward pass...")
    for tensor in params:
            tensor.grad = None
    ref_loss.backward()
    reference_grads = []
    for tensor in params:
        grad = tensor.grad
        if grad is None:
            raise AssertionError(f"Missing reference gradient for {scenario_name}")
        reference_grads.append(grad.detach().clone())


    
    # Clamp to [0,1]
    our_vol_save = our_vol.detach().squeeze().cpu().numpy()  
    our_vol_save = np.clip(our_vol_save, 0, 1)
    utils.save_slices_as_images(our_vol_save, "test_out/our_vol")

    ref_vol_save = ref_vol.detach().squeeze().cpu().numpy()
    ref_vol_save = np.clip(ref_vol_save, 0, 1)
    utils.save_slices_as_images(ref_vol_save, "test_out/reference_our_vol")
    
    torch.testing.assert_close(our_vol, ref_vol, rtol=1e-3, atol=5e-2)

    for ours_grad, ref_grad, name in zip(ours_grads, reference_grads, param_names):
        print(f"Testing gradient for {name}")
        torch.testing.assert_close(ours_grad, ref_grad, rtol=1e-4, atol=1e-3)
        assert torch.isfinite(ours_grad).all()
        assert torch.isfinite(ref_grad).all()
    

    

if __name__ == "__main__":
    test_voxelization()
    torch.cuda.empty_cache()
