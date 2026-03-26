"""
Utility script for profiling the gs_voxelizer implementation with Nsight.

Example usage:

    ncu -o ours_backward python test/profile_gs.py --passes backward
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from fused_ssim import fused_ssim3d
from gs_voxelizer import optim_to_render, voxelize
import utils


def sample_gaussians(num_gaussians: int, vol, requires_grad: bool):
    pos3d, scale3d, quat, intensity = utils.random_gauss_init(num_gaussians, vol)

    if requires_grad:
        pos3d = pos3d.requires_grad_()
        scale3d = scale3d.requires_grad_()
        quat = quat.requires_grad_()
        intensity = intensity.requires_grad_()

    return pos3d, scale3d, quat, intensity


def profile_forward(args, vol):
    pos3d, scale3d, quat, intensity = sample_gaussians(
        args.num_gaussians, vol, requires_grad=False
    )
    vol_size_voxel = (args.vol_size, args.vol_size, args.vol_size)
    vol_size_world = (1.0, 1.0, 1.0)
    vol_center_pos = (0.5, 0.5, 0.5)

    torch.cuda.synchronize()
    pos3d_viz_radii, conics, tile_min, tile_max, num_tiles_hit = optim_to_render.optim_to_render(
        pos3d,
        scale3d,
        quat,
        intensity,
        vol_size_voxel,
        vol_size_world,
        vol_center_pos,
    )

    voxelize.voxelize_gaussians(
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


def profile_backward(args, vol):
    pos3d, scale3d, quat, intensity = sample_gaussians(
        args.num_gaussians, vol, requires_grad=True
    )
    vol_size_voxel = (args.vol_size, args.vol_size, args.vol_size)
    vol_size_world = (1.0, 1.0, 1.0)
    vol_center_pos = (0.5, 0.5, 0.5)
    target_volume = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(pos3d.device)

    pos3d_viz_radii, conics, tile_min, tile_max, num_tiles_hit = optim_to_render.optim_to_render(
        pos3d,
        scale3d,
        quat,
        intensity,
        vol_size_voxel,
        vol_size_world,
        vol_center_pos,
    )
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
    loss.backward()
    torch.cuda.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the gs_voxelizer pipeline.")
    parser.add_argument("--vol-size", type=int, default=190, help="Cube volume size.")
    parser.add_argument(
        "--num-gaussians", type=int, default=5000, help="Number of Gaussians to voxelize."
    )
    parser.add_argument(
        "--passes",
        choices=["forward", "backward"],
        nargs="+",
        default=["forward", "backward"],
        help="Which passes to execute for profiling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for numpy/torch randomness to keep inputs reproducible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vol = utils.generate_test_volume(args.vol_size)

    for pass_name in args.passes:
        if pass_name == "forward":
            profile_forward(args, vol)
        elif pass_name == "backward":
            profile_backward(args, vol)


if __name__ == "__main__":
    main()
