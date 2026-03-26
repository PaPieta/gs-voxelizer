"""
Utility script for profiling the xray_gaussian_voxelization baseline with Nsight.

Run this script under Nsight Systems/Compute to capture the forward and/or backward
passes at a specific volume size and number of Gaussians, e.g.:

    nsys profile -o baseline_forward python test/profile_baseline.py --passes forward
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from fused_ssim import fused_ssim3d
import utils
from xray_gaussian_rasterization_voxelization import (
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)


def build_voxelizer(vol_size: int) -> GaussianVoxelizer:
    settings = GaussianVoxelizationSettings(
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
    return GaussianVoxelizer(settings)


def sample_gaussians(num_gaussians: int, vol_size: int, vol, requires_grad: bool):
    pos3d_t, scale3d_t, quat_t, intensity_t = utils.random_gauss_init(num_gaussians, vol)
    device = pos3d_t.device


    if requires_grad:
        pos3d_t.requires_grad_()
        scale3d_t.requires_grad_()
        quat_t.requires_grad_()
        intensity_t.requires_grad_()

    return pos3d_t, scale3d_t, quat_t, intensity_t


def profile_forward(voxelizer: GaussianVoxelizer, args, vol) -> None:
    pos3d, scale3d, quat, intensity = sample_gaussians(
        args.num_gaussians, args.vol_size, vol, requires_grad=False
    )

    torch.cuda.synchronize()
    voxelizer(
        means3D=pos3d,
        opacities=intensity,
        scales=scale3d,
        rotations=quat,
        cov3D_precomp=None,
    )
    torch.cuda.synchronize()


def profile_backward(voxelizer: GaussianVoxelizer, args, vol) -> None:
    pos3d, scale3d, quat, intensity = sample_gaussians(
        args.num_gaussians, args.vol_size, vol, requires_grad=True
    )
    target_volume = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(pos3d.device)

    out_image, _ = voxelizer(
        means3D=pos3d,
        opacities=intensity,
        scales=scale3d,
        rotations=quat,
        cov3D_precomp=None,
    )
    out_image = out_image.squeeze().unsqueeze(0).unsqueeze(0)
    loss = fused_ssim3d(out_image, target_volume)

    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the baseline voxelizer.")
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
        help="Seed for NumPy randomness to keep profiles reproducible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vol = utils.generate_test_volume([args.vol_size, args.vol_size, args.vol_size])
    voxelizer = build_voxelizer(args.vol_size)

    for pass_name in args.passes:
        if pass_name == "forward":
            profile_forward(voxelizer, args, vol)
        elif pass_name == "backward":
            profile_backward(voxelizer, args, vol)


if __name__ == "__main__":
    main()
