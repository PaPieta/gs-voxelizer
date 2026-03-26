"""
Torch profiler entry point for gs_voxelizer forward/backward passes.
This script mirrors the Nsight profiling utilities but emits TensorBoard traces
using torch.profiler, so you can inspect CPU/GPU timelines without Nsight.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

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
def run_forward(args, vol):
    pos3d, scale3d, quat, intensity = sample_gaussians(
        args.num_gaussians, vol, requires_grad=False
    )
    vol_size_voxel = (args.vol_size, args.vol_size, args.vol_size)
    vol_size_world = (1.0, 1.0, 1.0)
    vol_center_pos = (0.5, 0.5, 0.5)
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
def run_backward(args, vol):
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
    loss.backward()
def parse_args():
    parser = argparse.ArgumentParser(description="Torch profiler driver for gs_voxelizer.")
    parser.add_argument("--vol-size", type=int, default=190, help="Cube volume size.")
    parser.add_argument(
        "--num-gaussians", type=int, default=5000, help="Number of Gaussians to voxelize."
    )
    parser.add_argument(
        "--passes",
        choices=["forward", "backward"],
        nargs="+",
        default=["forward", "backward"],
        help="Which passes to execute under the profiler.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for numpy/torch.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations before tracing.")
    parser.add_argument("--cycles", type=int, default=1, help="Active iterations to capture.")
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="profiles/torch",
        help="Where to store TensorBoard trace files.",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Record tensor shapes in the profiler trace.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Track memory usage during profiling.",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Include stack traces for profiled operations.",
    )
    return parser.parse_args()
def build_profiler(args):
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    log_dir = Path(args.profile_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return profile(
        activities=activities,
        schedule=schedule(
            wait=0,
            warmup=max(0, args.warmup),
            active=max(1, args.cycles),
            repeat=1,
        ),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        on_trace_ready=tensorboard_trace_handler(str(log_dir)),
    )
def run_with_profiler(fn, warmup: int, cycles: int, prof):
    total_steps = max(0, warmup) + max(0, cycles)
    for _ in range(total_steps):
        fn()
        prof.step()
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    vol = utils.generate_test_volume([args.vol_size, args.vol_size, args.vol_size])
    with build_profiler(args) as prof:
        for pass_name in args.passes:
            if pass_name == "forward":
                run_with_profiler(lambda: run_forward(args, vol), args.warmup, args.cycles, prof)
            elif pass_name == "backward":
                run_with_profiler(lambda: run_backward(args, vol), args.warmup, args.cycles, prof)
    print(f"Torch profiler traces stored in: {Path(args.profile_dir).resolve()}")
if __name__ == "__main__":
    main()
