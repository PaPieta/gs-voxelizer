"""
Exercise the gs_voxelizer pipeline at increasingly large volume sizes.

The script scales the cubic volume resolution and the corresponding number of
Gaussians, runs the optim_to_render + voxelize passes once per configuration, and
reports timings for each stage.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np
import torch

from gs_voxelizer import optim_to_render, voxelize
import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark gs_voxelizer on increasingly large volumes."
    )
    parser.add_argument(
        "--edges",
        type=int,
        nargs="+",
        default=[96, 128, 256, 512, 1024, 1200],
        help="Cubic volume edges (in voxels) to test. Values are sorted internally.",
    )
    parser.add_argument(
        "--base-gaussians",
        type=int,
        default=6000,
        help="Number of Gaussians for the smallest volume; larger volumes scale linearly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for numpy/torch randomness to keep inputs reproducible.",
    )
    return parser.parse_args()


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def run_volume_case(
    vol_edge: int, num_gaussians: int, device: torch.device
) -> Dict[str, float]:
    vol_size_voxel = (vol_edge, vol_edge, vol_edge)
    vol_size_world = (1.0, 1.0, 1.0)
    vol_center_pos = (0.5, 0.5, 0.5)
    vol = utils.generate_noise_volume(list(vol_size_voxel), device=device)

    pos3d, scale3d, quat, intensity = utils.random_gauss_init(
        num_gaussians, vol, device=device, anisotropicScale=True
    )

    peak_mem_mb = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    maybe_synchronize(device)
    optim_start = time.perf_counter()
    (
        pos3d_viz_radii,
        conics,
        tile_min,
        tile_max,
        num_tiles_hit,
    ) = optim_to_render.optim_to_render(
        pos3d,
        scale3d,
        quat,
        intensity,
        vol_size_voxel,
        vol_size_world,
        vol_center_pos,
    )
    maybe_synchronize(device)
    optim_end = time.perf_counter()

    maybe_synchronize(device)
    voxel_start = time.perf_counter()
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
    maybe_synchronize(device)
    voxel_end = time.perf_counter()

    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    result = {
        "vol_edge": vol_edge,
        "num_gaussians": num_gaussians,
        "optim_ms": (optim_end - optim_start) * 1000.0,
        "voxel_ms": (voxel_end - voxel_start) * 1000.0,
        "total_ms": (voxel_end - optim_start) * 1000.0,
        "volume_shape": tuple(int(dim) for dim in voxelized_vol.shape),
        "peak_mem_mb": peak_mem_mb,
    }

    del (
        pos3d,
        scale3d,
        quat,
        intensity,
        pos3d_viz_radii,
        conics,
        tile_min,
        tile_max,
        num_tiles_hit,
        voxelized_vol,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main() -> None:
    args = parse_args()
    volume_edges = sorted(args.edges)
    base_edge = volume_edges[0]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: List[Dict[str, float]] = []
    for edge in volume_edges:
        scaled_gaussians = max(1, int(round(args.base_gaussians * edge / base_edge)))
        print(
            f"Running {edge}^3 volume with {scaled_gaussians} Gaussians...",
            flush=True,
        )
        result = run_volume_case(edge, scaled_gaussians, device)
        results.append(result)
        print(
            f"  -> total {result['total_ms']:.2f} ms "
            f"(optim_to_render {result['optim_ms']:.2f} ms, "
            f"voxelize {result['voxel_ms']:.2f} ms)"
        )
        if result["peak_mem_mb"] is not None:
            print(f"     peak GPU memory: {result['peak_mem_mb']:.2f} MB")

    print("\nSummary:")
    header = (
        "Volume  | Gaussians | Total ms | optim_to_render ms | voxelize ms | "
        "Peak MB | Output shape"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        volume_label = f"{result['vol_edge']}^3"
        peak_label = (
            f"{result['peak_mem_mb']:.2f}" if result["peak_mem_mb"] is not None else "N/A"
        )
        print(
            f"{volume_label:>7} | "
            f"{result['num_gaussians']:>9} | "
            f"{result['total_ms']:>8.2f} | "
            f"{result['optim_ms']:>15.2f} | "
            f"{result['voxel_ms']:>11.2f} | "
            f"{peak_label:>7} | "
            f"{result['volume_shape']}"
        )


if __name__ == "__main__":
    main()
