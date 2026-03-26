"""
Benchmark voxelization speed against the xray_gaussian_voxelization baseline.

This script measures forward and backward runtimes while varying either the
voxel volume size or the number of Gaussians. Results are saved as two plots
for quick visual comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import torch
import numpy as np

from fused_ssim import fused_ssim3d
from gs_voxelizer import optim_to_render, voxelize
import utils
from xray_gaussian_rasterization_voxelization import (
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)


def build_baseline_voxelizer(vol_size: int) -> GaussianVoxelizer:
    """Create an xray_gaussian_voxelization voxelizer for a cube volume."""
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


def _forward_baseline_run(
    voxelizer: GaussianVoxelizer,
    vol_size: int,
    num_gaussians: int,
    vol,
) -> float:
    pos3d, scale3d, quat, intensity = utils.random_gauss_init(num_gaussians, vol)

    torch.cuda.synchronize()
    start_time = time()
    voxelizer(
        means3D=pos3d,
        opacities=intensity,
        scales=scale3d,
        rotations=quat,
        cov3D_precomp=None,
    )
    torch.cuda.synchronize()
    end_time = time()
    return end_time - start_time


def _forward_gs_run(
    vol_size: int,
    num_gaussians: int,
    vol,
) -> float:
    pos3d, scale3d, quat, intensity = utils.random_gauss_init(num_gaussians, vol)
    vol_size_voxel = (vol_size, vol_size, vol_size)
    vol_size_world = (1.0, 1.0, 1.0)
    vol_center_pos = (0.5, 0.5, 0.5)

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
    end_time = time()
    return end_time - start_time


def _backward_baseline_run(
    voxelizer: GaussianVoxelizer,
    vol_size: int,
    num_gaussians: int,
    vol,
    target_volume: torch.Tensor,
) -> float:
    pos3d, scale3d, quat, intensity = utils.random_gauss_init(num_gaussians, vol)

    pos3d = pos3d.requires_grad_()
    scale3d = (scale3d).requires_grad_()
    quat = quat.requires_grad_()
    intensity = intensity.requires_grad_()

    out_image, _ = voxelizer(
        means3D=pos3d,
        opacities=intensity,
        scales=scale3d,
        rotations=quat,
        cov3D_precomp=None,
    )
    out_image = out_image.squeeze().unsqueeze(0).unsqueeze(0)
    loss = fused_ssim3d(out_image, target_volume.to(out_image.device))

    torch.cuda.synchronize()
    start_time = time()
    loss.backward()
    torch.cuda.synchronize()
    end_time = time()
    return end_time - start_time


def _backward_gs_run(
    vol_size: int,
    num_gaussians: int,
    vol,
    target_volume: torch.Tensor,
) -> float:
    pos3d, scale3d, quat, intensity = utils.random_gauss_init(num_gaussians, vol)
    vol_size_voxel = (vol_size, vol_size, vol_size)
    vol_size_world = (1.0, 1.0, 1.0)
    vol_center_pos = (0.5, 0.5, 0.5)

    pos3d = pos3d.requires_grad_()
    scale3d = scale3d.requires_grad_()
    quat = quat.requires_grad_()
    intensity = intensity.requires_grad_()
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
    loss = fused_ssim3d(voxelized_vol, target_volume.to(voxelized_vol.device))

    torch.cuda.synchronize()
    start_time = time()
    loss.backward()
    torch.cuda.synchronize()
    end_time = time()
    return end_time - start_time


def average_runtime(fn, repetitions: int, warmup: int) -> float:
    """Average runtime (seconds) while discarding the warmup iterations."""
    total = 0.0
    for idx in range(repetitions + warmup):
        duration = fn()
        if idx >= warmup:
            total += duration
    return total / repetitions


def benchmark_case(
    vol_size: int,
    num_gaussians: int,
    repetitions: int,
    warmup: int,
):
    """Measure forward/backward times for one pair of (vol_size, num_gaussians)."""
    vol = utils.generate_test_volume([vol_size, vol_size, vol_size])
    target_volume = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    xray_voxelizer = build_baseline_voxelizer(vol_size)

    forward_baseline = average_runtime(
        lambda: _forward_baseline_run(xray_voxelizer, vol_size, num_gaussians, vol),
        repetitions,
        warmup,
    )
    forward_gs = average_runtime(
        lambda: _forward_gs_run(vol_size, num_gaussians, vol),
        repetitions,
        warmup,
    )
    backward_baseline = average_runtime(
        lambda: _backward_baseline_run(
            xray_voxelizer, vol_size, num_gaussians, vol, target_volume
        ),
        repetitions,
        warmup,
    )
    backward_gs = average_runtime(
        lambda: _backward_gs_run(vol_size, num_gaussians, vol, target_volume),
        repetitions,
        warmup,
    )

    return {
        "forward": {
            "gs_voxelizer": forward_gs * 1000.0,
            "baseline": forward_baseline * 1000.0,
        },
        "backward": {
            "gs_voxelizer": backward_gs * 1000.0,
            "baseline": backward_baseline * 1000.0,
        },
    }


def plot_results(
    x_values,
    metrics,
    xlabel: str,
    title: str,
    output_path: Path,
):
    """Create a comparison plot and save it."""
    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")

    forward_color = np.array([57, 106, 177]) / 255
    backward_color = np.array([166, 23, 23]) / 255

    plt.plot(
        x_values,
        metrics["forward"]["baseline"],
        c=forward_color,
        linestyle="dashed",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        x_values,
        metrics["forward"]["gs_voxelizer"],
        c=forward_color,
        linestyle="solid",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        x_values,
        metrics["backward"]["baseline"],
        c=backward_color,
        linestyle="dashed",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        x_values,
        metrics["backward"]["gs_voxelizer"],
        c=backward_color,
        linestyle="solid",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    baseline_backward_marker = plt.plot(
        [], [],
        label="Baseline backward",
        c=backward_color,
        linestyle="dashed",
        linewidth=2
    )
    ours_backward_marker = plt.plot(
        [], [],
        label="Ours backward",
        c=backward_color,
        linestyle="solid",
        linewidth=2
    )
    baseline_forward_marker = plt.plot(
        [], [],
        label="Baseline forward",
        c=forward_color,
        linestyle="dashed",
        linewidth=2)
    ours_forward_marker = plt.plot(
        [], [],
        label="Ours forward",
        c=forward_color,
        linestyle="solid",
        linewidth=2,
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, which="major", axis="x", alpha=0.3)
    ax.grid(True, which="major", axis="y", alpha=0.3)

    plt.xlabel(xlabel)
    plt.ylabel("Execution time (ms)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def prepare_metric_storage():
    return {
        "forward": {"baseline": [], "gs_voxelizer": []},
        "backward": {"baseline": [], "gs_voxelizer": []},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare voxelization runtimes.")
    parser.add_argument("--repetitions", type=int, default=15, help="Measurement iterations per data point.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations to discard.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_out_speed"),
        help="Directory used to store the generated plots.",
    )
    parser.add_argument(
        "--volume-sizes",
        type=int,
        nargs="+",
        default=[48, 96, 128, 192, 256, 384],
        help="Volume sizes evaluated when varying the grid resolution.",
    )
    parser.add_argument(
        "--gaussian-counts",
        type=int,
        nargs="+",
        default=[1000, 2500, 5000, 7500, 10000, 20000, 40000, 80000, 160000, 320000],
        help="Gaussian counts evaluated when varying the number of Gaussians.",
    )
    parser.add_argument(
        "--fixed-volume-size",
        type=int,
        default=128,
        help="Volume size used when sweeping Gaussian counts.",
    )
    parser.add_argument(
        "--fixed-gaussian-count",
        type=int,
        default=20000,
        help="Number of Gaussians used when sweeping volume sizes.",
    )
    parser.add_argument(
        "--save-to-npy",
        type=Path,
        default=None,
        help="If set, skip plotting and save the collected metrics to this NumPy file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_to_npy is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.save_to_npy.parent.mkdir(parents=True, exist_ok=True)

    volume_metrics = prepare_metric_storage()
    gaussian_metrics = prepare_metric_storage()

    print("=== Sweep: varying volume size ===")
    for vol_size in args.volume_sizes:
        measurements = benchmark_case(
            vol_size=vol_size,
            num_gaussians=args.fixed_gaussian_count,
            repetitions=args.repetitions,
            warmup=args.warmup,
        )
        print(
            f"Volume {vol_size}: "
            f"baseline forward {measurements['forward']['baseline']:.2f} ms, "
            f"Ours forward {measurements['forward']['gs_voxelizer']:.2f} ms | "
            f"baseline backward {measurements['backward']['baseline']:.2f} ms, "
            f"Ours backward {measurements['backward']['gs_voxelizer']:.2f} ms"
        )
        for phase in ("forward", "backward"):
            for method in ("baseline", "gs_voxelizer"):
                volume_metrics[phase][method].append(measurements[phase][method])

    print("\n=== Sweep: varying number of Gaussians ===")
    for num_gaussians in args.gaussian_counts:
        measurements = benchmark_case(
            vol_size=args.fixed_volume_size,
            num_gaussians=num_gaussians,
            repetitions=args.repetitions,
            warmup=args.warmup,
        )
        print(
            f"Gaussians {num_gaussians}: "
            f"baseline forward {measurements['forward']['baseline']:.2f} ms, "
            f"Ours forward {measurements['forward']['gs_voxelizer']:.2f} ms | "
            f"baseline backward {measurements['backward']['baseline']:.2f} ms, "
            f"Ours backward {measurements['backward']['gs_voxelizer']:.2f} ms"
        )
        for phase in ("forward", "backward"):
            for method in ("baseline", "gs_voxelizer"):
                gaussian_metrics[phase][method].append(measurements[phase][method])

    if args.save_to_npy is not None:
        benchmark_data = {
            "volume_sizes": args.volume_sizes,
            "gaussian_counts": args.gaussian_counts,
            "fixed_volume_size": args.fixed_volume_size,
            "fixed_gaussian_count": args.fixed_gaussian_count,
            "volume_metrics": volume_metrics,
            "gaussian_metrics": gaussian_metrics,
        }
        np.save(args.save_to_npy, benchmark_data, allow_pickle=True)
        print(f"\nSaved benchmark data to: {args.save_to_npy}")
    else:
        volume_plot = args.output_dir / "speed_vs_volume.png"
        gaussian_plot = args.output_dir / "speed_vs_gaussians.png"
        plot_results(
            x_values=args.volume_sizes,
            metrics=volume_metrics,
            xlabel="Volume size (voxels)",
            title=f"Runtime vs. volume size (Gaussians={args.fixed_gaussian_count})",
            output_path=volume_plot,
        )
        plot_results(
            x_values=args.gaussian_counts,
            metrics=gaussian_metrics,
            xlabel="Number of Gaussians",
            title=f"Runtime vs. Gaussians (volume={args.fixed_volume_size}^3)",
            output_path=gaussian_plot,
        )

        print(f"\nSaved volume sweep plot to: {volume_plot}")
        print(f"Saved Gaussian sweep plot to: {gaussian_plot}")


if __name__ == "__main__":
    main()
