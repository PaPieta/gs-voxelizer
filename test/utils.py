import numpy as np
import os
from typing import Optional, Union

import skimage.io
import torch
import torch.nn.functional as F


def generate_test_volume(vol_size):
    """Generate a test volume with random spheres.

    Args:
        vol_size: size of the cubic volume (3-element iterable)

    Returns:
        3D numpy array (0-1 float values)
    """

    vol = np.ones((vol_size[0], vol_size[1], vol_size[2]), dtype=np.float32) * 0.1
    num_spheres = 20
    for _ in range(num_spheres):
        center = np.random.randint(0, [vol_size[0], vol_size[1], vol_size[2]], size=3)
        radius = np.random.randint(max(1, vol_size[0] // 10), max(2, vol_size[0] // 5))
        zz, yy, xx = np.ogrid[:vol_size[0], :vol_size[1], :vol_size[2]]
        mask = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= radius**2
        vol[mask] += np.random.uniform(0.5, 1.0)
    vol = np.clip(vol, 0, 1)
    return vol


def generate_noise_volume(
    vol_size,
    base_level: float = 0.1,
    contrast: float = 0.9,
    smooth_kernel: int = 5,
    device: Optional[Union[str, torch.device]] = None,
):
    """Generate a synthetic test volume using smoothed torch noise.

    Args:
        vol_size: Iterable with three ints describing (depth, height, width).
        base_level: Baseline intensity added to the noise.
        contrast: Multiplier applied to the noise component.
        smooth_kernel: Size of the averaging kernel used to impose structure.
        device: Optional torch device the temporary tensor lives on.

    Returns:
        3D numpy array (float32, 0-1 range).
    """

    if isinstance(vol_size, int):
        vol_dims = (vol_size, vol_size, vol_size)
    else:
        vol_dims = tuple(int(v) for v in vol_size)
    if len(vol_dims) != 3:
        raise ValueError(f"vol_size must have 3 entries, got {vol_dims}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    noise = torch.rand((1, 1, *vol_dims), device=device, dtype=torch.float32)
    if smooth_kernel and smooth_kernel > 1:
        padding = smooth_kernel // 2
        noise = F.avg_pool3d(noise, kernel_size=smooth_kernel, stride=1, padding=padding)

    volume = base_level + contrast * noise
    volume = volume.clamp(0.0, 1.0).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    return volume


def save_slices_as_images(volume, path_prefix):
    """Save central slices of the volume as images (0-255 uint8 pngs).
    Inputs:
        volume: 3D numpy array (0-1 float values)
        path_prefix: prefix for saving images"""

    # Create path if it doesn't exist
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)

    mid = np.array(volume.shape) // 2
    skimage.io.imsave(f"{path_prefix}_xy.png", (volume[mid[0], :, :] * 255).astype(np.uint8))
    # print(f"Saved slice {path_prefix}_xy.png, shape: {volume[mid[0], :, :].shape}")
    skimage.io.imsave(f"{path_prefix}_xz.png", (volume[:, mid[1], :] * 255).astype(np.uint8))
    # print(f"Saved slice {path_prefix}_xz.png, shape: {volume[:, mid[1], :].shape}")
    skimage.io.imsave(f"{path_prefix}_yz.png", (volume[:, :, mid[2]] * 255).astype(np.uint8))
    # print(f"Saved slice {path_prefix}_yz.png, shape: {volume[:, :, mid[2]].shape}")

def random_gauss_init(num_gaussians, vol, device: Optional[Union[str, torch.device]] = None, anisotropicScale=False, rotation=False):
    """Initialize Gaussians directly on the requested device.

    Args:
        num_gaussians: number of Gaussians to initialize.
        vol: 3D numpy array (0-1 float values) or torch tensor describing the volume.
        device: torch device identifier. Defaults to CUDA if available, otherwise CPU.
        anisotropicScale: If True, applies anisotropic scaling to the initialized Gaussians for more comprehensive testing.
        rotation: If True, applies random rotation to the initialized Gaussians for more comprehensive testing. Use in conjunction with anisotropicScale.

    Returns:
        Tuple of torch tensors (pos3d, scale3d, quat, intensity) resident on `device`.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if isinstance(vol, torch.Tensor):
        vol_tensor = vol.to(device=device, dtype=torch.float32)
    else:
        vol_tensor = torch.as_tensor(vol, dtype=torch.float32, device=device)

    num_gauss = int(num_gaussians)
    pos3d = torch.rand((num_gauss, 3), device=device, dtype=torch.float32)
    scale3d = torch.full((num_gauss, 3), 0.02, device=device, dtype=torch.float32)
    if anisotropicScale:
        # print("Applying anisotropic scaling to test Gaussians.")
        scale3d[:, 0] = scale3d[:, 0] * 1.5  #Anisotropic scaling for complete testing: stretch along x-axis
        scale3d[:, 1] = scale3d[:, 1] * 0.5  #Anisotropic scaling for complete testing: compress along y-axis

    quat = torch.zeros((num_gauss, 4), device=device, dtype=torch.float32)
    quat[:, 0] = 1.0
    if rotation:
        if anisotropicScale:
            # print("Applying rotation to test Gaussians.")
            angles = torch.rand((num_gauss, 3), device=device) * 360  # Random rotation angles in degrees
            # Convert angles to radians and create quaternions (assuming XYZ order)
            angles_rad = torch.deg2rad(angles)
            cx = torch.cos(angles_rad[:, 0] / 2)
            sx = torch.sin(angles_rad[:, 0] / 2)
            cy = torch.cos(angles_rad[:, 1] / 2)
            sy = torch.sin(angles_rad[:, 1] / 2)
            cz = torch.cos(angles_rad[:, 2] / 2)
            sz = torch.sin(angles_rad[:, 2] / 2)

            quat[:, 0] = cx * cy * cz + sx * sy * sz
            quat[:, 1] = sx * cy * cz - cx * sy * sz
            quat[:, 2] = cx * sy * cz + sx * cy * sz
            quat[:, 3] = cx * cy * sz - sx * sy * cz
        else:
            raise Warning("Rotation is enabled but anisotropicScale is False. No rotation will be applied to the test Gaussians")
        

    vol_shape = torch.tensor(
        [vol_tensor.shape[2], vol_tensor.shape[1], vol_tensor.shape[0]],
        device=device,
        dtype=torch.float32,
    )
    indices = torch.clamp((pos3d * (vol_shape - 1)).long(), min=0)
    intensity = vol_tensor[indices[:, 2], indices[:, 1], indices[:, 0]].unsqueeze(-1)

    return pos3d, scale3d, quat, intensity
