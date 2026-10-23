# /// script
# dependencies = ["torch>=2.2"]
# ///
from typing import Literal

import torch
import torch.nn.functional as F

samples: LATENT
width: int = 512
height: int = 512
method: Literal["nearest", "bilinear", "bicubic", "area"] = "bicubic"
crop: bool = False

if not isinstance(samples, dict):
    raise TypeError("samples must be a LATENT wrapper")
tensor = samples.get("samples")
if not isinstance(tensor, torch.Tensor):
    raise TypeError("LATENT is missing its samples tensor")
scale_factor = int(samples.get("vae_scale_factor", 8))
if scale_factor < 1:
    raise ValueError("LATENT has an invalid VAE scale factor")
if width < 8 or height < 8 or width > 16384 or height > 16384:
    raise ValueError("width and height must be between 8 and 16384")
if width % scale_factor != 0 or height % scale_factor != 0:
    raise ValueError(f"width and height must be multiples of {scale_factor}")

target_width = width // scale_factor
target_height = height // scale_factor
if crop:
    source_height, source_width = tensor.shape[-2:]
    target_ratio = target_width / target_height
    source_ratio = source_width / source_height
    if source_ratio > target_ratio:
        cropped_width = max(1, round(source_height * target_ratio))
        left = (source_width - cropped_width) // 2
        tensor = tensor[..., left : left + cropped_width]
    else:
        cropped_height = max(1, round(source_width / target_ratio))
        top = (source_height - cropped_height) // 2
        tensor = tensor[..., top : top + cropped_height]

kwargs = {"size": (target_height, target_width), "mode": method}
if method in {"bilinear", "bicubic"}:
    kwargs["align_corners"] = False
resized = F.interpolate(tensor, **kwargs)
latent = {
    **samples,
    "samples": resized,
    "width": width,
    "height": height,
    "batch_size": int(resized.shape[0]),
    "vae_scale_factor": scale_factor,
}

return latent: LATENT
