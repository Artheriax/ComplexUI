# /// script
# dependencies = ["torch>=2.2"]
# ///
from typing import Literal

import torch
import torch.nn.functional as F

samples: LATENT
scale_by: float = 1.5
method: Literal["nearest", "bilinear", "bicubic", "area"] = "bicubic"

if not isinstance(samples, dict):
    raise TypeError("samples must be a LATENT wrapper")
tensor = samples.get("samples")
if not isinstance(tensor, torch.Tensor):
    raise TypeError("LATENT is missing its samples tensor")
scale_factor = int(samples.get("vae_scale_factor", 8))
if scale_factor < 1:
    raise ValueError("LATENT has an invalid VAE scale factor")
if scale_by <= 0 or scale_by > 16:
    raise ValueError("scale_by must be greater than 0 and at most 16")

source_height, source_width = tensor.shape[-2:]
target_width = max(1, round(source_width * scale_by))
target_height = max(1, round(source_height * scale_by))
width = target_width * scale_factor
height = target_height * scale_factor
if width > 16384 or height > 16384:
    raise ValueError("Target dimensions must be at most 16384 pixels")

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
