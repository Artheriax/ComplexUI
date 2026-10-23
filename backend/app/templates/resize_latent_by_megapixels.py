# /// script
# dependencies = ["torch>=2.2"]
# ///
from math import sqrt
from typing import Literal

import torch
import torch.nn.functional as F

samples: LATENT
megapixels: float = 1.0
round_to: Literal[0, 8, 16, 32, 64] = 8
allow_upscale: bool = True
method: Literal["nearest", "bilinear", "bicubic", "area"] = "bicubic"

if not isinstance(samples, dict):
    raise TypeError("samples must be a LATENT wrapper")
tensor = samples.get("samples")
if not isinstance(tensor, torch.Tensor):
    raise TypeError("LATENT is missing its samples tensor")
scale_factor = int(samples.get("vae_scale_factor", 8))
if scale_factor < 1:
    raise ValueError("LATENT has an invalid VAE scale factor")
if megapixels <= 0 or megapixels > 100:
    raise ValueError("megapixels must be greater than 0 and at most 100")
if round_to not in {0, 8, 16, 32, 64}:
    raise ValueError("round_to must be off, 8, 16, 32 or 64")

source_width = int(samples.get("width") or tensor.shape[-1] * scale_factor)
source_height = int(samples.get("height") or tensor.shape[-2] * scale_factor)
ratio = source_width / source_height
target_height = sqrt(megapixels * 1_000_000 / ratio)
target_width = target_height * ratio
if round_to:
    target_width = max(round_to, int(round(target_width / round_to)) * round_to)
    target_height = max(round_to, int(round(target_height / round_to)) * round_to)
else:
    target_width = max(1, int(round(target_width)))
    target_height = max(1, int(round(target_height)))
if not allow_upscale:
    fit = min(1.0, source_width / target_width, source_height / target_height)
    target_width = max(1, int(target_width * fit))
    target_height = max(1, int(target_height * fit))
if target_width > 16384 or target_height > 16384:
    raise ValueError("Target dimensions must be at most 16384 pixels")

target_latent_width = max(1, round(target_width / scale_factor))
target_latent_height = max(1, round(target_height / scale_factor))
kwargs = {"size": (target_latent_height, target_latent_width), "mode": method}
if method in {"bilinear", "bicubic"}:
    kwargs["align_corners"] = False
resized = F.interpolate(tensor, **kwargs)
latent = {
    **samples,
    "samples": resized,
    "width": target_latent_width * scale_factor,
    "height": target_latent_height * scale_factor,
    "batch_size": int(resized.shape[0]),
    "vae_scale_factor": scale_factor,
}

return latent: LATENT
