# /// script
# dependencies = ["torch>=2.2"]
# ///
from math import sqrt
from typing import Literal

import torch
import torch.nn.functional as F

samples: LATENT
mode: Literal[
    "By Dimensions",
    "By Megapixels",
    "By Multiplier",
] = "By Dimensions"
width: int = ShowIf(default=512, mode="By Dimensions")
height: int = ShowIf(default=512, mode="By Dimensions")
crop: bool = ShowIf(default=False, mode="By Dimensions")
megapixel_preset: Literal[
    "0.26 MP - Preview",
    "0.36 MP - Small",
    "0.52 MP - SD",
    "0.65 MP - Balanced",
    "0.83 MP - HD",
    "1.00 MP - 1024p",
    "1.05 MP - HD+",
    "1.20 MP - HD++",
    "1.35 MP - 2K Lite",
    "1.55 MP - 2K",
    "1.65 MP - 2K+",
    "1.75 MP - QHD",
    "2.10 MP - FHD",
    "3.30 MP - QHD+",
    "4.75 MP - 2K Pro",
    "6.50 MP - Production",
    "8.30 MP - UHD",
    "Custom",
] = ShowIf(
    default="1.00 MP - 1024p",
    mode="By Megapixels",
)
custom_megapixels: float = ShowIf(
    default=1.0,
    mode="By Megapixels",
    megapixel_preset="Custom",
)
round_to: Literal[0, 8, 16, 32, 64] = ShowIf(
    default=8,
    mode="By Megapixels",
)
allow_upscale: bool = ShowIf(default=True, mode="By Megapixels")
scale_by: float = ShowIf(default=1.5, mode="By Multiplier")
method: Literal["nearest", "bilinear", "bicubic", "area"] = "bicubic"

if not isinstance(samples, dict):
    raise TypeError("samples must be a LATENT wrapper")
tensor = samples.get("samples")
if not isinstance(tensor, torch.Tensor):
    raise TypeError("LATENT is missing its samples tensor")
scale_factor = int(samples.get("vae_scale_factor", 8))
if scale_factor < 1:
    raise ValueError("LATENT has an invalid VAE scale factor")
if mode not in {"By Dimensions", "By Megapixels", "By Multiplier"}:
    raise ValueError(f"Unknown resize mode: {mode!r}")
if method not in {"nearest", "bilinear", "bicubic", "area"}:
    raise ValueError(f"Unknown interpolation method: {method!r}")
if round_to not in {0, 8, 16, 32, 64}:
    raise ValueError("round_to must be off, 8, 16, 32 or 64")

source_width = int(samples.get("width") or tensor.shape[-1] * scale_factor)
source_height = int(samples.get("height") or tensor.shape[-2] * scale_factor)


def megapixels_value() -> float:
    if megapixel_preset == "Custom":
        value = float(custom_megapixels)
    else:
        try:
            value = float(str(megapixel_preset).split(" ", 1)[0])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unknown megapixel preset: {megapixel_preset!r}"
            ) from exc
    if value <= 0:
        raise ValueError("megapixels must be greater than zero")
    if value > 100:
        raise ValueError("megapixels must be at most 100")
    return value


if mode == "By Dimensions":
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
    output_width = width
    output_height = height
elif mode == "By Megapixels":
    ratio = source_width / source_height
    target_height = sqrt(megapixels_value() * 1_000_000 / ratio)
    target_width = target_height * ratio
    if round_to:
        target_width = max(
            round_to,
            int(round(target_width / round_to)) * round_to,
        )
        target_height = max(
            round_to,
            int(round(target_height / round_to)) * round_to,
        )
    else:
        target_width = max(1, int(round(target_width)))
        target_height = max(1, int(round(target_height)))
    if not allow_upscale:
        fit = min(
            1.0,
            source_width / target_width,
            source_height / target_height,
        )
        target_width = max(1, int(target_width * fit))
        target_height = max(1, int(target_height * fit))
    if target_width > 16384 or target_height > 16384:
        raise ValueError("Target dimensions must be at most 16384 pixels")
    target_width = max(1, round(target_width / scale_factor))
    target_height = max(1, round(target_height / scale_factor))
    output_width = target_width * scale_factor
    output_height = target_height * scale_factor
else:
    if scale_by <= 0 or scale_by > 16:
        raise ValueError("scale_by must be greater than 0 and at most 16")
    target_width = max(1, round(tensor.shape[-1] * scale_by))
    target_height = max(1, round(tensor.shape[-2] * scale_by))
    output_width = target_width * scale_factor
    output_height = target_height * scale_factor
    if output_width > 16384 or output_height > 16384:
        raise ValueError("Target dimensions must be at most 16384 pixels")

kwargs = {"size": (target_height, target_width), "mode": method}
if method in {"bilinear", "bicubic"}:
    kwargs["align_corners"] = False
resized = F.interpolate(tensor, **kwargs)
latent = {
    **samples,
    "samples": resized,
    "width": output_width,
    "height": output_height,
    "batch_size": int(resized.shape[0]),
    "vae_scale_factor": scale_factor,
}

return latent: LATENT
