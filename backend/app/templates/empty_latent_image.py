# /// script
# dependencies = ["torch>=2.2"]
# ///
from math import sqrt
from typing import Literal

import torch

width: int = 512
height: int = 512
batch_size: int = 1
aspect_ratio: Literal[
    "custom",
    "1:1",
    "4:3",
    "3:2",
    "5:4",
    "16:10",
    "16:9",
    "21:9",
    "2:1",
] = "custom"
orientation: Literal["landscape", "portrait"] = "landscape"
megapixels: float = 0.0
divisible_by: Literal[8, 16, 32, 64] = 8

ASPECT_RATIOS = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:2": 3 / 2,
    "5:4": 5 / 4,
    "16:10": 16 / 10,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
    "2:1": 2.0,
}
if divisible_by not in {8, 16, 32, 64}:
    raise ValueError("divisible_by must be 8, 16, 32 or 64")
if megapixels < 0:
    raise ValueError("megapixels must be zero or greater")
if megapixels > 100:
    raise ValueError("megapixels must be at most 100")


def resolve_dimensions() -> tuple[int, int]:
    if aspect_ratio == "custom":
        if width < 8 or height < 8:
            raise ValueError("width and height must be at least 8 pixels")
        if width % divisible_by != 0 or height % divisible_by != 0:
            raise ValueError(
                f"width and height must be multiples of {divisible_by}"
            )
        return int(width), int(height)

    ratio = ASPECT_RATIOS[aspect_ratio]
    if orientation == "portrait" and ratio != 1.0:
        ratio = 1.0 / ratio
    pixel_count = (
        float(megapixels) * 1_000_000
        if megapixels > 0
        else float(width) * float(height)
    )
    target_height = sqrt(pixel_count / ratio)
    resolved_width = max(
        divisible_by,
        int(round(target_height * ratio / divisible_by)) * divisible_by,
    )
    resolved_height = max(
        divisible_by,
        int(round(target_height / divisible_by)) * divisible_by,
    )
    return resolved_width, resolved_height


width, height = resolve_dimensions()
if width < 8 or height < 8:
    raise ValueError("width and height must be at least 8 pixels")
if width > 16384 or height > 16384:
    raise ValueError("width and height must be at most 16384 pixels")
if batch_size < 1 or batch_size > 16:
    raise ValueError("batch_size must be between 1 and 16")
if width % divisible_by != 0 or height % divisible_by != 0:
    raise ValueError(f"width and height must be multiples of {divisible_by}")

scale_factor = 8
latent_width = width // scale_factor
latent_height = height // scale_factor
samples = torch.zeros(
    (batch_size, 4, latent_height, latent_width),
    dtype=torch.float32,
)
latent = {
    "kind": "latent",
    "samples": samples,
    "width": width,
    "height": height,
    "batch_size": batch_size,
    "vae_scale_factor": scale_factor,
}

return latent: LATENT
