# /// script
# dependencies = ["Pillow>=11.0"]
# ///
from math import sqrt
from typing import Literal

from PIL import Image

image: IMAGE
megapixels: float = 1.0
round_to: Literal[0, 8, 16, 32, 64] = 8
allow_upscale: bool = True
resample: Literal[
    "auto", "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"
] = "auto"

if megapixels <= 0 or megapixels > 100:
    raise ValueError("megapixels must be greater than 0 and at most 100")
if round_to not in {0, 8, 16, 32, 64}:
    raise ValueError("round_to must be off, 8, 16, 32 or 64")

source_width, source_height = image.size
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
    scale = min(1.0, source_width / target_width, source_height / target_height)
    target_width = max(1, int(target_width * scale))
    target_height = max(1, int(target_height * scale))
if target_width > 16384 or target_height > 16384:
    raise ValueError("Target dimensions must be at most 16384 pixels")
if resample == "auto":
    method = (
        Image.Resampling.LANCZOS
        if target_width * target_height <= source_width * source_height
        else Image.Resampling.BICUBIC
    )
else:
    method = {
        "nearest": Image.Resampling.NEAREST,
        "box": Image.Resampling.BOX,
        "bilinear": Image.Resampling.BILINEAR,
        "hamming": Image.Resampling.HAMMING,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }[resample]
result = image.resize((target_width, target_height), resample=method)

return result: IMAGE
