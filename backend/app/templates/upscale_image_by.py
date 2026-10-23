# /// script
# dependencies = ["Pillow>=11.0"]
# ///
from typing import Literal

from PIL import Image

image: IMAGE
scale_by: float = 1.5
resample: Literal[
    "auto", "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"
] = "auto"

if scale_by <= 0 or scale_by > 16:
    raise ValueError("scale_by must be greater than 0 and at most 16")
width = max(1, round(image.width * scale_by))
height = max(1, round(image.height * scale_by))
if width > 16384 or height > 16384:
    raise ValueError("Target dimensions must be at most 16384 pixels")

if resample == "auto":
    method = Image.Resampling.LANCZOS if scale_by < 1 else Image.Resampling.BICUBIC
else:
    method = {
        "nearest": Image.Resampling.NEAREST,
        "box": Image.Resampling.BOX,
        "bilinear": Image.Resampling.BILINEAR,
        "hamming": Image.Resampling.HAMMING,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }[resample]
result = image.resize((width, height), resample=method)

return result: IMAGE
