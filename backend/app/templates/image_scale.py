# /// script
# dependencies = ["Pillow>=11.0"]
# ///
from typing import Literal

from PIL import Image, ImageOps

image: IMAGE
width: int = 512
height: int = 512
method: Literal["lanczos", "bicubic", "bilinear", "nearest", "box"] = "lanczos"
crop: bool = False

if width < 8 or height < 8:
    raise ValueError("width and height must be at least 8 pixels")
if width > 16384 or height > 16384:
    raise ValueError("width and height must be at most 16384 pixels")

resampling = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
}[method]
if crop:
    scaled = ImageOps.fit(image, (width, height), method=resampling)
else:
    scaled = image.resize((width, height), resample=resampling)

return scaled: IMAGE
