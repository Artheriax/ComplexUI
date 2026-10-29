# /// script
# dependencies = ["Pillow>=11.0"]
# ///
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter

width: int = 512
height: int = 512
shape: Literal["ellipse", "rectangle"] = "ellipse"
margin: int = 96
feather: float = 0.0

if width < 8 or height < 8 or width > 16384 or height > 16384:
    raise ValueError("Mask dimensions must be between 8 and 16384 pixels")
if margin < 0 or margin * 2 >= min(width, height):
    raise ValueError("margin must leave a positive mask area")
if feather < 0 or feather > 0.5:
    raise ValueError("feather must be between 0 and 0.5")

mask = Image.new("L", (width, height), 0)
draw = ImageDraw.Draw(mask)
box = (margin, margin, width - margin - 1, height - margin - 1)
if shape == "rectangle":
    draw.rectangle(box, fill=255)
else:
    draw.ellipse(box, fill=255)

if feather > 0:
    radius = max(1, int(min(width, height) * feather))
    mask = mask.filter(ImageFilter.GaussianBlur(radius))

image = mask.convert("RGB")
return image: IMAGE
