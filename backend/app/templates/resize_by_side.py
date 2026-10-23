# /// script
# dependencies = ["Pillow>=11.0"]
# ///
from typing import Literal

from PIL import Image

image: IMAGE
side: Literal["longer", "shorter"] = "longer"
side_resolution: int = 1024
round_to: Literal[0, 8, 16, 32, 64] = 8
aspect_ratio: Literal[
    "keep",
    "1:1",
    "16:9",
    "9:16",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "21:9",
    "9:21",
    "1:2",
    "2:1",
    "custom",
] = "keep"
custom_ratio: str = "1:1"
crop_from: Literal[
    "center",
    "top-left",
    "top-center",
    "top-right",
    "left-center",
    "right-center",
    "bottom-left",
    "bottom-center",
    "bottom-right",
] = "center"
allow_upscale: bool = False
resample: Literal[
    "auto", "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"
] = "auto"

RATIOS = {
    "1:1": 1.0,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
    "3:4": 3 / 4,
    "4:3": 4 / 3,
    "4:5": 4 / 5,
    "5:4": 5 / 4,
    "21:9": 21 / 9,
    "9:21": 9 / 21,
    "1:2": 1 / 2,
    "2:1": 2.0,
}
if round_to not in {0, 8, 16, 32, 64}:
    raise ValueError("round_to must be off, 8, 16, 32 or 64")
if side_resolution < 8 or side_resolution > 16384:
    raise ValueError("side_resolution must be between 8 and 16384")
if aspect_ratio == "custom":
    parts = custom_ratio.replace(" ", "").split(":")
    if len(parts) != 2:
        raise ValueError("custom_ratio must use the form width:height")
    try:
        custom_width, custom_height = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError("custom_ratio must use numeric width:height") from exc
    if custom_width <= 0 or custom_height <= 0:
        raise ValueError("custom_ratio values must be positive")
    target_ratio = custom_width / custom_height
elif aspect_ratio == "keep":
    target_ratio = image.width / image.height
else:
    target_ratio = RATIOS[aspect_ratio]


def rounded_dimension(value: float) -> int:
    if round_to == 0:
        return max(1, int(round(value)))
    return max(round_to, int(round(value / round_to)) * round_to)


source_width, source_height = image.size
if aspect_ratio == "keep":
    scale = side_resolution / (
        max(source_width, source_height)
        if side == "longer"
        else min(source_width, source_height)
    )
    target_width = source_width * scale
    target_height = source_height * scale
else:
    if side == "longer":
        target_width = side_resolution if target_ratio >= 1 else side_resolution * target_ratio
        target_height = side_resolution / target_ratio if target_ratio >= 1 else side_resolution
    else:
        target_width = side_resolution * target_ratio if target_ratio >= 1 else side_resolution
        target_height = side_resolution if target_ratio >= 1 else side_resolution / target_ratio

if not allow_upscale:
    fit_scale = min(
        1.0,
        source_width / max(target_width, 1),
        source_height / max(target_height, 1),
    )
    target_width *= fit_scale
    target_height *= fit_scale

target_width = rounded_dimension(target_width)
target_height = rounded_dimension(target_height)
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

if aspect_ratio == "keep":
    result = image.resize((target_width, target_height), resample=method)
else:
    source_ratio = source_width / source_height
    if source_ratio > target_ratio:
        crop_height = source_height
        crop_width = max(1, round(source_height * target_ratio))
    else:
        crop_width = source_width
        crop_height = max(1, round(source_width / target_ratio))
    ANCHORS = {
        "center": ("center", "center"),
        "top-left": ("left", "top"),
        "top-center": ("center", "top"),
        "top-right": ("right", "top"),
        "left-center": ("left", "center"),
        "right-center": ("right", "center"),
        "bottom-left": ("left", "bottom"),
        "bottom-center": ("center", "bottom"),
        "bottom-right": ("right", "bottom"),
    }
    horizontal, vertical = ANCHORS[crop_from]
    if horizontal == "left":
        left = 0
    elif horizontal == "right":
        left = source_width - crop_width
    else:
        left = (source_width - crop_width) // 2
    if vertical == "top":
        top = 0
    elif vertical == "bottom":
        top = source_height - crop_height
    else:
        top = (source_height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    result = cropped.resize((target_width, target_height), resample=method)

return result: IMAGE
