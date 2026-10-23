# /// script
# dependencies = ["Pillow>=11.0"]
# ///
from math import sqrt
from typing import Literal

from PIL import Image, ImageOps

image: IMAGE
mode: Literal[
    "By Dimensions",
    "By Side",
    "By Megapixels",
    "By Multiplier",
] = "By Dimensions"
width: int = ShowIf(default=512, mode="By Dimensions")
height: int = ShowIf(default=512, mode="By Dimensions")
crop: bool = ShowIf(default=False, mode="By Dimensions")
side: Literal["longer", "shorter"] = ShowIf(default="longer", mode="By Side")
side_resolution: int = ShowIf(default=1024, mode="By Side")
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
] = ShowIf(default="keep", mode="By Side")
custom_ratio: str = ShowIf(
    default="1:1",
    mode="By Side",
    aspect_ratio="custom",
)
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
] = ShowIf(default="center", mode="By Side")
allow_upscale: bool = ShowIf(
    default=True,
    mode=["By Side", "By Megapixels"],
)
round_to: Literal[0, 8, 16, 32, 64] = ShowIf(
    default=8,
    mode=["By Side", "By Megapixels"],
)
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
scale_by: float = ShowIf(default=1.5, mode="By Multiplier")
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
if mode not in {
    "By Dimensions",
    "By Side",
    "By Megapixels",
    "By Multiplier",
}:
    raise ValueError(f"Unknown resize mode: {mode!r}")
if round_to not in {0, 8, 16, 32, 64}:
    raise ValueError("round_to must be off, 8, 16, 32 or 64")
if resample not in {
    "auto",
    "nearest",
    "box",
    "bilinear",
    "hamming",
    "bicubic",
    "lanczos",
}:
    raise ValueError(f"Unknown resample filter: {resample!r}")

source_width, source_height = image.size
if source_width < 1 or source_height < 1:
    raise ValueError("image must have a positive size")


def rounded_dimension(value: float) -> int:
    if round_to == 0:
        return max(1, int(round(value)))
    return max(round_to, int(round(value / round_to)) * round_to)


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


def custom_aspect_ratio() -> float:
    parts = str(custom_ratio).replace(" ", "").split(":")
    if len(parts) != 2:
        raise ValueError("custom_ratio must use the form width:height")
    try:
        custom_width, custom_height = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError("custom_ratio must use numeric width:height") from exc
    if custom_width <= 0 or custom_height <= 0:
        raise ValueError("custom_ratio values must be positive")
    return custom_width / custom_height


def resample_method(upscaling: bool) -> Image.Resampling:
    if resample == "auto":
        if mode == "By Dimensions":
            return Image.Resampling.LANCZOS
        return (
            Image.Resampling.BICUBIC
            if upscaling
            else Image.Resampling.LANCZOS
        )
    return {
        "nearest": Image.Resampling.NEAREST,
        "box": Image.Resampling.BOX,
        "bilinear": Image.Resampling.BILINEAR,
        "hamming": Image.Resampling.HAMMING,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }[resample]


if mode == "By Dimensions":
    if width < 8 or height < 8:
        raise ValueError("width and height must be at least 8 pixels")
    if width > 16384 or height > 16384:
        raise ValueError("width and height must be at most 16384 pixels")
    method = resample_method(width * height > source_width * source_height)
    if crop:
        result = ImageOps.fit(image, (width, height), method=method)
    else:
        result = image.resize((width, height), resample=method)
elif mode == "By Side":
    if side not in {"longer", "shorter"}:
        raise ValueError("side must be longer or shorter")
    if side_resolution < 8 or side_resolution > 16384:
        raise ValueError("side_resolution must be between 8 and 16384")
    if aspect_ratio not in {"keep", "custom", *RATIOS}:
        raise ValueError(f"Unknown aspect ratio: {aspect_ratio!r}")
    if crop_from not in ANCHORS:
        raise ValueError(f"Unknown crop anchor: {crop_from!r}")
    if aspect_ratio == "custom":
        target_ratio = custom_aspect_ratio()
    elif aspect_ratio == "keep":
        target_ratio = source_width / source_height
    else:
        target_ratio = RATIOS[aspect_ratio]
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
            target_width = (
                side_resolution
                if target_ratio >= 1
                else side_resolution * target_ratio
            )
            target_height = (
                side_resolution / target_ratio
                if target_ratio >= 1
                else side_resolution
            )
        else:
            target_width = (
                side_resolution * target_ratio
                if target_ratio >= 1
                else side_resolution
            )
            target_height = (
                side_resolution
                if target_ratio >= 1
                else side_resolution / target_ratio
            )
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
    method = resample_method(
        target_width * target_height > source_width * source_height
    )
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
        scale = min(
            1.0,
            source_width / target_width,
            source_height / target_height,
        )
        target_width = max(1, int(target_width * scale))
        target_height = max(1, int(target_height * scale))
    if target_width > 16384 or target_height > 16384:
        raise ValueError("Target dimensions must be at most 16384 pixels")
    method = resample_method(
        target_width * target_height > source_width * source_height
    )
    result = image.resize((target_width, target_height), resample=method)
else:
    if scale_by <= 0 or scale_by > 16:
        raise ValueError("scale_by must be greater than 0 and at most 16")
    target_width = max(1, round(source_width * scale_by))
    target_height = max(1, round(source_height * scale_by))
    if target_width > 16384 or target_height > 16384:
        raise ValueError("Target dimensions must be at most 16384 pixels")
    method = resample_method(
        target_width * target_height > source_width * source_height
    )
    result = image.resize((target_width, target_height), resample=method)

return result: IMAGE
