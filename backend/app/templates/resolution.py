from math import sqrt
from typing import Literal

size_mode: Literal["Resolution", "Megapixels"] = "Resolution"
width: int = ShowIf(default=512, size_mode="Resolution")
height: int = ShowIf(default=512, size_mode="Resolution")
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
    size_mode="Megapixels",
)
custom_megapixels: float = ShowIf(
    default=1.0,
    size_mode="Megapixels",
    megapixel_preset="Custom",
)
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
] = ShowIf(default="1:1", size_mode="Megapixels")
custom_ratio: str = ShowIf(
    default="1:1",
    size_mode="Megapixels",
    aspect_ratio="custom",
)
orientation: Literal["landscape", "portrait"] = ShowIf(
    default="landscape",
    size_mode="Megapixels",
)
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


def aspect_value() -> float:
    if orientation not in {"landscape", "portrait"}:
        raise ValueError("orientation must be landscape or portrait")
    if aspect_ratio != "custom" and aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"Unknown aspect ratio: {aspect_ratio!r}")
    if aspect_ratio == "custom":
        parts = str(custom_ratio).replace(" ", "").split(":")
        if len(parts) != 2:
            raise ValueError("custom_ratio must use the form width:height")
        try:
            custom_width, custom_height = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise ValueError(
                "custom_ratio must use numeric width:height"
            ) from exc
        if custom_width <= 0 or custom_height <= 0:
            raise ValueError("custom_ratio values must be positive")
        return custom_width / custom_height
    return ASPECT_RATIOS[aspect_ratio]


def resolve_dimensions() -> tuple[int, int]:
    if size_mode == "Resolution":
        if width < 8 or height < 8:
            raise ValueError("width and height must be at least 8 pixels")
        if width % divisible_by != 0 or height % divisible_by != 0:
            raise ValueError(
                f"width and height must be multiples of {divisible_by}"
            )
        return int(width), int(height)
    if size_mode != "Megapixels":
        raise ValueError("size_mode must be Resolution or Megapixels")

    ratio = aspect_value()
    if orientation == "portrait" and ratio != 1.0:
        ratio = 1.0 / ratio
    target_height = sqrt(megapixels_value() * 1_000_000 / ratio)
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
if width > 16384 or height > 16384:
    raise ValueError("width and height must be at most 16384 pixels")

return width: INT, height: INT
