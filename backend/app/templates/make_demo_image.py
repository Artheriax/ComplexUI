# /// script
# dependencies = ["Pillow>=11.0"]
# ///
import math
import random

from PIL import Image, ImageDraw

width: int = 640
height: int = 480
seed: int = 7

if width < 64 or height < 64 or width > 4096 or height > 4096:
    raise ValueError("Demo image dimensions must be between 64 and 4096 pixels")

rng = random.Random(seed)
image = Image.new("RGB", (width, height), (15, 23, 42))
draw = ImageDraw.Draw(image)

for y in range(height):
    ratio = y / max(height - 1, 1)
    color = (
        int(15 + 28 * ratio),
        int(23 + 52 * ratio),
        int(42 + 70 * ratio),
    )
    draw.line((0, y, width, y), fill=color)

for index in range(9):
    radius = rng.randint(min(width, height) // 14, min(width, height) // 5)
    cx = rng.randint(radius, width - radius)
    cy = rng.randint(radius, height - radius)
    hue = rng.random()
    color = (
        int(80 + 150 * hue),
        int(90 + 120 * (1 - hue)),
        int(120 + 100 * abs(math.sin(index))),
    )
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline=color,
        width=max(3, radius // 10),
    )
    if index % 2 == 0:
        draw.rectangle(
            (cx - radius // 2, cy - radius // 2, cx + radius // 2, cy + radius // 2),
            fill=(color[0] // 4, color[1] // 4, color[2] // 4),
            outline=color,
            width=2,
        )

for index in range(24):
    x = rng.randint(0, width)
    y = rng.randint(0, height)
    length = rng.randint(8, 36)
    draw.line((x, y, x + length, y + length), fill=(148, 163, 184), width=1)

draw.rounded_rectangle(
    (16, height - 62, 250, height - 16),
    radius=10,
    fill=(8, 15, 30),
    outline=(34, 211, 238),
    width=2,
)
draw.text((28, height - 48), "ComplexUI vision demo", fill=(226, 232, 240))

return image: IMAGE
