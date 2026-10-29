# /// script
# dependencies = ["Pillow>=11.0"]
# ///
from PIL import Image, ImageOps

path: ImageFile = ""
max_pixels: int = 50_000_000

with Image.open(path) as source:
    source = ImageOps.exif_transpose(source)
    width, height = source.size
    if width * height > max_pixels:
        raise ValueError(f"Image has {width * height:,} pixels; limit is {max_pixels:,}")
    source.load()
    image = source.convert("RGB")
    if "A" in source.getbands() or "transparency" in source.info:
        mask = source.convert("RGBA").getchannel("A")
    else:
        mask = source.convert("L")
    mask = mask.convert("RGB")

return image: IMAGE, mask: IMAGE
