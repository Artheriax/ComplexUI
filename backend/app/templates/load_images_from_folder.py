# /// script
# dependencies = ["Pillow>=11.0"]
# ///
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps

folder: FolderPath = ""
max_images: int = 32
max_pixels: int = 50_000_000
sort_by: Literal["name", "modified"] = "name"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
if not folder:
    raise ValueError("Choose a folder containing images")
if max_images < 1 or max_images > 256:
    raise ValueError("max_images must be between 1 and 256")
if sort_by not in {"name", "modified"}:
    raise ValueError("sort_by must be 'name' or 'modified'")
if max_pixels < 1:
    raise ValueError("max_pixels must be positive")
root = Path(folder)
if not root.is_dir():
    raise FileNotFoundError(f"Image folder does not exist: {folder}")

files = sorted(
    (
        path
        for path in root.iterdir()
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in IMAGE_SUFFIXES
    ),
    key=(
        (lambda path: path.name.lower())
        if sort_by == "name"
        else (lambda path: (path.stat().st_mtime_ns, path.name.lower()))
    ),
)
if not files:
    raise ValueError(f"No supported images found in {folder}")
if len(files) > max_images:
    raise ValueError(
        f"Folder has {len(files)} images; increase max_images (currently {max_images})"
    )

images = []
for path in files:
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        width, height = source.size
        if width * height > max_pixels:
            raise ValueError(
                f"{path.name} has {width * height:,} pixels; limit is {max_pixels:,}"
            )
        source.load()
        images.append(source.convert("RGB"))

return images: IMAGE_BATCH
