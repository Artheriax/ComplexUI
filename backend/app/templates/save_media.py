# /// script
# dependencies = ["Pillow>=11.0"]
# ///
import shutil
from pathlib import Path

from PIL import Image

media: MEDIA
path: OutputFile = "media.png"

target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
suffix = target.suffix.lower()
image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
video_suffixes = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}

if isinstance(media, Image.Image):
    if suffix not in image_suffixes:
        raise ValueError(f"Cannot save an image with extension '{suffix}'")
    image = media
    if suffix in {".jpg", ".jpeg"} and image.mode != "RGB":
        image = image.convert("RGB")
    image.save(target)
    saved = image
elif isinstance(media, dict) and str(media.get("media_type", "")).lower() == "video":
    if suffix not in video_suffixes:
        raise ValueError(f"Cannot save a video with extension '{suffix}'")
    source = Path(str(media.get("path", "")))
    if not source.is_file():
        raise FileNotFoundError(f"Video source does not exist: {source}")
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    saved = {**media, "path": f"outputs/{target.name}"}
else:
    raise TypeError("save_media expects an IMAGE or VIDEO value")

saved_path = f"outputs/{target.name}"

return saved_path: STRING, saved: MEDIA
