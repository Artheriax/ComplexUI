# /// script
# dependencies = ["Pillow>=11.0"]
# ///
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal

from PIL import Image

media: MEDIA
folder: str = "%date%"
filename: str = "%model%_%counter%"
model_name: str = "image"
format: Literal["png", "jpg", "webp", "bmp", "tiff", "gif"] = "png"
quality: int = 95
compress_level: int = 6
optimize: bool = True
save_nonce: int = 0

IMAGE_FORMATS = {
    "png": (".png", "PNG"),
    "jpg": (".jpg", "JPEG"),
    "webp": (".webp", "WEBP"),
    "bmp": (".bmp", "BMP"),
    "tiff": (".tiff", "TIFF"),
    "gif": (".gif", "GIF"),
}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_SUFFIXES = {suffix for suffix, _ in IMAGE_FORMATS.values()}
MEDIA_SUFFIXES = {*IMAGE_SUFFIXES, *VIDEO_SUFFIXES, ".jpeg", ".tif"}
TOKEN_PATTERN = re.compile(r"%([A-Za-z_]+)%")
DISABLED_TOKENS = {"seed", "input_name", "input_folder"}
MAX_OUTPUT_DEPTH = 8
MAX_OUTPUT_PATH_CHARS = 512

try:
    quality = int(quality)
    compress_level = int(compress_level)
except (TypeError, ValueError) as exc:
    raise ValueError("quality and compress_level must be integers") from exc
if quality < 1 or quality > 100:
    raise ValueError("quality must be between 1 and 100")
if compress_level < 0 or compress_level > 9:
    raise ValueError("compress_level must be between 0 and 9")
if format not in IMAGE_FORMATS:
    raise ValueError(f"Unsupported image format: {format!r}")

if isinstance(media, Image.Image):
    images = [media]
    video = None
elif isinstance(media, (list, tuple)) and media and all(
    isinstance(item, Image.Image) for item in media
):
    images = list(media)
    video = None
elif (
    isinstance(media, dict)
    and str(media.get("media_type", "")).lower() == "video"
):
    images = []
    video = media
else:
    raise TypeError("save_media expects an IMAGE, IMAGE_BATCH or VIDEO value")

if images:
    width: int | None = int(images[0].width)
    height: int | None = int(images[0].height)
    count = len(images)
else:
    width = int(video.get("width")) if video.get("width") else None
    height = int(video.get("height")) if video.get("height") else None
    count = 1

now = datetime.now()
day = now.strftime("%Y-%m-%d")
time_text = now.strftime("%H-%M-%S")
model = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name)).strip("._-")[:60]
if not model:
    model = "image" if video is None else "video"


def sanitize_segment(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip()
    cleaned = cleaned.strip(" .")
    return cleaned[:120]


def replace_tokens(
    template: str,
    *,
    allow_counter: bool,
    counter: int = 0,
    batch: int = 0,
) -> str:
    pieces: list[str] = []
    cursor = 0
    for match in TOKEN_PATTERN.finditer(template):
        pieces.append(template[cursor : match.start()])
        token = match.group(1).lower()
        if token == "counter":
            if not allow_counter:
                raise ValueError("%counter% can only be used in the filename")
            value = f"{counter:03d}"
        elif token in {"batch", "index"}:
            if not allow_counter:
                raise ValueError("%batch% can only be used in the filename")
            value = f"{batch:03d}"
        elif token == "date":
            value = day
        elif token == "time":
            value = time_text
        elif token == "model":
            value = model
        elif token == "width":
            if width is None:
                raise ValueError("%width% requires known media dimensions")
            value = str(width)
        elif token == "height":
            if height is None:
                raise ValueError("%height% requires known media dimensions")
            value = str(height)
        elif token in DISABLED_TOKENS:
            raise ValueError(
                f"%{token}% requires metadata support and is not available yet"
            )
        else:
            raise ValueError(f"Unknown filename token: %{token}%")
        pieces.append(value)
        cursor = match.end()
    pieces.append(template[cursor:])
    return "".join(pieces)


def has_token(template: str, name: str) -> bool:
    return any(
        match.group(1).lower() == name
        for match in TOKEN_PATTERN.finditer(template)
    )


def strip_media_suffix(value: str) -> str:
    lower = value.lower()
    for suffix in sorted(MEDIA_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return value[: -len(suffix)]
    return value


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def save_image(image: Image.Image, target: Path, image_format: str) -> None:
    if image_format in {"JPEG", "BMP"}:
        image = to_rgb(image)
    elif image_format == "GIF" and image.mode not in {"P", "L"}:
        image = image.convert("P", palette=Image.Palette.ADAPTIVE)
    options = {}
    if image_format in {"JPEG", "WEBP"}:
        options["quality"] = quality
    if image_format == "PNG":
        options["compress_level"] = compress_level
    if optimize and image_format in {"PNG", "JPEG"}:
        options["optimize"] = True
    image.save(target, format=image_format, **options)


folder_template = str(folder).strip()
folder_value = replace_tokens(folder_template, allow_counter=False)
output_root = Path(outputs_root).expanduser().resolve()
if folder_value.strip():
    folder_path = Path(folder_value).expanduser()
    is_absolute = folder_path.is_absolute()
    if not is_absolute:
        folder_path = output_root / folder_path
    folder_path = folder_path.resolve()
    if not is_absolute and not folder_path.is_relative_to(output_root):
        raise ValueError("Relative output folders must stay inside outputs")
else:
    folder_path = output_root
if folder_path.exists() and not folder_path.is_dir():
    raise ValueError(f"Output folder is not a directory: {folder_path}")
folder_path.mkdir(parents=True, exist_ok=True)

filename_template = str(filename).strip()
if not filename_template:
    raise ValueError("Filename template is required")
if not has_token(filename_template, "counter"):
    raise ValueError("Filename template must include %counter%")

raw_segments = [
    segment for segment in filename_template.split("/") if segment not in {"", "."}
]
if not raw_segments:
    raise ValueError("Filename template must contain a file name")
last_segment = strip_media_suffix(raw_segments[-1])
if count > 1 and not (
    has_token(filename_template, "batch")
    or has_token(filename_template, "index")
):
    last_segment = f"{last_segment}_%batch%"
raw_segments[-1] = last_segment

if video is not None:
    source = Path(str(video.get("path", "")))
    if not source.is_file():
        relative = str(video.get("relative_path") or video.get("path") or "")
        if relative.startswith("outputs/"):
            source = (output_root / relative[len("outputs/") :]).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video source does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError(f"Cannot save a video with extension '{suffix}'")
    image_format = "VIDEO"
else:
    suffix, image_format = IMAGE_FORMATS[format]


def build_targets(counter: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        current = folder_path
        expanded: list[str] = []
        for segment_template in raw_segments:
            value = replace_tokens(
                segment_template,
                allow_counter=True,
                counter=counter,
                batch=index if count > 1 else 0,
            ).strip()
            if value in {".", ".."}:
                raise ValueError("Filename segments may not be '.' or '..'")
            segment = sanitize_segment(value)
            if not segment:
                continue
            expanded.append(segment)
        if not expanded:
            raise ValueError("Filename resolved to an empty name")
        final_name = strip_media_suffix(expanded[-1])
        if not final_name:
            raise ValueError("Filename resolved to an empty name")
        for segment in expanded[:-1]:
            current = current / segment
        target = current / f"{final_name}{suffix}"
        try:
            relative = target.resolve().relative_to(output_root)
        except ValueError:
            pass
        else:
            if len(relative.parts) > MAX_OUTPUT_DEPTH + 1 or len(
                relative.as_posix()
            ) > MAX_OUTPUT_PATH_CHARS:
                raise ValueError(
                    "Output path is too long or too deep for the outputs library"
                )
        paths.append(target)
    return paths


targets = []
for counter in range(1, 10_001):
    candidate = build_targets(counter)
    if not any(target.exists() for target in candidate):
        targets = candidate
        break
if not targets:
    raise RuntimeError("Could not allocate a unique output file name")


def public_path(target: Path) -> str:
    resolved = target.resolve()
    try:
        relative = resolved.relative_to(output_root)
    except ValueError:
        return str(resolved)
    return f"outputs/{relative.as_posix()}"


for target in targets:
    target.parent.mkdir(parents=True, exist_ok=True)

if video is not None:
    target = targets[0]
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    saved_path = public_path(target)
    saved = {
        **video,
        "path": saved_path,
        "relative_path": saved_path if saved_path.startswith("outputs/") else None,
    }
else:
    saved_paths = []
    for image, target in zip(images, targets, strict=True):
        save_image(image, target, image_format)
        saved_paths.append(public_path(target))
    saved_path = saved_paths[0]
    saved = images[0] if len(images) == 1 else images

return saved_path: STRING, saved: MEDIA
