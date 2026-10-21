# /// script
# dependencies = ["opencv-python>=4.9", "imageio-ffmpeg>=0.5", "numpy>=1.26"]
# ///
import base64
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

path: OutputFile = "demo.mp4"
width: int = 320
height: int = 240
fps: int = 12
frames: int = 24
seed: int = 7

if not 64 <= int(width) <= 1920 or not 64 <= int(height) <= 1080:
    raise ValueError("Demo video dimensions must be between 64 and 1920x1080")
if not 1 <= int(fps) <= 60:
    raise ValueError("fps must be between 1 and 60")
if not 1 <= int(frames) <= 600:
    raise ValueError("frames must be between 1 and 600")

target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
suffix = target.suffix.lower()
if suffix not in {".mp4", ".mov", ".mkv"}:
    raise ValueError("Demo video path must use .mp4, .mov or .mkv")
encoded_width = int(width) + int(width) % 2
encoded_height = int(height) + int(height) % 2
encoder = imageio_ffmpeg.write_frames(
    str(target),
    (encoded_width, encoded_height),
    fps=float(fps),
    codec="libx264",
    pix_fmt_in="bgr24",
    pix_fmt_out="yuv420p",
    macro_block_size=2,
)
encoder.send(None)

rng = np.random.default_rng(int(seed))
objects = []
for index in range(4):
    objects.append(
        {
            "x": int(rng.integers(40, width - 40)),
            "y": int(rng.integers(40, height - 40)),
            "vx": int(rng.integers(-4, 5)),
            "vy": int(rng.integers(-4, 5)),
            "radius": int(rng.integers(12, 28)),
            "color": tuple(int(value) for value in rng.integers(90, 255, size=3)),
        }
    )

first_frame = None
try:
    for frame_number in range(int(frames)):
        ratio = frame_number / max(int(frames) - 1, 1)
        frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        frame[:, :] = (
            int(18 + 24 * ratio),
            int(24 + 40 * ratio),
            int(38 + 64 * ratio),
        )
        for index, item in enumerate(objects):
            item["x"] += item["vx"]
            item["y"] += item["vy"]
            if item["x"] <= item["radius"] or item["x"] >= width - item["radius"]:
                item["vx"] *= -1
            if item["y"] <= item["radius"] or item["y"] >= height - item["radius"]:
                item["vy"] *= -1
            center = (item["x"], item["y"])
            cv2.circle(frame, center, item["radius"], item["color"], -1)
            cv2.circle(frame, center, item["radius"], (240, 240, 240), 2)
            cv2.putText(
                frame,
                str(index + 1),
                (item["x"] - 5, item["y"] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (15, 23, 42),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"ComplexUI demo frame {frame_number + 1}/{frames}",
            (12, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (226, 232, 240),
            1,
            cv2.LINE_AA,
        )
        if (int(width), int(height)) != (encoded_width, encoded_height):
            frame = cv2.resize(
                frame,
                (encoded_width, encoded_height),
                interpolation=cv2.INTER_LINEAR,
            )
        encoder.send(frame.tobytes())
        if first_frame is None:
            first_frame = frame.copy()
finally:
    encoder.close()

if not target.is_file() or target.stat().st_size == 0:
    raise RuntimeError(f"Video writer produced no data for {target.name}")

poster = None
if first_frame is not None:
    encoded, buffer = cv2.imencode(".jpg", first_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if encoded:
        poster = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")

video = {
    "media_type": "video",
    "path": str(target),
    "relative_path": f"outputs/{target.name}",
    "width": encoded_width,
    "height": encoded_height,
    "fps": float(fps),
    "frames": int(frames),
    "duration": int(frames) / float(fps),
    "poster": poster,
}

return video: VIDEO
