# /// script
# dependencies = ["opencv-python>=4.9", "Pillow>=11.0"]
# ///
from pathlib import Path

import cv2
from PIL import Image

video: VIDEO
frame_index: int = 0

if not isinstance(video, dict) or str(video.get("media_type", "")).lower() != "video":
    raise TypeError("video must be a VIDEO value")
source = Path(str(video.get("path", "")))
if not source.is_file():
    raise FileNotFoundError(f"Video source does not exist: {source}")

frames = int(video.get("frames", 0) or 0)
index = max(0, int(frame_index))
if frames > 0:
    index = min(index, frames - 1)

capture = cv2.VideoCapture(str(source))
try:
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Could not read frame {index} from {source.name}")
finally:
    capture.release()

image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

return image: IMAGE
