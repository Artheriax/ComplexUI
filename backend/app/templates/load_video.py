# /// script
# dependencies = ["opencv-python>=4.9"]
# ///
import base64

import cv2

path: VideoFile = ""
max_pixels: int = 50_000_000

capture = cv2.VideoCapture(path)
if not capture.isOpened():
    raise FileNotFoundError(f"Could not open video: {path}")
try:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise ValueError("Video has invalid dimensions")
    if width * height > int(max_pixels):
        raise ValueError(f"Video frame has {width * height:,} pixels; limit is {max_pixels:,}")
    duration = frames / fps if fps > 0 and frames > 0 else 0.0

    poster = None
    ok, first = capture.read()
    if ok:
        scale = min(1.0, 640.0 / max(first.shape[0], first.shape[1]))
        if scale < 1.0:
            first = cv2.resize(
                first,
                (round(first.shape[1] * scale), round(first.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        encoded, buffer = cv2.imencode(".jpg", first, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if encoded:
            poster = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")
finally:
    capture.release()

video = {
    "media_type": "video",
    "path": str(path),
    "width": width,
    "height": height,
    "fps": fps,
    "frames": frames,
    "duration": duration,
    "poster": poster,
}

return video: VIDEO
