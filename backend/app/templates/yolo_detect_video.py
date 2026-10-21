# /// script
# dependencies = ["torch>=2.2", "ultralytics>=8.3", "onnxruntime>=1.18", "opencv-python>=4.9", "imageio-ffmpeg>=0.5"]
# ///
from pathlib import Path

import cv2
import imageio_ffmpeg
from ultralytics import YOLO

video: VIDEO
model: DETECTION_MODEL
output_path: OutputFile = "annotated-video.mp4"
score_threshold: float = 0.25
max_detections: int = 50
max_frames: int = 300
frame_step: int = 1

if not isinstance(model, dict) or model.get("backend") != "ultralytics":
    raise TypeError("model must be an Ultralytics detection descriptor")
if not isinstance(video, dict) or str(video.get("media_type", "")).lower() != "video":
    raise TypeError("video must be a VIDEO value")
source = Path(str(video.get("path", "")))
if not source.is_file():
    raise FileNotFoundError(f"Video source does not exist: {source}")
if not 0.0 <= float(score_threshold) <= 1.0:
    raise ValueError("score_threshold must be between 0 and 1")
if int(max_frames) < 1 or int(frame_step) < 1:
    raise ValueError("max_frames and frame_step must be positive")

target = Path(output_path)
target.parent.mkdir(parents=True, exist_ok=True)
if target.resolve() == source.resolve():
    raise ValueError("output_path must differ from the source video")
suffix = target.suffix.lower()
if suffix not in {".mp4", ".mov", ".mkv"}:
    raise ValueError("output_path must use .mp4, .mov or .mkv")

architecture = str(model.get("architecture", "yolo11n"))
weights_name = str(model.get("weights", "DEFAULT"))
device_name = str(model.get("device", "auto"))
device = None if device_name == "auto" else device_name
model_source = str(model.get("path") or "")
if not model_source:
    model_source = (
        f"{architecture}.yaml"
        if weights_name.lower() in {"none", "null", ""}
        else f"{architecture}.pt"
    )
limit = max(1, int(max_detections))
step = max(1, int(frame_step))
frame_cap = max(1, int(max_frames))

net = YOLO(model_source)
capture = cv2.VideoCapture(str(source))
if not capture.isOpened():
    raise FileNotFoundError(f"Could not open video: {source}")
fps = float(video.get("fps", 0) or 0)
width = int(video.get("width", 0) or 0)
height = int(video.get("height", 0) or 0)
if width <= 0 or height <= 0:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
if fps <= 0:
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0

encoded_width = width + width % 2
encoded_height = height + height % 2
encoder = imageio_ffmpeg.write_frames(
    str(target),
    (encoded_width, encoded_height),
    fps=fps,
    codec="libx264",
    pix_fmt_in="bgr24",
    pix_fmt_out="yuv420p",
    macro_block_size=2,
)
encoder.send(None)

palette = [
    (68, 68, 239),
    (246, 130, 59),
    (129, 185, 16),
    (11, 158, 245),
    (247, 85, 168),
    (212, 182, 6),
]
summary = []
count = 0
processed = 0
written = 0
try:
    while written < frame_cap:
        ok, frame = capture.read()
        if not ok:
            break
        if written % step == 0:
            result = net.predict(
                source=frame,
                conf=float(score_threshold),
                max_det=limit,
                device=device,
                verbose=False,
            )[0]
            boxes = result.boxes
            names = result.names
            if boxes is not None and len(boxes):
                coordinates = boxes.xyxy.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                classes = boxes.cls.cpu().tolist()
                for box, score, label_index in zip(
                    coordinates,
                    confidences,
                    classes,
                    strict=True,
                ):
                    index = int(label_index)
                    if isinstance(names, dict):
                        label = str(names.get(index, f"class_{index}"))
                    elif index < len(names):
                        label = str(names[index])
                    else:
                        label = f"class_{index}"
                    rounded = [round(float(value), 2) for value in box]
                    count += 1
                    if len(summary) < 200:
                        summary.append(
                            {
                                "label": label,
                                "label_id": index,
                                "score": float(score),
                                "box": rounded,
                                "frame": written,
                            }
                        )
                    x1, y1, x2, y2 = rounded
                    color = palette[index % len(palette)]
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        max(2, round(min(width, height) / 200)),
                    )
                    cv2.putText(
                        frame,
                        f"{label} {float(score):.2f}",
                        (int(x1) + 2, max(12, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            processed += 1
        if (width, height) != (encoded_width, encoded_height):
            frame = cv2.resize(
                frame,
                (encoded_width, encoded_height),
                interpolation=cv2.INTER_LINEAR,
            )
        encoder.send(frame.tobytes())
        written += 1
finally:
    capture.release()
    encoder.close()

if written == 0 or not target.is_file() or target.stat().st_size == 0:
    raise RuntimeError(f"Video writer produced no data for {target.name}")

detections = {
    "count": count,
    "detections": summary,
    "frames": processed,
    "truncated": written >= frame_cap,
    "width": encoded_width,
    "height": encoded_height,
    "fps": fps,
    "model": architecture,
    "device": device_name,
    "backend": "ultralytics",
    "source": str(model.get("source", "builtin")),
}
annotated = {
    "media_type": "video",
    "path": str(target),
    "relative_path": f"outputs/{target.name}",
    "width": encoded_width,
    "height": encoded_height,
    "fps": fps,
    "frames": written,
    "duration": written / fps if fps > 0 else 0.0,
}

return detections: DETECTIONS, annotated: VIDEO
