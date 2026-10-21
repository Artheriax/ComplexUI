# /// script
# dependencies = ["torch>=2.2", "ultralytics>=8.3", "onnxruntime>=1.18", "opencv-python>=4.9", "imageio-ffmpeg>=0.5", "numpy>=1.26"]
# ///
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

video: VIDEO
model: SEGMENTATION_MODEL
output_path: OutputFile = "segmented-video.mp4"
min_area: int = 64
max_masks: int = 20
max_frames: int = 300
frame_step: int = 1

if not isinstance(model, dict) or model.get("backend") != "ultralytics":
    raise TypeError("model must be an Ultralytics segmentation descriptor")
if not isinstance(video, dict) or str(video.get("media_type", "")).lower() != "video":
    raise TypeError("video must be a VIDEO value")
source = Path(str(video.get("path", "")))
if not source.is_file():
    raise FileNotFoundError(f"Video source does not exist: {source}")
if int(max_frames) < 1 or int(frame_step) < 1:
    raise ValueError("max_frames and frame_step must be positive")

target = Path(output_path)
target.parent.mkdir(parents=True, exist_ok=True)
if target.resolve() == source.resolve():
    raise ValueError("output_path must differ from the source video")
suffix = target.suffix.lower()
if suffix not in {".mp4", ".mov", ".mkv"}:
    raise ValueError("output_path must use .mp4, .mov or .mkv")

architecture = str(model.get("architecture", "yolo11n-seg"))
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
limit = max(1, int(max_masks))
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
    (22, 204, 132),
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
            result = net.predict(source=frame, device=device, verbose=False)[0]
            segments = result.masks
            boxes = result.boxes
            names = result.names
            candidates = []
            if segments is not None and len(segments):
                coordinates = (
                    boxes.xyxy.cpu().tolist() if boxes is not None and len(boxes) else []
                )
                confidences = (
                    boxes.conf.cpu().tolist() if boxes is not None and len(boxes) else []
                )
                classes = (
                    boxes.cls.cpu().tolist() if boxes is not None and len(boxes) else []
                )
                mask_data = segments.data
                for index in range(len(mask_data)):
                    array = mask_data[index].detach().cpu().numpy()
                    if array.shape != (height, width):
                        array = cv2.resize(
                            array.astype(np.float32),
                            (width, height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    mask = array >= 0.5
                    area = int(mask.sum())
                    if area < int(min_area):
                        continue
                    label_index = int(classes[index]) if index < len(classes) else index
                    if isinstance(names, dict):
                        label = str(names.get(label_index, f"class_{label_index}"))
                    elif label_index < len(names):
                        label = str(names[label_index])
                    else:
                        label = f"class_{label_index}"
                    score = float(confidences[index]) if index < len(confidences) else 0.0
                    box = coordinates[index] if index < len(coordinates) else [0, 0, 0, 0]
                    candidates.append(
                        {
                            "label": label,
                            "label_id": label_index,
                            "score": score,
                            "area": area,
                            "box": [round(float(value), 2) for value in box],
                            "mask_array": mask,
                        }
                    )
                candidates.sort(key=lambda item: item["area"], reverse=True)
            selected = candidates[:limit]
            for item in selected:
                mask = item.pop("mask_array")
                count += 1
                if len(summary) < 200:
                    summary.append({**item, "frame": written})
                color = np.array(palette[item["label_id"] % len(palette)], dtype=np.float32)
                frame[mask] = (
                    frame[mask].astype(np.float32) * 0.55 + color * 0.45
                ).astype(np.uint8)
                x1, y1, x2, y2 = item["box"]
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    tuple(int(value) for value in color),
                    max(2, round(min(width, height) / 200)),
                )
                cv2.putText(
                    frame,
                    f"{item['label']} {item['score']:.2f}",
                    (int(x1) + 2, max(12, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    tuple(int(value) for value in color),
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

masks = {
    "count": count,
    "masks": summary,
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

return masks: MASKS, annotated: VIDEO
