# /// script
# dependencies = ["torch>=2.2", "ultralytics>=8.3", "onnxruntime>=1.18", "Pillow>=11.0"]
# ///
from PIL import ImageDraw
from ultralytics import YOLO

image: IMAGE
model: DETECTION_MODEL
score_threshold: float = 0.25
max_detections: int = 50

if not isinstance(model, dict) or model.get("backend") != "ultralytics":
    raise TypeError("model must be an Ultralytics detection descriptor")
if not hasattr(image, "convert"):
    raise TypeError("image must be a PIL image")
if not 0.0 <= float(score_threshold) <= 1.0:
    raise ValueError("score_threshold must be between 0 and 1")

architecture = str(model.get("architecture", "yolo11n"))
weights_name = str(model.get("weights", "DEFAULT"))
device_name = str(model.get("device", "auto"))
source = str(model.get("path") or "")
if not source:
    source = (
        f"{architecture}.yaml"
        if weights_name.lower() in {"none", "null", ""}
        else f"{architecture}.pt"
    )
device = None if device_name == "auto" else device_name
limit = max(1, int(max_detections))

net = YOLO(source)
results = net.predict(
    source=image,
    conf=float(score_threshold),
    max_det=limit,
    device=device,
    verbose=False,
)
result = results[0]
names = result.names
boxes = result.boxes

palette = [
    (239, 68, 68),
    (59, 130, 246),
    (16, 185, 129),
    (245, 158, 11),
    (168, 85, 247),
    (6, 182, 212),
    (236, 72, 153),
]
annotated = image.convert("RGB").copy()
draw = ImageDraw.Draw(annotated)
line_width = max(2, round(min(annotated.size) / 200))
items = []
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
        items.append(
            {
                "label": label,
                "label_id": index,
                "score": float(score),
                "box": rounded,
            }
        )
        color = palette[index % len(palette)]
        x1, y1, x2, y2 = rounded
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        text_y = max(0, int(y1) - 12)
        draw.text((int(x1) + 2, text_y), f"{label} {float(score):.2f}", fill=color)

detections = {
    "count": len(items),
    "detections": items,
    "width": int(image.width),
    "height": int(image.height),
    "model": architecture,
    "device": device_name,
    "backend": "ultralytics",
    "source": str(model.get("source", "builtin")),
}

return detections: DETECTIONS, annotated: IMAGE
