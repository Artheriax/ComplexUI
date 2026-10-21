# /// script
# dependencies = ["torch>=2.2", "ultralytics>=8.3", "onnxruntime>=1.18", "Pillow>=11.0", "numpy>=1.26"]
# ///
import base64
import io

import numpy as np
from PIL import Image
from ultralytics import YOLO

image: IMAGE
model: SEGMENTATION_MODEL
min_area: int = 64
max_masks: int = 20

if not isinstance(model, dict) or model.get("backend") != "ultralytics":
    raise TypeError("model must be an Ultralytics segmentation descriptor")
if not hasattr(image, "convert"):
    raise TypeError("image must be a PIL image")

architecture = str(model.get("architecture", "yolo11n-seg"))
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
limit = max(1, int(max_masks))

net = YOLO(source)
results = net.predict(source=image, device=device, verbose=False)
result = results[0]
names = result.names
boxes = result.boxes
segments = result.masks
width, height = image.size

box_coordinates = boxes.xyxy.cpu().tolist() if boxes is not None and len(boxes) else []
box_scores = boxes.conf.cpu().tolist() if boxes is not None and len(boxes) else []
box_classes = boxes.cls.cpu().tolist() if boxes is not None and len(boxes) else []

candidates = []
if segments is not None and len(segments):
    mask_data = segments.data
    for index in range(len(mask_data)):
        array = mask_data[index].detach().cpu().numpy()
        if array.shape != (height, width):
            array = np.asarray(
                Image.fromarray(array.astype(np.float32)).resize(
                    (width, height),
                    Image.Resampling.BILINEAR,
                )
            )
        mask = array >= 0.5
        area = int(mask.sum())
        if area < int(min_area):
            continue
        ys, xs = np.nonzero(mask)
        bound = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        label_index = int(box_classes[index]) if index < len(box_classes) else index
        if isinstance(names, dict):
            label = str(names.get(label_index, f"class_{label_index}"))
        elif label_index < len(names):
            label = str(names[label_index])
        else:
            label = f"class_{label_index}"
        score = float(box_scores[index]) if index < len(box_scores) else 0.0
        box = box_coordinates[index] if index < len(box_coordinates) else bound
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

palette = [
    (239, 68, 68),
    (59, 130, 246),
    (16, 185, 129),
    (245, 158, 11),
    (168, 85, 247),
    (6, 182, 212),
    (236, 72, 153),
    (132, 204, 22),
]
rgb = np.zeros((height, width, 3), dtype=np.uint8)
alpha = np.zeros((height, width), dtype=np.uint8)
mask_items = []
for item in selected:
    mask = item.pop("mask_array")
    mask_bytes = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_bytes, format="PNG")
    mask_items.append(
        {
            **item,
            "mask": base64.b64encode(mask_bytes.getvalue()).decode("ascii"),
        }
    )
    color = palette[item["label_id"] % len(palette)]
    rgb[mask] = color
    alpha[mask] = 110

overlay = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
annotated = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

masks = {
    "width": int(width),
    "height": int(height),
    "model": architecture,
    "device": device_name,
    "backend": "ultralytics",
    "source": str(model.get("source", "builtin")),
    "masks": mask_items,
}

return masks: MASKS, annotated: IMAGE
