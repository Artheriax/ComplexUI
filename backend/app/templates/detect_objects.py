# /// script
# dependencies = ["torch>=2.2", "torchvision>=0.17", "safetensors>=0.4", "Pillow>=11.0"]
# ///
import torch
import torchvision
from PIL import ImageDraw
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
)

image: IMAGE
model: DETECTION_MODEL
score_threshold: float = 0.5
max_detections: int = 50

if not isinstance(model, dict):
    raise TypeError("model must be a torchvision detection descriptor")
if not hasattr(image, "convert"):
    raise TypeError("image must be a PIL image")

architecture = str(model.get("architecture", "fasterrcnn_mobilenet_v3_large_320_fpn"))
weights_name = str(model.get("weights", "DEFAULT"))
weight_enums = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    "fasterrcnn_mobilenet_v3_large_fpn": FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    "retinanet_resnet50_fpn_v2": RetinaNet_ResNet50_FPN_V2_Weights,
}
weight_enum = weight_enums.get(architecture)
weights = None
if weights_name.lower() not in {"none", "null", ""} and weight_enum is not None:
    weights = weight_enum[weights_name]
device = str(model.get("device", "auto"))
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"

local_path = str(model.get("path") or "")
local_format = str(model.get("format") or "").lower()
if local_path and local_format == "onnx":
    raise ValueError("ONNX torchvision detection is not supported; use a YOLO ONNX model")
if local_path and local_format not in {"pt", "pth", "safetensors"}:
    raise ValueError("Torchvision detection models must be .pt, .pth or .safetensors files")

net = torchvision.models.get_model(
    architecture,
    weights=None if local_path else weights,
)
if local_path:
    if local_format == "safetensors":
        from safetensors.torch import load_file

        state = load_file(local_path)
    else:
        state = torch.load(local_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                candidate = state.get(key)
                if isinstance(candidate, dict):
                    state = candidate
                    break
    net.load_state_dict(state)
net.eval().to(device)
if weights is not None:
    preprocess = weights.transforms()
else:
    preprocess = transforms.Compose([transforms.ToTensor()])

tensor = preprocess(image.convert("RGB")).to(device)
with torch.inference_mode():
    prediction = net([tensor])[0]

categories = list(weights.meta.get("categories", [])) if weights is not None else []

limit = max(1, int(max_detections))
threshold = float(score_threshold)
items = []
for box, label_index, score in zip(
    prediction["boxes"].detach().cpu().tolist(),
    prediction["labels"].detach().cpu().tolist(),
    prediction["scores"].detach().cpu().tolist(),
    strict=True,
):
    if len(items) >= limit or float(score) < threshold:
        continue
    label = (
        str(categories[label_index])
        if label_index < len(categories)
        else f"class_{label_index}"
    )
    items.append(
        {
            "label": label,
            "label_id": int(label_index),
            "score": float(score),
            "box": [round(float(value), 2) for value in box],
        }
    )

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
for item in items:
    x1, y1, x2, y2 = item["box"]
    color = palette[item["label_id"] % len(palette)]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    text = f"{item['label']} {item['score']:.2f}"
    text_y = max(0, int(y1) - 12)
    draw.text((int(x1) + 2, text_y), text, fill=color)

detections = {
    "count": len(items),
    "detections": items,
    "width": int(image.width),
    "height": int(image.height),
    "model": architecture,
    "device": device,
    "source": str(model.get("source", "builtin")),
}

return detections: DETECTIONS, annotated: IMAGE
