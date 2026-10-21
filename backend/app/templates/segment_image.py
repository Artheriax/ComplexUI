# /// script
# dependencies = ["torch>=2.2", "torchvision>=0.17", "safetensors>=0.4", "Pillow>=11.0", "numpy>=1.26"]
# ///
import base64
import io

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
)

image: IMAGE
model: SEGMENTATION_MODEL
min_area: int = 64
max_masks: int = 20

if not isinstance(model, dict):
    raise TypeError("model must be a torchvision segmentation descriptor")
if not hasattr(image, "convert"):
    raise TypeError("image must be a PIL image")

architecture = str(model.get("architecture", "deeplabv3_mobilenet_v3_large"))
weights_name = str(model.get("weights", "DEFAULT"))
weight_enums = {
    "deeplabv3_mobilenet_v3_large": DeepLabV3_MobileNet_V3_Large_Weights,
    "deeplabv3_resnet50": DeepLabV3_ResNet50_Weights,
    "fcn_resnet50": FCN_ResNet50_Weights,
    "lraspp_mobilenet_v3_large": LRASPP_MobileNet_V3_Large_Weights,
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
    raise ValueError("ONNX torchvision segmentation is not supported; use a YOLO ONNX model")
if local_path and local_format not in {"pt", "pth", "safetensors"}:
    raise ValueError(
        "Torchvision segmentation models must be .pt, .pth or .safetensors files"
    )

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
    logits = net(tensor.unsqueeze(0))["out"][0]
probabilities = logits.softmax(dim=0)
confidence, class_ids = probabilities.max(dim=0)

categories = list(weights.meta.get("categories", [])) if weights is not None else []

height, width = class_ids.shape
ids_array = class_ids.detach().cpu().numpy()
confidence_array = confidence.detach().cpu().numpy()
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
candidates = []
for class_index in np.unique(ids_array):
    index = int(class_index)
    if index == 0:
        continue
    mask = ids_array == index
    area = int(mask.sum())
    if area < int(min_area):
        continue
    ys, xs = np.nonzero(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    score = float(confidence_array[mask].mean())
    mask_bytes = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_bytes, format="PNG")
    encoded = base64.b64encode(mask_bytes.getvalue()).decode("ascii")
    label = (
        str(categories[index])
        if index < len(categories)
        else f"class_{index}"
    )
    candidates.append(
        {
            "label": label,
            "label_id": index,
            "score": score,
            "area": area,
            "box": box,
            "mask": encoded,
        }
    )
    color = palette[index % len(palette)]
    rgb[mask] = color
    alpha[mask] = 110

candidates.sort(key=lambda item: item["area"], reverse=True)
mask_items = candidates[: max(1, int(max_masks))]
overlay = np.dstack([rgb, alpha])
layer = Image.fromarray(overlay, mode="RGBA")
annotated = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")

masks = {
    "width": int(width),
    "height": int(height),
    "model": architecture,
    "device": device,
    "source": str(model.get("source", "builtin")),
    "masks": mask_items,
}

return masks: MASKS, annotated: IMAGE
