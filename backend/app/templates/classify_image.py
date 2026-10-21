# /// script
# dependencies = ["torch>=2.2", "torchvision>=0.17", "safetensors>=0.4", "onnxruntime>=1.18", "Pillow>=11.0"]
# ///
from pathlib import Path

import torch
import torchvision
from torchvision import transforms

image: IMAGE
model: CLASSIFICATION_MODEL
top_k: int = 5

if not isinstance(model, dict):
    raise TypeError("model must be a torchvision classification descriptor")
if not hasattr(image, "convert"):
    raise TypeError("image must be a PIL image")

architecture = str(model.get("architecture", "resnet18"))
weights_name = str(model.get("weights", "DEFAULT"))
weight_enums = {
    "resnet18": torchvision.models.ResNet18_Weights,
    "resnet50": torchvision.models.ResNet50_Weights,
    "mobilenet_v3_small": torchvision.models.MobileNet_V3_Small_Weights,
    "efficientnet_b0": torchvision.models.EfficientNet_B0_Weights,
    "vit_b_16": torchvision.models.ViT_B_16_Weights,
}
weight_enum = weight_enums.get(architecture)
weights = None
if weights_name.lower() not in {"none", "null", ""} and weight_enum is not None:
    weights = weight_enum[weights_name]
device = str(model.get("device", "auto"))
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"

local_path = str(model.get("path") or "")
local_format = str(model.get("format") or Path(local_path).suffix.lstrip(".")).lower()
if local_path and local_format not in {"pt", "pth", "onnx", "safetensors"}:
    raise ValueError(
        "Classification local models must be .pt, .pth, .onnx or .safetensors files"
    )

if weights is not None:
    preprocess = weights.transforms()
else:
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
categories = list(weights.meta.get("categories", [])) if weights is not None else []

onnx_session = None
if local_path and local_format == "onnx":
    import onnxruntime

    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    onnx_session = onnxruntime.InferenceSession(local_path, providers=providers)
    onnx_input = onnx_session.get_inputs()[0].name
    net = None
else:
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

tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
with torch.inference_mode():
    if onnx_session is not None:
        raw = onnx_session.run(None, {onnx_input: tensor.cpu().numpy()})[0]
        probabilities = torch.as_tensor(raw).softmax(dim=-1)
        if probabilities.ndim > 1:
            probabilities = probabilities[0]
    else:
        probabilities = net(tensor).softmax(dim=1)[0]

limit = max(1, min(int(top_k), int(probabilities.shape[0])))
scores, indices = torch.topk(probabilities, limit)

top = []
for score, index in zip(scores.tolist(), indices.tolist(), strict=True):
    label = str(categories[index]) if index < len(categories) else f"class_{index}"
    top.append({"label": label, "score": float(score), "index": int(index)})

classifications = {
    "task": "classification",
    "model": architecture,
    "device": device,
    "source": str(model.get("source", "builtin")),
    "width": int(image.width),
    "height": int(image.height),
    "top_k": top,
}

return classifications: CLASSIFICATIONS
