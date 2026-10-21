# /// script
# dependencies = ["torch>=2.2", "torchvision>=0.17", "safetensors>=0.4", "onnxruntime>=1.18", "Pillow>=11.0", "opencv-python>=4.9"]
# ///
from pathlib import Path

import cv2
import torch
import torchvision
from PIL import Image
from torchvision import transforms

video: VIDEO
model: CLASSIFICATION_MODEL
sample_frames: int = 8
top_k: int = 5

if not isinstance(model, dict):
    raise TypeError("model must be a torchvision classification descriptor")
if not isinstance(video, dict) or str(video.get("media_type", "")).lower() != "video":
    raise TypeError("video must be a VIDEO value")
source = Path(str(video.get("path", "")))
if not source.is_file():
    raise FileNotFoundError(f"Video source does not exist: {source}")

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
categories = (
    list(weights.meta.get("categories", []))
    if weights is not None
    else list(model.get("classes", []))
)

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

total = int(video.get("frames", 0) or 0)
count = max(1, int(sample_frames))
capture = cv2.VideoCapture(str(source))
if not capture.isOpened():
    raise FileNotFoundError(f"Could not open video: {source}")
if total <= 0:
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

indices = list(range(min(count, total))) if total > 0 else list(range(count))
if total > 0 and count > 1 and total > count:
    indices = [round(position * (total - 1) / (count - 1)) for position in range(count)]
indices = sorted(set(max(0, index) for index in indices))

probabilities = []
processed = 0
try:
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            if onnx_session is not None:
                raw = onnx_session.run(None, {onnx_input: tensor.cpu().numpy()})[0]
                frame_probabilities = torch.as_tensor(raw).softmax(dim=-1)
                if frame_probabilities.ndim > 1:
                    frame_probabilities = frame_probabilities[0]
                probabilities.append(frame_probabilities)
            else:
                probabilities.append(net(tensor).softmax(dim=1)[0])
        processed += 1
finally:
    capture.release()

if not probabilities:
    raise ValueError(f"Could not read frames from {source.name}")

mean = torch.stack(probabilities).mean(dim=0)
limit = max(1, min(int(top_k), int(mean.shape[0])))
scores, indices_tensor = torch.topk(mean, limit)

top = []
for score, index in zip(scores.tolist(), indices_tensor.tolist(), strict=True):
    label = str(categories[index]) if index < len(categories) else f"class_{index}"
    top.append({"label": label, "score": float(score), "index": int(index)})

classifications = {
    "task": "classification",
    "model": architecture,
    "device": device,
    "source": str(model.get("source", "builtin")),
    "width": int(video.get("width", 0) or 0),
    "height": int(video.get("height", 0) or 0),
    "fps": float(video.get("fps", 0.0) or 0.0),
    "frames": processed,
    "top_k": top,
}

return classifications: CLASSIFICATIONS
