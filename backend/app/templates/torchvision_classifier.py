# /// script
# dependencies = []
# ///
from pathlib import Path
from typing import Literal

local_model: ModelSelector = ModelSelector(type="classification")
architecture: Literal["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0", "vit_b_16"] = "resnet18"
weights: Literal["DEFAULT", "none"] = "DEFAULT"
device: Literal["auto", "cpu", "cuda"] = "auto"

model = {
    "backend": "torchvision",
    "task": "classification",
    "architecture": architecture,
    "weights": weights,
    "device": device,
    "classes": [],
    "source": "builtin",
    "model_type": "classification",
}

if local_model:
    suffix = Path(local_model).suffix.lower()
    if suffix not in {".pt", ".pth", ".onnx", ".safetensors"}:
        raise ValueError(
            "Classification local models must be .pt, .pth, .onnx or .safetensors files"
        )
    stat = Path(local_model).stat()
    model.update(
        {
            "source": "local",
            "path": str(local_model),
            "format": suffix.lstrip("."),
            "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
        }
    )

return model: CLASSIFICATION_MODEL
