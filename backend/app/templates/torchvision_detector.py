# /// script
# dependencies = []
# ///
from pathlib import Path
from typing import Literal

local_model: ModelSelector = ModelSelector(type="torchvision")
architecture: Literal["fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_mobilenet_v3_large_fpn", "retinanet_resnet50_fpn_v2"] = "fasterrcnn_mobilenet_v3_large_320_fpn"
weights: Literal["DEFAULT", "none"] = "DEFAULT"
device: Literal["auto", "cpu", "cuda"] = "auto"

model = {
    "backend": "torchvision",
    "task": "detection",
    "architecture": architecture,
    "weights": weights,
    "device": device,
    "classes": [],
    "source": "builtin",
    "model_type": "torchvision",
}

if local_model:
    suffix = Path(local_model).suffix.lower()
    if suffix not in {".pt", ".pth", ".safetensors"}:
        raise ValueError("Torchvision detection models must be .pt, .pth or .safetensors files")
    stat = Path(local_model).stat()
    model.update(
        {
            "source": "local",
            "path": str(local_model),
            "format": suffix.lstrip("."),
            "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
        }
    )

return model: DETECTION_MODEL
