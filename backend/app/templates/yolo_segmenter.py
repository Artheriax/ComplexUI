# /// script
# dependencies = []
# ///
from pathlib import Path
from typing import Literal

local_model: ModelSelector = ModelSelector(type="yolo")
architecture: Literal["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"] = "yolo11n-seg"
weights: Literal["DEFAULT", "none"] = "DEFAULT"
device: Literal["auto", "cpu", "cuda"] = "auto"

model = {
    "backend": "ultralytics",
    "task": "segmentation",
    "architecture": architecture,
    "weights": weights,
    "device": device,
    "classes": [],
    "source": "builtin",
    "model_type": "yolo",
}

if local_model:
    suffix = Path(local_model).suffix.lower()
    if suffix not in {".pt", ".onnx"}:
        raise ValueError("YOLO local models must be .pt or .onnx files")
    stat = Path(local_model).stat()
    model.update(
        {
            "source": "local",
            "path": str(local_model),
            "format": suffix.lstrip("."),
            "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
        }
    )

return model: SEGMENTATION_MODEL
