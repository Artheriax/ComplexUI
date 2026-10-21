# /// script
# dependencies = []
# ///
from pathlib import Path
from typing import Literal

local_model: ModelSelector = ModelSelector(type="yolo")
architecture: Literal["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"] = "yolo11n"
weights: Literal["DEFAULT", "none"] = "DEFAULT"
device: Literal["auto", "cpu", "cuda"] = "auto"

model = {
    "backend": "ultralytics",
    "task": "detection",
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

return model: DETECTION_MODEL
