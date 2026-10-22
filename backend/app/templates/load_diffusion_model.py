# /// script
# dependencies = ["torch>=2.2", "diffusers>=0.31", "safetensors>=0.4"]
# ///
from pathlib import Path
from typing import Literal

import torch
from diffusers import UNet2DConditionModel

component_path: ModelPath = ""
architecture: Literal["auto", "sd15", "sdxl"] = "auto"
device: Literal["auto", "cpu", "cuda"] = "auto"
dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"

DEFAULT_SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "steps_offset": 1,
}


def _detect_arch(config) -> str:
    cross_attention_dim = getattr(config, "cross_attention_dim", None)
    if isinstance(cross_attention_dim, (list, tuple)):
        cross_attention_dim = max(cross_attention_dim)
    if cross_attention_dim is not None and int(cross_attention_dim) >= 2048:
        return "sdxl"
    if getattr(config, "addition_embed_type", None):
        return "sdxl"
    return "sd15"


if not component_path:
    raise ValueError("Select a standalone UNet path")
path = Path(component_path)
if not path.exists():
    raise FileNotFoundError(f"Diffusion model does not exist: {path}")
if device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
if dtype == "auto":
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
elif dtype == "float16":
    torch_dtype = torch.float16
elif dtype == "bfloat16":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32
if device == "cpu" and torch_dtype == torch.float16:
    torch_dtype = torch.float32

report = globals().get("report_progress")
if callable(report):
    report(0.05, "loading diffusion model")
if path.is_dir():
    unet = UNet2DConditionModel.from_pretrained(
        str(path),
        torch_dtype=torch_dtype,
    )
else:
    try:
        unet = UNet2DConditionModel.from_single_file(
            str(path),
            config=str(path.parent) if (path.parent / "config.json").is_file() else None,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:
        raise ValueError(
            "Standalone UNet files need a config.json in the same folder. "
            "Use a diffusers component folder or add the config file."
        ) from exc
unet.to(device)
unet.eval()

detected = _detect_arch(unet.config)
if architecture != "auto" and architecture != detected:
    raise ValueError(
        f"Selected architecture '{architecture}' does not match UNet config '{detected}'"
    )
arch = architecture if architecture != "auto" else detected
if arch == "sdxl":
    recommended = {
        "steps": 25,
        "cfg": 7.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "width": 1024,
        "height": 1024,
    }
    preset = "sdxl"
else:
    recommended = {
        "steps": 25,
        "cfg": 7.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "width": 512,
        "height": 512,
    }
    preset = "sd15"

stat = path.stat()
model = {
    "kind": "model",
    "backend": "diffusers",
    "arch": arch,
    "name": path.stem if path.is_file() else path.name,
    "path": str(path.resolve()),
    "format": "diffusers" if path.is_dir() else path.suffix.lower().lstrip("."),
    "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
    "source": "local",
    "model_type": "unet",
    "device": device,
    "dtype": torch_dtype,
    "unet": unet,
    "scheduler_config": dict(DEFAULT_SCHEDULER_CONFIG),
    "vae_scale_factor": 8,
    "preset": preset,
    "recommended": recommended,
}
if callable(report):
    report(1.0, "diffusion model ready")

return model: MODEL
