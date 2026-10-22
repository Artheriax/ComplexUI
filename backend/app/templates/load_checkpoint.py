# /// script
# dependencies = [
#   "torch>=2.2",
#   "diffusers>=0.31",
#   "transformers>=4.46",
#   "accelerate>=1.0",
#   "safetensors>=0.4",
#   "Pillow>=11.0",
#   "numpy>=1.26",
# ]
# ///
import json
import struct
from pathlib import Path
from typing import Literal

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

local_model: ModelSelector = ModelSelector(type="diffusion")
device: Literal["auto", "cpu", "cuda"] = "auto"
dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"


def detect_architecture(checkpoint: Path) -> str:
    with checkpoint.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"Checkpoint header is truncated: {checkpoint.name}")
        length = struct.unpack("<Q", raw)[0]
        if length <= 0 or length > 16_000_000:
            raise ValueError(f"Checkpoint header size is invalid: {checkpoint.name}")
        payload = handle.read(length)
        if len(payload) != length:
            raise ValueError(f"Checkpoint header is truncated: {checkpoint.name}")
        header = json.loads(payload)
    keys = header.keys()
    if any(key.startswith("conditioner.embedders.1.") for key in keys):
        return "sdxl"
    if any(key.startswith("cond_stage_model.") for key in keys):
        return "sd15"
    raise ValueError(
        "Unsupported checkpoint layout; expected an SD1.5 or SDXL SafeTensors checkpoint"
    )


if not local_model:
    raise ValueError("Select a local diffusion checkpoint in models/diffusion")
checkpoint = Path(local_model)
if checkpoint.suffix.lower() != ".safetensors":
    raise ValueError("Diffusion checkpoints must use the .safetensors format")
if not checkpoint.is_file():
    raise FileNotFoundError(f"Diffusion checkpoint does not exist: {checkpoint}")

architecture = detect_architecture(checkpoint)
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
    report(0.05, "loading checkpoint")
pipeline_class = (
    StableDiffusionXLPipeline if architecture == "sdxl" else StableDiffusionPipeline
)
pipe = pipeline_class.from_single_file(
    str(checkpoint),
    torch_dtype=torch_dtype,
    use_safetensors=True,
)
pipe.to(device)
pipe.set_progress_bar_config(disable=True)
if hasattr(pipe, "safety_checker"):
    pipe.safety_checker = None

name = checkpoint.stem.lower()
if architecture == "sdxl" and any(
    token in name for token in ("lightning", "turbo", "lcm")
):
    preset = "sdxl_lightning"
    recommended = {
        "steps": 6,
        "cfg": 2.0,
        "sampler": "dpmpp_sde",
        "scheduler": "karras",
        "width": 1024,
        "height": 1024,
    }
elif architecture == "sdxl":
    preset = "sdxl"
    recommended = {
        "steps": 25,
        "cfg": 7.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "width": 1024,
        "height": 1024,
    }
else:
    preset = "sd15"
    recommended = {
        "steps": 25,
        "cfg": 7.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "width": 512,
        "height": 512,
    }

stat = checkpoint.stat()
model = {
    "kind": "model",
    "backend": "diffusers",
    "arch": architecture,
    "name": checkpoint.stem,
    "path": str(checkpoint.resolve()),
    "format": "safetensors",
    "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
    "source": "local",
    "model_type": "diffusion",
    "device": device,
    "dtype": torch_dtype,
    "unet": pipe.unet,
    "scheduler_config": dict(pipe.scheduler.config),
    "vae_scale_factor": int(getattr(pipe, "vae_scale_factor", 8)),
    "preset": preset,
    "recommended": recommended,
}
clip = {
    "kind": "clip",
    "backend": "diffusers",
    "arch": architecture,
    "text_encoder": pipe.text_encoder,
    "tokenizer": pipe.tokenizer,
    "text_encoder_2": getattr(pipe, "text_encoder_2", None),
    "tokenizer_2": getattr(pipe, "tokenizer_2", None),
    "device": device,
    "dtype": torch_dtype,
}
vae = {
    "kind": "vae",
    "backend": "diffusers",
    "arch": architecture,
    "vae": pipe.vae,
    "scaling_factor": float(pipe.vae.config.scaling_factor),
    "shift_factor": float(getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0),
    "device": device,
    "dtype": torch_dtype,
}
if callable(report):
    report(1.0, "checkpoint ready")
del pipe

return model: MODEL, clip: CLIP, vae: VAE
