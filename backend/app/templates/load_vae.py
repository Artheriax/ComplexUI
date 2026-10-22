# /// script
# dependencies = ["torch>=2.2", "diffusers>=0.31", "safetensors>=0.4"]
# ///
from pathlib import Path
from typing import Literal

import torch
from diffusers import AutoencoderKL

component_path: ModelPath = ""
architecture: Literal["auto", "sd15", "sdxl"] = "auto"
device: Literal["auto", "cpu", "cuda"] = "auto"
dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"


def _detect_arch(vae, path: Path) -> str:
    scaling = float(getattr(vae.config, "scaling_factor", 0.0) or 0.0)
    if scaling > 0.14:
        return "sd15"
    if scaling > 0.0:
        return "sdxl"
    name = path.name.lower()
    return "sdxl" if "xl" in name or "sdxl" in name else "sd15"


if not component_path:
    raise ValueError("Select a standalone VAE path")
path = Path(component_path)
if not path.exists():
    raise FileNotFoundError(f"VAE does not exist: {path}")
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
    report(0.05, "loading VAE")
if path.is_dir():
    vae_model = AutoencoderKL.from_pretrained(str(path), torch_dtype=torch_dtype)
else:
    try:
        vae_model = AutoencoderKL.from_single_file(
            str(path),
            config=str(path.parent) if (path.parent / "config.json").is_file() else None,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:
        raise ValueError(
            "Standalone VAE files need a config.json in the same folder. "
            "Use a diffusers component folder or add the config file."
        ) from exc
vae_model.to(device)
vae_model.eval()

detected = _detect_arch(vae_model, path)
if architecture != "auto" and architecture != detected:
    raise ValueError(
        f"Selected architecture '{architecture}' does not match VAE config '{detected}'"
    )
arch = architecture if architecture != "auto" else detected
block_channels = getattr(vae_model.config, "block_out_channels", (128, 256, 512, 512))
vae_scale_factor = 2 ** (len(block_channels) - 1)
scaling_factor = float(
    getattr(vae_model.config, "scaling_factor", 0.13025 if arch == "sdxl" else 0.18215)
)
shift_factor = float(getattr(vae_model.config, "shift_factor", 0.0) or 0.0)
vae = {
    "kind": "vae",
    "backend": "diffusers",
    "arch": arch,
    "vae": vae_model,
    "scaling_factor": scaling_factor,
    "shift_factor": shift_factor,
    "vae_scale_factor": vae_scale_factor,
    "device": device,
    "dtype": torch_dtype,
    "name": path.stem if path.is_file() else path.name,
    "path": str(path.resolve()),
}
if callable(report):
    report(1.0, "VAE ready")

return vae: VAE
