# /// script
# dependencies = ["torch>=2.2", "transformers>=4.46", "safetensors>=0.4"]
# ///
from pathlib import Path
from typing import Literal

import torch
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

clip_path: ModelPath = ""
architecture: Literal["auto", "sd15", "sdxl"] = "auto"
device: Literal["auto", "cpu", "cuda"] = "auto"
dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"

if not clip_path:
    raise ValueError("Select a CLIP component folder")
root = Path(clip_path)
if not root.is_dir():
    raise FileNotFoundError(f"CLIP folder does not exist: {root}")
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

text_dir = root / "text_encoder" if (root / "text_encoder").is_dir() else root
tokenizer_dir = root / "tokenizer" if (root / "tokenizer").is_dir() else root
text2_dir = root / "text_encoder_2"
tokenizer2_dir = root / "tokenizer_2"
detected = "sdxl" if text2_dir.is_dir() else "sd15"
if architecture != "auto" and architecture != detected:
    raise ValueError(
        f"Selected architecture '{architecture}' does not match CLIP folder '{detected}'"
    )
arch = architecture if architecture != "auto" else detected
if arch == "sdxl" and not tokenizer2_dir.is_dir():
    raise ValueError("SDXL CLIP folders require tokenizer_2")

report = globals().get("report_progress")
if callable(report):
    report(0.05, "loading CLIP")
text_encoder = CLIPTextModel.from_pretrained(
    str(text_dir),
    torch_dtype=torch_dtype,
).to(device)
text_encoder.eval()
tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_dir))
text_encoder_2 = None
tokenizer_2 = None
if arch == "sdxl":
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        str(text2_dir),
        torch_dtype=torch_dtype,
    ).to(device)
    text_encoder_2.eval()
    tokenizer_2 = CLIPTokenizer.from_pretrained(str(tokenizer2_dir))

clip = {
    "kind": "clip",
    "backend": "diffusers",
    "arch": arch,
    "text_encoder": text_encoder,
    "tokenizer": tokenizer,
    "text_encoder_2": text_encoder_2,
    "tokenizer_2": tokenizer_2,
    "device": device,
    "dtype": torch_dtype,
    "name": root.name,
    "path": str(root.resolve()),
}
if callable(report):
    report(1.0, "CLIP ready")

return clip: CLIP
