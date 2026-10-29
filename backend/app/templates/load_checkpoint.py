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
import os
import struct
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

checkpoint_name: ModelSelector = ModelSelector(type="diffusion")


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


def configure_vae_tiling(
    vae_model,
    enabled,
    tile_size,
    overlap,
    temporal_size,
    temporal_overlap,
    scale_factor,
):
    if not hasattr(vae_model, "enable_tiling"):
        return
    if not enabled:
        if hasattr(vae_model, "disable_tiling"):
            vae_model.disable_tiling()
        return
    vae_model.enable_tiling()
    if hasattr(vae_model, "tile_sample_min_size"):
        vae_model.tile_sample_min_size = tile_size
    if hasattr(vae_model, "tile_latent_min_size"):
        vae_model.tile_latent_min_size = max(1, tile_size // scale_factor)
    if hasattr(vae_model, "tile_overlap_factor"):
        vae_model.tile_overlap_factor = overlap / max(tile_size, 1)
    if hasattr(vae_model, "temporal_size"):
        vae_model.temporal_size = temporal_size
    if hasattr(vae_model, "temporal_overlap"):
        vae_model.temporal_overlap = temporal_overlap


if not checkpoint_name:
    raise ValueError("Select a local diffusion checkpoint in models/diffusion")
checkpoint = Path(checkpoint_name)
if checkpoint.suffix.lower() != ".safetensors":
    raise ValueError("Diffusion checkpoints must use the .safetensors format")
if not checkpoint.is_file():
    raise FileNotFoundError(f"Diffusion checkpoint does not exist: {checkpoint}")

architecture = detect_architecture(checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
pipeline_class = (
    StableDiffusionXLPipeline if architecture == "sdxl" else StableDiffusionPipeline
)
quantization_info = None
quantized_report = []
try:
    from app.quantization import detect_quantization, validate_quantization
    from app.single_file_quantization import load_quantized_single_file
except ImportError:
    pass
else:
    capability = None
    if device == "cuda":
        try:
            capability = torch.cuda.get_device_capability()
        except Exception:
            capability = None
    quantization_info = detect_quantization(checkpoint, "diffusion")
    if quantization_info is not None and quantization_info.format != "none":
        validate_quantization(
            quantization_info,
            model_type="diffusion",
            device=device,
            capability=capability,
        )

report = globals().get("report_progress")
if callable(report):
    report(0.05, "loading checkpoint")
if quantization_info is not None and quantization_info.format != "none":
    pipe, quantized_report = load_quantized_single_file(
        checkpoint,
        pipeline_class,
        torch_dtype=torch_dtype,
    )
    if quantized_report:
        try:
            from app.single_file_quantization import quantization_mode
        except ImportError:
            mode = "emulated"
        else:
            mode = quantization_mode(quantized_report)
        print(
            "checkpoint quantization: "
            f"{sum(1 for entry in quantized_report if entry.native)} native, "
            f"{sum(1 for entry in quantized_report if not entry.native)} emulated"
        )
        quantization_info = quantization_info.model_copy(
            update={
                "mode": mode,
                "accumulation": "compute",
            }
        )
    else:
        quantization_info = quantization_info.model_copy(
            update={"mode": "emulated", "accumulation": "compute"}
        )
else:
    pipe = pipeline_class.from_single_file(
        str(checkpoint),
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
pipe.to(device)
pipe.set_progress_bar_config(disable=True)
if hasattr(pipe, "safety_checker"):
    pipe.safety_checker = None

attention = {
    "backend": "sdpa",
    "reason": "standalone export default",
    "approximates": False,
}
try:
    from app.attention_dispatch import apply_attention, benchmark_attention
except Exception:
    pass
else:
    requested = os.environ.get("COMPLEXUI_ATTENTION", "auto").strip().lower()
    benchmark = os.environ.get(
        "COMPLEXUI_ATTENTION_BENCHMARK", ""
    ).strip().lower() in {"1", "true", "on", "yes"}
    if benchmark and requested in {"auto", "sdpa"}:
        try:
            results = benchmark_attention(torch, device, "auto")
            attention = apply_attention(
                torch, str(results["selected"]), device
            )
            attention["benchmark"] = results
        except Exception as exc:
            print(f"attention benchmark failed; using auto ({exc})")
            attention = apply_attention(torch, "auto", device)
    else:
        attention = apply_attention(torch, requested, device)

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
elif architecture == "sdxl" and "hyper" in name:
    preset = "sdxl_hyper"
    recommended = {
        "steps": 8,
        "cfg": 1.0,
        "sampler": "dpmpp_sde",
        "scheduler": "karras",
        "width": 1024,
        "height": 1024,
    }
elif architecture == "sdxl" and "pony" in name:
    preset = "sdxl_pony"
    recommended = {
        "steps": 30,
        "cfg": 7.0,
        "sampler": "euler_ancestral",
        "scheduler": "karras",
        "width": 1024,
        "height": 1024,
    }
elif architecture == "sdxl" and "illustrious" in name:
    preset = "sdxl_illustrious"
    recommended = {
        "steps": 28,
        "cfg": 5.5,
        "sampler": "euler_ancestral",
        "scheduler": "karras",
        "width": 1024,
        "height": 1024,
    }
elif architecture == "sdxl" and "noob" in name:
    preset = "sdxl_noobai"
    recommended = {
        "steps": 28,
        "cfg": 5.0,
        "sampler": "euler_ancestral",
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
memory = None
try:
    from app.memory_estimate import estimate_model_memory
except ImportError:
    pass
else:
    memory = estimate_model_memory(
        size_bytes=stat.st_size,
        quantization=quantization_info,
    )
model = {
    "kind": "model",
    "backend": "diffusers",
    "arch": architecture,
    "name": checkpoint.stem,
    "path": str(checkpoint.resolve()),
    "format": "safetensors",
    "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
    "size_bytes": stat.st_size,
    "memory": memory.model_dump() if memory is not None else None,
    "source": "local",
    "model_type": "diffusion",
    "device": device,
    "dtype": torch_dtype,
    "quantization": (
        quantization_info.model_dump()
        if quantization_info is not None
        else None
    ),
    "attention": attention,
    "unet": pipe.unet,
    "transformer": None,
    "transformer_class": type(pipe.unet).__name__,
    "scheduler": getattr(pipe, "scheduler", None),
    "scheduler_config": dict(pipe.scheduler.config),
    "vae_scale_factor": int(getattr(pipe, "vae_scale_factor", 8)),
    "latent_space": "sd",
    "latent_channels": 4,
    "latent_packing": 1,
    "preset": preset,
    "is_distilled": "lightning" in preset or "hyper" in preset,
    "capabilities": {
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": False,
        "max_references": 1,
        "negative_prompt": "required",
        "guidance": "cfg",
    },
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
pipe_vae = pipe.vae
vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 8))
scaling_factor = float(pipe_vae.config.scaling_factor)
shift_factor = float(getattr(pipe_vae.config, "shift_factor", 0.0) or 0.0)


def encode_pixels(
    pixel_tensor,
    tiled=False,
    tile_size=512,
    overlap=64,
    temporal_size=64,
    temporal_overlap=8,
):
    if not isinstance(pixel_tensor, torch.Tensor):
        raise TypeError("pixel_tensor must be a torch.Tensor")
    if tile_size < 8 or overlap < 0 or temporal_size < 1 or temporal_overlap < 0:
        raise ValueError("Invalid VAE tiling parameters")
    configure_vae_tiling(
        pipe_vae,
        bool(tiled),
        int(tile_size),
        int(overlap),
        int(temporal_size),
        int(temporal_overlap),
        vae_scale_factor,
    )
    with torch.inference_mode():
        distribution = pipe_vae.encode(
            pixel_tensor.to(device=device, dtype=torch_dtype)
        ).latent_dist
        mode = getattr(distribution, "mode", None)
        samples = mode() if callable(mode) else distribution.mean
        samples = (samples - shift_factor) * scaling_factor
    return samples.detach()


def decode_latents(
    latent_samples,
    tiled=False,
    tile_size=512,
    overlap=64,
    temporal_size=64,
    temporal_overlap=8,
):
    if not isinstance(latent_samples, torch.Tensor):
        raise TypeError("latent_samples must be a torch.Tensor")
    if tile_size < 8 or overlap < 0 or temporal_size < 1 or temporal_overlap < 0:
        raise ValueError("Invalid VAE tiling parameters")
    configure_vae_tiling(
        pipe_vae,
        bool(tiled),
        int(tile_size),
        int(overlap),
        int(temporal_size),
        int(temporal_overlap),
        vae_scale_factor,
    )
    latents = latent_samples.to(device=device, dtype=torch_dtype)
    with torch.inference_mode():
        decoded = pipe_vae.decode(
            latents / scaling_factor + shift_factor,
            return_dict=False,
        )[0]
        decoded = torch.nan_to_num(decoded, nan=0.0, posinf=0.0, neginf=0.0)
    return decoded.detach()


vae = {
    "kind": "vae",
    "backend": "diffusers",
    "arch": architecture,
    "vae": pipe_vae,
    "vae_class": type(pipe_vae).__name__,
    "scaling_factor": scaling_factor,
    "shift_factor": shift_factor,
    "vae_scale_factor": vae_scale_factor,
    "latent_channels": int(getattr(pipe_vae.config, "latent_channels", 4)),
    "latent_space": "sd",
    "latent_packing": 1,
    "device": device,
    "dtype": torch_dtype,
    "encode": encode_pixels,
    "decode": decode_latents,
}
if callable(report):
    report(1.0, "checkpoint ready")
del pipe

return model: MODEL, clip: CLIP, vae: VAE
