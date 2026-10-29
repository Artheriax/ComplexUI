# /// script
# dependencies = [
#   "torch>=2.2",
#   "diffusers>=0.42.0",
#   "transformers>=5.0",
#   "accelerate>=1.0",
#   "safetensors>=0.4",
# ]
# ///
import json
import os
import time
from pathlib import Path
from typing import Literal

import diffusers
import torch

vae_name: ModelSelector = ModelSelector(
    type="vae",
    select="any",
    required=True,
    include=["diffusion"],
)
weight_dtype: Literal["default", "fp16", "bf16", "fp32"] = "default"
compute_dtype: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
allow_custom_code: bool = False

PIPELINE_ARCH = {
    "StableDiffusionPipeline": "sd15",
    "StableDiffusionImg2ImgPipeline": "sd15",
    "StableDiffusionInpaintPipeline": "sd15",
    "StableDiffusionXLPipeline": "sdxl",
    "StableDiffusionXLImg2ImgPipeline": "sdxl",
    "StableDiffusionXLInpaintPipeline": "sdxl",
    "StableDiffusionXLModularPipeline": "sdxl",
    "FluxPipeline": "flux1",
    "FluxImg2ImgPipeline": "flux1",
    "FluxInpaintPipeline": "flux1",
    "FluxFillPipeline": "flux1",
    "FluxKontextPipeline": "flux1",
    "FluxKontextInpaintPipeline": "flux1",
    "FluxModularPipeline": "flux1",
    "FluxKontextModularPipeline": "flux1",
    "Flux2Pipeline": "flux2",
    "Flux2KleinPipeline": "flux2",
    "Flux2ModularPipeline": "flux2",
    "Flux2KleinModularPipeline": "flux2",
    "Flux2KleinBaseModularPipeline": "flux2",
    "Flux3ModularPipeline": "flux3",
    "Flux3DistilledModularPipeline": "flux3",
    "QwenImage21Pipeline": "qwen_image",
    "QwenImage21ModularPipeline": "qwen_image",
    "QwenImage21TurboModularPipeline": "qwen_image",
    "QwenImagePipeline": "qwen_image",
    "QwenImageModularPipeline": "qwen_image",
    "QwenImageEditPipeline": "qwen_image",
    "QwenImageEditPlusPipeline": "qwen_image",
    "ZImagePipeline": "z_image",
    "ZImageModularPipeline": "z_image",
    "ZImageImg2ImgPipeline": "z_image",
    "ZImageInpaintPipeline": "z_image",
    "Krea2Pipeline": "krea2",
    "Krea2ModularPipeline": "krea2",
    "Krea2TurboModularPipeline": "krea2",
    "Ideogram4Pipeline": "ideogram4",
    "Ideogram4ModularPipeline": "ideogram4",
    "BriaFibo2Pipeline": "fibo2",
    "BriaFiboEditPipeline": "fibo2",
    "AnimaModularPipeline": "anima",
    "MOSSImagePipeline": "moss_image",
}

VAE_CLASS_ARCH = {
    "AutoencoderKLFlux2": "flux2",
    "AutoencoderKLQwenImage": "qwen_image",
    "AutoencoderKLQwenImage21": "qwen_image",
    "AutoencoderKLWan": "wan",
}

DEFAULT_VAE_CLASS = {
    "sd15": "AutoencoderKL",
    "sdxl": "AutoencoderKL",
    "flux1": "AutoencoderKL",
    "flux2": "AutoencoderKLFlux2",
    "flux3": "AutoencoderKLFlux2",
    "qwen_image": "AutoencoderKLQwenImage21",
    "z_image": "AutoencoderKL",
    "krea2": "AutoencoderKLQwenImage",
    "ideogram4": "AutoencoderKLFlux2",
    "fibo2": "AutoencoderKLFlux2",
    "anima": "AutoencoderKLQwenImage",
    "moss_image": "AutoencoderKLFlux2",
    "wan": "AutoencoderKLWan",
}


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _config(path: Path):
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
    return _read_json(config_path) or {}


def _class_name(path: Path):
    config = _config(path)
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    return config.get("_class_name")


def _entries(root: Path):
    entries = {}
    index_class = None
    standard = _read_json(root / "model_index.json")
    modular = _read_json(root / "modular_model_index.json")
    if standard is not None:
        index_class = standard.get("_class_name")
        for name, value in standard.items():
            if name.startswith("_") or not isinstance(value, (list, tuple)):
                continue
            if len(value) != 2:
                continue
            entries[name] = {"class_name": str(value[1]), "subfolder": name}
    if modular is not None:
        index_class = index_class or modular.get("_class_name")
        for name, value in modular.items():
            if name.startswith("_") or not isinstance(value, dict):
                continue
            hint = value.get("type_hint")
            if not isinstance(hint, (list, tuple)) or len(hint) != 2:
                continue
            entries[name] = {
                "class_name": str(hint[1]),
                "subfolder": str(value.get("subfolder") or name),
            }
    return index_class, entries


def _infer_arch(root: Path, index_class, entries, class_name: str):
    if isinstance(index_class, str) and index_class in PIPELINE_ARCH:
        return PIPELINE_ARCH[index_class]
    name = root.name.lower()
    if class_name == "AutoencoderKLFlux2":
        for token, arch in (
            ("moss", "moss_image"),
            ("fibo", "fibo2"),
            ("ideogram", "ideogram4"),
            ("flux3", "flux3"),
        ):
            if token in name:
                return arch
        return "flux2"
    if class_name == "AutoencoderKLQwenImage21":
        return "qwen_image"
    if class_name == "AutoencoderKLQwenImage":
        for token, arch in (("anima", "anima"), ("krea", "krea2")):
            if token in name:
                return arch
        return "qwen_image"
    if class_name == "AutoencoderKLWan":
        return "wan"
    if class_name == "AutoencoderKL":
        for token, arch in (
            ("flux", "flux1"),
            ("zimage", "z_image"),
            ("z_image", "z_image"),
            ("sdxl", "sdxl"),
            ("sd15", "sd15"),
        ):
            if token in name:
                return arch
        config = _config(root)
        try:
            shift = float(config.get("shift_factor", 0.0) or 0.0)
            scaling = float(config.get("scaling_factor", 0.0) or 0.0)
        except (TypeError, ValueError):
            shift = 0.0
            scaling = 0.0
        if shift > 0:
            return "flux1"
        if scaling > 0.14:
            return "sd15"
        if scaling > 0:
            return "sdxl"
    if "vae" in entries:
        entry_class = str(entries["vae"].get("class_name") or "")
        if entry_class in VAE_CLASS_ARCH:
            return VAE_CLASS_ARCH[entry_class]
    for token, arch in (
        ("anima", "anima"),
        ("fibo", "fibo2"),
        ("ideogram", "ideogram4"),
        ("krea", "krea2"),
        ("qwen", "qwen_image"),
        ("zimage", "z_image"),
        ("z_image", "z_image"),
        ("flux3", "flux3"),
        ("flux2", "flux2"),
        ("flux", "flux1"),
        ("wan", "wan"),
        ("sdxl", "sdxl"),
        ("sd15", "sd15"),
    ):
        if token in name:
            return arch
    return "sd15"


def _load_vae_component(path: Path, class_name: str, torch_dtype, trust: bool):
    errors = []
    if class_name:
        component_class = getattr(diffusers, class_name, None)
        if component_class is not None and hasattr(component_class, "from_pretrained"):
            try:
                return component_class.from_pretrained(
                    str(path),
                    torch_dtype=torch_dtype,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    trust_remote_code=trust,
                )
            except Exception as exc:
                errors.append(str(exc))
    auto_model = getattr(diffusers, "AutoModel", None)
    if auto_model is not None:
        try:
            return auto_model.from_pretrained(
                str(path),
                torch_dtype=torch_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                trust_remote_code=trust,
            )
        except Exception as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "unknown loader error"
    raise ValueError(
        f"Could not load VAE component '{path.name}'. Expected a local "
        f"diffusers folder. Details: {detail}"
    )


def _load_vae_file(path: Path, arch: str, torch_dtype):
    class_name = _class_name(path) or DEFAULT_VAE_CLASS.get(arch, "AutoencoderKL")
    vae_class = getattr(diffusers, class_name, None)
    if vae_class is None:
        raise ValueError(f"diffusers does not expose the VAE class '{class_name}'")
    if not hasattr(vae_class, "from_single_file"):
        raise ValueError(
            f"'{class_name}' cannot load a single file. Use a diffusers VAE "
            "component folder with config.json."
        )
    kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
    if (path.parent / "config.json").is_file():
        kwargs["config"] = str(path.parent)
    try:
        return vae_class.from_single_file(str(path), **kwargs)
    except Exception as exc:
        raise ValueError(
            f"Could not load VAE file '{path.name}'. Single VAE files need the "
            f"matching config.json in the same folder. Details: {exc}"
        ) from exc


def _configure_tiling(
    vae_model,
    enabled: bool,
    tile_size: int,
    overlap: int,
    temporal_size: int,
    temporal_overlap: int,
    latent_factor: int,
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
        vae_model.tile_latent_min_size = max(1, tile_size // latent_factor)
    if hasattr(vae_model, "tile_overlap_factor"):
        vae_model.tile_overlap_factor = overlap / max(tile_size, 1)
    if hasattr(vae_model, "temporal_size"):
        vae_model.temporal_size = temporal_size
    if hasattr(vae_model, "temporal_overlap"):
        vae_model.temporal_overlap = temporal_overlap


def _retrieve_latents(distribution):
    mode = getattr(distribution, "mode", None)
    if callable(mode):
        return mode()
    mean = getattr(distribution, "mean", None)
    if mean is not None:
        return mean
    sample = getattr(distribution, "sample", None)
    if callable(sample):
        return sample()
    raise TypeError("VAE encode returned an unsupported latent distribution")


def _patchify(latents):
    batch, channels, height, width = latents.shape
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch, channels * 4, height // 2, width // 2)


def _unpatchify(latents):
    batch, channels, height, width = latents.shape
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch, channels // 4, height * 2, width * 2)


def _image_to_4d(latents):
    if latents.ndim == 5:
        if latents.shape[2] != 1:
            raise ValueError("Image VAE encoding produced more than one temporal frame")
        return latents[:, :, 0]
    return latents


def _memory_profile(device):
    value = os.environ.get("COMPLEXUI_MEMORY_PROFILE", "auto").strip().lower()
    allowed = {"auto", "low", "balanced", "resident"}
    if value not in allowed:
        value = "auto"
    if value != "auto":
        return value
    if device != "cuda":
        return "resident"
    try:
        total = float(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return "resident"
    gib = total / (1024.0**3)
    if gib < 10.0:
        return "low"
    if gib < 20.0:
        return "balanced"
    return "resident"


if not vae_name:
    raise ValueError("Select a VAE in models/vae or models/diffusion")
selected = Path(vae_name)
if not selected.exists():
    raise FileNotFoundError(f"VAE does not exist: {selected}")
load_started = time.perf_counter()
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    from app.quantization import require_float_checkpoint
except ImportError:
    pass
else:
    capability = None
    if device == "cuda":
        try:
            capability = torch.cuda.get_device_capability()
        except Exception:
            capability = None
    require_float_checkpoint(
        selected,
        model_type="vae",
        device=device,
        capability=capability,
    )
if weight_dtype == "default":
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
elif weight_dtype == "fp16":
    torch_dtype = torch.float16
elif weight_dtype == "bf16":
    torch_dtype = torch.bfloat16
elif weight_dtype == "fp32":
    torch_dtype = torch.float32
else:
    raise ValueError(f"Unsupported weight_dtype: {weight_dtype}")
storage_dtype = (
    weight_dtype
    if weight_dtype in {"fp32", "fp16", "bf16"}
    else ("fp16" if torch_dtype is torch.float16 else "fp32")
)
if compute_dtype == "auto":
    resolved_dtype = torch_dtype
elif compute_dtype == "fp16":
    if device == "cpu":
        raise ValueError("FP16 VAE inference requires CUDA; choose bf16 or fp32")
    resolved_dtype = torch.float16
elif compute_dtype == "bf16":
    resolved_dtype = torch.bfloat16
elif compute_dtype == "fp32":
    resolved_dtype = torch.float32
else:
    raise ValueError(f"Unsupported compute_dtype: {compute_dtype}")
memory_profile = _memory_profile(device)
offload = memory_profile in {"low", "balanced"}
try:
    from app.attention_dispatch import apply_attention
except ImportError:
    pass
else:
    apply_attention(
        torch,
        os.environ.get("COMPLEXUI_ATTENTION", "auto"),
        device,
    )

report = globals().get("report_progress")
if callable(report):
    report(0.05, "loading VAE")

index_class = None
entries = {}
if selected.is_dir():
    index_class, entries = _entries(selected)
    if "vae" in entries:
        component_path = selected / str(entries["vae"].get("subfolder") or "vae")
        component_class = str(entries["vae"].get("class_name") or "")
    else:
        component_path = selected
        component_class = str(_class_name(selected) or "")
    if not component_path.exists():
        raise FileNotFoundError(f"VAE component does not exist: {component_path}")
    vae_model = _load_vae_component(
        component_path, component_class, torch_dtype, allow_custom_code
    )
else:
    component_class = str(_class_name(selected) or "")
    vae_model = _load_vae_file(
        selected,
        _infer_arch(selected, None, {}, component_class),
        torch_dtype,
    )

torch_dtype = resolved_dtype
vae_model.to(dtype=torch_dtype)
vae_model.to("cpu" if offload else device)
vae_model.eval()
class_name = component_class or vae_model.__class__.__name__
arch = _infer_arch(selected, index_class, entries, class_name or component_class)
channels_last = arch in {"sd15", "sdxl"} and class_name == "AutoencoderKL"
if channels_last:
    vae_model.to(memory_format=torch.channels_last)
load_time_ms = round((time.perf_counter() - load_started) * 1000, 1)
config = getattr(vae_model, "config", None)
latent_channels = int(
    getattr(config, "latent_channels", None)
    or getattr(config, "z_dim", None)
    or (4 if class_name == "AutoencoderKL" else 32)
)
if class_name == "AutoencoderKLFlux2":
    latent_channels *= 4
block_channels = getattr(config, "block_out_channels", (128, 256, 512, 512))
if class_name == "AutoencoderKLQwenImage21":
    vae_scale_factor = 16
elif class_name in {"AutoencoderKLQwenImage", "AutoencoderKLWan"}:
    vae_scale_factor = 8
else:
    vae_scale_factor = 2 ** (len(block_channels) - 1)
scaling_factor = float(
    getattr(config, "scaling_factor", None)
    or (0.13025 if arch == "sdxl" else 0.18215)
)
shift_factor = float(getattr(config, "shift_factor", 0.0) or 0.0)
latents_mean = getattr(config, "latents_mean", None)
latents_std = getattr(config, "latents_std", None)
if latents_mean is not None:
    latents_mean = [float(value) for value in latents_mean]
if latents_std is not None:
    latents_std = [float(value) for value in latents_std]
latent_space = "sd"
latent_packing = 1
if class_name == "AutoencoderKLFlux2":
    latent_space = "moss" if arch == "moss_image" else "flux2"
    latent_packing = 2
elif class_name in {"AutoencoderKLQwenImage", "AutoencoderKLQwenImage21"}:
    latent_space = "qwen"
elif class_name == "AutoencoderKLWan":
    latent_space = "wan"


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
    _configure_tiling(
        vae_model,
        bool(tiled),
        int(tile_size),
        int(overlap),
        int(temporal_size),
        int(temporal_overlap),
        vae_scale_factor,
    )
    with torch.inference_mode():
        distribution = vae_model.encode(
            pixel_tensor.to(device=device, dtype=torch_dtype)
        ).latent_dist
        latents = _retrieve_latents(distribution)
        if class_name == "AutoencoderKLFlux2":
            latents = _patchify(latents)
            batch_norm = getattr(vae_model, "bn", None)
            if batch_norm is not None:
                mean = batch_norm.running_mean.view(1, -1, 1, 1).to(latents)
                variance = batch_norm.running_var.view(1, -1, 1, 1).to(latents)
                epsilon = float(getattr(config, "batch_norm_eps", 1e-5) or 1e-5)
                latents = (latents - mean) / torch.sqrt(variance + epsilon)
        elif class_name in {"AutoencoderKLQwenImage", "AutoencoderKLQwenImage21"}:
            latents = _image_to_4d(latents)
            if latents_mean is not None and latents_std is not None:
                mean = torch.tensor(
                    latents_mean,
                    device=latents.device,
                    dtype=latents.dtype,
                ).view(1, -1, 1, 1)
                std = torch.tensor(
                    latents_std,
                    device=latents.device,
                    dtype=latents.dtype,
                ).view(1, -1, 1, 1)
                latents = (latents - mean) / std
        elif class_name == "AutoencoderKLWan":
            if latents.ndim == 4:
                latents = latents.unsqueeze(2)
        else:
            latents = _image_to_4d(latents)
            latents = (latents - shift_factor) * scaling_factor
    return latents.detach()


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
    _configure_tiling(
        vae_model,
        bool(tiled),
        int(tile_size),
        int(overlap),
        int(temporal_size),
        int(temporal_overlap),
        vae_scale_factor,
    )
    latents = latent_samples.to(device=device, dtype=torch_dtype)
    with torch.inference_mode():
        if class_name == "AutoencoderKLFlux2":
            batch_norm = getattr(vae_model, "bn", None)
            if batch_norm is not None:
                mean = batch_norm.running_mean.view(1, -1, 1, 1).to(latents)
                variance = batch_norm.running_var.view(1, -1, 1, 1).to(latents)
                epsilon = float(getattr(config, "batch_norm_eps", 1e-5) or 1e-5)
                latents = latents * torch.sqrt(variance + epsilon) + mean
            latents = _unpatchify(latents)
        elif class_name in {"AutoencoderKLQwenImage", "AutoencoderKLQwenImage21"}:
            if latents_mean is not None and latents_std is not None:
                mean = torch.tensor(
                    latents_mean,
                    device=latents.device,
                    dtype=latents.dtype,
                ).view(1, -1, 1, 1)
                std = torch.tensor(
                    latents_std,
                    device=latents.device,
                    dtype=latents.dtype,
                ).view(1, -1, 1, 1)
                latents = latents * std + mean
            if latents.ndim == 4:
                latents = latents.unsqueeze(2)
        elif class_name == "AutoencoderKLWan":
            if latents.ndim == 4:
                latents = latents.unsqueeze(2)
        else:
            latents = latents / scaling_factor + shift_factor
        decoded = vae_model.decode(latents, return_dict=False)[0]
        if decoded.ndim == 5:
            decoded = decoded[:, :, 0]
    return decoded.detach()


stat = selected.stat()
vae = {
    "kind": "vae",
    "backend": "diffusers",
    "arch": arch,
    "name": selected.stem if selected.is_file() else selected.name,
    "path": str(selected.resolve()),
    "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
    "source": "local",
    "model_type": "vae",
    "vae": vae_model,
    "vae_class": class_name,
    "device": device,
    "dtype": torch_dtype,
    "storage_dtype": storage_dtype,
    "compute_dtype": str(torch_dtype),
    "memory_profile": memory_profile,
    "offload": bool(offload),
    "resident_device": "cpu" if offload else device,
    "channels_last": bool(channels_last),
    "load_time_ms": load_time_ms,
    "scaling_factor": scaling_factor,
    "shift_factor": shift_factor,
    "latents_mean": latents_mean,
    "latents_std": latents_std,
    "vae_scale_factor": vae_scale_factor,
    "latent_channels": latent_channels,
    "latent_space": latent_space,
    "latent_packing": latent_packing,
    "encode": encode_pixels,
    "decode": decode_latents,
}
if callable(report):
    report(1.0, "VAE ready")

return vae: VAE
