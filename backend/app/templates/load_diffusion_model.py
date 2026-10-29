# /// script
# dependencies = [
#   "torch>=2.2",
#   "diffusers>=0.42.0",
#   "transformers>=5.0",
#   "accelerate>=1.0",
#   "safetensors>=0.4",
# ]
# ///
import importlib.util
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Literal

import diffusers
import torch

LOGGER = logging.getLogger("complexui.attention")

unet_name: ModelSelector = ModelSelector(
    type="diffusion",
    select="any",
    required=True,
    include=["unet"],
)
type: Literal[
    "auto",
    "stable_diffusion",
    "sd15",
    "sdxl",
    "flux",
    "flux1",
    "flux2",
    "flux3",
    "qwen_image",
    "z_image",
    "krea2",
    "ideogram4",
    "fibo2",
    "anima",
    "moss_image",
] = "auto"
weight_dtype: Literal[
    "default",
    "fp16",
    "bf16",
    "fp32",
    "fp8_e4m3fn",
    "fp8",
    "e4m3fn_last",
    "fp8_e5m2",
] = "default"
allow_custom_code: bool = False

DEFAULT_SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "steps_offset": 1,
}

ARCH_ALIASES = {
    "stable_diffusion": "sd15",
    "sd15": "sd15",
    "sd1.5": "sd15",
    "sd_15": "sd15",
    "sdxl": "sdxl",
    "sd_xl": "sdxl",
    "flux": "flux1",
    "flux1": "flux1",
    "flux_1": "flux1",
    "flux2": "flux2",
    "flux_2": "flux2",
    "flux3": "flux3",
    "flux_3": "flux3",
    "qwen_image": "qwen_image",
    "qwenimage": "qwen_image",
    "qwen_image_21": "qwen_image",
    "qwen_image_2.1": "qwen_image",
    "z_image": "z_image",
    "zimage": "z_image",
    "z-image": "z_image",
    "krea2": "krea2",
    "krea_2": "krea2",
    "ideogram4": "ideogram4",
    "ideogram_4": "ideogram4",
    "fibo2": "fibo2",
    "fibo_2": "fibo2",
    "bria_fibo2": "fibo2",
    "anima": "anima",
    "anima_base": "anima",
    "moss_image": "moss_image",
    "moss": "moss_image",
}

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
    "FluxControlPipeline": "flux1",
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
    "ZImageControlNetPipeline": "z_image",
    "ZImageControlNetInpaintPipeline": "z_image",
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

CLASS_ARCH = {
    "UNet2DConditionModel": "sd15",
    "FluxTransformer2DModel": "flux1",
    "Flux2Transformer2DModel": "flux2",
    "Flux3Transformer2DModel": "flux3",
    "QwenImageTransformer2DModel": "qwen_image",
    "QwenImage21Transformer2DModel": "qwen_image",
    "ZImageTransformer2DModel": "z_image",
    "Krea2Transformer2DModel": "krea2",
    "Ideogram4Transformer2DModel": "ideogram4",
    "BriaFibo2Transformer2DModel": "fibo2",
    "CosmosTransformer3DModel": "anima",
    "MOSSImageTransformer": "moss_image",
}

DEFAULT_TRANSFORMER_CLASS = {
    "sd15": "UNet2DConditionModel",
    "sdxl": "UNet2DConditionModel",
    "flux1": "FluxTransformer2DModel",
    "flux2": "Flux2Transformer2DModel",
    "flux3": "Flux3Transformer2DModel",
    "qwen_image": "QwenImage21Transformer2DModel",
    "z_image": "ZImageTransformer2DModel",
    "krea2": "Krea2Transformer2DModel",
    "ideogram4": "Ideogram4Transformer2DModel",
    "fibo2": "BriaFibo2Transformer2DModel",
    "anima": "CosmosTransformer3DModel",
    "moss_image": "MOSSImageTransformer",
}

ARCH_INFO = {
    "sd15": {
        "label": "Stable Diffusion 1.5",
        "width": 512,
        "height": 512,
        "steps": 25,
        "cfg": 7.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "latent_space": "sd",
        "latent_channels": 4,
        "latent_packing": 1,
        "vae_scale_factor": 8,
        "negative_prompt": "required",
        "guidance": "cfg",
        "max_references": 0,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": False,
    },
    "sdxl": {
        "label": "Stable Diffusion XL",
        "width": 1024,
        "height": 1024,
        "steps": 25,
        "cfg": 7.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "latent_space": "sd",
        "latent_channels": 4,
        "latent_packing": 1,
        "vae_scale_factor": 8,
        "negative_prompt": "required",
        "guidance": "cfg",
        "max_references": 0,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": False,
    },
    "flux1": {
        "label": "FLUX.1",
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "simple",
        "latent_space": "flux",
        "latent_channels": 16,
        "latent_packing": 2,
        "vae_scale_factor": 8,
        "negative_prompt": "optional",
        "guidance": "true_cfg",
        "guidance_default": 3.5,
        "max_references": 2,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": True,
    },
    "flux2": {
        "label": "FLUX.2",
        "width": 1024,
        "height": 1024,
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "latent_space": "flux2",
        "latent_channels": 32,
        "latent_packing": 2,
        "vae_scale_factor": 8,
        "negative_prompt": "none",
        "guidance": "distilled",
        "guidance_default": 4.0,
        "max_references": 10,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": True,
    },
    "flux3": {
        "label": "FLUX.3",
        "width": 1024,
        "height": 1024,
        "steps": 16,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "latent_space": "flux3",
        "latent_channels": 32,
        "latent_packing": 2,
        "vae_scale_factor": 8,
        "negative_prompt": "none",
        "guidance": "distilled",
        "guidance_default": 1.0,
        "max_references": 10,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": True,
    },
    "qwen_image": {
        "label": "Qwen Image 2.1",
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "cfg": 4.0,
        "sampler": "euler",
        "scheduler": "simple",
        "latent_space": "qwen",
        "latent_channels": 64,
        "latent_packing": 1,
        "vae_scale_factor": 16,
        "negative_prompt": "optional",
        "guidance": "true_cfg",
        "max_references": 10,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": True,
    },
    "z_image": {
        "label": "Z-Image",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "cfg": 5.0,
        "sampler": "euler",
        "scheduler": "simple",
        "latent_space": "z_image",
        "latent_channels": 16,
        "latent_packing": 1,
        "vae_scale_factor": 8,
        "negative_prompt": "optional",
        "guidance": "cfg",
        "max_references": 0,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": False,
    },
    "krea2": {
        "label": "Krea 2",
        "width": 1024,
        "height": 1024,
        "steps": 25,
        "cfg": 4.5,
        "sampler": "euler",
        "scheduler": "simple",
        "latent_space": "krea2",
        "latent_channels": 16,
        "latent_packing": 2,
        "vae_scale_factor": 8,
        "negative_prompt": "optional",
        "guidance": "cfg",
        "max_references": 0,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": False,
    },
    "ideogram4": {
        "label": "Ideogram 4.0",
        "width": 2048,
        "height": 2048,
        "steps": 20,
        "cfg": 7.0,
        "sampler": "euler",
        "scheduler": "simple",
        "latent_space": "ideogram4",
        "latent_channels": 32,
        "latent_packing": 2,
        "vae_scale_factor": 8,
        "negative_prompt": "none",
        "guidance": "dual_branch",
        "guidance_default": 7.0,
        "max_references": 0,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": False,
    },
    "fibo2": {
        "label": "Fibo 2.0",
        "width": 1024,
        "height": 1024,
        "steps": 25,
        "cfg": 5.0,
        "sampler": "euler",
        "scheduler": "simple",
        "latent_space": "fibo2",
        "latent_channels": 32,
        "latent_packing": 2,
        "vae_scale_factor": 8,
        "negative_prompt": "optional",
        "guidance": "cfg",
        "guidance_default": 5.0,
        "max_references": 5,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": True,
    },
    "anima": {
        "label": "Anima",
        "width": 1024,
        "height": 1024,
        "steps": 25,
        "cfg": 4.0,
        "sampler": "euler",
        "scheduler": "simple",
        "latent_space": "anima",
        "latent_channels": 16,
        "latent_packing": 1,
        "vae_scale_factor": 8,
        "negative_prompt": "required",
        "guidance": "cfg",
        "guidance_default": 4.0,
        "max_references": 0,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": False,
    },
    "moss_image": {
        "label": "MOSS Image",
        "width": 1024,
        "height": 1024,
        "steps": 50,
        "cfg": 3.0,
        "sampler": "euler",
        "scheduler": "simple",
        "latent_space": "moss",
        "latent_channels": 32,
        "latent_packing": 2,
        "vae_scale_factor": 8,
        "negative_prompt": "optional",
        "guidance": "cfg",
        "guidance_default": 3.0,
        "max_references": 0,
        "text_to_image": True,
        "img2img": True,
        "inpaint": True,
        "edit": False,
    },
}


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _canonical_type(value: str):
    if value == "auto":
        return None
    canonical = ARCH_ALIASES.get(value.strip().lower())
    if canonical is None:
        raise ValueError(f"Unsupported architecture type: {value}")
    return canonical


def _config_class(path: Path):
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
    config = _read_json(config_path)
    if config is None:
        return None
    return config.get("_class_name") or config.get("architectures", [None])[0]


def _component_entries(root: Path):
    entries = {}
    standard = _read_json(root / "model_index.json")
    modular = _read_json(root / "modular_model_index.json")
    index_class = None
    if standard is not None:
        index_class = standard.get("_class_name")
        for name, value in standard.items():
            if name.startswith("_") or not isinstance(value, (list, tuple)):
                continue
            if len(value) != 2:
                continue
            entries[name] = {
                "class_name": str(value[1]),
                "subfolder": name,
            }
    if modular is not None:
        index_class = index_class or modular.get("_class_name")
        for name, value in modular.items():
            if name.startswith("_") or not isinstance(value, dict):
                continue
            type_hint = value.get("type_hint")
            class_name = None
            if isinstance(type_hint, (list, tuple)) and len(type_hint) == 2:
                class_name = str(type_hint[1])
            if class_name is None:
                continue
            entries[name] = {
                "class_name": class_name,
                "subfolder": str(value.get("subfolder") or name),
            }
    if not entries and root.is_dir():
        class_name = _config_class(root)
        if class_name:
            entries["component"] = {"class_name": class_name, "subfolder": "."}
    return index_class, entries


def _unet_is_sdxl(config) -> bool:
    if not isinstance(config, dict):
        return False
    cross_attention_dim = config.get("cross_attention_dim")
    if isinstance(cross_attention_dim, (list, tuple)):
        cross_attention_dim = max(
            [int(item) for item in cross_attention_dim if item is not None] or [0]
        )
    if cross_attention_dim is not None and int(cross_attention_dim) >= 2048:
        return True
    return bool(config.get("addition_embed_type"))


def _detect_arch(root: Path, explicit, index_class, entries) -> str:
    if explicit is not None:
        return explicit
    if isinstance(index_class, str) and index_class in PIPELINE_ARCH:
        return PIPELINE_ARCH[index_class]
    for entry in entries.values():
        class_name = entry.get("class_name")
        if class_name == "UNet2DConditionModel":
            has_second = "text_encoder_2" in entries or "tokenizer_2" in entries
            config = _read_json(
                root / str(entry.get("subfolder") or "") / "config.json"
            )
            return "sdxl" if has_second or _unet_is_sdxl(config) else "sd15"
        if class_name in CLASS_ARCH and class_name != "UNet2DConditionModel":
            return CLASS_ARCH[class_name]
    name = root.name.lower()
    for token, arch in (
        ("flux3", "flux3"),
        ("flux2", "flux2"),
        ("flux", "flux1"),
        ("qwen", "qwen_image"),
        ("krea", "krea2"),
        ("ideogram", "ideogram4"),
        ("fibo", "fibo2"),
        ("anima", "anima"),
        ("moss", "moss_image"),
        ("sdxl", "sdxl"),
        ("sd15", "sd15"),
    ):
        if token in name:
            return arch
    if root.is_file():
        class_name = _config_class(root)
        if class_name in CLASS_ARCH:
            return CLASS_ARCH[class_name]
    raise ValueError(
        "Could not detect the diffusion architecture. Select type explicitly "
        "or use a diffusers repository/component folder with config.json."
    )


def _load_repo_component_class(path: Path, class_name, torch_dtype):
    """Load a component class declared by a repository's custom pipeline file."""
    if not class_name:
        return None
    for source in sorted(path.parent.glob("pipeline*.py")):
        module_name = f"complexui_custom_{source.stem}_{abs(hash(source))}"
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            continue
        component_class = getattr(module, class_name, None)
        if component_class is None or not hasattr(
            component_class,
            "from_pretrained",
        ):
            continue
        try:
            return component_class.from_pretrained(
                str(path),
                torch_dtype=torch_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except Exception:
            continue
    return None


def _load_quantized_component_dir(path: Path, torch_dtype, trust_remote_code: bool):
    try:
        from app.quantized_loading import load_quantized_component
    except ImportError as exc:
        raise ValueError(
            "Pre-quantized diffusion weights require the comfy-kitchen runtime; "
            "install it in the backend environment"
        ) from exc
    auto_model = getattr(diffusers, "AutoModel", None)
    if auto_model is None:
        raise ValueError(
            "This diffusers build cannot build a component from config for "
            "pre-quantized weights"
        )
    model = auto_model.from_config(
        str(path),
        trust_remote_code=trust_remote_code,
    )
    loaded, report = load_quantized_component(
        model,
        path,
        compute_dtype=torch_dtype,
        device="cpu",
    )
    print(
        "quantized component: "
        f"{sum(1 for entry in report if entry.native)} native, "
        f"{sum(1 for entry in report if not entry.native)} emulated layer(s)"
    )
    return loaded


def _load_component_dir(path: Path, torch_dtype, trust_remote_code: bool):
    class_name = _config_class(path)
    try:
        from app.quantization import detect_quantization, validate_quantization
    except ImportError:
        quantized = None
    else:
        quantized = detect_quantization(path, "diffusion")
        if quantized is not None and quantized.format != "none":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            capability = None
            if device_name == "cuda":
                try:
                    capability = torch.cuda.get_device_capability()
                except Exception:
                    capability = None
            validate_quantization(
                quantized,
                model_type="diffusion",
                device=device_name,
                capability=capability,
            )
            return _load_quantized_component_dir(path, torch_dtype, trust_remote_code)
    errors = []
    auto_model = getattr(diffusers, "AutoModel", None)
    if auto_model is not None:
        try:
            return auto_model.from_pretrained(
                str(path),
                torch_dtype=torch_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            errors.append(str(exc))
    if class_name:
        component_class = getattr(diffusers, class_name, None)
        if component_class is not None:
            try:
                return component_class.from_pretrained(
                    str(path),
                    torch_dtype=torch_dtype,
                    local_files_only=True,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as exc:
                errors.append(str(exc))
    if trust_remote_code:
        custom = _load_repo_component_class(path, class_name, torch_dtype)
        if custom is not None:
            return custom
    detail = errors[-1] if errors else "unknown loader error"
    raise ValueError(
        f"Could not load diffusion component '{path.name}'. "
        f"Expected a local diffusers folder. Details: {detail}"
    )


def _load_component_file(path: Path, arch: str, torch_dtype):
    class_name = _config_class(path) or DEFAULT_TRANSFORMER_CLASS.get(arch)
    if not class_name:
        raise ValueError(
            f"Single-file loading is not available for architecture '{arch}'"
        )
    component_class = getattr(diffusers, class_name, None)
    if component_class is None:
        raise ValueError(
            f"diffusers does not expose the '{class_name}' component class. "
            "Upgrade diffusers or use a full component folder."
        )
    if not hasattr(component_class, "from_single_file"):
        raise ValueError(
            f"'{class_name}' cannot load a single file. Use a diffusers component "
            "folder containing config.json and weights."
        )
    kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
    if (path.parent / "config.json").is_file():
        kwargs["config"] = str(path.parent)
    try:
        return component_class.from_single_file(str(path), **kwargs)
    except Exception as exc:
        raise ValueError(
            f"Could not load '{path.name}' as {class_name}. Single diffusion "
            "component files need the matching config.json in the same folder. "
            f"Details: {exc}"
        ) from exc


def _load_scheduler(root: Path, entries, arch: str, trust_remote_code: bool):
    scheduler_entry = entries.get("scheduler")
    if scheduler_entry is None:
        config = (
            dict(DEFAULT_SCHEDULER_CONFIG)
            if arch in {"sd15", "sdxl"}
            else {}
        )
        return None, config
    subfolder = str(scheduler_entry.get("subfolder") or "scheduler")
    scheduler_path = root / subfolder
    config = _read_json(scheduler_path / "scheduler_config.json") or {}
    scheduler_mixin = getattr(diffusers, "SchedulerMixin", None)
    if scheduler_mixin is not None and scheduler_path.is_dir():
        try:
            scheduler = scheduler_mixin.from_pretrained(
                str(scheduler_path),
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            )
            return scheduler, dict(scheduler.config)
        except Exception:
            pass
    if config:
        return None, config
    return None, (
        dict(DEFAULT_SCHEDULER_CONFIG) if arch in {"sd15", "sdxl"} else {}
    )


def _load_optional_component(
    root: Path,
    entries,
    name: str,
    torch_dtype,
    trust_remote_code: bool,
):
    entry = entries.get(name)
    if entry is None:
        return None
    path = root / str(entry.get("subfolder") or name)
    if not path.exists():
        return None
    return _load_component_dir(path, torch_dtype, trust_remote_code)


def _patchify_latents(latents):
    """Pack 2x2 latent patches into channels, matching the FLUX family layout."""
    batch, channels, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError("Latent height and width must be divisible by two")
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch, channels * 4, height // 2, width // 2)


def _unpatchify_latents(latents):
    """Reverse ``_patchify_latents`` without changing channel ordering."""
    batch, channels, height, width = latents.shape
    if channels % 4:
        raise ValueError("Packed latent channels must be divisible by four")
    latents = latents.view(batch, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch, channels // 4, height * 2, width * 2)


def _squeeze_latent_frames(samples):
    if samples.ndim == 5:
        if samples.shape[2] != 1:
            raise ValueError("Video latents are not supported by image samplers")
        return samples[:, :, 0]
    if samples.ndim != 4:
        raise ValueError("LATENT samples must be a four- or five-dimensional tensor")
    return samples


def _to_latent_4d(
    samples,
    channels,
    height,
    width,
    device,
    dtype,
    *,
    patchified=False,
):
    """Normalize an incoming LATENT to the raw (unpacked) VAE latent layout."""
    batch = int(samples.shape[0])
    samples = _squeeze_latent_frames(samples)
    expected = (channels, height, width)
    if tuple(samples.shape[1:]) == expected:
        return samples.to(device=device, dtype=dtype)
    packed_expected = (channels * 4, height // 2, width // 2)
    if patchified and tuple(samples.shape[1:]) == packed_expected:
        return _unpatchify_latents(samples).to(device=device, dtype=dtype)
    if samples.numel() > 0 and bool(torch.count_nonzero(samples).item() == 0):
        return torch.zeros(
            (batch, channels, height, width),
            device=device,
            dtype=dtype,
        )
    if tuple(samples.shape[1:]) == (channels * 4, height // 2, width // 2):
        return _unpatchify_latents(samples).to(device=device, dtype=dtype)
    raise ValueError(
        "LATENT shape does not match the selected model: "
        f"got {tuple(samples.shape)}, expected ({batch}, {channels}, {height}, {width})"
    )


def _to_patchified_4d(samples, height, width, device, dtype):
    """Normalize to the 128-channel packed VAE layout used by FLUX.2/3."""
    batch = int(samples.shape[0])
    samples = _squeeze_latent_frames(samples)
    if tuple(samples.shape[1:]) == (128, height, width):
        return samples.to(device=device, dtype=dtype)
    if tuple(samples.shape[1:]) == (32, height * 2, width * 2):
        return _patchify_latents(samples).to(device=device, dtype=dtype)
    if samples.numel() > 0 and bool(torch.count_nonzero(samples).item() == 0):
        return torch.zeros((batch, 128, height, width), device=device, dtype=dtype)
    raise ValueError(
        "FLUX LATENT shape does not match the selected model: "
        f"got {tuple(samples.shape)}, expected ({batch}, 128, {height}, {width})"
    )


def _to_patchified_tensor(samples, device, dtype):
    """Normalize a VAE-encoded reference to the 128-channel packed layout."""
    samples = _squeeze_latent_frames(samples)
    if samples.shape[1] == 128:
        return samples.to(device=device, dtype=dtype)
    if samples.shape[1] == 32:
        return _patchify_latents(samples).to(device=device, dtype=dtype)
    raise ValueError(
        "Reference latents must use 32 raw or 128 packed channels for this model"
    )


def _pack_tokens(latents):
    batch, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).transpose(1, 2)


def _unpack_tokens(tokens, channels, height, width):
    batch = int(tokens.shape[0])
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def _latent_grid_ids(height, width, device, time_coord=0.0):
    """FLUX-style 3D image ids for one latent grid."""
    ids = torch.zeros(height, width, 3, device=device)
    ids[..., 0] = float(time_coord)
    ids[..., 1] = torch.arange(height, device=device)[:, None]
    ids[..., 2] = torch.arange(width, device=device)[None, :]
    return ids.reshape(height * width, 3)


def _spatial_ids(
    height,
    width,
    device,
    *,
    scale=1.0,
    time_coord=0.0,
    layer_coord=0.0,
):
    """FLUX.2/FLUX.3 4D image ids (T, H, W, L) for one latent grid."""
    time_ids = torch.tensor([float(time_coord)], device=device)
    row_ids = torch.arange(height, device=device, dtype=torch.float32) * scale
    column_ids = torch.arange(width, device=device, dtype=torch.float32) * scale
    layer_ids = torch.tensor([float(layer_coord)], device=device)
    return torch.cartesian_prod(
        time_ids,
        row_ids,
        column_ids,
        layer_ids,
    )


def _token_mask(mask, height, width, device, patch_size=1):
    """Reduce a pixel mask to the token grid used by an adapter."""
    if not isinstance(mask, torch.Tensor):
        return None
    mask = mask.float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        raise ValueError("inpaint mask must be a 2D, 3D or 4D tensor")
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    mask = torch.nn.functional.interpolate(
        mask,
        size=(height, width),
        mode="area",
    )
    if patch_size > 1:
        if height % patch_size or width % patch_size:
            raise ValueError("Inpaint mask grid must be divisible by the patch size")
        mask = mask.unfold(2, patch_size, patch_size)
        mask = mask.unfold(3, patch_size, patch_size)
        mask = mask.mean(dim=(-1, -2))
    return mask.reshape(mask.shape[0], -1, 1).to(device)


def _text_ids_3d(length, device, dtype):
    return torch.zeros(length, 3, device=device, dtype=dtype)


def _text_ids_4d(length, device):
    ids = torch.zeros(length, 4, dtype=torch.long, device=device)
    ids[:, 3] = torch.arange(length, device=device)
    return ids.unsqueeze(0)


def _conditioning(value, device, dtype):
    if not isinstance(value, dict):
        raise TypeError("conditioning input must be a CONDITIONING wrapper")
    cond = value.get("cond")
    if not isinstance(cond, torch.Tensor):
        raise TypeError("conditioning wrapper is missing its cond tensor")
    pooled = value.get("pooled")
    mask = value.get("attention_mask")
    image_pad_mask = value.get("image_pad_mask")
    return {
        "cond": cond.to(device=device, dtype=dtype),
        "pooled": (
            pooled.to(device=device, dtype=dtype)
            if isinstance(pooled, torch.Tensor)
            else None
        ),
        "attention_mask": (
            mask.to(device=device)
            if isinstance(mask, torch.Tensor)
            else None
        ),
        "image_pad_mask": (
            image_pad_mask.to(device=device)
            if isinstance(image_pad_mask, torch.Tensor)
            else None
        ),
    }


def _apply_cfg(positive, negative, scale, formula="standard"):
    if negative is None or float(scale) <= 1.0:
        return positive
    if formula == "additive":
        return positive + float(scale) * (positive - negative)
    return negative + float(scale) * (positive - negative)


def _expand_conditioning(current, batch):
    if batch <= 1:
        return current
    expanded = dict(current)
    for key in ("cond", "pooled", "attention_mask", "image_pad_mask"):
        value = expanded.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] == 1:
            expanded[key] = value.expand(batch, *value.shape[1:])
    return expanded


def _run_flux1(
    transformer,
    state,
    timestep,
    conditioning,
    cfg,
    negative,
    guidance,
    *,
    allow_cfg_with_guidance=False,
):
    device = state["latents"].device
    dtype = state["latents"].dtype
    if guidance is not None:
        guidance = torch.as_tensor(
            guidance,
            device=device,
            dtype=torch.float32,
        ).reshape(-1)
    text_ids = _text_ids_3d(conditioning["cond"].shape[1], device, dtype)
    refs = state.get("ref_latents") or []
    hidden = state["latents"]
    image_ids = state["img_ids"]
    if refs:
        ref_tokens = []
        ref_ids = []
        for raw in refs:
            packed = _patchify_latents(raw).flatten(2).transpose(1, 2)
            if packed.shape[0] != hidden.shape[0]:
                packed = packed.expand(hidden.shape[0], -1, -1)
            ref_tokens.append(packed)
            ref_ids.append(
                _latent_grid_ids(
                    raw.shape[2] // 2,
                    raw.shape[3] // 2,
                    device,
                    time_coord=1.0,
                ).to(dtype=dtype)
            )
        hidden = torch.cat([hidden, *ref_tokens], dim=1)
        image_ids = torch.cat([image_ids, *ref_ids], dim=0)
    if state.get("inpaint_tokens") is not None:
        hidden = torch.cat([hidden, state["inpaint_tokens"]], dim=2)
    timestep_values = (timestep / 1000).expand(hidden.shape[0]).to(dtype)
    conditioning = _expand_conditioning(conditioning, hidden.shape[0])
    negative = (
        _expand_conditioning(negative, hidden.shape[0])
        if isinstance(negative, dict)
        else negative
    )

    def run(current):
        output = transformer(
            hidden_states=hidden,
            encoder_hidden_states=current["cond"],
            pooled_projections=current["pooled"],
            timestep=timestep_values,
            img_ids=image_ids,
            txt_ids=text_ids,
            guidance=(
                guidance.expand(hidden.shape[0])
                if guidance is not None and guidance.numel() == 1
                else guidance
            ),
            return_dict=False,
        )[0]
        return output[
            :,
            : state["latents"].shape[1],
            : state.get("output_channels", output.shape[2]),
        ]

    positive = run(conditioning)
    if (
        (guidance is not None and not allow_cfg_with_guidance)
        or float(cfg) <= 1.0
        or negative is None
    ):
        return positive
    negative = run(negative)
    return _apply_cfg(positive, negative, cfg)


def _make_flux1_adapter(transformer, latent_channels, vae_scale_factor, edit_capable):
    guidance_embeds = bool(
        getattr(getattr(transformer, "config", None), "guidance_embeds", False)
    )
    transformer_in = int(
        getattr(
            getattr(transformer, "config", None),
            "in_channels",
            latent_channels * 4,
        )
    )
    fill_capable = transformer_in > latent_channels * 4

    def prepare(samples, width, height, device, dtype, references=(), inpaint=None, model=None):
        if references and not edit_capable:
            raise ValueError(
                "This FLUX.1 model does not accept reference images; use a "
                "FLUX.1 Kontext checkpoint"
            )
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        raw = _to_latent_4d(
            samples,
            latent_channels,
            latent_height,
            latent_width,
            device,
            dtype,
        )
        packed = _patchify_latents(raw).flatten(2).transpose(1, 2)
        ids = _latent_grid_ids(
            latent_height // 2,
            latent_width // 2,
            device,
        ).to(dtype=dtype)
        refs = []
        for reference in references:
            if not isinstance(reference, torch.Tensor):
                continue
            raw_ref = _squeeze_latent_frames(reference)
            if raw_ref.shape[1] != latent_channels:
                raise ValueError(
                    "Reference latents must use the selected model's latent width"
                )
            refs.append(raw_ref.to(device=device, dtype=dtype))
        inpaint_tokens = None
        noise_mask = None
        if inpaint is not None:
            mask = inpaint.get("mask")
            masked = inpaint.get("masked_latents")
            if fill_capable:
                if not isinstance(masked, torch.Tensor):
                    raise TypeError("inpaint conditioning is missing masked latents")
                masked_raw = _to_latent_4d(
                    masked,
                    latent_channels,
                    latent_height,
                    latent_width,
                    device,
                    dtype,
                )
                mask_values = _token_mask(
                    mask,
                    latent_height,
                    latent_width,
                    device,
                    patch_size=2,
                )
                if mask_values is None:
                    raise TypeError("inpaint conditioning is missing its mask")
                mask_tokens = mask_values.expand(
                    -1, -1, max(1, transformer_in - latent_channels * 8)
                )
                masked_tokens = _patchify_latents(masked_raw).flatten(
                    2
                ).transpose(1, 2)
                inpaint_tokens = torch.cat(
                    [masked_tokens, mask_tokens],
                    dim=2,
                )
            else:
                noise_mask = _token_mask(
                    mask,
                    latent_height,
                    latent_width,
                    device,
                    patch_size=2,
                )
        return {
            "latents": packed,
            "img_ids": ids,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "ref_latents": refs,
            "inpaint_tokens": inpaint_tokens,
            "noise_mask": noise_mask,
            "output_channels": latent_channels * 4,
        }

    def predict(
        latents,
        state,
        timestep,
        positive,
        negative,
        cfg,
        next_timestep=None,
        guidance=None,
        guidance_explicit=False,
        step_fraction=None,
    ):
        current = _conditioning(positive, latents.device, latents.dtype)
        current_negative = (
            _conditioning(negative, latents.device, latents.dtype)
            if isinstance(negative, dict)
            else None
        )
        if guidance is None and guidance_embeds:
            guidance = torch.full(
                [1],
                float(cfg),
                device=latents.device,
                dtype=torch.float32,
            )
        return _run_flux1(
            transformer,
            state,
            timestep,
            current,
            cfg,
            current_negative,
            guidance,
            allow_cfg_with_guidance=guidance_explicit,
        )

    def finalize(latents, state):
        packed = latents.transpose(1, 2).reshape(
            latents.shape[0],
            latent_channels * 4,
            state["latent_height"] // 2,
            state["latent_width"] // 2,
        )
        return _unpatchify_latents(packed)

    return {
        "arch": "flux1",
        "prepare_latents": prepare,
        "predict": predict,
        "finalize_latents": finalize,
        "supports_references": True,
        "supports_guidance": bool(guidance_embeds),
        "supports_inpaint": True,
        "edit_capable": bool(edit_capable),
    }


def _run_flux2(
    transformer,
    state,
    timestep,
    conditioning,
    cfg,
    negative,
    guidance,
    *,
    distilled=False,
    next_timestep=None,
    allow_cfg_with_guidance=False,
):
    device = state["latents"].device
    dtype = state["latents"].dtype
    if guidance is not None:
        guidance = torch.as_tensor(
            guidance,
            device=device,
            dtype=torch.float32,
        ).reshape(-1)
    hidden = state["latents"]
    image_ids = state["img_ids"]
    ref_tokens = state.get("ref_tokens")
    ref_ids = state.get("ref_ids")
    if ref_tokens is not None and ref_ids is not None:
        hidden = torch.cat([hidden, ref_tokens], dim=1)
        image_ids = torch.cat([image_ids, ref_ids], dim=1)
    text_ids = _text_ids_4d(
        conditioning["cond"].shape[1],
        device,
    ).expand(hidden.shape[0], -1, -1)
    conditioning = _expand_conditioning(conditioning, hidden.shape[0])
    negative = (
        _expand_conditioning(negative, hidden.shape[0])
        if isinstance(negative, dict)
        else negative
    )
    timestep_values = (timestep / 1000).expand(hidden.shape[0]).to(dtype)
    target_values = None
    if distilled:
        target = (
            torch.zeros_like(timestep)
            if next_timestep is None
            else next_timestep
        )
        target_values = (target / 1000).expand(hidden.shape[0]).to(dtype)

    def run(current):
        kwargs = {
            "hidden_states": hidden,
            "encoder_hidden_states": current["cond"],
            "timestep": timestep_values,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "guidance": (
                guidance.expand(hidden.shape[0])
                if guidance is not None and guidance.numel() == 1
                else guidance
            ),
            "return_dict": False,
        }
        if target_values is not None:
            kwargs["target_timestep"] = target_values
        output = transformer(**kwargs)[0]
        return output[:, : state["latents"].shape[1]]

    positive = run(conditioning)
    if (
        (guidance is not None and not allow_cfg_with_guidance)
        or distilled
        or float(cfg) <= 1.0
        or negative is None
    ):
        return positive
    negative = run(negative)
    return _apply_cfg(positive, negative, cfg)


def _make_flux2_adapter(
    transformer,
    vae_scale_factor,
    *,
    distilled,
    klein,
    edit_capable,
):
    guidance_embeds = not bool(klein)

    def prepare(samples, width, height, device, dtype, references=(), inpaint=None, model=None):
        latent_height = height // (vae_scale_factor * 2)
        latent_width = width // (vae_scale_factor * 2)
        packed = _to_patchified_4d(
            samples,
            latent_height,
            latent_width,
            device,
            dtype,
        )
        tokens = _pack_tokens(packed)
        ids = _spatial_ids(
            latent_height,
            latent_width,
            device,
        ).unsqueeze(0).repeat(tokens.shape[0], 1, 1)
        ref_tokens = []
        ref_ids = []
        for index, reference in enumerate(references):
            if not isinstance(reference, torch.Tensor):
                continue
            current = _to_patchified_tensor(reference, device, dtype)
            ref_tokens.append(_pack_tokens(current).squeeze(0))
            ref_ids.append(
                _spatial_ids(
                    current.shape[2],
                    current.shape[3],
                    device,
                    time_coord=10 + 10 * index,
                )
            )
        noise_mask = None
        if inpaint is not None:
            noise_mask = _token_mask(
                inpaint.get("mask"),
                latent_height,
                latent_width,
                device,
            )
        return {
            "latents": tokens,
            "img_ids": ids,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "ref_tokens": (
                torch.cat(ref_tokens, dim=0)
                .unsqueeze(0)
                .repeat(tokens.shape[0], 1, 1)
                if ref_tokens
                else None
            ),
            "ref_ids": (
                torch.cat(ref_ids, dim=0)
                .unsqueeze(0)
                .repeat(tokens.shape[0], 1, 1)
                if ref_ids
                else None
            ),
            "noise_mask": noise_mask,
        }

    def predict(
        latents,
        state,
        timestep,
        positive,
        negative,
        cfg,
        next_timestep=None,
        guidance=None,
        guidance_explicit=False,
        step_fraction=None,
    ):
        current = _conditioning(positive, latents.device, latents.dtype)
        current_negative = (
            _conditioning(negative, latents.device, latents.dtype)
            if isinstance(negative, dict)
            else None
        )
        if guidance is None and guidance_embeds:
            guidance = torch.full(
                [1],
                float(cfg),
                device=latents.device,
                dtype=torch.float32,
            )
        return _run_flux2(
            transformer,
            state,
            timestep,
            current,
            cfg,
            current_negative,
            guidance,
            distilled=distilled,
            next_timestep=next_timestep,
            allow_cfg_with_guidance=guidance_explicit,
        )

    def finalize(latents, state):
        return _unpack_tokens(
            latents,
            128,
            state["latent_height"],
            state["latent_width"],
        )

    return {
        "arch": "flux2",
        "prepare_latents": prepare,
        "predict": predict,
        "finalize_latents": finalize,
        "supports_references": True,
        "supports_guidance": bool(guidance_embeds),
        "supports_inpaint": True,
        "edit_capable": bool(edit_capable),
    }


def _run_flux3(
    transformer,
    state,
    timestep,
    conditioning,
    cfg,
    negative,
    flow_map,
    next_timestep,
):
    device = state["latents"].device
    dtype = state["latents"].dtype
    hidden = state["latents"]
    image_ids = state["img_ids"]
    ref_tokens = state.get("ref_tokens")
    ref_ids = state.get("ref_ids")
    text_ids = _text_ids_4d(
        conditioning["cond"].shape[1],
        device,
    ).squeeze(0).to(torch.float32)
    conditioning = _expand_conditioning(conditioning, hidden.shape[0])
    negative = (
        _expand_conditioning(negative, hidden.shape[0])
        if isinstance(negative, dict)
        else negative
    )
    timestep_values = (timestep / 1000).expand(hidden.shape[0]).to(dtype)
    target_values = None
    if flow_map:
        target = (
            torch.zeros_like(timestep)
            if next_timestep is None
            else next_timestep
        )
        target_values = (target / 1000).expand(hidden.shape[0]).to(dtype)

    def run(current):
        kwargs = {
            "hidden_states": hidden,
            "encoder_hidden_states": current["cond"],
            "timestep": timestep_values,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "return_dict": False,
        }
        if ref_tokens is not None and ref_ids is not None:
            kwargs["image_cond_hidden_states"] = ref_tokens
            kwargs["image_cond_ids"] = ref_ids
        if target_values is not None:
            kwargs["target_timestep"] = target_values
        output = transformer(**kwargs)[0]
        return output[:, : state["latents"].shape[1]]

    positive = run(conditioning)
    if flow_map or float(cfg) <= 1.0 or negative is None:
        return positive
    negative = run(negative)
    return _apply_cfg(positive, negative, cfg)


def _make_flux3_adapter(transformer, vae_scale_factor, *, edit_capable):
    flow_map = bool(
        getattr(getattr(transformer, "config", None), "flow_map", False)
    )
    def prepare(samples, width, height, device, dtype, references=(), inpaint=None, model=None):
        latent_height = height // (vae_scale_factor * 2)
        latent_width = width // (vae_scale_factor * 2)
        packed = _to_patchified_4d(
            samples,
            latent_height,
            latent_width,
            device,
            dtype,
        )
        tokens = _pack_tokens(packed)
        ref_tokens = []
        ref_ids = []
        for index, reference in enumerate(references):
            if not isinstance(reference, torch.Tensor):
                continue
            current = _to_patchified_tensor(reference, device, dtype)
            ref_tokens.append(_pack_tokens(current).squeeze(0))
            ref_ids.append(
                _spatial_ids(
                    current.shape[2],
                    current.shape[3],
                    device,
                    scale=0.5,
                    layer_coord=-(1 + index),
                )
            )
        noise_mask = None
        if inpaint is not None:
            noise_mask = _token_mask(
                inpaint.get("mask"),
                latent_height,
                latent_width,
                device,
            )
        return {
            "latents": tokens,
            "img_ids": _spatial_ids(
                latent_height,
                latent_width,
                device,
                scale=0.5,
            ),
            "latent_height": latent_height,
            "latent_width": latent_width,
            "ref_tokens": (
                torch.cat(ref_tokens, dim=0)
                .unsqueeze(0)
                .repeat(tokens.shape[0], 1, 1)
                if ref_tokens
                else None
            ),
            "ref_ids": (
                torch.cat(ref_ids, dim=0)
                if ref_ids
                else None
            ),
            "noise_mask": noise_mask,
        }

    def predict(
        latents,
        state,
        timestep,
        positive,
        negative,
        cfg,
        next_timestep=None,
        guidance=None,
        guidance_explicit=False,
        step_fraction=None,
    ):
        current = _conditioning(positive, latents.device, latents.dtype)
        current_negative = (
            _conditioning(negative, latents.device, latents.dtype)
            if isinstance(negative, dict)
            else None
        )
        return _run_flux3(
            transformer,
            state,
            timestep,
            current,
            cfg,
            current_negative,
            flow_map,
            next_timestep,
        )

    def finalize(latents, state):
        return _unpack_tokens(
            latents,
            128,
            state["latent_height"],
            state["latent_width"],
        )

    return {
        "arch": "flux3",
        "prepare_latents": prepare,
        "predict": predict,
        "finalize_latents": finalize,
        "supports_references": True,
        "supports_guidance": False,
        "supports_inpaint": True,
        "edit_capable": bool(edit_capable),
    }


def _run_qwen(
    transformer,
    state,
    timestep,
    conditioning,
    cfg,
    negative,
):
    device = state["latents"].device
    dtype = state["latents"].dtype
    hidden = state["latents"]
    ref_tokens = state.get("ref_tokens")
    if ref_tokens is not None:
        hidden = torch.cat([ref_tokens, hidden], dim=1)
    conditioning = _expand_conditioning(conditioning, hidden.shape[0])
    negative = (
        _expand_conditioning(negative, hidden.shape[0])
        if isinstance(negative, dict)
        else negative
    )
    timestep_values = (timestep / 1000).expand(hidden.shape[0]).to(dtype)
    img_shapes = state["img_shapes"]
    if ref_tokens is not None:
        img_shapes = [
            [*state["ref_shapes"], *shape]
            for shape in img_shapes
        ]
    step_latent_count = state["latents"].shape[1]
    target_slots = max(1, step_latent_count // 4)

    def run(current):
        prompt_mask = current.get("image_pad_mask")
        if (
            isinstance(prompt_mask, torch.Tensor)
            and prompt_mask.shape[-1] == current["cond"].shape[1]
        ):
            prompt_mask = prompt_mask.to(device=device).bool()
        else:
            prompt_mask = torch.zeros(
                hidden.shape[0],
                current["cond"].shape[1],
                dtype=torch.bool,
                device=device,
            )
        image_mask = torch.cat(
            [
                prompt_mask,
                torch.ones(
                    hidden.shape[0],
                    target_slots,
                    dtype=torch.bool,
                    device=device,
                ),
            ],
            dim=1,
        )
        output = transformer(
            hidden_states=hidden,
            encoder_hidden_states=current["cond"],
            timestep=timestep_values,
            img_shapes=img_shapes,
            img_mask=image_mask,
            encoder_hidden_states_mask=current["attention_mask"],
            return_dict=False,
        )[0]
        return output[:, -step_latent_count:]

    positive = run(conditioning)
    if float(cfg) <= 1.0 or negative is None:
        return positive
    negative = run(negative)
    return _apply_cfg(positive, negative, cfg)


def _make_qwen_adapter(transformer, vae_scale_factor, *, edit_capable):
    def prepare(samples, width, height, device, dtype, references=(), inpaint=None, model=None):
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        raw = _to_latent_4d(
            samples,
            64,
            latent_height,
            latent_width,
            device,
            dtype,
        )
        ref_tokens = []
        ref_shapes = []
        for reference in references:
            if not isinstance(reference, torch.Tensor):
                continue
            raw_ref = _squeeze_latent_frames(reference)
            if raw_ref.shape[1] != 64:
                raise ValueError("Qwen reference latents must have 64 channels")
            ref_tokens.append(
                _pack_tokens(raw_ref.to(device=device, dtype=dtype))
            )
            ref_shapes.append(
                (1, int(raw_ref.shape[2]), int(raw_ref.shape[3]))
            )
        noise_mask = None
        if inpaint is not None:
            noise_mask = _token_mask(
                inpaint.get("mask"),
                latent_height,
                latent_width,
                device,
            )
        return {
            "latents": _pack_tokens(raw),
            "latent_height": latent_height,
            "latent_width": latent_width,
            "img_shapes": [
                [(1, latent_height, latent_width)]
                for _ in range(int(raw.shape[0]))
            ],
            "ref_tokens": (
                torch.cat(ref_tokens, dim=1)
                .repeat(int(raw.shape[0]), 1, 1)
                if ref_tokens
                else None
            ),
            "ref_shapes": ref_shapes,
            "noise_mask": noise_mask,
        }

    def predict(
        latents,
        state,
        timestep,
        positive,
        negative,
        cfg,
        next_timestep=None,
        guidance=None,
        guidance_explicit=False,
        step_fraction=None,
    ):
        current = _conditioning(positive, latents.device, latents.dtype)
        current_negative = (
            _conditioning(negative, latents.device, latents.dtype)
            if isinstance(negative, dict)
            else None
        )
        return _run_qwen(
            transformer,
            state,
            timestep,
            current,
            cfg,
            current_negative,
        )

    def finalize(latents, state):
        return _unpack_tokens(
            latents,
            64,
            state["latent_height"],
            state["latent_width"],
        )

    return {
        "arch": "qwen_image",
        "prepare_latents": prepare,
        "predict": predict,
        "finalize_latents": finalize,
        "supports_references": True,
        "supports_guidance": False,
        "supports_inpaint": True,
        "edit_capable": bool(edit_capable),
    }


def _make_zimage_adapter(transformer, vae_scale_factor):
    def prepare(samples, width, height, device, dtype, references=(), inpaint=None, model=None):
        if references:
            raise ValueError("Z-Image does not accept reference images")
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        raw = _to_latent_4d(
            samples,
            16,
            latent_height,
            latent_width,
            device,
            dtype,
        )
        noise_mask = None
        if inpaint is not None:
            mask_values = _token_mask(
                inpaint.get("mask"),
                latent_height,
                latent_width,
                device,
            )
            if mask_values is not None:
                noise_mask = mask_values.transpose(1, 2).reshape(
                    mask_values.shape[0],
                    1,
                    latent_height,
                    latent_width,
                )
        return {
            "latents": raw,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "noise_mask": noise_mask,
        }

    def prompt_list(current, count):
        cond = current["cond"]
        mask = current["attention_mask"]
        if mask is None:
            return [cond[0]] * count
        valid = mask[0].bool()
        return [cond[0][valid]] * count

    def run(state, timestep, current, count):
        latents = state["latents"]
        latent_list = [latent.unsqueeze(1) for latent in latents]
        timestep_values = ((1000 - timestep) / 1000).expand(count).to(
            latents.dtype
        )
        outputs = transformer(
            latent_list,
            timestep_values,
            prompt_list(current, count),
            return_dict=False,
        )[0]
        return torch.stack([item.float() for item in outputs], dim=0)

    def predict(
        latents,
        state,
        timestep,
        positive,
        negative,
        cfg,
        next_timestep=None,
        guidance=None,
        guidance_explicit=False,
        step_fraction=None,
    ):
        current = _conditioning(positive, latents.device, latents.dtype)
        count = int(latents.shape[0])
        positive_noise = run(state, timestep, current, count)
        if float(cfg) <= 1.0 or not isinstance(negative, dict):
            return (-positive_noise).squeeze(2)
        current_negative = _conditioning(
            negative, latents.device, latents.dtype
        )
        negative_noise = run(state, timestep, current_negative, count)
        return (
            -(
                positive_noise
                + float(cfg) * (positive_noise - negative_noise)
            )
        ).squeeze(2)

    def finalize(latents, state):
        return latents

    return {
        "arch": "z_image",
        "prepare_latents": prepare,
        "predict": predict,
        "finalize_latents": finalize,
        "supports_references": False,
        "supports_guidance": False,
        "supports_inpaint": True,
        "edit_capable": False,
    }


def _make_krea_adapter(transformer, vae_scale_factor):
    def prepare(samples, width, height, device, dtype, references=(), inpaint=None, model=None):
        if references:
            raise ValueError("Krea 2 does not accept reference images")
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        raw = _to_latent_4d(
            samples,
            16,
            latent_height,
            latent_width,
            device,
            dtype,
        )
        packed = _patchify_latents(raw)
        noise_mask = None
        if inpaint is not None:
            noise_mask = _token_mask(
                inpaint.get("mask"),
                latent_height // 2,
                latent_width // 2,
                device,
                patch_size=1,
            )
        return {
            "latents": _pack_tokens(packed),
            "latent_height": latent_height,
            "latent_width": latent_width,
            "noise_mask": noise_mask,
        }

    def predict(
        latents,
        state,
        timestep,
        positive,
        negative,
        cfg,
        next_timestep=None,
        guidance=None,
        guidance_explicit=False,
        step_fraction=None,
    ):
        current = _conditioning(positive, latents.device, latents.dtype)
        if current["cond"].ndim != 4:
            raise ValueError("Krea 2 conditioning must contain stacked text layers")
        current = _expand_conditioning(current, latents.shape[0])
        grid_height = state["latent_height"] // 2
        grid_width = state["latent_width"] // 2
        text_length = int(current["cond"].shape[1])
        position_ids = torch.cat(
            [
                torch.zeros(text_length, 3, device=latents.device),
                _latent_grid_ids(
                    grid_height,
                    grid_width,
                    latents.device,
                ),
            ],
            dim=0,
        )
        timestep_values = (timestep / 1000).expand(latents.shape[0]).to(
            latents.dtype
        )

        def run(current_value):
            return transformer(
                hidden_states=latents,
                encoder_hidden_states=current_value["cond"],
                timestep=timestep_values,
                position_ids=position_ids,
                encoder_attention_mask=current_value["attention_mask"],
                return_dict=False,
            )[0]

        positive_noise = run(current)
        if float(cfg) <= 1.0 or not isinstance(negative, dict):
            return positive_noise
        current_negative = _conditioning(
            negative, latents.device, latents.dtype
        )
        current_negative = _expand_conditioning(
            current_negative, latents.shape[0]
        )
        negative_noise = run(current_negative)
        return positive_noise + float(cfg) * (
            positive_noise - negative_noise
        )

    def finalize(latents, state):
        packed = _unpack_tokens(
            latents,
            64,
            state["latent_height"] // 2,
            state["latent_width"] // 2,
        )
        return _unpatchify_latents(packed)

    return {
        "arch": "krea2",
        "prepare_latents": prepare,
        "predict": predict,
        "finalize_latents": finalize,
        "supports_references": False,
        "supports_guidance": False,
        "supports_inpaint": True,
        "edit_capable": False,
    }


def _raw_latent_tensor(samples, channels, device, dtype):
    """Normalize a reference latent to raw (unpacked) VAE channels."""
    samples = _squeeze_latent_frames(samples)
    if samples.shape[1] == channels:
        return samples.to(device=device, dtype=dtype)
    if samples.shape[1] == channels * 4:
        return _unpatchify_latents(samples).to(device=device, dtype=dtype)
    raise ValueError(
        "Reference latents must use the model's raw or packed latent width"
    )


def _ideogram4_layout(text_mask, grid_height, grid_width, device):
    """Build the packed text+image ids/layout consumed by Ideogram 4."""
    batch, max_text_tokens = text_mask.shape
    num_image_tokens = grid_height * grid_width
    total = max_text_tokens + num_image_tokens
    position_ids = torch.zeros(
        batch,
        total,
        3,
        dtype=torch.long,
        device=device,
    )
    segment_ids = torch.full(
        (batch, total),
        -1,
        dtype=torch.long,
        device=device,
    )
    indicator = torch.zeros(batch, total, dtype=torch.long, device=device)

    rows = torch.arange(grid_height, device=device)
    columns = torch.arange(grid_width, device=device)
    row_ids = rows[:, None].expand(grid_height, grid_width).reshape(-1)
    column_ids = columns[None, :].expand(grid_height, grid_width).reshape(-1)
    time_ids = torch.zeros_like(row_ids)
    image_positions = torch.stack(
        (time_ids, row_ids, column_ids),
        dim=1,
    ) + 65536

    lengths = text_mask.to(torch.long).sum(dim=1).tolist()
    for index, text_length in enumerate(lengths):
        offset = max_text_tokens - text_length
        if text_length:
            text_positions = torch.arange(text_length, device=device)
            position_ids[index, offset : offset + text_length] = torch.stack(
                (text_positions, text_positions, text_positions),
                dim=1,
            )
            indicator[index, offset : offset + text_length] = 3
        position_ids[index, offset + text_length :] = image_positions
        indicator[index, offset + text_length :] = 2
        segment_ids[index, offset : offset + text_length + num_image_tokens] = 1
    return position_ids, segment_ids, indicator


def _make_ideogram4_adapter(
    transformer,
    unconditional_transformer,
    vae_scale_factor,
):
    def prepare(samples, width, height, device, dtype, references=(), inpaint=None, model=None):
        if references:
            raise ValueError("Ideogram 4 does not accept reference images")
        grid_height = height // (vae_scale_factor * 2)
        grid_width = width // (vae_scale_factor * 2)
        source = samples
        if inpaint is not None:
            masked = inpaint.get("masked_latents")
            if isinstance(masked, torch.Tensor):
                source = masked
        packed = _to_patchified_4d(
            source,
            grid_height,
            grid_width,
            device,
            dtype,
        )
        return {
            "latents": _pack_tokens(packed),
            "grid_height": grid_height,
            "grid_width": grid_width,
        }

    def predict(
        latents,
        state,
        timestep,
        positive,
        negative,
        cfg,
        next_timestep=None,
        guidance=None,
        guidance_explicit=False,
        step_fraction=None,
    ):
        current = _conditioning(positive, latents.device, latents.dtype)
        if current["cond"].ndim != 3:
            raise ValueError("Ideogram 4 conditioning must contain text features")
        current = _expand_conditioning(current, latents.shape[0])
        text_mask = current["attention_mask"]
        if not isinstance(text_mask, torch.Tensor):
            text_mask = torch.ones(
                current["cond"].shape[:2],
                dtype=torch.bool,
                device=latents.device,
            )
        else:
            text_mask = text_mask.to(device=latents.device).bool()
        max_text_tokens = int(current["cond"].shape[1])
        if int(text_mask.shape[1]) != max_text_tokens:
            raise ValueError(
                "Ideogram 4 text attention mask does not match its features"
            )
        grid_height = int(state["grid_height"])
        grid_width = int(state["grid_width"])
        num_image_tokens = grid_height * grid_width
        position_ids, segment_ids, indicator = _ideogram4_layout(
            text_mask,
            grid_height,
            grid_width,
            latents.device,
        )
        model_timestep = (1.0 - timestep / 1000.0).expand(
            latents.shape[0]
        ).to(latents.dtype)

        def run(current_value, branch):
            cond = current_value["cond"]
            current_mask = current_value["attention_mask"]
            if not isinstance(current_mask, torch.Tensor):
                current_mask = torch.ones(
                    cond.shape[:2],
                    dtype=torch.bool,
                    device=latents.device,
                )
            else:
                current_mask = current_mask.to(device=latents.device).bool()
            current_positions, current_segments, current_indicators = (
                _ideogram4_layout(
                    current_mask,
                    grid_height,
                    grid_width,
                    latents.device,
                )
            )
            hidden = torch.cat(
                [
                    torch.zeros(
                        latents.shape[0],
                        current_positions.shape[1] - num_image_tokens,
                        latents.shape[2],
                        device=latents.device,
                        dtype=latents.dtype,
                    ),
                    latents,
                ],
                dim=1,
            )
            features = torch.cat(
                [
                    cond,
                    torch.zeros(
                        latents.shape[0],
                        num_image_tokens,
                        cond.shape[2],
                        device=latents.device,
                        dtype=latents.dtype,
                    ),
                ],
                dim=1,
            )
            output = branch(
                hidden_states=hidden,
                timestep=model_timestep,
                encoder_hidden_states=features,
                position_ids=current_positions,
                segment_ids=current_segments,
                indicator=current_indicators,
                return_dict=False,
            )[0]
            return output[:, current_positions.shape[1] - num_image_tokens :]

        positive_noise = run(current, transformer)
        if not isinstance(negative, dict) or float(cfg) == 1.0:
            # The Ideogram transformer predicts data-to-noise velocity; the
            # diffusers scheduler expects noise-to-data velocity.
            return -positive_noise
        current_negative = _conditioning(
            negative,
            latents.device,
            latents.dtype,
        )
        current_negative = _expand_conditioning(
            current_negative,
            latents.shape[0],
        )
        negative_features = torch.zeros(
            latents.shape[0],
            num_image_tokens,
            current_negative["cond"].shape[2],
            device=latents.device,
            dtype=latents.dtype,
        )
        branch = (
            unconditional_transformer
            if unconditional_transformer is not None
            else transformer
        )
        negative_noise = branch(
            hidden_states=latents,
            timestep=model_timestep,
            encoder_hidden_states=negative_features,
            position_ids=position_ids[:, -num_image_tokens:],
            segment_ids=segment_ids[:, -num_image_tokens:],
            indicator=indicator[:, -num_image_tokens:],
            return_dict=False,
        )[0]
        blended = negative_noise + float(cfg) * (
            positive_noise - negative_noise
        )
        return -blended

    def finalize(latents, state):
        return _unpack_tokens(
            latents,
            128,
            state["grid_height"],
            state["grid_width"],
        )

    return {
        "arch": "ideogram4",
        "prepare_latents": prepare,
        "predict": predict,
        "finalize_latents": finalize,
        "supports_references": False,
        "supports_guidance": False,
        "supports_inpaint": True,
        "edit_capable": False,
    }


def _make_fibo2_adapter(transformer, vae_scale_factor):
    def prepare(samples, width, height, device, dtype, references=(), inpaint=None, model=None):
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        source = samples
        if inpaint is not None:
            masked = inpaint.get("masked_latents")
            if isinstance(masked, torch.Tensor):
                source = masked
        latents = _to_latent_4d(
            source,
            32,
            latent_height,
            latent_width,
            device,
            dtype,
            patchified=True,
        )
        contexts = []
        for reference in references:
            if isinstance(reference, torch.Tensor):
                contexts.append(
                    _raw_latent_tensor(reference, 32, device, dtype)
                )
        if len(contexts) > 5:
            raise ValueError(
                "Fibo 2 accepts at most five reference images"
            )
        if inpaint is not None:
            masked = inpaint.get("masked_latents")
            if isinstance(masked, torch.Tensor):
                contexts.append(
                    _to_latent_4d(
                        masked,
                        32,
                        latent_height,
                        latent_width,
                        device,
                        dtype,
                        patchified=True,
                    )
                )
        return {
            "latents": latents,
            "context_latents": contexts or None,
            "latent_height": latent_height,
            "latent_width": latent_width,
        }

    def predict(
        latents,
        state,
        timestep,
        positive,
        negative,
        cfg,
        next_timestep=None,
        guidance=None,
        guidance_explicit=False,
        step_fraction=None,
    ):
        current = _conditioning(positive, latents.device, latents.dtype)
        if current["cond"].ndim != 4:
            raise ValueError("Fibo 2 conditioning must contain stacked text layers")
        current = _expand_conditioning(current, latents.shape[0])
        timestep_values = (timestep / 1000.0).expand(latents.shape[0]).to(
            latents.dtype
        )

        def run(current_value):
            attention_mask = current_value["attention_mask"]
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(latents.device).bool()
            return transformer(
                hidden_states=latents,
                timestep=timestep_values,
                encoder_hidden_states=current_value["cond"],
                encoder_attention_mask=attention_mask,
                context_latents=state["context_latents"],
                return_dict=False,
            )[0]

        positive_noise = run(current)
        if float(cfg) <= 1.0 or not isinstance(negative, dict):
            return positive_noise
        current_negative = _conditioning(
            negative,
            latents.device,
            latents.dtype,
        )
        current_negative = _expand_conditioning(
            current_negative,
            latents.shape[0],
        )
        negative_noise = run(current_negative)
        return _apply_cfg(positive_noise, negative_noise, cfg)

    def finalize(latents, state):
        return _patchify_latents(latents)

    return {
        "arch": "fibo2",
        "prepare_latents": prepare,
        "predict": predict,
        "finalize_latents": finalize,
        "supports_references": True,
        "supports_guidance": False,
        "supports_inpaint": True,
        "edit_capable": True,
    }


def _make_anima_adapter(transformer, vae_scale_factor):
    def prepare(samples, width, height, device, dtype, references=(), inpaint=None, model=None):
        if references:
            raise ValueError("Anima does not accept reference images")
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        source = samples
        if inpaint is not None:
            masked = inpaint.get("masked_latents")
            if isinstance(masked, torch.Tensor):
                source = masked
        raw = _to_latent_4d(
            source,
            16,
            latent_height,
            latent_width,
            device,
            dtype,
        )
        latents = raw.unsqueeze(2)
        return {
            "latents": latents,
            "padding_mask": torch.zeros(
                latents.shape[0],
                1,
                height,
                width,
                device=device,
                dtype=dtype,
            ),
            "latent_height": latent_height,
            "latent_width": latent_width,
        }

    def predict(
        latents,
        state,
        timestep,
        positive,
        negative,
        cfg,
        next_timestep=None,
        guidance=None,
        guidance_explicit=False,
        step_fraction=None,
    ):
        current = _conditioning(positive, latents.device, latents.dtype)
        if current["cond"].ndim != 3:
            raise ValueError("Anima conditioning must contain text embeddings")
        current = _expand_conditioning(current, latents.shape[0])
        timestep_values = (timestep / 1000.0).expand(latents.shape[0]).to(
            latents.dtype
        )

        def run(current_value):
            return transformer(
                hidden_states=latents,
                timestep=timestep_values,
                encoder_hidden_states=current_value["cond"],
                padding_mask=state["padding_mask"],
                return_dict=False,
            )[0]

        positive_noise = run(current)
        if float(cfg) <= 1.0 or not isinstance(negative, dict):
            return positive_noise
        current_negative = _conditioning(
            negative,
            latents.device,
            latents.dtype,
        )
        current_negative = _expand_conditioning(
            current_negative,
            latents.shape[0],
        )
        negative_noise = run(current_negative)
        return _apply_cfg(positive_noise, negative_noise, cfg)

    def finalize(latents, state):
        if latents.ndim == 5:
            return latents[:, :, 0]
        return latents

    return {
        "arch": "anima",
        "prepare_latents": prepare,
        "predict": predict,
        "finalize_latents": finalize,
        "supports_references": False,
        "supports_guidance": False,
        "supports_inpaint": True,
        "edit_capable": False,
    }


def _make_moss_adapter(transformer, vae_scale_factor):
    defaults = {
        "uncond_skip_layers": (22,),
        "slg_layers": (),
        "slg_scale": 0.0,
        "slg_start": 0.0,
        "slg_stop": 1.0,
    }

    def prepare(
        samples,
        width,
        height,
        device,
        dtype,
        references=(),
        inpaint=None,
        model=None,
    ):
        if references:
            raise ValueError("MOSS Image does not accept reference images")
        latent_height = height // (vae_scale_factor * 2)
        latent_width = width // (vae_scale_factor * 2)
        source = samples
        if inpaint is not None:
            masked = inpaint.get("masked_latents")
            if isinstance(masked, torch.Tensor):
                source = masked
        options = defaults
        if isinstance(model, dict):
            configured = model.get("moss_guidance")
            if isinstance(configured, dict):
                options = {**defaults, **configured}
        return {
            "latents": _to_patchified_4d(
                source,
                latent_height,
                latent_width,
                device,
                dtype,
            ),
            "latent_height": latent_height,
            "latent_width": latent_width,
            "moss_guidance": options,
        }

    def predict(
        latents,
        state,
        timestep,
        positive,
        negative,
        cfg,
        next_timestep=None,
        guidance=None,
        guidance_explicit=False,
        step_fraction=None,
    ):
        options = state.get("moss_guidance")
        if not isinstance(options, dict):
            options = defaults
        uncond_skip_layers = tuple(
            int(item) for item in options.get("uncond_skip_layers", ())
        )
        slg_layers = tuple(
            int(item) for item in options.get("slg_layers", ())
        )
        slg_scale = float(options.get("slg_scale", 0.0))
        slg_start = float(options.get("slg_start", 0.0))
        slg_stop = float(options.get("slg_stop", 1.0))
        if step_fraction is None:
            step_fraction = 1.0 - min(
                max(float(timestep) / 1000.0, 0.0),
                1.0,
            )
        skip_active = slg_start <= float(step_fraction) < slg_stop

        current = _conditioning(positive, latents.device, latents.dtype)
        if current["cond"].ndim != 3:
            raise ValueError("MOSS Image conditioning must contain text features")
        current = _expand_conditioning(current, latents.shape[0])
        timestep_values = timestep.expand(latents.shape[0]).to(torch.float32)

        def attention_mask(current_value):
            mask = current_value["attention_mask"]
            if isinstance(mask, torch.Tensor):
                return mask.to(latents.device).bool()
            return torch.ones(
                current_value["cond"].shape[:2],
                dtype=torch.bool,
                device=latents.device,
            )

        def run(current_value, *, skip_layers=None):
            kwargs = {}
            if skip_layers:
                kwargs["skip_layers"] = skip_layers
            return transformer(
                latents,
                timestep_values,
                current_value["cond"],
                attention_mask(current_value),
                return_dict=False,
                **kwargs,
            )[0].float()

        positive_noise = run(current)
        if float(cfg) == 1.0 or not isinstance(negative, dict):
            if slg_layers and slg_scale > 0 and skip_active:
                skipped = run(current, skip_layers=slg_layers)
                return skipped + slg_scale * (positive_noise - skipped)
            return positive_noise
        current_negative = _conditioning(
            negative,
            latents.device,
            latents.dtype,
        )
        current_negative = _expand_conditioning(
            current_negative,
            latents.shape[0],
        )
        negative_noise = run(
            current_negative,
            skip_layers=uncond_skip_layers if skip_active else (),
        )
        prediction = negative_noise + float(cfg) * (
            positive_noise - negative_noise
        )
        if slg_layers and slg_scale > 0 and skip_active:
            skipped = run(current, skip_layers=slg_layers)
            prediction = prediction + slg_scale * (
                positive_noise - skipped
            )
        return prediction

    def finalize(latents, state):
        return latents

    return {
        "arch": "moss_image",
        "prepare_latents": prepare,
        "predict": predict,
        "finalize_latents": finalize,
        "supports_references": False,
        "supports_guidance": False,
        "supports_inpaint": True,
        "edit_capable": False,
    }


def _build_sampler_adapter(
    arch,
    info,
    transformer,
    *,
    distilled,
    index_name,
    unconditional_transformer=None,
):
    try:
        latent_channels = int(info["latent_channels"])
        scale = int(info["vae_scale_factor"])
        edit_capable = any(
            token in index_name
            for token in ("kontext", "edit", "inpaint", "fill")
        )
        if arch == "flux1":
            return _make_flux1_adapter(
                transformer,
                latent_channels,
                scale,
                edit_capable,
            )
        if arch == "flux2":
            return _make_flux2_adapter(
                transformer,
                scale,
                distilled=distilled,
                klein="klein" in index_name,
                edit_capable=edit_capable or "flux2" in index_name,
            )
        if arch == "flux3":
            return _make_flux3_adapter(
                transformer,
                scale,
                edit_capable=True,
            )
        if arch == "ideogram4":
            return _make_ideogram4_adapter(
                transformer,
                unconditional_transformer,
                scale,
            )
        if arch == "fibo2":
            return _make_fibo2_adapter(transformer, scale)
        if arch == "anima":
            return _make_anima_adapter(transformer, scale)
        if arch == "moss_image":
            return _make_moss_adapter(transformer, scale)
        if arch == "qwen_image":
            return _make_qwen_adapter(
                transformer,
                scale,
                edit_capable=True,
            )
        if arch == "z_image":
            return _make_zimage_adapter(transformer, scale)
        if arch == "krea2":
            return _make_krea_adapter(transformer, scale)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _attention_backend():
    value = os.environ.get("COMPLEXUI_ATTENTION", "auto").strip().lower()
    allowed = {
        "auto",
        "sdpa",
        "sdpa-flash",
        "sdpa-mem-efficient",
        "sdpa-cudnn",
        "sdpa-math",
        "xformers",
        "flash-attn",
        "flash-attn4",
        "sageattention",
        "comfy_kitchen_sol",
        "comfy_kitchen_int8",
    }
    return value if value in allowed else "auto"


def _attention_selection(device):
    requested = _attention_backend()
    if requested in {
        "xformers",
        "flash-attn",
        "flash-attn4",
        "sageattention",
        "comfy_kitchen_sol",
        "comfy_kitchen_int8",
    }:
        raise ValueError(
            f"Attention backend '{requested}' requires the ComplexUI runtime; "
            "exported standalone scripts fall back to SDPA"
        )
    capability = None
    arch = "cpu"
    if device == "cuda":
        try:
            capability = torch.cuda.get_device_capability()
        except Exception:
            capability = None
        if capability and tuple(capability) >= (10, 0):
            arch = "blackwell"
        elif capability and tuple(capability) >= (9, 0):
            arch = "hopper"
        elif capability and tuple(capability) >= (8, 9):
            arch = "ada"
        elif capability and tuple(capability) >= (8, 0):
            arch = "ampere"

    def flags(flash=False, mem=False, cudnn=False, math=True):
        return {
            "flash": bool(flash),
            "mem_efficient": bool(mem),
            "cudnn": bool(cudnn),
            "math": bool(math),
        }

    if requested in {"auto", "sdpa"}:
        if device != "cuda":
            selection_flags = flags()
            reason = "cpu default dispatch"
        elif capability and tuple(capability) >= (9, 0):
            selection_flags = flags(flash=True, mem=True, cudnn=True)
            reason = f"{arch} prefers flash/cuDNN SDPA"
        elif capability and tuple(capability) >= (8, 0):
            selection_flags = flags(flash=True, mem=True)
            reason = f"{arch} prefers flash SDPA"
        else:
            selection_flags = flags(mem=True)
            reason = "pre-Ampere safe SDPA fallback"
        return {
            "requested": requested,
            "backend": "sdpa",
            "reason": reason,
            "approximates": False,
            "flags": selection_flags,
            "device": device,
            "capability": capability,
            "arch": arch,
        }
    if requested == "sdpa-math":
        return {
            "requested": requested,
            "backend": requested,
            "reason": "explicit reference math attention",
            "approximates": False,
            "flags": flags(),
            "device": device,
            "capability": capability,
            "arch": arch,
        }
    if device != "cuda":
        raise ValueError(
            f"Attention backend '{requested}' requires CUDA; choose auto, "
            "sdpa or sdpa-math on CPU"
        )
    if requested == "sdpa-flash":
        if not capability or tuple(capability) < (8, 0):
            raise ValueError(
                f"Attention backend 'sdpa-flash' requires sm80 or newer; "
                f"detected {capability}"
            )
        selection_flags = flags(flash=True)
        reason = "explicit flash SDPA"
    elif requested == "sdpa-mem-efficient":
        selection_flags = flags(mem=True)
        reason = "explicit memory-efficient SDPA"
    elif requested == "sdpa-cudnn":
        if not capability or tuple(capability) < (9, 0):
            raise ValueError(
                f"Attention backend 'sdpa-cudnn' requires sm90 or newer; "
                f"detected {capability}"
            )
        if not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            raise ValueError(
                "Attention backend 'sdpa-cudnn' is unavailable in the "
                "installed PyTorch build"
            )
        selection_flags = flags(cudnn=True)
        reason = "explicit cuDNN SDPA"
    else:
        raise ValueError(f"Unsupported attention backend: {requested}")
    return {
        "requested": requested,
        "backend": requested,
        "reason": reason,
        "approximates": False,
        "flags": selection_flags,
        "device": device,
        "capability": capability,
        "arch": arch,
    }


def _apply_attention_selection(selection):
    flags = selection["flags"]
    torch.backends.cuda.enable_flash_sdp(bool(flags["flash"]))
    torch.backends.cuda.enable_mem_efficient_sdp(
        bool(flags["mem_efficient"])
    )
    torch.backends.cuda.enable_math_sdp(bool(flags["math"]))
    enable_cudnn = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if callable(enable_cudnn):
        enable_cudnn(bool(flags["cudnn"]))
    message = (
        f"attention: backend={selection['backend']} "
        f"reason={selection['reason']} device={selection['device']} "
        f"flags={selection['flags']}"
    )
    print(message)
    LOGGER.info(message)


def _apply_attention_runtime(device):
    try:
        from app.attention_dispatch import apply_attention, benchmark_attention
    except Exception:
        selection = _attention_selection(device)
        _apply_attention_selection(selection)
        return selection
    requested = _attention_backend()
    benchmark = os.environ.get(
        "COMPLEXUI_ATTENTION_BENCHMARK", ""
    ).strip().lower() in {"1", "true", "on", "yes"}
    if benchmark and requested in {"auto", "sdpa"}:
        try:
            results = benchmark_attention(torch, device, "auto")
            selection = apply_attention(
                torch, str(results["selected"]), device
            )
            selection["benchmark"] = results
            return selection
        except Exception as exc:
            print(f"attention benchmark failed; using auto ({exc})")
    return apply_attention(torch, requested, device)


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


def _compile_mode():
    value = os.environ.get("COMPLEXUI_TORCH_COMPILE", "off").strip().lower()
    allowed = {
        "off",
        "auto",
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
    return value if value in allowed else "off"


def _recompile_limit():
    raw = os.environ.get("COMPLEXUI_TORCH_COMPILE_RECOMPILE_LIMIT", "8")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 8
    return max(1, min(value, 64))


def _compile_component(component, mode):
    if mode == "off":
        return {"mode": "off", "compiled": False, "error": None}
    resolved = "default" if mode == "auto" else mode
    limit = _recompile_limit()
    dynamo = getattr(torch, "_dynamo", None)
    config = getattr(dynamo, "config", None) if dynamo is not None else None
    if config is not None and hasattr(config, "cache_size_limit"):
        config.cache_size_limit = limit
    started = time.perf_counter()
    try:
        try:
            component.forward = torch.compile(
                component.forward,
                mode=resolved,
                recompile_limit=limit,
            )
        except TypeError:
            component.forward = torch.compile(component.forward, mode=resolved)
    except Exception as exc:
        if dynamo is not None and hasattr(dynamo, "reset"):
            dynamo.reset()
        if mode != "auto":
            raise ValueError(
                f"torch.compile failed for mode '{resolved}': {exc}"
            ) from exc
        message = f"compile: failed, continuing eager ({exc})"
        print(message)
        LOGGER.warning("compile: eager fallback after failure: %s", exc)
        return {
            "mode": resolved,
            "compiled": False,
            "compile_ms": round((time.perf_counter() - started) * 1000, 1),
            "recompile_limit": limit,
            "error": str(exc),
        }
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    message = f"compile: mode={resolved} compile_ms={elapsed}"
    print(message)
    LOGGER.info(message)
    return {
        "mode": resolved,
        "compiled": True,
        "compile_ms": elapsed,
        "recompile_limit": limit,
        "error": None,
    }


def _load_quantized_single_file_component(path: Path, arch: str, torch_dtype):
    if arch not in {"sd15", "sdxl"}:
        raise ValueError(
            "Single-file quantized loading currently supports SD1.5 and SDXL "
            "checkpoints. Use a diffusers component folder for other architectures."
        )
    try:
        from app.single_file_quantization import load_quantized_single_file
    except ImportError as exc:
        raise ValueError(
            "Pre-quantized single-file loading requires the comfy-kitchen runtime"
        ) from exc
    pipeline_class = (
        diffusers.StableDiffusionXLPipeline
        if arch == "sdxl"
        else diffusers.StableDiffusionPipeline
    )
    pipe, report = load_quantized_single_file(
        path,
        pipeline_class,
        torch_dtype=torch_dtype,
    )
    print(
        "quantized single-file checkpoint: "
        f"{sum(1 for entry in report if entry.native)} native, "
        f"{sum(1 for entry in report if not entry.native)} emulated module(s)"
    )
    return pipe.unet, report


def _component_quantization(path: Path):
    try:
        from app.quantization import detect_quantization
    except ImportError:
        return None
    return detect_quantization(path, "diffusion")


def _path_weight_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for file in sorted(path.rglob("*.safetensors"))[:256]:
        try:
            total += file.stat().st_size
        except OSError:
            continue
    return total


def _move_component(component, device, offload):
    if component is None or not hasattr(component, "to"):
        return
    target = "cpu" if offload else device
    try:
        component.to(target)
    except Exception as exc:
        raise ValueError(
            f"Could not move diffusion component to {target}: {exc}"
        ) from exc


if not unet_name:
    raise ValueError("Select a diffusion model in models/diffusion")
selected = Path(unet_name)
if not selected.exists():
    raise FileNotFoundError(f"Diffusion model does not exist: {selected}")
load_started = time.perf_counter()
device = "cuda" if torch.cuda.is_available() else "cpu"
quantization_info = None
try:
    from app.quantization import detect_quantization, validate_quantization
except ImportError:
    pass
else:
    capability = None
    if device == "cuda":
        try:
            capability = torch.cuda.get_device_capability()
        except Exception:
            capability = None
    quantization_info = detect_quantization(selected, "diffusion")
    if quantization_info is not None and quantization_info.format != "none":
        validate_quantization(
            quantization_info,
            model_type="diffusion",
            device=device,
            capability=capability,
        )
requested_weight_dtype = str(weight_dtype)
if weight_dtype == "default":
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
elif weight_dtype == "fp16":
    torch_dtype = torch.float16
elif weight_dtype == "bf16":
    torch_dtype = torch.bfloat16
elif weight_dtype == "fp32":
    torch_dtype = torch.float32
elif weight_dtype in {"fp8_e4m3fn", "fp8", "e4m3fn_last", "fp8_e5m2"}:
    raise ValueError(
        "FP8 diffusion weights need the quantized runtime path, which is not "
        "available in this build. Choose fp16, bf16 or fp32; pre-quantized "
        "format support is planned separately."
    )
else:
    raise ValueError(f"Unsupported weight_dtype: {weight_dtype}")
storage_dtype = (
    requested_weight_dtype
    if requested_weight_dtype in {"fp32", "fp16", "bf16"}
    else ("fp16" if torch_dtype is torch.float16 else "fp32")
)
memory_profile = _memory_profile(device)
offload = memory_profile == "low"

report = globals().get("report_progress")
if callable(report):
    report(0.05, "loading diffusion model")

explicit_arch = _canonical_type(type)
entries = {}
index_class = None
if selected.is_dir():
    index_class, entries = _component_entries(selected)
arch = _detect_arch(selected, explicit_arch, index_class, entries)
if explicit_arch is not None:
    try:
        detected_arch = _detect_arch(selected, None, index_class, entries)
    except ValueError:
        detected_arch = None
    if (
        detected_arch is not None
        and detected_arch != explicit_arch
        and not ({detected_arch, explicit_arch} <= {"sd15", "sdxl"})
    ):
        raise ValueError(
            f"Selected type '{type}' does not match the detected architecture "
            f"'{detected_arch}'"
        )
if arch not in ARCH_INFO:
    raise ValueError(f"Unsupported diffusion architecture: {arch}")
if arch == "moss_image" and not allow_custom_code:
    raise ValueError(
        "MOSS Image loading requires allow_custom_code=True for its "
        "repository-local pipeline/components."
    )
info = ARCH_INFO[arch]

if selected.is_file():
    if quantization_info is not None and quantization_info.format != "none":
        transformer, quantized_report = _load_quantized_single_file_component(
            selected, arch, torch_dtype
        )
        try:
            from app.single_file_quantization import quantization_mode
        except ImportError:
            mode = "emulated"
        else:
            mode = quantization_mode(quantized_report)
        quantization_info = quantization_info.model_copy(
            update={"mode": mode, "accumulation": "compute"}
        )
    else:
        transformer = _load_component_file(selected, arch, torch_dtype)
    scheduler = None
    scheduler_config = (
        dict(DEFAULT_SCHEDULER_CONFIG) if arch in {"sd15", "sdxl"} else {}
    )
    component_paths = {"component": str(selected.name)}
    text_conditioner = None
    unconditional_transformer = None
elif "unet" in entries:
    unet_entry = entries["unet"]
    transformer = _load_component_dir(
        selected / str(unet_entry.get("subfolder") or "unet"),
        torch_dtype,
        allow_custom_code,
    )
    scheduler, scheduler_config = _load_scheduler(
        selected, entries, arch, allow_custom_code
    )
    component_paths = {
        "unet": str(unet_entry.get("subfolder") or "unet"),
        "scheduler": str(
            entries.get("scheduler", {}).get("subfolder") or "scheduler"
        ),
    }
    text_conditioner = None
    unconditional_transformer = None
elif "transformer" in entries:
    transformer_entry = entries["transformer"]
    transformer = _load_component_dir(
        selected / str(transformer_entry.get("subfolder") or "transformer"),
        torch_dtype,
        allow_custom_code,
    )
    scheduler, scheduler_config = _load_scheduler(
        selected, entries, arch, allow_custom_code
    )
    component_paths = {
        "transformer": str(
            transformer_entry.get("subfolder") or "transformer"
        ),
        "scheduler": str(
            entries.get("scheduler", {}).get("subfolder") or "scheduler"
        ),
    }
    text_conditioner = _load_optional_component(
        selected, entries, "text_conditioner", torch_dtype, allow_custom_code
    )
    unconditional_transformer = _load_optional_component(
        selected,
        entries,
        "unconditional_transformer",
        torch_dtype,
        allow_custom_code,
    )
else:
    transformer = _load_component_dir(selected, torch_dtype, allow_custom_code)
    scheduler = None
    scheduler_config = (
        dict(DEFAULT_SCHEDULER_CONFIG) if arch in {"sd15", "sdxl"} else {}
    )
    component_paths = {"component": selected.name}
    text_conditioner = None
    unconditional_transformer = None

transformer.eval()
for optional in (text_conditioner, unconditional_transformer):
    if optional is not None:
        optional.eval()

attention_selection = _apply_attention_runtime(device)
_move_component(transformer, device, offload)
for optional in (text_conditioner, unconditional_transformer):
    _move_component(optional, device, offload)
channels_last = arch in {"sd15", "sdxl"}
if channels_last and hasattr(transformer, "to"):
    transformer.to(memory_format=torch.channels_last)
if device == "cuda" and not offload and torch.cuda.is_available():
    torch.cuda.empty_cache()
compile_mode = _compile_mode()
attention_backend = str(attention_selection.get("backend", "sdpa"))
optional_attention = attention_backend in {
    "xformers",
    "flash-attn",
    "flash-attn4",
    "sageattention",
    "comfy_kitchen_sol",
    "comfy_kitchen_int8",
}
if compile_mode != "off" and offload:
    if compile_mode == "auto":
        compile_info = {
            "mode": "auto",
            "compiled": False,
            "error": "skipped because the model is offloaded",
        }
        print("compile: skipped because the model is offloaded")
    else:
        raise ValueError(
            "torch.compile requires resident diffusion weights; use the "
            "resident memory profile or disable --torch-compile"
        )
elif compile_mode != "off" and optional_attention:
    if compile_mode == "auto":
        compile_info = {
            "mode": "auto",
            "compiled": False,
            "error": "skipped because an optional attention kernel is active",
        }
        print(
            "compile: skipped because optional attention kernels can graph-break "
            "or fall back to SDPA"
        )
    else:
        compile_info = _compile_component(transformer, compile_mode)
        if isinstance(compile_info, dict):
            compile_info["warning"] = (
                "optional attention kernels may graph-break or fall back to SDPA; "
                "use the sdpa attention backend for stable compilation"
            )
else:
    compile_info = _compile_component(transformer, compile_mode)
load_time_ms = round((time.perf_counter() - load_started) * 1000, 1)

class_name = transformer.__class__.__name__
index_name = str(index_class or "").lower()
pipeline_config = (
    _read_json(selected / "model_index.json")
    or _read_json(selected / "modular_model_index.json")
    or {}
) if selected.is_dir() else {}
is_distilled = bool(pipeline_config.get("is_distilled", False)) or (
    "distilled" in index_name
)
recommended = {
    "steps": int(info["steps"]),
    "cfg": float(info["cfg"]),
    "guidance": float(info.get("guidance_default", info["cfg"])),
    "sampler": str(info["sampler"]),
    "scheduler": str(info["scheduler"]),
    "width": int(info["width"]),
    "height": int(info["height"]),
}
capabilities = {
    "text_to_image": bool(info["text_to_image"]),
    "img2img": bool(info["img2img"]),
    "inpaint": bool(info["inpaint"]),
    "edit": bool(info["edit"]),
    "max_references": int(info["max_references"]),
    "negative_prompt": str(info["negative_prompt"]),
    "guidance": str(info["guidance"]),
    "guidance_default": float(info.get("guidance_default", info["cfg"])),
    "dual_branch_guidance": arch == "ideogram4",
    "skip_layer_guidance": arch == "moss_image",
}
sampler_adapter = _build_sampler_adapter(
    arch,
    info,
    transformer,
    distilled=is_distilled,
    index_name=f"{index_name}:{selected.name.lower()}",
    unconditional_transformer=unconditional_transformer,
)
guidance_available = bool(
    isinstance(sampler_adapter, dict)
    and sampler_adapter.get("supports_guidance", False)
)
capabilities["model_guidance"] = guidance_available
stat = selected.stat()
if selected.is_dir():
    component_dir = selected
    for candidate in ("transformer", "unet"):
        candidate_path = selected / candidate
        if candidate_path.is_dir():
            component_dir = candidate_path
            break
    quantization_info = _component_quantization(component_dir)
size_bytes = _path_weight_size(selected)
memory = None
try:
    from app.memory_estimate import estimate_model_memory
except ImportError:
    pass
else:
    memory = estimate_model_memory(
        size_bytes=size_bytes,
        quantization=quantization_info,
        is_directory=selected.is_dir(),
    )
model = {
    "kind": "model",
    "backend": "diffusers",
    "arch": arch,
    "arch_label": str(info["label"]),
    "name": selected.stem if selected.is_file() else selected.name,
    "path": str(selected.resolve()),
    "format": selected.suffix.lower().lstrip(".") if selected.is_file() else "diffusers",
    "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
    "size_bytes": size_bytes,
    "source": "local",
    "model_type": "diffusion",
    "device": device,
    "dtype": torch_dtype,
    "storage_dtype": storage_dtype,
    "compute_dtype": str(torch_dtype),
    "accumulation": "default",
    "quantization": (
        quantization_info.model_dump()
        if quantization_info is not None
        else None
    ),
    "memory": memory.model_dump() if memory is not None else None,
    "memory_profile": memory_profile,
    "offload": bool(offload),
    "offload_mode": "model" if offload else "none",
    "resident_device": "cpu" if offload else device,
    "channels_last": bool(channels_last),
    "attention": attention_selection,
    "compiled": bool(compile_info.get("compiled", False)),
    "compile_mode": str(compile_info.get("mode", "off")),
    "compile": compile_info,
    "load_time_ms": load_time_ms,
    "unet": transformer if arch in {"sd15", "sdxl"} else None,
    "transformer": transformer,
    "transformer_class": class_name,
    "scheduler": scheduler,
    "scheduler_config": dict(scheduler_config or {}),
    "text_conditioner": text_conditioner,
    "unconditional_transformer": unconditional_transformer,
    "component_paths": component_paths,
    "latent_space": str(info["latent_space"]),
    "latent_channels": int(info["latent_channels"]),
    "latent_packing": int(info["latent_packing"]),
    "vae_scale_factor": int(info["vae_scale_factor"]),
    "preset": arch,
    "is_distilled": is_distilled,
    "guidance_embeds": bool(
        getattr(getattr(transformer, "config", None), "guidance_embeds", False)
    ),
    "guidance_available": guidance_available,
    "capabilities": capabilities,
    "recommended": recommended,
    "sampler_adapter": sampler_adapter,
    "sampler_supported": bool(
        sampler_adapter is not None or arch in {"sd15", "sdxl"}
    ),
}
if callable(report):
    report(1.0, f"{info['label']} loaded")

return model: MODEL
