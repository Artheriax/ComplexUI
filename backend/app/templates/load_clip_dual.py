# /// script
# dependencies = [
#   "torch>=2.2",
#   "diffusers>=0.42.0",
#   "transformers>=5.0",
#   "sentencepiece>=0.2",
#   "protobuf>=5.0",
#   "safetensors>=0.4",
# ]
# ///
import json
import os
from pathlib import Path
from typing import Literal

import torch
import transformers

clip_name1: ModelSelector = ModelSelector(
    type="clip",
    select="any",
    required=True,
    include=["diffusion"],
)
clip_name2: ModelSelector = ModelSelector(
    type="clip",
    select="any",
    required=True,
    include=["diffusion"],
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
device: Literal["default", "cpu"] = "default"
max_sequence_length: int = 512
allow_custom_code: bool = False

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
    "z_image": "z_image",
    "zimage": "z_image",
    "krea2": "krea2",
    "ideogram4": "ideogram4",
    "fibo2": "fibo2",
    "anima": "anima",
    "moss_image": "moss_image",
}

PIPELINE_ARCH = {
    "StableDiffusionXLPipeline": "sdxl",
    "StableDiffusionXLImg2ImgPipeline": "sdxl",
    "StableDiffusionXLInpaintPipeline": "sdxl",
    "StableDiffusionXLModularPipeline": "sdxl",
    "FluxPipeline": "flux1",
    "FluxImg2ImgPipeline": "flux1",
    "FluxInpaintPipeline": "flux1",
    "FluxKontextPipeline": "flux1",
    "FluxKontextInpaintPipeline": "flux1",
    "FluxModularPipeline": "flux1",
    "FluxKontextModularPipeline": "flux1",
}

CLASS_ROLE = {
    "CLIPTextModel": "clip",
    "CLIPTextModelWithProjection": "clip_projection",
    "T5EncoderModel": "t5",
}

DUAL_ARCHS = {"sdxl", "flux1"}


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


def _canonical_type(value: str):
    if value == "auto":
        return None
    canonical = ARCH_ALIASES.get(value.strip().lower())
    if canonical is None:
        raise ValueError(f"Unsupported architecture type: {value}")
    return canonical


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


def _load_quantized_text_model(path: Path, class_name: str, torch_dtype):
    try:
        from app.quantized_loading import load_quantized_component
    except ImportError as exc:
        raise ValueError(
            "Pre-quantized text encoders require the comfy-kitchen runtime; "
            "install it in the backend environment"
        ) from exc
    config_class = getattr(transformers, "AutoConfig", None)
    if config_class is None:
        raise ValueError("transformers AutoConfig is unavailable")
    config = config_class.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=allow_custom_code,
    )
    model_class = getattr(transformers, class_name, None) if class_name else None
    if model_class is None or not hasattr(model_class, "from_config"):
        model_class = getattr(transformers, "AutoModel", None)
    if model_class is None or not hasattr(model_class, "from_config"):
        raise ValueError("This transformers build cannot build a text encoder from config")
    model = model_class.from_config(config)
    loaded, report = load_quantized_component(
        model,
        path,
        compute_dtype=torch_dtype,
        device="cpu",
    )
    print(
        "quantized text encoder: "
        f"{sum(1 for entry in report if entry.native)} native, "
        f"{sum(1 for entry in report if not entry.native)} emulated layer(s)"
    )
    return loaded


def _load_text_model(path: Path, class_name: str, torch_dtype):
    try:
        from app.quantization import detect_quantization, validate_quantization
    except ImportError:
        quantized = None
    else:
        quantized = detect_quantization(path, "clip")
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
                model_type="clip",
                device=device_name,
                capability=capability,
            )
            return _load_quantized_text_model(path, class_name, torch_dtype)
    model_class = getattr(transformers, class_name, None) if class_name else None
    errors = []
    if model_class is not None:
        try:
            return model_class.from_pretrained(
                str(path),
                torch_dtype=torch_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                trust_remote_code=allow_custom_code,
            )
        except Exception as exc:
            errors.append(str(exc))
    auto_model = getattr(transformers, "AutoModel", None)
    if auto_model is not None:
        try:
            return auto_model.from_pretrained(
                str(path),
                torch_dtype=torch_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                trust_remote_code=allow_custom_code,
            )
        except Exception as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "unknown loader error"
    raise ValueError(f"Could not load text encoder '{path.name}': {detail}")


def _load_tokenizer(path: Path):
    errors = []
    tokenizer_class = getattr(transformers, "AutoTokenizer", None)
    if tokenizer_class is not None:
        try:
            return tokenizer_class.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=allow_custom_code,
            )
        except Exception as exc:
            errors.append(str(exc))
    processor_class = getattr(transformers, "AutoProcessor", None)
    if processor_class is not None:
        try:
            return processor_class.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=allow_custom_code,
            )
        except Exception as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "unknown tokenizer error"
    raise ValueError(f"Could not load tokenizer '{path.name}': {detail}")


def _tokenizer_for(selected: Path, encoder_name: str):
    if "t5" in encoder_name or "encoder_2" in encoder_name:
        candidates = [
            selected / "tokenizer_2",
            selected.parent / "tokenizer_2",
            selected / "tokenizer",
            selected.parent / "tokenizer",
        ]
    else:
        candidates = [
            selected / "tokenizer",
            selected.parent / "tokenizer",
            selected / "tokenizer_2",
            selected.parent / "tokenizer_2",
        ]
    for candidate in candidates:
        if candidate.is_dir() and candidate != selected:
            return candidate
    return None


def _load_bundle(value: str):
    selected = Path(value)
    if not selected.is_dir():
        raise ValueError(f"CLIP component folder does not exist: {selected}")
    index_class, entries = _entries(selected)
    encoders = []
    if entries:
        arch = PIPELINE_ARCH.get(str(index_class or ""))
        names = [
            name
            for name in ("text_encoder", "text_encoder_2")
            if name in entries and entries[name].get("class_name") in CLASS_ROLE
        ]
        if not names:
            raise ValueError(
                f"'{selected.name}' does not contain a supported CLIP/T5 text encoder"
            )
        for name in names:
            entry = entries[name]
            path = selected / str(entry.get("subfolder") or name)
            class_name = str(entry.get("class_name") or "")
            tokenizer_name = "tokenizer_2" if name == "text_encoder_2" else "tokenizer"
            tokenizer_entry = entries.get(tokenizer_name)
            tokenizer_path = (
                selected / str(tokenizer_entry.get("subfolder") or tokenizer_name)
                if tokenizer_entry is not None
                else _tokenizer_for(path, name)
            )
            if tokenizer_path is None or not tokenizer_path.is_dir():
                raise ValueError(f"Text encoder '{name}' is missing its tokenizer")
            encoders.append(
                {
                    "name": name,
                    "class_name": class_name,
                    "role": CLASS_ROLE.get(class_name, "text"),
                    "model": _load_text_model(path, class_name, torch_dtype),
                    "tokenizer": _load_tokenizer(tokenizer_path),
                }
            )
        return {"arch": arch, "encoders": encoders}

    class_name = str(_class_name(selected) or "")
    if class_name not in CLASS_ROLE:
        raise ValueError(
            f"'{selected.name}' is not a supported CLIP/T5 text-encoder folder"
        )
    tokenizer_path = _tokenizer_for(selected, selected.name)
    if tokenizer_path is None:
        raise ValueError(f"Text encoder '{selected.name}' is missing its tokenizer")
    return {
        "arch": None,
        "encoders": [
            {
                "name": selected.name,
                "class_name": class_name,
                "role": CLASS_ROLE[class_name],
                "model": _load_text_model(selected, class_name, torch_dtype),
                "tokenizer": _load_tokenizer(tokenizer_path),
            }
        ],
    }


if not clip_name1 or not clip_name2:
    raise ValueError("Load CLIP (Dual) requires two local text-encoder selections")
if Path(clip_name1).resolve() == Path(clip_name2).resolve():
    raise ValueError("Load CLIP (Dual) requires two different model selections")
device_name = "cpu" if device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
target_device = torch.device(device_name)
torch_dtype = torch.float16 if device_name == "cuda" else torch.float32
try:
    from app.quantization import detect_quantization, validate_quantization
except ImportError:
    clip_quantization = None
else:
    capability = None
    if device_name == "cuda":
        try:
            capability = torch.cuda.get_device_capability()
        except Exception:
            capability = None
    clip_quantization = None
    for clip_selection in (clip_name1, clip_name2):
        quantization_info = detect_quantization(Path(clip_selection), "clip")
        if quantization_info is not None and quantization_info.format != "none":
            validate_quantization(
                quantization_info,
                model_type="clip",
                device=device_name,
                capability=capability,
            )
            if clip_quantization is None:
                clip_quantization = quantization_info
try:
    from app.attention_dispatch import apply_attention
except ImportError:
    pass
else:
    apply_attention(
        torch,
        os.environ.get("COMPLEXUI_ATTENTION", "auto"),
        device_name,
    )
if max_sequence_length < 8 or max_sequence_length > 8192:
    raise ValueError("max_sequence_length must be between 8 and 8192")

report = globals().get("report_progress")
if callable(report):
    report(0.05, "loading dual text encoders")

first = _load_bundle(clip_name1)
second = _load_bundle(clip_name2)
if len(first["encoders"]) != 1 or len(second["encoders"]) != 1:
    raise ValueError(
        "Load CLIP (Dual) expects one text encoder per selection. "
        "A full repo that already contains both encoders should use Load CLIP."
    )
combined = first["encoders"] + second["encoders"]
explicit_arch = _canonical_type(type)
roles = {entry["role"] for entry in combined}
if explicit_arch is not None:
    if explicit_arch not in DUAL_ARCHS:
        raise ValueError(
            f"Architecture '{explicit_arch}' has a single text encoder and cannot "
            "use Load CLIP (Dual)"
        )
    arch = explicit_arch
elif roles == {"clip", "t5"}:
    arch = "flux1"
elif "t5" in roles:
    raise ValueError("FLUX.1 dual CLIP requires one CLIP encoder and one T5 encoder")
elif roles <= {"clip", "clip_projection"} and len(combined) == 2:
    arch = "sdxl"
else:
    raise ValueError(
        "Could not infer a supported dual-CLIP architecture. Use SDXL "
        "(CLIP-L + CLIP-G) or FLUX.1 (CLIP + T5), or select type explicitly."
    )

if arch == "sdxl":
    projection = [entry for entry in combined if entry["role"] == "clip_projection"]
    plain = [entry for entry in combined if entry["role"] == "clip"]
    if len(projection) != 1 or len(plain) != 1:
        raise ValueError(
            "SDXL dual CLIP requires one CLIPTextModel and one "
            "CLIPTextModelWithProjection (CLIP-L + CLIP-G)"
        )
    first_encoder = plain[0]
    second_encoder = projection[0]
elif arch == "flux1":
    clip_encoder = [entry for entry in combined if entry["role"] in {"clip", "clip_projection"}]
    t5_encoder = [entry for entry in combined if entry["role"] == "t5"]
    if len(clip_encoder) != 1 or len(t5_encoder) != 1:
        raise ValueError("FLUX.1 dual CLIP requires one CLIP encoder and one T5 encoder")
    first_encoder = clip_encoder[0]
    second_encoder = t5_encoder[0]
else:
    raise ValueError(f"Unsupported dual CLIP architecture: {arch}")

for entry in (first_encoder, second_encoder):
    entry["model"] = entry["model"].to(target_device)
    entry["model"].eval()


def _inputs(text: str, tokenizer, limit: int):
    model_limit = getattr(tokenizer, "model_max_length", limit)
    if isinstance(model_limit, int) and 0 < model_limit < 10000:
        limit = min(limit, model_limit)
    kwargs = {
        "padding": "max_length",
        "max_length": limit,
        "truncation": True,
        "return_tensors": "pt",
    }
    if hasattr(tokenizer, "tokenizer"):
        try:
            batch = tokenizer(text=[text], **kwargs)
        except TypeError:
            batch = tokenizer([text], **kwargs)
    else:
        batch = tokenizer([text], **kwargs)
    return batch.input_ids, getattr(batch, "attention_mask", None)


def _forward(model, input_ids, attention_mask):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
    }
    try:
        return model(**kwargs, use_cache=False)
    except TypeError:
        return model(**kwargs)


def _hidden(output):
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
    return hidden


def encode_text(text: str):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    with torch.inference_mode():
        first_ids, first_mask = _inputs(text, first_encoder["tokenizer"], 77)
        first_ids = first_ids.to(target_device)
        first_mask = first_mask.to(target_device) if first_mask is not None else None
        first_output = _forward(first_encoder["model"], first_ids, first_mask)
        if arch == "sdxl":
            second_ids, second_mask = _inputs(text, second_encoder["tokenizer"], 77)
            second_ids = second_ids.to(target_device)
            second_mask = second_mask.to(target_device) if second_mask is not None else None
            second_output = _forward(second_encoder["model"], second_ids, second_mask)
            cond = torch.cat([_hidden(first_output), _hidden(second_output)], dim=-1)
            pooled = getattr(second_output, "text_embeds", None)
            if pooled is None:
                pooled = getattr(second_output, "pooler_output", None)
            if pooled is None:
                raise ValueError("CLIP-G did not return pooled text embeddings")
            return {
                "kind": "conditioning",
                "arch": "sdxl",
                "cond": cond.to(torch_dtype).detach().cpu(),
                "pooled": pooled.to(torch_dtype).detach().cpu(),
                "text": text,
            }
        pooled = getattr(first_output, "pooler_output", None)
        if pooled is None:
            pooled = _hidden(first_output)[:, 0]
        second_ids, second_mask = _inputs(
            text, second_encoder["tokenizer"], max_sequence_length
        )
        second_ids = second_ids.to(target_device)
        second_mask = second_mask.to(target_device) if second_mask is not None else None
        second_output = _forward(second_encoder["model"], second_ids, second_mask)
        hidden = _hidden(second_output)
        return {
            "kind": "conditioning",
            "arch": "flux1",
            "cond": hidden.to(torch_dtype).detach().cpu(),
            "pooled": pooled.to(torch_dtype).detach().cpu(),
            "text": text,
            "attention_mask": (
                second_mask.to(torch.float32).detach().cpu()
                if second_mask is not None
                else None
            ),
            "encoders": {
                "clip": {
                    "hidden": _hidden(first_output).to(torch_dtype).detach().cpu(),
                    "pooled": pooled.to(torch_dtype).detach().cpu(),
                },
                "t5": {
                    "hidden": hidden.to(torch_dtype).detach().cpu(),
                    "attention_mask": (
                        second_mask.to(torch.float32).detach().cpu()
                        if second_mask is not None
                        else None
                    ),
                },
            },
        }


stat = Path(clip_name1).stat()
clip = {
    "kind": "clip",
    "backend": "diffusers",
    "arch": arch,
    "name": f"{Path(clip_name1).name} + {Path(clip_name2).name}",
    "path": str(Path(clip_name1).resolve()),
    "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
    "source": "local",
    "model_type": "clip",
    "device": device_name,
    "dtype": torch_dtype,
    "quantization": (
        clip_quantization.model_dump()
        if clip_quantization is not None
        else None
    ),
    "text_encoder": first_encoder["model"],
    "tokenizer": first_encoder["tokenizer"],
    "text_encoder_2": second_encoder["model"],
    "tokenizer_2": second_encoder["tokenizer"],
    "encoders": [
        {
            "role": first_encoder["role"],
            "class_name": first_encoder["class_name"],
            "model": first_encoder["model"],
            "tokenizer": first_encoder["tokenizer"],
        },
        {
            "role": second_encoder["role"],
            "class_name": second_encoder["class_name"],
            "model": second_encoder["model"],
            "tokenizer": second_encoder["tokenizer"],
        },
    ],
    "max_sequence_length": max_sequence_length,
    "encode": encode_text,
}
if callable(report):
    report(1.0, f"{arch} dual text encoders ready")

return clip: CLIP
