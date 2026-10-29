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
import math
import os
from pathlib import Path
from typing import Literal

import diffusers
import torch
import transformers

clip_name: ModelSelector = ModelSelector(
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

CLASS_ROLE = {
    "CLIPTextModel": "clip",
    "CLIPTextModelWithProjection": "clip_projection",
    "T5EncoderModel": "t5",
    "Mistral3ForConditionalGeneration": "mistral",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3VLForConditionalGeneration": "qwen3_vl",
    "Qwen3VLModel": "qwen3_vl_text",
    "Qwen3Model": "qwen3_text",
    "Qwen3_5ForConditionalGeneration": "qwen3_5",
    "Qwen3_5Model": "qwen3_5",
    "SmolLM3Model": "smollm3",
    "SmolLM3ForCausalLM": "smollm3",
}

MAX_LENGTHS = {
    "sd15": 77,
    "sdxl": 77,
    "flux1": 512,
    "flux2": 512,
    "flux3": 512,
    "qwen_image": 1024,
    "z_image": 512,
    "krea2": 512,
    "ideogram4": 512,
    "fibo2": 4096,
    "anima": 512,
    "moss_image": 4096,
}

COMPONENT_NAMES = {
    "sd15": (("text_encoder",), ("tokenizer",)),
    "sdxl": (("text_encoder", "text_encoder_2"), ("tokenizer", "tokenizer_2")),
    "flux1": (("text_encoder", "text_encoder_2"), ("tokenizer", "tokenizer_2")),
    "flux2": (("text_encoder",), ("processor", "tokenizer")),
    "flux3": (("text_encoder",), ("processor", "tokenizer")),
    "qwen_image": (("text_encoder",), ("processor", "tokenizer")),
    "z_image": (("text_encoder",), ("tokenizer",)),
    "krea2": (("text_encoder",), ("tokenizer",)),
    "ideogram4": (("text_encoder",), ("tokenizer",)),
    "fibo2": (("text_encoder",), ("processor", "tokenizer")),
    "anima": (("text_encoder",), ("tokenizer",)),
    "moss_image": (("text_encoder",), ("processor", "tokenizer")),
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


def _component_path(root: Path, entries, names):
    for name in names:
        entry = entries.get(name)
        if entry is not None:
            path = root / str(entry.get("subfolder") or name)
            if path.exists():
                return path, str(entry.get("class_name") or _class_name(path) or "")
        path = root / name
        if path.exists():
            return path, str(_class_name(path) or "")
    return None, None


def _sibling_component(selected: Path, names):
    for name in names:
        candidate = selected.parent / name
        if candidate.is_dir() and candidate != selected:
            return candidate, str(_class_name(candidate) or "")
    return None, None


def _infer_arch(selected: Path, explicit, index_class, entries):
    if explicit is not None:
        return explicit
    if isinstance(index_class, str) and index_class in PIPELINE_ARCH:
        return PIPELINE_ARCH[index_class]
    classes = {str(entry.get("class_name") or "") for entry in entries.values()}
    if not classes:
        classes = {str(_class_name(selected) or "")}
    sibling_classes = set()
    for sibling_name in ("text_encoder", "text_encoder_2", "t5", "t5_encoder"):
        sibling = selected.parent / sibling_name
        if sibling.is_dir() and sibling != selected:
            class_name = str(_class_name(sibling) or "")
            if class_name:
                sibling_classes.add(class_name)
    if "Mistral3ForConditionalGeneration" in classes or "Qwen3ForCausalLM" in classes:
        return "flux2"
    if "T5EncoderModel" in classes or "T5EncoderModel" in sibling_classes:
        return "flux1"
    if "CLIPTextModelWithProjection" in classes or "text_encoder_2" in entries:
        return "sdxl"
    if sibling_classes & {"CLIPTextModel", "CLIPTextModelWithProjection"}:
        return "sdxl"
    if "CLIPTextModel" in classes:
        return "sd15"
    if "Qwen3VLForConditionalGeneration" in classes:
        name = selected.name.lower()
        if "fibo" in name:
            return "fibo2"
        if "flux3" in name or "flux_3" in name:
            return "flux3"
        return "qwen_image"
    if "Qwen3Model" in classes:
        return "anima" if "anima" in selected.name.lower() else "z_image"
    if "Qwen3_5ForConditionalGeneration" in classes or "Qwen3_5Model" in classes:
        if "moss" in selected.name.lower():
            return "moss_image"
    if "Qwen3VLModel" in classes:
        return "ideogram4" if "ideogram" in selected.name.lower() else "krea2"
    name = selected.name.lower()
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
        ("sdxl", "sdxl"),
        ("sd15", "sd15"),
    ):
        if token in name:
            return arch
    raise ValueError(
        "Could not detect the text-encoder architecture. Select type explicitly "
        "or use a diffusers repository/component folder with config.json."
    )


def _load_quantized_text_model(
    path: Path,
    class_name: str,
    torch_dtype,
    trust_remote_code: bool,
):
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
        trust_remote_code=trust_remote_code,
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


def _load_text_model(path: Path, class_name: str, torch_dtype, trust_remote_code: bool):
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
            return _load_quantized_text_model(
                path, class_name, torch_dtype, trust_remote_code
            )
    model_class = getattr(transformers, class_name, None) if class_name else None
    errors = []
    if model_class is not None:
        try:
            return model_class.from_pretrained(
                str(path),
                torch_dtype=torch_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
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
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "unknown loader error"
    raise ValueError(
        f"Could not load text encoder '{path.name}'. Expected a local "
        f"transformers/diffusers component folder. Details: {detail}"
    )


def _load_tokenizer(path: Path, prefer_processor: bool, trust_remote_code: bool):
    errors = []
    if prefer_processor:
        processor_class = getattr(transformers, "AutoProcessor", None)
        if processor_class is not None:
            try:
                return processor_class.from_pretrained(
                    str(path),
                    local_files_only=True,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as exc:
                errors.append(str(exc))
    tokenizer_class = getattr(transformers, "AutoTokenizer", None)
    if tokenizer_class is not None:
        try:
            return tokenizer_class.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            errors.append(str(exc))
    if not prefer_processor:
        processor_class = getattr(transformers, "AutoProcessor", None)
        if processor_class is not None:
            try:
                return processor_class.from_pretrained(
                    str(path),
                    local_files_only=True,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as exc:
                errors.append(str(exc))
    detail = errors[-1] if errors else "unknown tokenizer error"
    raise ValueError(
        f"Could not load tokenizer/processor '{path.name}'. Details: {detail}"
    )


def _move_batch(batch, target_device):
    if hasattr(batch, "to"):
        return batch.to(target_device)
    return batch


def _tokenizer_call(tokenizer, text, max_length: int):
    kwargs = {
        "padding": "max_length",
        "max_length": int(max_length),
        "truncation": True,
        "return_tensors": "pt",
    }
    if hasattr(tokenizer, "tokenizer"):
        try:
            return tokenizer(text=text, **kwargs)
        except TypeError:
            return tokenizer(text, **kwargs)
    return tokenizer(text, **kwargs)


def _inputs_from_text(text: str, tokenizer, max_length: int, chat: bool):
    token_limit = int(max_length)
    if chat and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": text}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch = _tokenizer_call(tokenizer, rendered, token_limit)
        except (TypeError, ValueError):
            batch = _tokenizer_call(tokenizer, [text], token_limit)
    else:
        batch = _tokenizer_call(tokenizer, [text], token_limit)
    return batch.input_ids, getattr(batch, "attention_mask", None)


def _hidden(output):
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        if isinstance(output, (tuple, list)):
            hidden = output[0]
        else:
            hidden = output
    return hidden


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


def _stack_layers(output, layers):
    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states:
        raise ValueError("Text encoder did not return hidden states")
    selected = torch.stack([hidden_states[index] for index in layers], dim=1)
    batch, count, seq, width = selected.shape
    return selected.permute(0, 2, 1, 3).reshape(batch, seq, count * width)


def _to_cpu(value, dtype):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype).detach().cpu()
    return value


def _plain_tokenizer(tokenizer):
    return getattr(tokenizer, "tokenizer", tokenizer)


def _batch_get(batch, key):
    if isinstance(batch, dict):
        return batch[key]
    return getattr(batch, key)


def _module_device(module):
    device = getattr(module, "device", None)
    if device is not None:
        return device
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    return torch.device("cpu")


def _chat_text(tokenizer, prompt: str, enable_thinking: bool | None = None) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def _forward_with_mask(model, input_ids, attention_mask, **kwargs):
    call_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        **kwargs,
    }
    try:
        return model(**call_kwargs, use_cache=False)
    except TypeError:
        return model(**call_kwargs)


def _flux3_text_embeds(prompt: str, model, tokenizer, limit: int):
    text = _chat_text(tokenizer, prompt)
    base_tokenizer = _plain_tokenizer(tokenizer)
    real_length = int(
        _batch_get(
            base_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=limit,
            ),
            "input_ids",
        ).shape[1]
    )
    padded_length = min(
        max(80, math.ceil(max(real_length, 1) / 80) * 80),
        limit,
    )
    encoder_inputs = base_tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=padded_length,
    )
    input_ids = _batch_get(encoder_inputs, "input_ids").to(_module_device(model))
    attention_mask = _batch_get(encoder_inputs, "attention_mask").to(
        _module_device(model)
    )
    position_ids = torch.arange(
        attention_mask.shape[1],
        device=_module_device(model),
    ).unsqueeze(0).masked_fill(attention_mask == 0, 1)
    output = _forward_with_mask(
        model,
        input_ids,
        attention_mask,
        position_ids=position_ids,
    )
    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states:
        raise ValueError("FLUX.3 text encoder did not return hidden states")
    layers = (4, 8, 12, 16, 20, 24, 28, 32)
    selected = torch.stack(
        [hidden_states[index] for index in layers],
        dim=1,
    )
    batch, count, sequence, width = selected.shape
    return (
        selected.permute(0, 2, 1, 3).reshape(
            batch,
            sequence,
            count * width,
        ),
        attention_mask,
    )


def _krea_text_embeds(prompt: str, model, tokenizer, limit: int):
    base_tokenizer = _plain_tokenizer(tokenizer)
    prefix = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, "
        "size, texture, quantity, text, spatial relationships of the objects and "
        "background:<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    text = [prefix + prompt]
    text_tokens = base_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=limit + 34 - 5,
        return_tensors="pt",
    )
    suffix_tokens = base_tokenizer([suffix], return_tensors="pt")
    input_ids = torch.cat(
        [
            _batch_get(text_tokens, "input_ids"),
            _batch_get(suffix_tokens, "input_ids"),
        ],
        dim=1,
    ).to(_module_device(model))
    attention_mask = torch.cat(
        [
            _batch_get(text_tokens, "attention_mask"),
            _batch_get(suffix_tokens, "attention_mask"),
        ],
        dim=1,
    ).bool().to(_module_device(model))
    position_ids = (
        attention_mask.long().cumsum(dim=-1) - 1
    ).clamp(min=0)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    output = _forward_with_mask(
        model,
        input_ids,
        attention_mask,
        position_ids=position_ids,
    )
    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states:
        raise ValueError("Krea 2 text encoder did not return hidden states")
    layers = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    hidden = torch.stack(
        [hidden_states[index] for index in layers],
        dim=2,
    )
    return hidden[:, 34:], attention_mask[:, 34:]


def _ideogram4_text_embeds(prompt: str, model, tokenizer, limit: int):
    base_tokenizer = _plain_tokenizer(tokenizer)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    try:
        text = base_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except (AttributeError, TypeError, ValueError):
        text = _chat_text(base_tokenizer, prompt)
    try:
        batch = base_tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
    except TypeError:
        batch = base_tokenizer(text, return_tensors="pt")
    tokens = _batch_get(batch, "input_ids")
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    tokens = tokens[0]
    token_count = int(tokens.shape[0])
    if token_count > limit:
        raise ValueError(
            f"prompt has {token_count} tokens, exceeds "
            f"max_sequence_length={limit}"
        )
    input_ids = torch.zeros(1, limit, dtype=torch.long)
    attention_mask = torch.zeros(1, limit, dtype=torch.long)
    text_position_ids = torch.zeros(1, limit, dtype=torch.long)
    input_ids[0, limit - token_count :] = tokens
    attention_mask[0, limit - token_count :] = 1
    text_position_ids[0, limit - token_count :] = torch.arange(token_count)

    device = _module_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    text_position_ids = text_position_ids.to(device)
    layers = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
    selected = None
    language_model = getattr(model, "language_model", None)
    try:
        from transformers.masking_utils import create_causal_mask

        language_model = language_model or model
        inputs_embeds = language_model.embed_tokens(input_ids)
        position_ids_4d = text_position_ids[None, ...].expand(
            4,
            text_position_ids.shape[0],
            -1,
        )
        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = language_model.rotary_emb(
            inputs_embeds,
            position_ids_4d[1:],
        )
        captured = {}
        hidden_states = inputs_embeds
        for index, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if index in layers:
                captured[index] = hidden_states
        selected = [captured[index] for index in layers]
    except (AttributeError, ImportError, TypeError, ValueError):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = getattr(output, "hidden_states", None)
        if not hidden_states:
            raise ValueError("Ideogram 4 text encoder did not return hidden states")
        selected = [hidden_states[index + 1] for index in layers]
    features = torch.stack(selected, dim=0).permute(1, 2, 3, 0).reshape(
        1,
        limit,
        -1,
    )
    features = features * attention_mask.to(features.dtype).unsqueeze(-1)
    return features.to(torch.float32), attention_mask.bool()


def _fibo2_text_embeds(prompt: str, model, tokenizer, limit: int):
    base_tokenizer = _plain_tokenizer(tokenizer)
    text = (
        "<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    batch = base_tokenizer(
        [text],
        padding="longest",
        max_length=limit,
        truncation=True,
        return_tensors="pt",
    )
    device = _module_device(model)
    input_ids = _batch_get(batch, "input_ids").to(device)
    attention_mask = _batch_get(batch, "attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states:
        raise ValueError("Fibo 2 text encoder did not return hidden states")
    layer_groups = (
        (9, 20, 31),
        (5, 16, 27),
        (7, 18, 29),
        (9, 20, 31),
        (11, 22, 33),
        (13, 24, 35),
    )
    normalized = {}
    for index in sorted({item for group in layer_groups for item in group}):
        states = hidden_states[index].float()
        normalized[index] = (
            states
            * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        ).to(hidden_states[index].dtype)
    bundles = [
        torch.cat([normalized[index] for index in group], dim=-1)
        for group in layer_groups
    ]
    embeds = torch.stack(bundles, dim=1)
    return embeds, attention_mask.bool()


def _moss_text_embeds(prompt: str, model, tokenizer, limit: int):
    base_tokenizer = _plain_tokenizer(tokenizer)
    try:
        text = _chat_text(base_tokenizer, prompt, enable_thinking=True)
    except TypeError:
        text = _chat_text(base_tokenizer, prompt)
    batch = base_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=limit,
    )
    device = _module_device(model)
    input_ids = _batch_get(batch, "input_ids").to(device)
    attention_mask = _batch_get(batch, "attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = _hidden(output)
    if hidden is None:
        raise ValueError("MOSS Image text encoder returned no hidden states")
    return hidden, attention_mask.bool()


def _anima_text_embeds(
    prompt: str,
    model,
    tokenizer,
    t5_tokenizer,
    conditioner,
    limit: int,
):
    if conditioner is None or t5_tokenizer is None:
        raise ValueError(
            "Anima text encoding requires its text_conditioner and "
            "t5_tokenizer components"
        )
    base_tokenizer = _plain_tokenizer(tokenizer)
    batch = base_tokenizer(
        [prompt],
        padding="longest",
        max_length=limit,
        truncation=True,
        return_tensors="pt",
    )
    device = _module_device(model)
    input_ids = _batch_get(batch, "input_ids").to(device)
    attention_mask = _batch_get(batch, "attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)
    output = _forward_with_mask(model, input_ids, attention_mask)
    hidden = _hidden(output)
    if hidden is None:
        raise ValueError("Anima text encoder returned no hidden states")
    hidden = hidden * attention_mask.to(hidden.dtype).unsqueeze(-1)
    t5_batch = t5_tokenizer(
        [prompt],
        padding="longest",
        max_length=limit,
        truncation=True,
        return_tensors="pt",
    )
    t5_ids = _batch_get(t5_batch, "input_ids")
    t5_mask = _batch_get(t5_batch, "attention_mask")
    if t5_mask is None:
        t5_mask = torch.ones_like(t5_ids)
    parameter = next(conditioner.parameters(), None)
    conditioner_dtype = (
        parameter.dtype if parameter is not None else hidden.dtype
    )
    conditioner_device = (
        parameter.device if parameter is not None else hidden.device
    )
    conditioned = conditioner(
        source_hidden_states=hidden.to(conditioner_dtype),
        target_input_ids=t5_ids.to(conditioner_device),
        target_attention_mask=t5_mask.to(conditioner_device),
        source_attention_mask=attention_mask,
    )
    return conditioned, t5_ids, t5_mask


def _load_diffusion_component(
    path: Path,
    class_name: str,
    torch_dtype,
    trust_remote_code: bool,
):
    errors = []
    component_class = getattr(diffusers, class_name, None) if class_name else None
    if component_class is not None:
        try:
            return component_class.from_pretrained(
                str(path),
                torch_dtype=torch_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
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
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "unknown loader error"
    raise ValueError(
        f"Could not load diffusion component '{path.name}'. Details: {detail}"
    )


if not clip_name:
    raise ValueError("Select a CLIP/text-encoder bundle in models/clip or models/diffusion")
selected = Path(clip_name)
if not selected.is_dir():
    raise ValueError("Load CLIP expects a diffusers repo or component folder")
device_name = "cpu" if device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
target_device = torch.device(device_name)
torch_dtype = torch.float16 if device_name == "cuda" else torch.float32
try:
    from app.quantization import detect_quantization, validate_quantization
except ImportError:
    quantization_info = None
else:
    capability = None
    if device_name == "cuda":
        try:
            capability = torch.cuda.get_device_capability()
        except Exception:
            capability = None
    quantization_info = detect_quantization(selected, "clip")
    if quantization_info is not None and quantization_info.format != "none":
        validate_quantization(
            quantization_info,
            model_type="clip",
            device=device_name,
            capability=capability,
        )
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
    report(0.05, "loading text encoders")

repo_root = selected
index_class, entries = _entries(repo_root)
if not entries and (selected.parent / "model_index.json").is_file():
    repo_root = selected.parent
    index_class, entries = _entries(repo_root)
elif not entries and (
    (selected.parent / "text_encoder").is_dir()
    or (selected.parent / "text_encoder_2").is_dir()
):
    repo_root = selected.parent

explicit_arch = _canonical_type(type)
arch = _infer_arch(selected, explicit_arch, index_class, entries)
if explicit_arch is not None:
    try:
        detected_arch = _infer_arch(selected, None, index_class, entries)
    except ValueError:
        detected_arch = None
    if detected_arch is not None and detected_arch != explicit_arch:
        raise ValueError(
            f"Selected type '{type}' does not match the detected text-encoder "
            f"architecture '{detected_arch}'"
        )
if arch not in COMPONENT_NAMES:
    raise ValueError(f"Unsupported CLIP architecture: {arch}")
if arch == "moss_image" and not allow_custom_code:
    raise ValueError(
        "MOSS Image text encoding requires allow_custom_code=True for its "
        "repository-local pipeline/components."
    )

encoder_names, tokenizer_names = COMPONENT_NAMES[arch]
primary_path, primary_class = _component_path(repo_root, entries, encoder_names[:1])
if primary_path is None and selected.is_dir():
    primary_class = str(_class_name(selected) or "")
    if primary_class in CLASS_ROLE:
        primary_path = selected
if primary_path is None or not primary_class:
    raise ValueError(
        f"Could not find a text encoder for architecture '{arch}' in "
        f"{repo_root}. Expected one of {list(encoder_names)!r}."
    )

secondary_path = None
secondary_class = None
if len(encoder_names) > 1:
    secondary_path, secondary_class = _component_path(
        repo_root, entries, encoder_names[1:]
    )
    if secondary_path is None:
        raise ValueError(
            f"Architecture '{arch}' requires a second text encoder "
            f"({encoder_names[1]})."
        )

prefer_processor = arch in {"flux2", "flux3", "qwen_image", "fibo2", "moss_image"}
tokenizer_candidates = list(tokenizer_names)
if "text_encoder_2" in selected.name.lower() or "t5" in selected.name.lower():
    tokenizer_candidates = [
        name for name in ("tokenizer_2", "processor_2", "tokenizer", "processor")
        if name in tokenizer_names
    ]
tokenizer_path, _ = _component_path(repo_root, entries, tokenizer_candidates)
if tokenizer_path is None and selected.is_dir():
    for candidate in tokenizer_candidates:
        candidate_path = selected / candidate
        if candidate_path.is_dir():
            tokenizer_path = candidate_path
            break
if tokenizer_path is None:
    tokenizer_path, _ = _sibling_component(selected, tokenizer_candidates)
if tokenizer_path is None:
    raise ValueError(
        f"Could not find a tokenizer/processor for architecture '{arch}'. "
        f"Expected one of {list(tokenizer_candidates)!r} next to the text encoder."
    )

primary_model = _load_text_model(
    primary_path, primary_class, torch_dtype, allow_custom_code
).to(target_device)
primary_model.eval()
try:
    from app.quantization import detect_quantization as _detect_quantization
except ImportError:
    pass
else:
    quantization_info = _detect_quantization(primary_path, "clip")
primary_tokenizer = _load_tokenizer(
    tokenizer_path, prefer_processor, allow_custom_code
)

secondary_model = None
secondary_tokenizer = None
if secondary_path is not None:
    secondary_model = _load_text_model(
        secondary_path, secondary_class or "", torch_dtype, allow_custom_code
    ).to(target_device)
    secondary_model.eval()
    second_tokenizer_path, _ = _component_path(
        repo_root, entries, ("tokenizer_2", "processor_2", "tokenizer")
    )
    if second_tokenizer_path is None:
        for candidate in ("tokenizer_2", "tokenizer"):
            candidate_path = secondary_path.parent / candidate
            if candidate_path.is_dir():
                second_tokenizer_path = candidate_path
                break
    if second_tokenizer_path is None:
        raise ValueError("The second text encoder is missing its tokenizer")
    secondary_tokenizer = _load_tokenizer(
        second_tokenizer_path, False, allow_custom_code
    )

t5_tokenizer = None
text_conditioner = None
if arch == "anima":
    t5_path, _ = _component_path(repo_root, entries, ("t5_tokenizer",))
    if t5_path is None:
        t5_path = repo_root / "t5_tokenizer"
    if t5_path.exists():
        t5_tokenizer = _load_tokenizer(t5_path, False, allow_custom_code)
    conditioner_path, conditioner_class = _component_path(
        repo_root,
        entries,
        ("text_conditioner",),
    )
    if conditioner_path is None:
        candidate = repo_root / "text_conditioner"
        if candidate.exists():
            conditioner_path = candidate
            conditioner_class = str(_class_name(candidate) or "")
    if conditioner_path is not None:
        text_conditioner = _load_diffusion_component(
            conditioner_path,
            str(conditioner_class or ""),
            torch_dtype,
            allow_custom_code,
        ).to(target_device)
        text_conditioner.eval()

primary_role = CLASS_ROLE.get(primary_class, "text")
secondary_role = CLASS_ROLE.get(secondary_class or "", "text")
max_length = max_sequence_length
token_limit = getattr(primary_tokenizer, "model_max_length", max_length)
if isinstance(token_limit, int) and 0 < token_limit < 10000:
    max_length = min(max_length, token_limit)


def encode_text(text: str):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    prompt = text if text.strip() else " "
    encoders = {}
    with torch.inference_mode():
        if arch in {"sd15", "sdxl"}:
            input_ids, attention_mask = _inputs_from_text(
                prompt, primary_tokenizer, 77, False
            )
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device) if attention_mask is not None else None
            output = _forward(primary_model, input_ids, attention_mask)
            cond = _hidden(output)
            if secondary_model is not None and secondary_tokenizer is not None:
                second_ids, second_mask = _inputs_from_text(
                    prompt, secondary_tokenizer, 77, False
                )
                second_ids = second_ids.to(target_device)
                second_mask = second_mask.to(target_device) if second_mask is not None else None
                second_output = _forward(secondary_model, second_ids, second_mask)
                cond = torch.cat([cond, _hidden(second_output)], dim=-1)
                pooled = getattr(second_output, "text_embeds", None)
                if pooled is None:
                    pooled = getattr(second_output, "pooler_output", None)
            else:
                pooled = getattr(output, "pooler_output", None)
                if arch == "sdxl":
                    raise ValueError("SDXL text encoding requires two CLIP encoders")
            return {
                "kind": "conditioning",
                "arch": arch,
                "cond": _to_cpu(cond, torch_dtype),
                "pooled": _to_cpu(pooled, torch_dtype),
                "text": text,
            }
        if arch == "flux1":
            clip_ids, clip_mask = _inputs_from_text(prompt, primary_tokenizer, 77, False)
            clip_ids = clip_ids.to(target_device)
            clip_mask = clip_mask.to(target_device) if clip_mask is not None else None
            clip_output = _forward(primary_model, clip_ids, clip_mask)
            pooled = getattr(clip_output, "pooler_output", None)
            if pooled is None:
                pooled = _hidden(clip_output)[:, 0]
            if secondary_model is None or secondary_tokenizer is None:
                raise ValueError("FLUX.1 text encoding requires CLIP and T5 encoders")
            t5_ids, t5_mask = _inputs_from_text(
                prompt, secondary_tokenizer, max_length, False
            )
            t5_ids = t5_ids.to(target_device)
            t5_mask = t5_mask.to(target_device) if t5_mask is not None else None
            t5_output = _forward(secondary_model, t5_ids, t5_mask)
            t5_hidden = _hidden(t5_output)
            encoders["clip"] = {
                "hidden": _to_cpu(_hidden(clip_output), torch_dtype),
                "pooled": _to_cpu(pooled, torch_dtype),
            }
            encoders["t5"] = {
                "hidden": _to_cpu(t5_hidden, torch_dtype),
                "attention_mask": _to_cpu(t5_mask, torch.float32),
            }
            return {
                "kind": "conditioning",
                "arch": arch,
                "cond": _to_cpu(t5_hidden, torch_dtype),
                "pooled": _to_cpu(pooled, torch_dtype),
                "text": text,
                "encoders": encoders,
                "attention_mask": _to_cpu(t5_mask, torch.float32),
            }
        if arch == "qwen_image":
            try:
                from diffusers.modular_pipelines.qwenimage21.encoders import (
                    get_qwenimage21_prompt_embeds,
                )

                hidden, mask, image_mask = get_qwenimage21_prompt_embeds(
                    primary_model,
                    primary_tokenizer,
                    prompt,
                    device=target_device,
                )
            except (ImportError, AttributeError, TypeError, ValueError):
                pass
            else:
                encoders["text"] = {
                    "hidden": _to_cpu(hidden, torch_dtype),
                    "attention_mask": _to_cpu(mask, torch.float32),
                }
                return {
                    "kind": "conditioning",
                    "arch": arch,
                    "cond": _to_cpu(hidden, torch_dtype),
                    "pooled": None,
                    "text": text,
                    "encoders": encoders,
                    "attention_mask": _to_cpu(mask, torch.float32),
                    "image_pad_mask": _to_cpu(image_mask, torch.bool),
                }
        if arch == "flux3":
            hidden, mask = _flux3_text_embeds(
                prompt,
                primary_model,
                primary_tokenizer,
                max_length,
            )
            encoders["text"] = {
                "hidden": _to_cpu(hidden, torch_dtype),
                "attention_mask": _to_cpu(mask, torch.float32),
            }
            return {
                "kind": "conditioning",
                "arch": arch,
                "cond": _to_cpu(hidden, torch_dtype),
                "pooled": None,
                "text": text,
                "encoders": encoders,
                "attention_mask": _to_cpu(mask, torch.float32),
            }
        if arch == "krea2":
            hidden, mask = _krea_text_embeds(
                prompt,
                primary_model,
                primary_tokenizer,
                max_length,
            )
            encoders["text"] = {
                "hidden": _to_cpu(hidden, torch_dtype),
                "attention_mask": _to_cpu(mask, torch.float32),
            }
            return {
                "kind": "conditioning",
                "arch": arch,
                "cond": _to_cpu(hidden, torch_dtype),
                "pooled": None,
                "text": text,
                "encoders": encoders,
                "attention_mask": _to_cpu(mask, torch.float32),
            }
        if arch == "ideogram4":
            hidden, mask = _ideogram4_text_embeds(
                prompt,
                primary_model,
                primary_tokenizer,
                max_length,
            )
            encoders["text"] = {
                "hidden": _to_cpu(hidden, torch_dtype),
                "attention_mask": _to_cpu(mask, torch.float32),
            }
            return {
                "kind": "conditioning",
                "arch": arch,
                "cond": _to_cpu(hidden, torch_dtype),
                "pooled": None,
                "text": text,
                "encoders": encoders,
                "attention_mask": _to_cpu(mask, torch.float32),
            }
        if arch == "fibo2":
            fibo_prompt = prompt
            if not prompt.strip():
                fibo_prompt = (
                    '{"short_description":"","objects":[],'
                    '"background_setting":"","lighting":{},'
                    '"aesthetics":{},"photographic_characteristics":{},'
                    '"style_medium":"","context":"","artistic_style":""}'
                )
            hidden, mask = _fibo2_text_embeds(
                fibo_prompt,
                primary_model,
                primary_tokenizer,
                max_length,
            )
            encoders["text"] = {
                "hidden": _to_cpu(hidden, torch_dtype),
                "attention_mask": _to_cpu(mask, torch.float32),
            }
            return {
                "kind": "conditioning",
                "arch": arch,
                "cond": _to_cpu(hidden, torch_dtype),
                "pooled": None,
                "text": text,
                "encoders": encoders,
                "attention_mask": _to_cpu(mask, torch.float32),
            }
        if arch == "moss_image":
            hidden, mask = _moss_text_embeds(
                prompt,
                primary_model,
                primary_tokenizer,
                max_length,
            )
            encoders["text"] = {
                "hidden": _to_cpu(hidden, torch_dtype),
                "attention_mask": _to_cpu(mask, torch.float32),
            }
            return {
                "kind": "conditioning",
                "arch": arch,
                "cond": _to_cpu(hidden, torch_dtype),
                "pooled": None,
                "text": text,
                "encoders": encoders,
                "attention_mask": _to_cpu(mask, torch.float32),
            }
        if arch == "anima":
            hidden, t5_ids, mask = _anima_text_embeds(
                prompt,
                primary_model,
                primary_tokenizer,
                t5_tokenizer,
                text_conditioner,
                max_length,
            )
            encoders["text"] = {
                "hidden": _to_cpu(hidden, torch_dtype),
                "attention_mask": _to_cpu(mask, torch.float32),
            }
            return {
                "kind": "conditioning",
                "arch": arch,
                "cond": _to_cpu(hidden, torch_dtype),
                "pooled": None,
                "text": text,
                "encoders": encoders,
                "attention_mask": _to_cpu(mask, torch.float32),
                "t5_input_ids": t5_ids.detach().cpu(),
                "t5_attention_mask": mask.detach().cpu(),
            }
        if arch == "flux2":
            try:
                from diffusers import Flux2KleinPipeline, Flux2Pipeline

                if primary_class == "Mistral3ForConditionalGeneration":
                    hidden = Flux2Pipeline._get_mistral_3_small_prompt_embeds(
                        primary_model,
                        primary_tokenizer,
                        prompt,
                        dtype=torch_dtype,
                        device=target_device,
                        max_sequence_length=max_length,
                    )
                else:
                    hidden = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                        primary_model,
                        primary_tokenizer,
                        prompt,
                        dtype=torch_dtype,
                        device=target_device,
                        max_sequence_length=max_length,
                    )
            except (ImportError, AttributeError, TypeError, ValueError):
                pass
            else:
                encoders["text"] = {
                    "hidden": _to_cpu(hidden, torch_dtype),
                }
                return {
                    "kind": "conditioning",
                    "arch": arch,
                    "cond": _to_cpu(hidden, torch_dtype),
                    "pooled": None,
                    "text": text,
                    "encoders": encoders,
                    "attention_mask": None,
                }
            input_ids, attention_mask = _inputs_from_text(
                prompt, primary_tokenizer, max_length, True
            )
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device) if attention_mask is not None else None
            output = _forward(primary_model, input_ids, attention_mask)
            layers = (10, 20, 30) if primary_role == "mistral" else (9, 18, 27)
            hidden = _stack_layers(output, layers)
            encoders["text"] = {
                "hidden": _to_cpu(hidden, torch_dtype),
                "attention_mask": _to_cpu(attention_mask, torch.float32),
            }
            return {
                "kind": "conditioning",
                "arch": arch,
                "cond": _to_cpu(hidden, torch_dtype),
                "pooled": None,
                "text": text,
                "encoders": encoders,
                "attention_mask": _to_cpu(attention_mask, torch.float32),
            }
        if arch == "z_image":
            base_tokenizer = _plain_tokenizer(primary_tokenizer)
            rendered = _chat_text(
                primary_tokenizer,
                prompt,
                enable_thinking=True,
            )
            batch = base_tokenizer(
                [rendered],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = batch.input_ids.to(target_device)
            attention_mask = getattr(batch, "attention_mask", None)
            attention_mask = (
                attention_mask.to(target_device)
                if attention_mask is not None
                else None
            )
            output = _forward_with_mask(
                primary_model,
                input_ids,
                attention_mask,
            )
            hidden_states = getattr(output, "hidden_states", None)
            hidden = hidden_states[-2] if hidden_states else _hidden(output)
            encoders["qwen"] = {
                "hidden": _to_cpu(hidden, torch_dtype),
                "attention_mask": _to_cpu(attention_mask, torch.float32),
            }
            return {
                "kind": "conditioning",
                "arch": arch,
                "cond": _to_cpu(hidden, torch_dtype),
                "pooled": None,
                "text": text,
                "encoders": encoders,
                "attention_mask": _to_cpu(attention_mask, torch.float32),
            }
        input_ids, attention_mask = _inputs_from_text(
            prompt, primary_tokenizer, max_length, arch not in {"anima", "moss_image"}
        )
        input_ids = input_ids.to(target_device)
        attention_mask = attention_mask.to(target_device) if attention_mask is not None else None
        output = _forward(primary_model, input_ids, attention_mask)
        hidden = _hidden(output)
        if hidden is None:
            raise ValueError("Text encoder returned no hidden states")
        encoders["text"] = {
            "hidden": _to_cpu(hidden, torch_dtype),
            "attention_mask": _to_cpu(attention_mask, torch.float32),
        }
        result = {
            "kind": "conditioning",
            "arch": arch,
            "cond": _to_cpu(hidden, torch_dtype),
            "pooled": None,
            "text": text,
            "encoders": encoders,
            "attention_mask": _to_cpu(attention_mask, torch.float32),
        }
        if arch == "anima" and t5_tokenizer is not None:
            t5_batch = t5_tokenizer(
                [prompt],
                padding="longest",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            result["t5_input_ids"] = t5_batch.input_ids.detach().cpu()
            result["t5_attention_mask"] = t5_batch.attention_mask.detach().cpu()
        return result


encoders = [
    {
        "role": primary_role,
        "class_name": primary_class,
        "model": primary_model,
        "tokenizer": primary_tokenizer,
    }
]
if secondary_model is not None:
    encoders.append(
        {
            "role": secondary_role,
            "class_name": secondary_class or "",
            "model": secondary_model,
            "tokenizer": secondary_tokenizer,
        }
    )
stat = selected.stat()
clip = {
    "kind": "clip",
    "backend": "diffusers",
    "arch": arch,
    "name": selected.name,
    "path": str(selected.resolve()),
    "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
    "source": "local",
    "model_type": "clip",
    "device": device_name,
    "dtype": torch_dtype,
    "quantization": (
        quantization_info.model_dump()
        if quantization_info is not None
        else None
    ),
    "text_encoder": primary_model,
    "tokenizer": primary_tokenizer,
    "text_encoder_2": secondary_model,
    "tokenizer_2": secondary_tokenizer,
    "processor": primary_tokenizer if prefer_processor else None,
    "t5_tokenizer": t5_tokenizer,
    "encoders": encoders,
    "max_sequence_length": max_length,
    "encode": encode_text,
}
if callable(report):
    report(1.0, f"{arch} text encoders ready")

return clip: CLIP
