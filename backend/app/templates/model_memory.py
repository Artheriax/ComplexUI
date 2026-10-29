# /// script
# dependencies = ["torch>=2.2", "diffusers>=0.42.0"]
# ///
import logging
import os
from typing import Literal

import torch

model: MODEL
profile: Literal["server", "auto", "resident", "balanced", "low"] = "server"
offload: Literal[
    "server", "none", "model", "sequential", "block", "group"
] = "server"
group_size: int = ShowIf(default=2, offload=["block", "group"])
attention_slicing: bool = False
channels_last: Literal["auto", "on", "off"] = "auto"

LOGGER = logging.getLogger("complexui.memory")
PROFILES = {"auto", "low", "balanced", "resident"}


def _server_profile():
    value = os.environ.get("COMPLEXUI_MEMORY_PROFILE", "auto").strip().lower()
    return value if value in PROFILES else "auto"


def _total_vram():
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        try:
            return int(torch.cuda.mem_get_info()[1])
        except Exception:
            return None


def _resolve_profile(requested):
    value = _server_profile() if requested == "server" else requested
    if value in PROFILES and value != "auto":
        return value
    total = _total_vram()
    if total is None or total <= 0:
        return "resident"
    gib = float(total) / (1024.0**3)
    if gib < 10.0:
        return "low"
    if gib < 20.0:
        return "balanced"
    return "resident"


if not isinstance(model, dict):
    raise TypeError("model must be a MODEL wrapper")
component = model.get("transformer") or model.get("unet")
if component is None:
    raise ValueError("MODEL wrapper is missing its transformer/UNet")
resolved_profile = _resolve_profile(str(profile))
if offload == "server":
    resolved_offload = "model" if resolved_profile == "low" else "none"
elif offload in {"model", "sequential", "block", "group"}:
    resolved_offload = offload
else:
    resolved_offload = "none"
if resolved_offload in {"block", "group"} and (
    group_size < 1 or group_size > 64
):
    raise ValueError("group_size must be between 1 and 64")
if resolved_offload != "none" and model.get("compiled"):
    raise ValueError(
        "Offloading cannot be combined with an existing compiled graph; "
        "compile only resident models"
    )
components = []
seen_components = set()
for key in ("transformer", "unet", "unconditional_transformer", "text_conditioner"):
    value = model.get(key)
    if value is None or not hasattr(value, "to"):
        continue
    if id(value) in seen_components:
        continue
    seen_components.add(id(value))
    components.append(value)
arch = str(model.get("arch", ""))
resolved_channels_last = channels_last == "on" or (
    channels_last == "auto" and arch in {"sd15", "sdxl"}
)
if resolved_channels_last:
    component.to(memory_format=torch.channels_last)
    LOGGER.info("memory: channels-last enabled for %s", arch or "model")
if attention_slicing:
    set_slice = getattr(component, "set_attention_slice", None)
    if not callable(set_slice):
        raise ValueError(
            "Attention slicing is not supported by this model architecture"
        )
    set_slice("auto")
    LOGGER.info("memory: attention slicing enabled")
target_device = str(model.get("device", "cpu"))
resolved_group_size = 1 if resolved_offload == "block" else int(group_size)
offload_stream = resolved_offload in {"sequential", "block", "group"} and (
    target_device.startswith("cuda") and torch.cuda.is_available()
)
if resolved_offload in {"sequential", "block", "group"}:
    try:
        from diffusers.hooks import apply_group_offloading
    except Exception as exc:
        raise ValueError(
            "Group/sequential offload requires diffusers>=0.42 with hooks"
        ) from exc
    offload_type = "leaf_level" if resolved_offload == "sequential" else "block_level"
    for value in components:
        apply_group_offloading(
            value,
            onload_device=torch.device(target_device),
            offload_device=torch.device("cpu"),
            offload_type=offload_type,
            num_blocks_per_group=(
                None if resolved_offload == "sequential" else resolved_group_size
            ),
            non_blocking=offload_stream,
            use_stream=offload_stream,
            record_stream=False,
            low_cpu_mem_usage=False,
        )
    LOGGER.info(
        "memory: %s offload enabled (stream=%s, group_size=%s)",
        resolved_offload,
        offload_stream,
        resolved_group_size,
    )
elif resolved_offload == "model":
    for value in components:
        value.to("cpu")
else:
    for value in components:
        value.to(target_device)
memory = {
    "profile": resolved_profile,
    "offload": resolved_offload != "none",
    "offload_mode": resolved_offload,
    "group_size": resolved_group_size,
    "offload_stream": bool(offload_stream),
    "channels_last": bool(resolved_channels_last),
    "attention_slicing": bool(attention_slicing),
}
print(
    "memory: profile={profile} offload={offload_mode} group_size={group_size} "
    "stream={offload_stream} channels_last={channels_last} "
    "attention_slicing={attention_slicing}".format(**memory)
)
patched = dict(model)
patched["memory_profile"] = resolved_profile
patched["memory"] = memory
patched["offload"] = resolved_offload == "model"
patched["offload_mode"] = resolved_offload
patched["offload_group_size"] = resolved_group_size
patched["offload_stream"] = bool(offload_stream)
patched["channels_last"] = bool(resolved_channels_last)
patched["attention_slicing"] = bool(attention_slicing)
model = patched
return model: MODEL
