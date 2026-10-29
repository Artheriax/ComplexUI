# /// script
# dependencies = ["torch>=2.2"]
# ///
import logging
import os
from typing import Literal

import torch

model: MODEL
backend: Literal[
    "server",
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
] = "server"
benchmark: bool = False
sol_tau: float = ShowIf(default=1.0, backend="comfy_kitchen_sol")

ATTENTION_BACKENDS = (
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
)
OPTIONAL_BACKENDS = {
    "xformers",
    "flash-attn",
    "flash-attn4",
    "sageattention",
    "comfy_kitchen_sol",
    "comfy_kitchen_int8",
}
LOGGER = logging.getLogger("complexui.attention")


def _device_info():
    if not torch.cuda.is_available():
        return "cpu", None, "cpu"
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        major, minor = 0, 0
    if (major, minor) >= (10, 0):
        arch = "blackwell"
    elif (major, minor) >= (9, 0):
        arch = "hopper"
    elif (major, minor) >= (8, 9):
        arch = "ada"
    elif (major, minor) >= (8, 0):
        arch = "ampere"
    else:
        arch = f"sm{major}{minor}"
    return "cuda", (major, minor), arch


def _server_backend():
    value = os.environ.get("COMPLEXUI_ATTENTION", "auto").strip().lower()
    return value if value in ATTENTION_BACKENDS else "auto"


def _flags(flash=False, mem_efficient=False, cudnn=False, math=True):
    return {
        "flash": bool(flash),
        "mem_efficient": bool(mem_efficient),
        "cudnn": bool(cudnn),
        "math": bool(math),
    }


def _unavailable(requested, device, capability, reason):
    capability_text = (
        f"sm{capability[0]}{capability[1]}" if capability else device
    )
    raise ValueError(
        f"Attention backend '{requested}' is unavailable ({reason}; device="
        f"{capability_text}). Choose 'auto', 'sdpa' or 'sdpa-math', or use a "
        "compatible CUDA device."
    )


def _select(requested):
    device, capability, arch = _device_info()
    if requested == "server":
        requested = _server_backend()
    if requested in OPTIONAL_BACKENDS:
        raise ValueError(
            f"Attention backend '{requested}' requires the ComplexUI runtime; "
            "exported standalone scripts fall back to SDPA"
        )
    if requested == "auto":
        if device == "cpu":
            return {
                "requested": requested,
                "backend": "sdpa",
                "reason": "cpu default dispatch",
                "approximates": False,
                "flags": _flags(mem_efficient=False, math=True),
                "device": device,
                "capability": capability,
                "arch": arch,
            }
        if capability and capability >= (9, 0):
            return {
                "requested": requested,
                "backend": "sdpa",
                "reason": f"{arch} prefers flash/cuDNN SDPA",
                "approximates": False,
                "flags": _flags(
                    flash=True,
                    mem_efficient=True,
                    cudnn=True,
                    math=True,
                ),
                "device": device,
                "capability": capability,
                "arch": arch,
            }
        if capability and capability >= (8, 0):
            return {
                "requested": requested,
                "backend": "sdpa",
                "reason": f"{arch} prefers flash SDPA",
                "approximates": False,
                "flags": _flags(flash=True, mem_efficient=True, math=True),
                "device": device,
                "capability": capability,
                "arch": arch,
            }
        return {
            "requested": requested,
            "backend": "sdpa",
            "reason": "pre-Ampere safe SDPA fallback",
            "approximates": False,
            "flags": _flags(mem_efficient=True, math=True),
            "device": device,
            "capability": capability,
            "arch": arch,
        }
    if requested == "sdpa":
        return {
            "requested": requested,
            "backend": "sdpa",
            "reason": "PyTorch SDPA default dispatch",
            "approximates": False,
            "flags": _flags(flash=True, mem_efficient=True, cudnn=True, math=True),
            "device": device,
            "capability": capability,
            "arch": arch,
        }
    if requested == "sdpa-flash":
        if device != "cuda":
            _unavailable(requested, device, capability, "CUDA is required")
        if not capability or capability < (8, 0):
            _unavailable(requested, device, capability, "requires sm80+")
        return {
            "requested": requested,
            "backend": requested,
            "reason": "explicit flash SDPA",
            "approximates": False,
            "flags": _flags(flash=True, math=True),
            "device": device,
            "capability": capability,
            "arch": arch,
        }
    if requested == "sdpa-mem-efficient":
        if device != "cuda":
            _unavailable(requested, device, capability, "CUDA is required")
        return {
            "requested": requested,
            "backend": requested,
            "reason": "explicit memory-efficient SDPA",
            "approximates": False,
            "flags": _flags(mem_efficient=True, math=True),
            "device": device,
            "capability": capability,
            "arch": arch,
        }
    if requested == "sdpa-cudnn":
        if device != "cuda":
            _unavailable(requested, device, capability, "CUDA is required")
        if not capability or capability < (9, 0):
            _unavailable(requested, device, capability, "requires sm90+")
        if not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            _unavailable(
                requested,
                device,
                capability,
                "installed torch has no cuDNN SDPA switch",
            )
        return {
            "requested": requested,
            "backend": requested,
            "reason": "explicit cuDNN SDPA",
            "approximates": False,
            "flags": _flags(cudnn=True, math=True),
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
            "flags": _flags(math=True),
            "device": device,
            "capability": capability,
            "arch": arch,
        }
    raise ValueError("Unsupported attention backend: " + str(requested))


def _apply_flags(selection):
    flags = selection["flags"]
    torch.backends.cuda.enable_flash_sdp(bool(flags["flash"]))
    torch.backends.cuda.enable_mem_efficient_sdp(
        bool(flags["mem_efficient"])
    )
    torch.backends.cuda.enable_math_sdp(bool(flags["math"]))
    enable_cudnn = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if callable(enable_cudnn):
        enable_cudnn(bool(flags["cudnn"]))


if not isinstance(model, dict):
    raise TypeError("model must be a MODEL wrapper")
if sol_tau < 0 or sol_tau > 4:
    raise ValueError("sol_tau must be between 0 and 4")
device_name = _device_info()[0]
try:
    from app.attention_dispatch import (
        benchmark_attention,
        ensure_attention_dispatch,
        resolve_attention,
    )
except Exception:
    resolve_attention = None
    benchmark_attention = None
    ensure_attention_dispatch = None
if resolve_attention is not None:
    requested = str(backend)
    benchmark_results = None
    if benchmark and requested in {"server", "auto", "sdpa"}:
        try:
            benchmark_results = benchmark_attention(torch, device_name, "auto")
            requested = str(benchmark_results["selected"])
        except Exception as exc:
            print(f"attention benchmark failed; using auto ({exc})")
            requested = "auto"
    selection = resolve_attention(requested, device=device_name).as_dict()
    selection["sol_tau"] = float(sol_tau)
    if benchmark_results is not None:
        selection["benchmark"] = benchmark_results
    if ensure_attention_dispatch is not None:
        ensure_attention_dispatch()
else:
    selection = _select(str(backend))
    _apply_flags(selection)
    selection["sol_tau"] = float(sol_tau)
message = (
    f"attention: backend={selection['backend']} "
    f"reason={selection['reason']} device={selection['device']} "
    f"flags={selection['flags']}"
)
print(message)
LOGGER.info(message)
patched = dict(model)
patched["attention"] = selection
model = patched
return model: MODEL
