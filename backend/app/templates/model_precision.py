# /// script
# dependencies = ["torch>=2.2"]
# ///
import logging
import os
from typing import Literal

import torch

model: MODEL
compute_dtype: Literal["keep", "fp32", "fp16", "bf16"] = "keep"
accumulation: Literal["default", "fp16"] = "default"

LOGGER = logging.getLogger("complexui.precision")


def _dtype(value):
    if value == "fp32":
        return torch.float32
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    raise ValueError("Unsupported compute dtype: " + str(value))


def _components(wrapper):
    seen = set()
    components = []
    for key in ("transformer", "unet", "text_conditioner", "unconditional_transformer"):
        value = wrapper.get(key)
        if value is None or not hasattr(value, "to"):
            continue
        marker = id(value)
        if marker in seen:
            continue
        seen.add(marker)
        components.append((key, value))
    return components


if not isinstance(model, dict):
    raise TypeError("model must be a MODEL wrapper")
target_device = str(model.get("device", "cpu"))
if compute_dtype == "fp16" and target_device == "cpu":
    raise ValueError(
        "FP16 diffusion inference requires CUDA; choose bf16 or fp32 on CPU"
    )
if accumulation == "fp16" and not os.environ.get(
    "COMPLEXUI_EXPERIMENTAL_FP16_ACCUM"
):
    raise ValueError(
        "FP16 accumulation is experimental and disabled. Restart with "
        "--experimental-fp16-accumulation to unlock it, then validate output "
        "quality and overflow behavior."
    )
if model.get("compiled") and (
    compute_dtype != "keep" or accumulation != "default"
):
    raise ValueError(
        "Apply Model Precision before Compile Model; changing dtype or "
        "accumulation after torch.compile would invalidate the compiled graph"
    )

storage_dtype = str(model.get("storage_dtype") or model.get("dtype") or "unknown")
resolved_dtype = model.get("dtype", torch.float32)
if compute_dtype != "keep":
    resolved_dtype = _dtype(compute_dtype)
    for key, component in _components(model):
        component.to(dtype=resolved_dtype)
        LOGGER.info("precision: %s -> %s", key, resolved_dtype)

if accumulation == "fp16":
    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul = (
        getattr(cuda_backend, "matmul", None)
        if cuda_backend is not None
        else None
    )
    if matmul is not None and hasattr(
        matmul, "allow_fp16_reduced_precision_reduction"
    ):
        matmul.allow_fp16_reduced_precision_reduction = True
    LOGGER.warning("precision: experimental FP16 accumulation enabled")

precision = {
    "storage": storage_dtype,
    "compute": str(resolved_dtype),
    "accumulation": accumulation,
}
print(
    "precision: storage={storage} compute={compute} accumulation={accumulation}".format(
        **precision
    )
)
patched = dict(model)
patched["dtype"] = resolved_dtype
patched["compute_dtype"] = str(resolved_dtype)
patched["storage_dtype"] = storage_dtype
patched["accumulation"] = accumulation
patched["precision"] = precision
model = patched
return model: MODEL
