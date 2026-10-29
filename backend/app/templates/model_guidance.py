# /// script
# dependencies = ["torch>=2.2"]
# ///
from typing import Literal

model: MODEL
guidance_mode: Literal["architecture", "override"] = "architecture"
guidance: float = ShowIf(default=3.5, guidance_mode="override")

if not isinstance(model, dict):
    raise TypeError("model must be a MODEL wrapper")
if guidance_mode not in {"architecture", "override"}:
    raise ValueError("guidance_mode must be 'architecture' or 'override'")
if guidance < 0 or guidance > 50:
    raise ValueError("guidance must be between 0 and 50")
adapter = model.get("sampler_adapter")
if not isinstance(adapter, dict) or not adapter.get("supports_guidance", False):
    raise ValueError(
        f"Architecture '{model.get('arch', 'unknown')}' does not use a "
        "guidance embedding; set CFG on the sampler instead"
    )

value = float(guidance)
if guidance_mode == "architecture":
    recommended = model.get("recommended")
    if isinstance(recommended, dict) and recommended.get("guidance") is not None:
        value = float(recommended["guidance"])
    else:
        capabilities = model.get("capabilities")
        if (
            isinstance(capabilities, dict)
            and capabilities.get("guidance_default") is not None
        ):
            value = float(capabilities["guidance_default"])
if value < 0 or value > 50:
    raise ValueError("architecture guidance default must be between 0 and 50")

guided = dict(model)
guided["guidance"] = value
guided["guidance_mode"] = guidance_mode
guided["guidance_explicit"] = True
guided["guidance_source"] = f"model_guidance_{guidance_mode}"

model = guided
return model: MODEL
