# /// script
# dependencies = ["torch>=2.2"]
# ///
from typing import Literal

model: MODEL
positive: CONDITIONING
negative: CONDITIONING
cfg: float = 7.0
guidance_mode: Literal["inherit", "override"] = "inherit"
guidance: float = ShowIf(default=3.5, guidance_mode="override")
start_percent: float = 0.0
end_percent: float = 1.0

if not isinstance(model, dict):
    raise TypeError("model must be a MODEL wrapper")
if not isinstance(positive, dict) or not isinstance(negative, dict):
    raise TypeError("positive and negative must be CONDITIONING wrappers")
if cfg < 0 or cfg > 50:
    raise ValueError("cfg must be between 0 and 50")
if guidance_mode not in {"inherit", "override"}:
    raise ValueError("guidance_mode must be 'inherit' or 'override'")
if guidance < 0 or guidance > 50:
    raise ValueError("guidance must be between 0 and 50")
if start_percent < 0 or start_percent > 1:
    raise ValueError("start_percent must be between 0 and 1")
if end_percent < 0 or end_percent > 1:
    raise ValueError("end_percent must be between 0 and 1")
if start_percent >= end_percent:
    raise ValueError("start_percent must be less than end_percent")

resolved_guidance = model.get("guidance")
if guidance_mode == "override":
    adapter = model.get("sampler_adapter")
    if not isinstance(adapter, dict) or not adapter.get("supports_guidance", False):
        raise ValueError(
            f"Architecture '{model.get('arch', 'unknown')}' does not use a "
            "guidance embedding; keep guidance_mode='inherit' or use CFG"
        )
    model = dict(model)
    model["guidance"] = float(guidance)
    model["guidance_explicit"] = True
    model["guidance_source"] = "cfg_guider_override"
    resolved_guidance = float(guidance)

guider = {
    "kind": "guider",
    "guider_type": "cfg",
    "model": model,
    "positive": positive,
    "negative": negative,
    "cfg": float(cfg),
    "guidance_mode": guidance_mode,
    "guidance": (
        float(resolved_guidance) if resolved_guidance is not None else None
    ),
    "start_percent": float(start_percent),
    "end_percent": float(end_percent),
}

return guider: GUIDER
