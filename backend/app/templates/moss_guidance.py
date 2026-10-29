# /// script
# dependencies = []
# ///

model: MODEL
uncond_skip_layers: str = "22"
slg_layers: str = ""
slg_scale: float = 0.0
slg_start: float = 0.0
slg_stop: float = 1.0

if not isinstance(model, dict):
    raise TypeError("model must be a MODEL wrapper")
if str(model.get("arch", "")) != "moss_image":
    raise ValueError("MOSS Guidance only applies to MOSS Image models")
if slg_scale < 0 or slg_scale > 20:
    raise ValueError("slg_scale must be between 0 and 20")
if slg_start < 0 or slg_start > 1:
    raise ValueError("slg_start must be between 0 and 1")
if slg_stop < 0 or slg_stop > 1:
    raise ValueError("slg_stop must be between 0 and 1")
if slg_start >= slg_stop:
    raise ValueError("slg_start must be less than slg_stop")


def parse_layers(value: str, name: str):
    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a comma-separated list of block indices"
            ) from exc
        if index < 0 or index > 255:
            raise ValueError(f"{name} block indices must be between 0 and 255")
        layers.append(index)
    return tuple(sorted(set(layers)))


skip_layers = parse_layers(uncond_skip_layers, "uncond_skip_layers")
positive_layers = parse_layers(slg_layers, "slg_layers")
if slg_scale > 0 and not positive_layers:
    raise ValueError("slg_layers must be set when slg_scale is greater than 0")
transformer = model.get("transformer")
depth = len(transformer.layers) if hasattr(transformer, "layers") else None
if depth is not None:
    for index in (*skip_layers, *positive_layers):
        if index >= depth:
            raise ValueError(
                f"skip-layer index {index} is outside the model's {depth} blocks"
            )

guided = dict(model)
guided["moss_guidance"] = {
    "uncond_skip_layers": skip_layers,
    "slg_layers": positive_layers,
    "slg_scale": float(slg_scale),
    "slg_start": float(slg_start),
    "slg_stop": float(slg_stop),
}
guided["guidance_source"] = "moss_guidance"

model = guided
return model: MODEL
