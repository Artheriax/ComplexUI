model: MODEL
positive: CONDITIONING
negative: CONDITIONING
cfg: float = 7.0

if not isinstance(model, dict):
    raise TypeError("model must be a MODEL wrapper")
if not isinstance(positive, dict) or not isinstance(negative, dict):
    raise TypeError("positive and negative must be CONDITIONING wrappers")
if cfg < 0 or cfg > 50:
    raise ValueError("cfg must be between 0 and 50")

guider = {
    "kind": "guider",
    "guider_type": "cfg",
    "model": model,
    "positive": positive,
    "negative": negative,
    "cfg": float(cfg),
}

return guider: GUIDER
