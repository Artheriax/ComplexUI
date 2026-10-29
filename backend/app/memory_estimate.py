"""Static memory estimates for local models and workflows.

These are deliberately heuristics. They use checkpoint metadata and file sizes
without loading weights; run-history peak metrics remain the measured source of
truth.
"""

from __future__ import annotations

from .schemas import ModelMemoryEstimate, ModelSummary, QuantizationInfo

__all__ = [
    "estimate_model_memory",
    "estimate_model_summary_memory",
]

_RUNTIME_OVERHEAD_FRACTION = 0.12
_MIN_RUNTIME_OVERHEAD = 192 * 1024 * 1024
_OFFLOAD_PINNED_FRACTION = 0.05


def _round_bytes(value: float) -> int:
    return max(0, int(round(value)))


def estimate_model_memory(
    *,
    size_bytes: int,
    quantization: QuantizationInfo | None = None,
    parameter_count: int | None = None,
    is_directory: bool = False,
) -> ModelMemoryEstimate:
    """Estimate weights, resident VRAM and offloaded RAM for one model."""
    weights = max(0, int(size_bytes))
    info = quantization
    quantized = info is not None and info.format != "none"
    if info is not None and info.mode in {"emulated", "mixed"} and info.weight_bits:
        weights = _round_bytes(weights * 16 / int(info.weight_bits))
    overhead = 0
    runtime_overhead = max(
        _MIN_RUNTIME_OVERHEAD,
        _round_bytes(weights * _RUNTIME_OVERHEAD_FRACTION),
    )
    resident = weights + overhead + runtime_overhead
    offload_ram = weights + overhead + _round_bytes(weights * _OFFLOAD_PINNED_FRACTION)
    assumptions = [
        (
            "weights estimate expands the packed checkpoint to compute width"
            if info is not None and info.mode in {"emulated", "mixed"}
            else "weights use the on-disk checkpoint size"
        ),
        (
            "scale and metadata tensors are included in the checkpoint size"
            if quantized
            and (info is None or info.mode not in {"emulated", "mixed"})
            else "runtime overhead includes allocator and activation headroom"
        ),
    ]
    if parameter_count:
        assumptions.append("parameter count read from SafeTensors headers")
    else:
        assumptions.append("parameter count unavailable")
    if is_directory:
        assumptions.append("folder size includes all component weight files")
    if info is not None and info.weight_bits:
        assumptions.append(f"declared weight width: {info.weight_bits}-bit")
    if info is not None and info.mode == "pre_quantized":
        assumptions.append("pre-quantized checkpoint; weights are not re-quantized")
    elif info is not None and info.mode in {"emulated", "mixed"}:
        assumptions.append("single-file compatibility path expands weights to compute dtype")
    confidence = "low"
    if parameter_count and not quantized:
        confidence = "medium"
    elif info is not None and quantized:
        confidence = "medium" if info.valid else "low"
    return ModelMemoryEstimate(
        weights_bytes=weights,
        quantization_overhead_bytes=overhead,
        resident_estimate_bytes=resident,
        offload_ram_estimate_bytes=offload_ram,
        confidence=confidence,
        assumptions=assumptions[:16],
        unknown=weights <= 0,
    )


def estimate_model_summary_memory(summary: ModelSummary) -> ModelMemoryEstimate:
    """Estimate memory from a ModelSummary without loading the model."""
    return estimate_model_memory(
        size_bytes=summary.size_bytes,
        quantization=summary.quantization,
        parameter_count=summary.parameter_count,
        is_directory=summary.is_directory,
    )
