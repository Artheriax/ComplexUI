# /// script
# dependencies = ["torch>=2.2", "numpy>=1.26", "Pillow>=11.0"]
# ///
import numpy as np
import torch
from PIL import Image

samples: LATENT
vae: VAE
tiled: bool = False
tile_size: int = 512
overlap: int = 64
temporal_size: int = 64
temporal_overlap: int = 8
sliced: bool = False
auto_tiled_retry: bool = True

if not isinstance(vae, dict):
    raise TypeError("vae must be a VAE wrapper")
if not isinstance(samples, dict):
    raise TypeError("samples must be a LATENT wrapper")
vae_model = vae.get("vae")
latent_samples = samples.get("samples")
if tile_size < 8 or overlap < 0 or temporal_size < 1 or temporal_overlap < 0:
    raise ValueError("Invalid VAE tiling parameters")
if vae_model is None:
    raise ValueError("VAE wrapper is missing its model")
if not isinstance(latent_samples, torch.Tensor):
    raise TypeError("samples wrapper is missing its samples tensor")
latent_samples = torch.nan_to_num(
    latent_samples,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
overlap = min(int(overlap), max(0, int(tile_size) // 4))
if int(temporal_size) < 2 * int(temporal_overlap):
    temporal_overlap = int(temporal_size) // 2
device = torch.device(str(vae.get("device", "cpu")))
dtype = vae.get("dtype", torch.float32)
offload = bool(vae.get("offload", False))
release_device = str(vae.get("resident_device") or "cpu")
scale_factor = int(vae.get("vae_scale_factor", 8))
scaling_factor = float(vae.get("scaling_factor", 0.18215))


def _is_oom(exc):
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()


def _configure_tiling(enabled):
    if not hasattr(vae_model, "enable_tiling"):
        return
    if not enabled:
        if hasattr(vae_model, "disable_tiling"):
            vae_model.disable_tiling()
        return
    vae_model.enable_tiling()
    if hasattr(vae_model, "tile_sample_min_size"):
        vae_model.tile_sample_min_size = int(tile_size)
    if hasattr(vae_model, "tile_latent_min_size"):
        vae_model.tile_latent_min_size = max(1, int(tile_size) // scale_factor)
    if hasattr(vae_model, "tile_overlap_factor"):
        vae_model.tile_overlap_factor = int(overlap) / max(int(tile_size), 1)
    if hasattr(vae_model, "temporal_size"):
        vae_model.temporal_size = int(temporal_size)
    if hasattr(vae_model, "temporal_overlap"):
        vae_model.temporal_overlap = int(temporal_overlap)


def _decode_once(sample, use_tiled):
    decoder = vae.get("decode")
    if callable(decoder):
        result = decoder(
            sample,
            tiled=bool(use_tiled),
            tile_size=int(tile_size),
            overlap=int(overlap),
            temporal_size=int(temporal_size),
            temporal_overlap=int(temporal_overlap),
        )
        if not isinstance(result, torch.Tensor):
            raise TypeError("VAE decode callable must return a torch tensor")
        return result
    _configure_tiling(bool(use_tiled))
    with torch.inference_mode():
        decoded = vae_model.decode(
            sample.to(device=device, dtype=dtype) / scaling_factor
        ).sample
    return torch.nan_to_num(decoded, nan=0.0, posinf=0.0, neginf=0.0)


report = globals().get("report_progress")
if callable(report):
    report(0.1, "decoding latent")
if offload and hasattr(vae_model, "to"):
    vae_model.to(device)
try:
    items = (
        [latent_samples[index : index + 1] for index in range(latent_samples.shape[0])]
        if sliced and latent_samples.shape[0] > 1
        else [latent_samples]
    )
    decoded_parts = []
    for item in items:
        use_tiled = bool(tiled)
        try:
            decoded_parts.append(_decode_once(item, use_tiled))
        except Exception as exc:
            if auto_tiled_retry and not use_tiled and _is_oom(exc):
                print("regular VAE decoding failed; retrying with tiled decoding")
                decoded_parts.append(_decode_once(item, True))
            else:
                raise
    decoded = (
        decoded_parts[0]
        if len(decoded_parts) == 1
        else torch.cat(decoded_parts, dim=0)
    )
finally:
    if offload and hasattr(vae_model, "to"):
        vae_model.to(release_device if release_device != "cuda" else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if decoded.ndim == 5:
    decoded = decoded[:, :, 0]
decoded = torch.nan_to_num(decoded, nan=0.0, posinf=0.0, neginf=0.0)
decoded = (decoded / 2 + 0.5).clamp(0, 1)
array = decoded.detach().cpu().permute(0, 2, 3, 1).float().numpy()
image = [Image.fromarray((frame * 255).round().astype(np.uint8)) for frame in array]
if callable(report):
    report(1.0, "images decoded")

return image: IMAGE_BATCH
