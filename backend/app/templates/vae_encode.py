# /// script
# dependencies = ["torch>=2.2", "numpy>=1.26", "Pillow>=11.0"]
# ///
import numpy as np
import torch
from PIL import Image

pixels: IMAGE
vae: VAE
tiled: bool = False
tile_size: int = 512
overlap: int = 64
temporal_size: int = 64
temporal_overlap: int = 8
auto_tiled_retry: bool = True

if not isinstance(vae, dict):
    raise TypeError("vae must be a VAE wrapper")
if not isinstance(pixels, Image.Image):
    raise TypeError("pixels must be a PIL image")
if tile_size < 8 or overlap < 0 or temporal_size < 1 or temporal_overlap < 0:
    raise ValueError("Invalid VAE tiling parameters")
vae_model = vae.get("vae")
if vae_model is None:
    raise ValueError("VAE wrapper is missing its model")
overlap = min(int(overlap), max(0, int(tile_size) // 4))
if int(temporal_size) < 2 * int(temporal_overlap):
    temporal_overlap = int(temporal_size) // 2
device = torch.device(str(vae.get("device", "cpu")))
dtype = vae.get("dtype", torch.float32)
offload = bool(vae.get("offload", False))
release_device = str(vae.get("resident_device") or "cpu")
scale_factor = int(vae.get("vae_scale_factor", 8))
latent_packing = int(vae.get("latent_packing", 1))
width = int(pixels.width)
height = int(pixels.height)
pixel_multiple = max(1, scale_factor * latent_packing)
if width % pixel_multiple != 0 or height % pixel_multiple != 0:
    raise ValueError(
        f"Pixel dimensions must be multiples of {pixel_multiple} for this VAE"
    )


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


def _encode_once(pixel_tensor, use_tiled):
    encoder = vae.get("encode")
    if callable(encoder):
        samples = encoder(
            pixel_tensor,
            tiled=bool(use_tiled),
            tile_size=int(tile_size),
            overlap=int(overlap),
            temporal_size=int(temporal_size),
            temporal_overlap=int(temporal_overlap),
        )
        if not isinstance(samples, torch.Tensor):
            raise TypeError("VAE encode callable must return a torch tensor")
        return samples.detach().cpu()
    _configure_tiling(bool(use_tiled))
    with torch.inference_mode():
        distribution = vae_model.encode(pixel_tensor).latent_dist
        mode = getattr(distribution, "mode", None)
        samples = mode() if callable(mode) else distribution.mean
        samples = samples * float(vae.get("scaling_factor", 0.18215))
    return samples.detach().cpu()


report = globals().get("report_progress")
if callable(report):
    report(0.1, "encoding image")
array = np.asarray(pixels.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
pixel_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
pixel_tensor = pixel_tensor.to(device=device, dtype=dtype)
if offload and hasattr(vae_model, "to"):
    vae_model.to(device)
try:
    use_tiled = bool(tiled)
    try:
        samples = _encode_once(pixel_tensor, use_tiled)
    except Exception as exc:
        if auto_tiled_retry and not use_tiled and _is_oom(exc):
            print("regular VAE encoding failed; retrying with tiled encoding")
            samples = _encode_once(pixel_tensor, True)
        else:
            raise
finally:
    if offload and hasattr(vae_model, "to"):
        vae_model.to(release_device if release_device != "cuda" else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

latent = {
    "kind": "latent",
    "samples": samples,
    "width": width,
    "height": height,
    "batch_size": int(samples.shape[0]),
    "vae_scale_factor": scale_factor,
    "arch": str(vae.get("arch", "")),
    "latent_space": str(vae.get("latent_space", "sd")),
    "latent_channels": int(vae.get("latent_channels", samples.shape[1])),
    "latent_packing": latent_packing,
    "scaling_factor": float(vae.get("scaling_factor", 0.18215)),
    "shift_factor": float(vae.get("shift_factor", 0.0) or 0.0),
    "latents_mean": vae.get("latents_mean"),
    "latents_std": vae.get("latents_std"),
}
if callable(report):
    report(1.0, "image encoded")

return latent: LATENT
