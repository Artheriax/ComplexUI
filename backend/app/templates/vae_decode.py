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
device = torch.device(str(vae.get("device", "cpu")))
dtype = vae.get("dtype", torch.float32)

report = globals().get("report_progress")
if callable(report):
    report(0.1, "decoding latent")
if hasattr(vae_model, "enable_tiling"):
    if tiled:
        vae_model.enable_tiling()
        if hasattr(vae_model, "tile_sample_min_size"):
            vae_model.tile_sample_min_size = tile_size
        if hasattr(vae_model, "tile_latent_min_size"):
            vae_model.tile_latent_min_size = max(1, tile_size // 8)
        if hasattr(vae_model, "tile_overlap_factor"):
            vae_model.tile_overlap_factor = overlap / max(tile_size, 1)
        if hasattr(vae_model, "temporal_size"):
            vae_model.temporal_size = temporal_size
        if hasattr(vae_model, "temporal_overlap"):
            vae_model.temporal_overlap = temporal_overlap
    else:
        vae_model.disable_tiling()
scaling_factor = float(vae.get("scaling_factor", 0.18215))
with torch.inference_mode():
    decoded = vae_model.decode(latent_samples.to(device=device, dtype=dtype) / scaling_factor).sample
    decoded = torch.nan_to_num(decoded, nan=0.0, posinf=0.0, neginf=0.0)
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    array = decoded.detach().cpu().permute(0, 2, 3, 1).float().numpy()
image = [Image.fromarray((frame * 255).round().astype(np.uint8)) for frame in array]
if callable(report):
    report(1.0, "images decoded")

return image: IMAGE_BATCH
