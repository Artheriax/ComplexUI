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

if not isinstance(vae, dict):
    raise TypeError("vae must be a VAE wrapper")
if not isinstance(pixels, Image.Image):
    raise TypeError("pixels must be a PIL image")
if tile_size < 8 or overlap < 0 or temporal_size < 1 or temporal_overlap < 0:
    raise ValueError("Invalid VAE tiling parameters")
vae_model = vae.get("vae")
if vae_model is None:
    raise ValueError("VAE wrapper is missing its model")
device = torch.device(str(vae.get("device", "cpu")))
dtype = vae.get("dtype", torch.float32)
scale_factor = int(vae.get("vae_scale_factor", 8))
width = int(pixels.width)
height = int(pixels.height)
if width % scale_factor != 0 or height % scale_factor != 0:
    raise ValueError(
        f"Pixel dimensions must be multiples of {scale_factor} for this VAE"
    )

report = globals().get("report_progress")
if callable(report):
    report(0.1, "encoding image")
if hasattr(vae_model, "enable_tiling"):
    if tiled:
        vae_model.enable_tiling()
        if hasattr(vae_model, "tile_sample_min_size"):
            vae_model.tile_sample_min_size = tile_size
        if hasattr(vae_model, "tile_latent_min_size"):
            vae_model.tile_latent_min_size = max(1, tile_size // scale_factor)
        if hasattr(vae_model, "tile_overlap_factor"):
            vae_model.tile_overlap_factor = overlap / max(tile_size, 1)
        if hasattr(vae_model, "temporal_size"):
            vae_model.temporal_size = temporal_size
        if hasattr(vae_model, "temporal_overlap"):
            vae_model.temporal_overlap = temporal_overlap
    else:
        vae_model.disable_tiling()

array = np.asarray(pixels.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
pixel_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
pixel_tensor = pixel_tensor.to(device=device, dtype=dtype)
with torch.inference_mode():
    distribution = vae_model.encode(pixel_tensor).latent_dist
    mode = getattr(distribution, "mode", None)
    samples = mode() if callable(mode) else distribution.mean
    samples = samples * float(vae.get("scaling_factor", 0.18215))
samples = samples.detach().cpu()
latent = {
    "kind": "latent",
    "samples": samples,
    "width": width,
    "height": height,
    "batch_size": int(samples.shape[0]),
    "vae_scale_factor": scale_factor,
}
if callable(report):
    report(1.0, "image encoded")

return latent: LATENT
