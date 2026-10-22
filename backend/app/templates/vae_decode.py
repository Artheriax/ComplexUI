# /// script
# dependencies = ["torch>=2.2", "numpy>=1.26", "Pillow>=11.0"]
# ///
import numpy as np
import torch
from PIL import Image

vae: VAE
latent: LATENT
tiling: bool = False

if not isinstance(vae, dict):
    raise TypeError("vae must be a VAE wrapper")
if not isinstance(latent, dict):
    raise TypeError("latent must be a LATENT wrapper")
vae_model = vae.get("vae")
samples = latent.get("samples")
if vae_model is None:
    raise ValueError("VAE wrapper is missing its model")
if not isinstance(samples, torch.Tensor):
    raise TypeError("latent wrapper is missing its samples tensor")
device = torch.device(str(vae.get("device", "cpu")))
dtype = vae.get("dtype", torch.float32)

report = globals().get("report_progress")
if callable(report):
    report(0.1, "decoding latent")
if hasattr(vae_model, "enable_tiling"):
    if tiling:
        vae_model.enable_tiling()
    else:
        vae_model.disable_tiling()
scaling_factor = float(vae.get("scaling_factor", 0.18215))
with torch.inference_mode():
    decoded = vae_model.decode(samples.to(device=device, dtype=dtype) / scaling_factor).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    array = decoded.detach().cpu().permute(0, 2, 3, 1).float().numpy()
images = [Image.fromarray((frame * 255).round().astype(np.uint8)) for frame in array]
if callable(report):
    report(1.0, "images decoded")

return images: IMAGE_BATCH
