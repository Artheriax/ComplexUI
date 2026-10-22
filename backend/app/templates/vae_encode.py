# /// script
# dependencies = ["torch>=2.2", "numpy>=1.26", "Pillow>=11.0"]
# ///
import numpy as np
import torch
from PIL import Image

vae: VAE
image: IMAGE
tiling: bool = False

if not isinstance(vae, dict):
    raise TypeError("vae must be a VAE wrapper")
if not isinstance(image, Image.Image):
    raise TypeError("image must be a PIL image")
vae_model = vae.get("vae")
if vae_model is None:
    raise ValueError("VAE wrapper is missing its model")
device = torch.device(str(vae.get("device", "cpu")))
dtype = vae.get("dtype", torch.float32)
scale_factor = int(vae.get("vae_scale_factor", 8))
if image.width % scale_factor != 0 or image.height % scale_factor != 0:
    raise ValueError(
        f"Image dimensions must be multiples of {scale_factor} for this VAE"
    )

report = globals().get("report_progress")
if callable(report):
    report(0.1, "encoding image")
if hasattr(vae_model, "enable_tiling"):
    if tiling:
        vae_model.enable_tiling()
    else:
        vae_model.disable_tiling()

array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
pixels = pixels.to(device=device, dtype=dtype)
with torch.inference_mode():
    distribution = vae_model.encode(pixels).latent_dist
    mode = getattr(distribution, "mode", None)
    samples = mode() if callable(mode) else distribution.mean
    samples = samples * float(vae.get("scaling_factor", 0.18215))
samples = samples.detach().cpu()
latent = {
    "kind": "latent",
    "samples": samples,
    "width": int(image.width),
    "height": int(image.height),
    "batch_size": int(samples.shape[0]),
    "vae_scale_factor": scale_factor,
}
if callable(report):
    report(1.0, "image encoded")

return latent: LATENT
