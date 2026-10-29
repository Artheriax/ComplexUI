# /// script
# dependencies = ["torch>=2.2", "numpy>=1.26", "Pillow>=11.0"]
# ///
from typing import Optional

import numpy as np
import torch
from PIL import Image

conditioning: CONDITIONING
vae: VAE
image1: Optional[IMAGE] = None
image2: Optional[IMAGE] = None
image3: Optional[IMAGE] = None
image4: Optional[IMAGE] = None
image5: Optional[IMAGE] = None
image6: Optional[IMAGE] = None
image7: Optional[IMAGE] = None
image8: Optional[IMAGE] = None
image9: Optional[IMAGE] = None
image10: Optional[IMAGE] = None
max_pixels: int = 16_000_000

if not isinstance(conditioning, dict):
    raise TypeError("conditioning must be a CONDITIONING wrapper")
if not isinstance(vae, dict):
    raise TypeError("vae must be a VAE wrapper")
vae_model = vae.get("vae")
if vae_model is None:
    raise ValueError("VAE wrapper is missing its model")
if max_pixels < 1 or max_pixels > 100_000_000:
    raise ValueError("max_pixels must be between 1 and 100,000,000")
device = torch.device(str(vae.get("device", "cpu")))
dtype = vae.get("dtype", torch.float32)
offload = bool(vae.get("offload", False))
release_device = str(vae.get("resident_device") or "cpu")
scale_factor = int(vae.get("vae_scale_factor", 8))
latent_packing = int(vae.get("latent_packing", 1))
multiple = max(1, scale_factor * latent_packing)
encoder = vae.get("encode")
if not callable(encoder) and not hasattr(vae_model, "encode"):
    raise ValueError("VAE wrapper cannot encode reference images")
if offload and hasattr(vae_model, "to"):
    vae_model.to(device)

references = [
    image
    for image in (
        image1,
        image2,
        image3,
        image4,
        image5,
        image6,
        image7,
        image8,
        image9,
        image10,
    )
    if image is not None
]
if not references:
    raise ValueError("Connect at least one reference image")
for image in references:
    if not isinstance(image, Image.Image):
        raise TypeError("reference inputs must be PIL images")

report = globals().get("report_progress")
if callable(report):
    report(0.05, "encoding reference images")

latents = []
for index, image in enumerate(references):
    if image.width * image.height > max_pixels:
        raise ValueError(
            f"Reference image {index + 1} has {image.width * image.height:,} "
            f"pixels; limit is {max_pixels:,}"
        )
    prepared = image.convert("RGB")
    target_width = max(
        multiple,
        int(round(prepared.width / multiple)) * multiple,
    )
    target_height = max(
        multiple,
        int(round(prepared.height / multiple)) * multiple,
    )
    if (target_width, target_height) != prepared.size:
        prepared = prepared.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS,
        )
    array = np.asarray(prepared, dtype=np.float32) / 127.5 - 1.0
    pixels = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    with torch.inference_mode():
        if callable(encoder):
            reference = encoder(pixels)
        else:
            distribution = vae_model.encode(pixels).latent_dist
            mode = getattr(distribution, "mode", None)
            reference = mode() if callable(mode) else distribution.mean
            reference = reference * float(vae.get("scaling_factor", 0.18215))
    if not isinstance(reference, torch.Tensor):
        raise TypeError("VAE reference encoding did not return a tensor")
    if reference.ndim not in {4, 5}:
        raise ValueError(
            "VAE reference encoding must return a 4D or 5D latent tensor"
        )
    latents.append(reference.detach().cpu())
    if callable(report):
        report(
            0.05 + 0.9 * (index + 1) / len(references),
            f"encoded reference {index + 1}/{len(references)}",
        )

if offload and hasattr(vae_model, "to"):
    vae_model.to(release_device if release_device != "cuda" else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

result = dict(conditioning)
result["ref_latents"] = latents
result["ref_count"] = len(latents)
result["ref_sizes"] = [
    (int(reference.shape[-2]), int(reference.shape[-1]))
    for reference in latents
]

conditioning = result
return conditioning: CONDITIONING
