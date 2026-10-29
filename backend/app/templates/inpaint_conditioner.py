# /// script
# dependencies = ["torch>=2.2", "numpy>=1.26", "Pillow>=11.0"]
# ///
import base64
import io
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter

positive: CONDITIONING
negative: CONDITIONING
vae: VAE
pixels: IMAGE
mask: Optional[IMAGE] = None
masks: Optional[MASKS] = None
grow_mask_by: int = 6

if not isinstance(positive, dict) or not isinstance(negative, dict):
    raise TypeError("positive and negative must be CONDITIONING wrappers")
if not isinstance(vae, dict):
    raise TypeError("vae must be a VAE wrapper")
if not isinstance(pixels, Image.Image):
    raise TypeError("pixels must be a PIL image")
if mask is None and masks is None:
    raise ValueError("Connect either mask or masks")
if mask is not None and masks is not None:
    raise ValueError("Connect either mask or masks, not both")
if mask is not None and not isinstance(mask, Image.Image):
    raise TypeError("mask must be a PIL image")
if grow_mask_by < 0 or grow_mask_by > 256:
    raise ValueError("grow_mask_by must be between 0 and 256")
if masks is not None:
    if not isinstance(masks, dict):
        raise TypeError("masks must be a MASKS value")
    entries = masks.get("masks")
    if not isinstance(entries, list) or not entries:
        raise ValueError("masks does not contain any mask candidates")
    encoded = entries[0].get("mask") if isinstance(entries[0], dict) else None
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("the selected segmentation mask is missing PNG data")
    try:
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as source:
            source.load()
            mask = source.convert("L")
    except (OSError, ValueError) as exc:
        raise ValueError("the selected segmentation mask is not a valid image") from exc
vae_model = vae.get("vae")
if vae_model is None:
    raise ValueError("VAE wrapper is missing its model")
device = torch.device(str(vae.get("device", "cpu")))
dtype = vae.get("dtype", torch.float32)
offload = bool(vae.get("offload", False))
release_device = str(vae.get("resident_device") or "cpu")
encoder = vae.get("encode")
if not callable(encoder) and not hasattr(vae_model, "encode"):
    raise ValueError("VAE wrapper cannot encode inpainting pixels")
scale_factor = int(vae.get("vae_scale_factor", 8))
latent_packing = int(vae.get("latent_packing", 1))
pixel_multiple = max(1, scale_factor * latent_packing)
if pixels.width % pixel_multiple or pixels.height % pixel_multiple:
    raise ValueError(
        f"Pixel dimensions must be multiples of {pixel_multiple} for this VAE"
    )

report = globals().get("report_progress")
if callable(report):
    report(0.05, "preparing inpainting mask")

mask_image = mask.convert("L")
if mask_image.size != pixels.size:
    mask_image = mask_image.resize(pixels.size, Image.Resampling.NEAREST)
if grow_mask_by:
    mask_image = mask_image.filter(
        ImageFilter.MaxFilter(grow_mask_by * 2 + 1)
    )


def encode_image(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    tensor = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    with torch.inference_mode():
        if callable(encoder):
            latent = encoder(tensor)
        else:
            distribution = vae_model.encode(tensor).latent_dist
            mode = getattr(distribution, "mode", None)
            latent = mode() if callable(mode) else distribution.mean
            shift = float(vae.get("shift_factor", 0.0) or 0.0)
            latent = (latent - shift) * float(
                vae.get("scaling_factor", 0.18215)
            )
    if not isinstance(latent, torch.Tensor):
        raise TypeError("VAE encode callable must return a torch tensor")
    if latent.ndim not in {4, 5}:
        raise ValueError("VAE encoding must return a 4D or 5D latent tensor")
    return latent.detach().cpu()


mask_array = np.asarray(mask_image, dtype=np.float32) / 255.0
mask_tensor = (
    torch.from_numpy(mask_array)
    .unsqueeze(0)
    .unsqueeze(0)
    .clamp(0.0, 1.0)
)
masked_pixels = Image.fromarray(
    (
        np.asarray(pixels.convert("RGB"), dtype=np.float32)
        * (1.0 - mask_array[..., None])
    )
    .round()
    .astype(np.uint8)
)

if offload and hasattr(vae_model, "to"):
    vae_model.to(device)
image_latents = encode_image(pixels)
masked_latents = encode_image(masked_pixels)
if offload and hasattr(vae_model, "to"):
    vae_model.to(release_device if release_device != "cuda" else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
if callable(report):
    report(0.75, "encoded inpainting pixels")

inpaint = {
    "kind": "inpaint",
    "mask": mask_tensor,
    "image_latents": image_latents,
    "masked_latents": masked_latents,
    "width": int(pixels.width),
    "height": int(pixels.height),
    "grow_mask_by": int(grow_mask_by),
}

positive_conditioning = dict(positive)
positive_conditioning["inpaint"] = inpaint
negative_conditioning = dict(negative)
negative_conditioning["inpaint"] = dict(inpaint)

if callable(report):
    report(1.0, "inpainting conditioning ready")

positive = positive_conditioning
negative = negative_conditioning
return positive: CONDITIONING, negative: CONDITIONING
