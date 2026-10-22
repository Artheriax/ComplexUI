# /// script
# dependencies = ["torch>=2.2"]
# ///
import torch

checkpoint: MODEL
width: int = 512
height: int = 512
batch_size: int = 1

if not isinstance(checkpoint, dict):
    raise TypeError("checkpoint must be a MODEL wrapper")
if width < 8 or height < 8:
    raise ValueError("width and height must be at least 8 pixels")
if width > 16384 or height > 16384:
    raise ValueError("width and height must be at most 16384 pixels")
if batch_size < 1 or batch_size > 16:
    raise ValueError("batch_size must be between 1 and 16")
scale_factor = int(checkpoint.get("vae_scale_factor", 8))
if width % scale_factor != 0 or height % scale_factor != 0:
    raise ValueError(
        f"width and height must be multiples of {scale_factor} for this checkpoint"
    )

latent_width = width // scale_factor
latent_height = height // scale_factor
samples = torch.zeros(
    (batch_size, 4, latent_height, latent_width),
    dtype=torch.float32,
)
latent = {
    "kind": "latent",
    "samples": samples,
    "width": width,
    "height": height,
    "batch_size": batch_size,
    "vae_scale_factor": scale_factor,
}

return latent: LATENT
