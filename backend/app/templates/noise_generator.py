from typing import Literal

noise_type: Literal[
    "fractal",
    "gaussian",
    "gaussian_backwards",
    "uniform",
    "pyramid-cascade_B",
    "pyramid-interpolated",
    "pyramid-bilinear",
    "pyramid-bicubic",
    "pyramid-nearest",
    "hires-pyramid-bilinear",
    "hires-pyramid-bicubic",
    "hires-pyramid-nearest",
    "brownian",
    "laplacian",
    "studentt",
    "wavelet",
    "perlin",
] = "gaussian"
noise_seed: int = 0
alpha: float = 1.0
k: float = 1.0
scale: float = 0.1

if alpha < 0 or alpha > 10:
    raise ValueError("alpha must be between 0 and 10")
if k <= 0 or k > 100:
    raise ValueError("k must be greater than 0 and at most 100")
if scale < 0 or scale > 10:
    raise ValueError("scale must be between 0 and 10")

noise = {
    "kind": "noise",
    "noise_type": noise_type,
    "noise_seed": int(noise_seed),
    "alpha": float(alpha),
    "k": float(k),
    "scale": float(scale),
}

return noise: NOISE
