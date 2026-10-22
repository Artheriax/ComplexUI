# /// script
# dependencies = [
#   "torch>=2.2",
#   "diffusers>=0.31",
#   "numpy>=1.26",
#   "torchsde>=0.2.6",
#   "PyWavelets>=1.6",
# ]
# ///
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
)

guider: GUIDER
sampler: SAMPLER
sigmas: SIGMAS
noise: NOISE
latent_image: LATENT
add_noise: bool = True

SCHEDULER_CLASSES = {
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "dpm_2": KDPM2DiscreteScheduler,
    "dpm_2_ancestral": KDPM2AncestralDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "dpmpp_2s_ancestral": DPMSolverSinglestepScheduler,
    "dpmpp_sde": DPMSolverSDEScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "dpmpp_2m_sde": DPMSolverMultistepScheduler,
    "dpmpp_3m_sde": DPMSolverMultistepScheduler,
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "uni_pc": UniPCMultistepScheduler,
}

NOISE_TYPES = (
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
)


def _normalize(noise: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    noise = torch.nan_to_num(noise.float(), nan=0.0, posinf=0.0, neginf=0.0)
    noise = (noise - noise.mean()) / (noise.std() + 1e-6)
    return noise.to(dtype=dtype)


def _pyramid(shape: tuple[int, ...], generator: torch.Generator, dtype: torch.dtype, device: torch.device, mode: str, levels: int, discount: float) -> torch.Tensor:
    batch, channels, height, width = shape
    noise = torch.randn(
        shape, generator=generator, dtype=torch.float32, device=device
    ) * 0.5
    for index in range(levels):
        factor = 2 ** (index + 1)
        coarse_h = max(1, height // factor)
        coarse_w = max(1, width // factor)
        coarse = torch.randn(
            (batch, channels, coarse_h, coarse_w),
            generator=generator,
            dtype=torch.float32,
            device=device,
        ) * (0.5**index)
        if coarse_h == height and coarse_w == width:
            noise = noise + coarse
        else:
            noise = noise + F.interpolate(
                coarse, size=(height, width), mode=mode
            ) * (discount**index)
    return _normalize(noise, dtype)


def _perlin(shape: tuple[int, ...], generator: torch.Generator, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    batch, channels, height, width = shape
    noise = torch.randn(
        shape, generator=generator, dtype=torch.float32, device=device
    ) * 0.25
    for divisor in (16, 8, 4, 2):
        coarse = torch.randn(
            (batch, channels, max(1, height // divisor), max(1, width // divisor)),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        noise = noise + F.interpolate(
            coarse, size=(height, width), mode="bicubic", align_corners=False
        )
    return _normalize(noise, dtype)


def _wavelet(shape: tuple[int, ...], generator: torch.Generator, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    try:
        import pywt
    except ImportError:
        return _pyramid(shape, generator, dtype, device, "bilinear", 3, 0.7)
    raw = torch.randn(
        shape, generator=generator, dtype=torch.float32, device=device
    )
    frames = []
    for batch in range(raw.shape[0]):
        channels = []
        for channel in range(raw.shape[1]):
            coefficients = pywt.wavedecn(
                raw[batch, channel].cpu().numpy(),
                "haar",
                mode="periodization",
                level=2,
            )
            restored = pywt.waverecn(
                coefficients,
                "haar",
                mode="periodization",
            )
            channels.append(torch.from_numpy(restored))
        frames.append(torch.stack(channels))
    return _normalize(torch.stack(frames).to(device), dtype)


def _generate_noise(noise_type: str, shape: tuple[int, ...], seed: int, dtype: torch.dtype, device: torch.device, alpha: float, k: float, scale: float) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    if noise_type == "fractal":
        noise = torch.randn(
            shape, generator=generator, dtype=torch.float32, device=device
        )
        height, width = shape[-2:]
        y_freq = torch.fft.fftfreq(height, 1 / height, device=device)
        x_freq = torch.fft.fftfreq(width, 1 / width, device=device)
        frequency = torch.sqrt(
            y_freq[:, None] ** 2 + x_freq[None, :] ** 2
        ).clamp(min=1e-10)
        density = k / torch.pow(frequency, alpha * scale)
        density[0, 0] = 0
        modified = torch.fft.fftn(noise) * density
        noise = torch.fft.ifftn(modified).real
        return _normalize(noise, dtype)
    if noise_type == "gaussian":
        return _normalize(
            torch.randn(shape, generator=generator, dtype=torch.float32, device=device),
            dtype,
        )
    if noise_type == "gaussian_backwards":
        backwards = torch.Generator(device=device).manual_seed(int(seed) - 1)
        return _normalize(
            torch.randn(
                shape, generator=backwards, dtype=torch.float32, device=device
            ),
            dtype,
        )
    if noise_type == "uniform":
        return _normalize(
            torch.rand(
                shape, generator=generator, dtype=torch.float32, device=device
            )
            * 2
            - 1,
            dtype,
        )
    if noise_type == "brownian":
        noise = torch.randn(
            shape, generator=generator, dtype=torch.float32, device=device
        )
        noise = torch.cumsum(noise, dim=-1)
        return _normalize(noise, dtype)
    if noise_type == "laplacian":
        torch.manual_seed(int(seed))
        noise = torch.distributions.Laplace(0.0, 1.0).sample(shape)
        return _normalize(noise.to(device), dtype)
    if noise_type == "studentt":
        torch.manual_seed(int(seed))
        noise = torch.distributions.StudentT(2.0).sample(shape)
        limit = torch.quantile(noise.flatten(1).abs(), 0.75, dim=-1)
        limit = limit.reshape(-1, *([1] * (noise.dim() - 1)))
        noise = noise.clamp(-limit, limit)
        return _normalize(noise.to(device), dtype)
    if noise_type == "perlin":
        return _perlin(shape, generator, dtype, device)
    if noise_type == "wavelet":
        return _wavelet(shape, generator, dtype, device)
    mode = "nearest"
    levels = 4
    discount = 0.7
    if noise_type == "pyramid-cascade_B":
        levels, discount, mode = 8, 0.75, "nearest"
    elif noise_type == "pyramid-interpolated":
        levels, discount, mode = 4, 0.7, "nearest"
    elif noise_type.endswith("-bilinear"):
        mode = "bilinear"
    elif noise_type.endswith("-bicubic"):
        mode = "bicubic"
    elif noise_type.endswith("-nearest"):
        mode = "nearest"
    if noise_type.startswith("hires-"):
        levels, discount = 5, 0.65
    return _pyramid(shape, generator, dtype, device, mode, levels, discount)


def _conditioning(value: Any, arch: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("guider conditionings must be CONDITIONING wrappers")
    cond = value.get("cond")
    if not isinstance(cond, torch.Tensor):
        raise TypeError("conditioning wrapper is missing its cond tensor")
    result = {"cond": cond.to(device=device, dtype=dtype), "pooled": None}
    pooled = value.get("pooled")
    if pooled is not None:
        result["pooled"] = pooled.to(device=device, dtype=dtype)
    elif arch == "sdxl":
        raise ValueError("SDXL conditioning is missing its pooled text embedding")
    return result


if not isinstance(guider, dict):
    raise TypeError("guider must be a GUIDER wrapper")
if not isinstance(sampler, dict):
    raise TypeError("sampler must be a SAMPLER wrapper")
if not isinstance(sigmas, dict):
    raise TypeError("sigmas must be a SIGMAS wrapper")
if not isinstance(noise, dict):
    raise TypeError("noise must be a NOISE wrapper")
if not isinstance(latent_image, dict):
    raise TypeError("latent_image must be a LATENT wrapper")

model = guider.get("model")
positive = guider.get("positive")
negative = guider.get("negative")
if not isinstance(model, dict):
    raise ValueError("guider is missing its diffusion model")
if not isinstance(model.get("unet"), torch.nn.Module):
    raise ValueError("guider model is missing its UNet")
arch = str(model.get("arch", ""))
device = torch.device(str(model.get("device", "cpu")))
dtype = model.get("dtype", torch.float32)
unet = model.get("unet")
config = dict(model.get("scheduler_config") or {})
if not config:
    raise ValueError("guider model is missing its scheduler configuration")

sigmas_tensor = sigmas.get("sigmas")
if not isinstance(sigmas_tensor, torch.Tensor):
    raise TypeError("sigmas wrapper is missing its sigmas tensor")
sigmas_tensor = sigmas_tensor.to(device=device, dtype=torch.float32)
if sigmas_tensor.ndim != 1 or len(sigmas_tensor) < 2:
    raise ValueError("sigmas must contain at least two values")
samples = latent_image.get("samples")
if not isinstance(samples, torch.Tensor):
    raise TypeError("latent_image wrapper is missing its samples tensor")

sampler_name = str(sampler.get("name", ""))
custom_sampler = sampler.get("sample")
scheduler_class = SCHEDULER_CLASSES.get(sampler_name)
if scheduler_class is None and not callable(custom_sampler):
    raise ValueError(f"Unsupported sampler: {sampler_name}")
noise_type = str(noise.get("noise_type", "gaussian"))
custom_noise_generator = noise.get("generate")
if not callable(custom_noise_generator) and noise_type not in NOISE_TYPES:
    raise ValueError(f"Unsupported noise type: {noise_type}")
noise_seed = int(noise.get("seed", 0))

config.update(
    {
        "num_train_timesteps": int(config.get("num_train_timesteps", 1000)),
        "beta_start": float(config.get("beta_start", 0.00085)),
        "beta_end": float(config.get("beta_end", 0.012)),
        "beta_schedule": str(config.get("beta_schedule", "scaled_linear")),
        "prediction_type": str(config.get("prediction_type", "epsilon")),
        "timestep_spacing": str(config.get("timestep_spacing", "leading")),
        "steps_offset": int(config.get("steps_offset", 1)),
    }
)
if sampler_name in {"dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde"}:
    config["algorithm_type"] = (
        "sde-dpmsolver++" if sampler_name.endswith("_sde") else "dpmsolver++"
    )
    config["solver_order"] = 3 if "3m" in sampler_name else 2

step_count = int(len(sigmas_tensor) - 1)
sampler_object = None
timesteps = None
if scheduler_class is not None:
    sampler_object = scheduler_class.from_config(config)
    sampler_object.set_timesteps(step_count, device=device)
    if hasattr(sampler_object, "sigmas") and sampler_object.sigmas is not None:
        sampler_object.sigmas = sigmas_tensor
    if hasattr(sampler_object, "set_begin_index"):
        sampler_object.set_begin_index(0)
    if hasattr(sampler_object, "_step_index"):
        sampler_object._step_index = None
    timesteps = sampler_object.timesteps[:step_count]
    sampler_object.timesteps = timesteps
else:
    timesteps = torch.linspace(1, 0, step_count, device=device)

use_cfg = float(guider.get("cfg", 7.0)) > 1.0
positive_conditioning = _conditioning(positive, arch, device, dtype)
negative_conditioning = _conditioning(negative, arch, device, dtype)
text_conditioning = torch.cat(
    [negative_conditioning["cond"], positive_conditioning["cond"]]
    if use_cfg
    else [positive_conditioning["cond"]],
    dim=0,
)
pooled_conditioning = None
if arch == "sdxl":
    if positive_conditioning["pooled"] is None or negative_conditioning["pooled"] is None:
        raise ValueError("SDXL conditioning is missing pooled text embeddings")
    pooled_conditioning = torch.cat(
        [negative_conditioning["pooled"], positive_conditioning["pooled"]]
        if use_cfg
        else [positive_conditioning["pooled"]],
        dim=0,
    )

height = int(latent_image.get("height", 0))
width = int(latent_image.get("width", 0))
if height <= 0 or width <= 0:
    raise ValueError("latent_image wrapper is missing its dimensions")
latents = samples.to(device=device, dtype=dtype).clone()
torch.manual_seed(noise_seed)
if add_noise:
    if callable(custom_noise_generator):
        generated = custom_noise_generator(
            shape=tuple(latents.shape),
            seed=noise_seed,
            dtype=dtype,
            device=device,
        )
        generated = generated.to(device=device, dtype=dtype)
    else:
        generated = _generate_noise(
            noise_type,
            tuple(latents.shape),
            noise_seed,
            dtype,
            device,
            float(noise.get("alpha", 1.0)),
            float(noise.get("k", 1.0)),
            float(noise.get("scale", 0.1)),
        )
    unit_init = {
        "ddpm",
        "ddim",
        "dpmpp_2s_ancestral",
        "dpmpp_2m",
        "dpmpp_2m_sde",
        "dpmpp_3m_sde",
        "uni_pc",
    }
    initial_sigma = (
        float(sigmas_tensor[0]) if sampler_name not in unit_init else 1.0
    )
    latents = latents + initial_sigma * generated

batch_size = int(latents.shape[0])
total_steps = max(len(timesteps), 1)
height_ids = torch.tensor(
    [height, width, 0, 0, height, width],
    device=device,
    dtype=dtype,
)
time_ids = height_ids.unsqueeze(0).expand(batch_size, -1)
if use_cfg:
    time_ids = height_ids.unsqueeze(0).expand(batch_size * 2, -1)
added_conditioning = (
    {"text_embeds": pooled_conditioning, "time_ids": time_ids}
    if pooled_conditioning is not None
    else None
)

prepare = globals().get("report_progress")
cancel = globals().get("raise_if_cancelled")
if callable(prepare):
    prepare(0.02, "advanced sampler ready")

with torch.inference_mode():
    for index, timestep in enumerate(timesteps):
        if callable(cancel):
            cancel()
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        if sampler_object is not None:
            model_input = sampler_object.scale_model_input(model_input, timestep)
        noise_prediction = unet(
            model_input,
            timestep.reshape(1),
            encoder_hidden_states=text_conditioning,
            added_cond_kwargs=added_conditioning,
        )
        if hasattr(noise_prediction, "sample"):
            noise_prediction = noise_prediction.sample
        if use_cfg:
            unconditioned, conditioned = noise_prediction.chunk(2)
            noise_prediction = unconditioned + float(guider.get("cfg", 7.0)) * (
                conditioned - unconditioned
            )
        if callable(custom_sampler):
            latents = custom_sampler(
                model=model,
                noise_prediction=noise_prediction,
                timestep=timestep,
                latents=latents,
                sigma=sigmas_tensor[index],
                sigma_next=sigmas_tensor[index + 1],
                step=index,
                total_steps=total_steps,
            )
        else:
            step_kwargs = {}
            if sampler_name in {"ddpm", "ddim"}:
                step_kwargs["generator"] = torch.Generator(
                    device=device
                ).manual_seed(noise_seed + index)
            elif sampler_name in {"dpmpp_2m_sde", "dpmpp_3m_sde"}:
                if callable(custom_noise_generator):
                    variance_noise = custom_noise_generator(
                        shape=tuple(latents.shape),
                        seed=noise_seed + index + 1,
                        dtype=dtype,
                        device=device,
                    )
                else:
                    variance_noise = _generate_noise(
                        noise_type,
                        tuple(latents.shape),
                        noise_seed + index + 1,
                        dtype,
                        device,
                        float(noise.get("alpha", 1.0)),
                        float(noise.get("k", 1.0)),
                        float(noise.get("scale", 0.1)),
                    )
                step_kwargs["variance_noise"] = variance_noise.to(
                    device=device, dtype=torch.float32
                )
            latents = sampler_object.step(
                noise_prediction,
                timestep,
                latents,
                return_dict=False,
                **step_kwargs,
            )[0]
        latents = latents.to(dtype=dtype)
        if callable(prepare):
            prepare(
                0.05 + 0.9 * (index + 1) / total_steps,
                f"advanced sampling {index + 1}/{total_steps}",
            )

output = {
    "kind": "latent",
    "samples": latents.detach().cpu(),
    "width": width,
    "height": height,
    "batch_size": batch_size,
    "vae_scale_factor": int(latent_image.get("vae_scale_factor", 8)),
}
denoised = {**output, "kind": "latent", "samples": output["samples"].clone()}

return output: LATENT, denoised: LATENT
