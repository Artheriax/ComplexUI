# /// script
# dependencies = ["torch>=2.2", "diffusers>=0.31", "numpy>=1.26", "torchsde>=0.2.6"]
# ///
from typing import Any, Literal

import torch
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

model: MODEL
positive: CONDITIONING
negative: CONDITIONING
latent_image: LATENT
seed: int = 0
control_after_generate: Literal[
    "fixed",
    "increment",
    "decrement",
    "randomize",
] = "randomize"
steps: int = 25
denoise: float = 1.0
cfg: float = 7.0
sampler: Literal[
    "euler",
    "euler_ancestral",
    "heun",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "ddpm",
    "ddim",
    "uni_pc",
] = "dpmpp_2m"
scheduler: Literal[
    "normal",
    "karras",
    "exponential",
    "beta",
] = "karras"

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


def _flow_sigmas(step_count: int, karras: bool) -> torch.Tensor:
    start, end = (0.999, 0.001) if karras else (1.0, 0.0)
    slope = (end - start) / max(step_count - 1, 1)
    values = [start + slope * index for index in range(step_count)]
    return torch.tensor(values + [0.0], dtype=torch.float32)


def _conditioning(value: Any, arch: str, device: torch.device, dtype: torch.dtype):
    if not isinstance(value, dict):
        raise TypeError("conditioning input must be a CONDITIONING wrapper")
    cond = value.get("cond")
    if not isinstance(cond, torch.Tensor):
        raise TypeError("conditioning wrapper is missing its cond tensor")
    prepared = {
        "cond": cond.to(device=device, dtype=dtype),
        "pooled": None,
    }
    pooled = value.get("pooled")
    if pooled is not None:
        prepared["pooled"] = pooled.to(device=device, dtype=dtype)
    elif arch == "sdxl":
        raise ValueError("SDXL conditioning is missing its pooled text embedding")
    return prepared


if not isinstance(model, dict):
    raise TypeError("model must be a MODEL wrapper")
if not isinstance(latent_image, dict):
    raise TypeError("latent_image must be a LATENT wrapper")
if steps < 1 or steps > 200:
    raise ValueError("steps must be between 1 and 200")
if cfg < 0 or cfg > 50:
    raise ValueError("cfg must be between 0 and 50")
if denoise <= 0 or denoise > 1:
    raise ValueError("denoise must be greater than 0 and at most 1")

arch = str(model.get("arch", ""))
device = torch.device(str(model.get("device", "cpu")))
dtype = model.get("dtype", torch.float32)
unet = model.get("unet")
scheduler_config = model.get("scheduler_config")
if unet is None:
    raise ValueError("model wrapper is missing its UNet")
if not isinstance(scheduler_config, dict):
    raise ValueError("model wrapper is missing its scheduler configuration")
latent = latent_image.get("samples")
if not isinstance(latent, torch.Tensor):
    raise TypeError("latent_image wrapper is missing its samples tensor")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(int(seed))
generator = torch.Generator(device=device).manual_seed(int(seed))
prepare = globals().get("report_progress")
cancel = globals().get("raise_if_cancelled")
if callable(prepare):
    prepare(0.02, "building sampler")

scheduler_class = SCHEDULER_CLASSES.get(sampler)
if scheduler_class is None:
    raise ValueError(f"Unsupported sampler: {sampler}")
scheduler_config = dict(scheduler_config)
scheduler_config.update(
    {
        "num_train_timesteps": int(
            scheduler_config.get("num_train_timesteps", 1000)
        ),
        "beta_start": float(scheduler_config.get("beta_start", 0.00085)),
        "beta_end": float(scheduler_config.get("beta_end", 0.012)),
        "beta_schedule": str(scheduler_config.get("beta_schedule", "scaled_linear")),
        "prediction_type": str(
            scheduler_config.get("prediction_type", "epsilon")
        ),
        "timestep_spacing": str(
            scheduler_config.get("timestep_spacing", "leading")
        ),
        "steps_offset": int(scheduler_config.get("steps_offset", 1)),
        "use_karras_sigmas": scheduler == "karras",
        "use_exponential_sigmas": scheduler == "exponential",
        "use_beta_sigmas": scheduler == "beta",
        "sigma_max": None,
        "sigma_min": None,
    }
)
if sampler in {"dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde"}:
    scheduler_config["algorithm_type"] = (
        "sde-dpmsolver++" if sampler.endswith("_sde") else "dpmsolver++"
    )
    scheduler_config["solver_order"] = 3 if "3m" in sampler else 2
sampler_object = scheduler_class.from_config(scheduler_config)
sampler_object.set_timesteps(int(steps), device=device)
if hasattr(sampler_object, "set_begin_index"):
    sampler_object.set_begin_index(0)
if hasattr(sampler_object, "_step_index"):
    sampler_object._step_index = None

timesteps = sampler_object.timesteps
if sampler in {
    "euler",
    "heun",
    "lms",
    "dpm_2",
    "dpm_2_ancestral",
    "dpmpp_2s_ancestral",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "dpmpp_sde",
} and scheduler in {"normal", "karras"}:
    sampler_object.sigmas = _flow_sigmas(
        len(timesteps), karras=scheduler == "karras"
    ).to(device)

if denoise < 1.0:
    start_index = max(0, int(round(int(steps) * (1.0 - float(denoise)))))
    timesteps = timesteps[start_index:]
    sampler_object.timesteps = timesteps
    if hasattr(sampler_object, "sigmas") and sampler_object.sigmas is not None:
        sampler_object.sigmas = sampler_object.sigmas[start_index:]
    if hasattr(sampler_object, "set_begin_index"):
        sampler_object.set_begin_index(0)
    if hasattr(sampler_object, "_step_index"):
        sampler_object._step_index = None

use_cfg = float(cfg) > 1.0
positive_conditioning = _conditioning(positive, arch, device, dtype)
negative_conditioning = _conditioning(negative, arch, device, dtype)
if positive_conditioning["cond"].shape[0] != 1:
    raise ValueError("positive conditioning must contain one batch entry")
if negative_conditioning["cond"].shape[0] != 1:
    raise ValueError("negative conditioning must contain one batch entry")
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
    raise ValueError("latent_image wrapper is missing its output dimensions")
latents = latent.to(device=device, dtype=dtype).clone()
initial_sigmas = getattr(sampler_object, "sigmas", None)
initial_sigma = (
    float(initial_sigmas[0])
    if initial_sigmas is not None and len(initial_sigmas) > 0
    else 1.0
)
noise = torch.randn(
    latents.shape,
    generator=generator,
    device=device,
    dtype=dtype,
)
latents = latents + initial_sigma * noise
batch_size = int(latents.shape[0])
total_steps = max(len(timesteps), 1)
height_ids = torch.tensor([height, width, 0, 0, height, width], device=device, dtype=dtype)
time_ids = height_ids.unsqueeze(0).expand(batch_size, -1)
if use_cfg:
    time_ids = height_ids.unsqueeze(0).expand(batch_size * 2, -1)
added_conditioning = (
    {"text_embeds": pooled_conditioning, "time_ids": time_ids}
    if pooled_conditioning is not None
    else None
)

with torch.inference_mode():
    for index, timestep in enumerate(timesteps):
        if callable(cancel):
            cancel()
        model_input = torch.cat([latents] * 2) if use_cfg else latents
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
            noise_prediction = unconditioned + float(cfg) * (
                conditioned - unconditioned
            )
        step_kwargs = (
            {"generator": generator} if sampler in {"ddpm", "ddim"} else {}
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
                f"sampling {index + 1}/{total_steps}",
            )

latent = {
    "kind": "latent",
    "samples": latents.detach().cpu(),
    "width": width,
    "height": height,
    "batch_size": batch_size,
    "vae_scale_factor": int(latent_image.get("vae_scale_factor", 8)),
}

return latent: LATENT
