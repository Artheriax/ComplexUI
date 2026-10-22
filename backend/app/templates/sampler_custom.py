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
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
)

model: MODEL
positive: CONDITIONING
negative: CONDITIONING
sampler: SAMPLER
sigmas: SIGMAS
latent_image: LATENT
add_noise: bool = True
noise_seed: int = 0
control_after_generated: Literal[
    "fixed",
    "increment",
    "decrement",
    "randomize",
] = "randomize"
cfg: float = 7.0

SCHEDULER_CLASSES = {
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "dpm_2": KDPM2DiscreteScheduler,
    "dpm_2_ancestral": KDPM2AncestralDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "dpmpp_2s_ancestral": DPMSolverSinglestepScheduler,
    "dpmpp_sde": DPMSolverSDEScheduler,
    "dpmpp_sde_gpu": DPMSolverSDEScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "dpmpp_2m_sde": DPMSolverMultistepScheduler,
    "dpmpp_2m_sde_gpu": DPMSolverMultistepScheduler,
    "dpmpp_3m_sde": DPMSolverMultistepScheduler,
    "dpmpp_3m_sde_gpu": DPMSolverMultistepScheduler,
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "uni_pc": UniPCMultistepScheduler,
    "lcm": LCMScheduler,
    "ipndm": IPNDMScheduler,
    "ipndm_v": IPNDMScheduler,
}


def _conditioning(value: Any, arch: str, device: torch.device, dtype: torch.dtype):
    if not isinstance(value, dict):
        raise TypeError("conditioning input must be a CONDITIONING wrapper")
    cond = value.get("cond")
    if not isinstance(cond, torch.Tensor):
        raise TypeError("conditioning wrapper is missing its cond tensor")
    prepared = {"cond": cond.to(device=device, dtype=dtype), "pooled": None}
    pooled = value.get("pooled")
    if pooled is not None:
        prepared["pooled"] = pooled.to(device=device, dtype=dtype)
    elif arch == "sdxl":
        raise ValueError("SDXL conditioning is missing its pooled text embedding")
    return prepared


if not isinstance(model, dict):
    raise TypeError("model must be a MODEL wrapper")
if not isinstance(sampler, dict):
    raise TypeError("sampler must be a SAMPLER wrapper")
if not isinstance(sigmas, dict):
    raise TypeError("sigmas must be a SIGMAS wrapper")
if not isinstance(latent_image, dict):
    raise TypeError("latent_image must be a LATENT wrapper")
if cfg < 0 or cfg > 50:
    raise ValueError("cfg must be between 0 and 50")

arch = str(model.get("arch", ""))
device = torch.device(str(model.get("device", "cpu")))
dtype = model.get("dtype", torch.float32)
unet = model.get("unet")
config = model.get("scheduler_config")
if unet is None or not isinstance(config, dict):
    raise ValueError("model wrapper is missing its UNet or scheduler configuration")
sampler_name = str(sampler.get("name", ""))
custom_sampler = sampler.get("sample")
scheduler_class = SCHEDULER_CLASSES.get(sampler_name)
if scheduler_class is None and not callable(custom_sampler):
    raise ValueError(f"Unsupported sampler: {sampler_name}")
sigmas_tensor = sigmas.get("sigmas")
if not isinstance(sigmas_tensor, torch.Tensor):
    raise TypeError("sigmas wrapper is missing its sigmas tensor")
sigmas_tensor = sigmas_tensor.to(device=device, dtype=torch.float32)
if sigmas_tensor.ndim != 1 or len(sigmas_tensor) < 2:
    raise ValueError("sigmas must contain at least two values")
samples = latent_image.get("samples")
if not isinstance(samples, torch.Tensor):
    raise TypeError("latent_image wrapper is missing its samples tensor")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(int(noise_seed))
generator = torch.Generator(device=device).manual_seed(int(noise_seed))
prepare = globals().get("report_progress")
cancel = globals().get("raise_if_cancelled")
if callable(prepare):
    prepare(0.02, "building custom sampler")

config = dict(config)
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
if sampler_name in {
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
}:
    config["algorithm_type"] = (
        "sde-dpmsolver++" if "_sde" in sampler_name else "dpmsolver++"
    )
    config["solver_order"] = 3 if "3m" in sampler_name else 2
step_count = int(len(sigmas_tensor) - 1)
timesteps_tensor = sigmas.get("timesteps")
if timesteps_tensor is not None:
    if not isinstance(timesteps_tensor, torch.Tensor):
        raise TypeError("sigmas wrapper timesteps must be a tensor")
    timesteps_tensor = timesteps_tensor.to(device=device, dtype=torch.float32)
    if timesteps_tensor.ndim != 1 or len(timesteps_tensor) != step_count:
        raise ValueError("sigmas timesteps must match the sigma schedule length")
sampler_object = None
if scheduler_class is not None:
    sampler_object = scheduler_class.from_config(config)
    sampler_object.set_timesteps(step_count, device=device)
    if hasattr(sampler_object, "sigmas"):
        sampler_object.sigmas = sigmas_tensor
    if timesteps_tensor is not None:
        sampler_object.timesteps = timesteps_tensor.to(dtype=sampler_object.timesteps.dtype)
    if hasattr(sampler_object, "set_begin_index"):
        sampler_object.set_begin_index(0)
    if hasattr(sampler_object, "_step_index"):
        sampler_object._step_index = None
    timesteps = sampler_object.timesteps[:step_count]
    sampler_object.timesteps = timesteps
else:
    timesteps = (
        timesteps_tensor
        if timesteps_tensor is not None
        else torch.linspace(1, 0, step_count, device=device)
    )

use_cfg = float(cfg) > 1.0
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
if add_noise:
    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if sampler_object is not None and hasattr(sampler_object, "add_noise"):
        latents = sampler_object.add_noise(latents, noise, timesteps[0:1])
    else:
        initial_sigma = float(sigmas_tensor[0])
        latents = latents + initial_sigma * noise
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
            noise_prediction = unconditioned + float(cfg) * (
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
            step_kwargs = (
                {"generator": generator} if sampler_name in {"ddpm", "ddim"} else {}
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
                f"custom sampling {index + 1}/{total_steps}",
            )

latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)

output = {
    "kind": "latent",
    "samples": latents.detach().cpu(),
    "width": width,
    "height": height,
    "batch_size": batch_size,
    "vae_scale_factor": int(latent_image.get("vae_scale_factor", 8)),
}
denoised_output = {**output, "samples": output["samples"].clone()}

return output: LATENT, denoised_output: LATENT
