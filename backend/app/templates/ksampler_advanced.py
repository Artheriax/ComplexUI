# /// script
# dependencies = ["torch>=2.2", "diffusers>=0.31", "numpy>=1.26", "scipy>=1.12", "torchsde>=0.2.6"]
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
from scipy.stats import beta as beta_distribution

model: MODEL
positive: CONDITIONING
negative: CONDITIONING
latent_image: LATENT
add_noise: bool = True
noise_seed: int = 0
control_after_generated: Literal[
    "fixed",
    "increment",
    "decrement",
    "randomize",
] = "randomize"
steps: int = 25
cfg: float = 7.0
sampler_name: Literal[
    "euler",
    "euler_ancestral",
    "heun",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "ddim",
    "uni_pc",
    "lcm",
    "ipndm",
    "ipndm_v",
] = "dpmpp_2m"
scheduler: Literal[
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
    "beta",
    "beta57",
    "linear_quadratic",
    "kl_optimal",
] = "karras"
start_at_step: int = 0
end_at_step: int = 10000
return_with_leftover_noise: bool = False

SAMPLER_CLASSES = {
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


def _base_sigmas(config: dict[str, Any]) -> torch.Tensor:
    count = int(config.get("num_train_timesteps", 1000) or 1000)
    beta_start = float(config.get("beta_start", 0.00085) or 0.00085)
    beta_end = float(config.get("beta_end", 0.012) or 0.012)
    beta_schedule = str(config.get("beta_schedule", "scaled_linear"))
    if beta_schedule == "scaled_linear":
        betas = torch.linspace(
            beta_start**0.5,
            beta_end**0.5,
            count,
            dtype=torch.float64,
        ).square()
    elif beta_schedule == "squaredcos_cap_v2":
        indexes = torch.arange(count, dtype=torch.float64)
        alphas = torch.cos(
            ((indexes / count + 0.008) / 1.008) * torch.pi / 2
        ).square()
        alphas_cumprod = alphas / alphas[0]
        return ((1 - alphas_cumprod) / alphas_cumprod).sqrt().flip(0).float()
    else:
        betas = torch.linspace(
            beta_start,
            beta_end,
            count,
            dtype=torch.float64,
        )
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)
    return ((1 - alphas_cumprod) / alphas_cumprod).sqrt().flip(0).float()


def _scheduled_sigmas(base_sigmas: torch.Tensor, scheduler_name: str, step_count: int) -> torch.Tensor:
    sigma_max = float(base_sigmas[0])
    sigma_min = float(base_sigmas[-1])
    if scheduler_name == "karras":
        rho = 7.0
        ramp = torch.linspace(0.0, 1.0, step_count, dtype=torch.float64)
        values = (
            sigma_max ** (1.0 / rho)
            + ramp * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho
    elif scheduler_name == "exponential":
        values = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(sigma_max, dtype=torch.float64)),
                torch.log(torch.tensor(sigma_min, dtype=torch.float64)),
                step_count,
            )
        )
    elif scheduler_name == "sgm_uniform":
        values = torch.linspace(sigma_max, 0.0, step_count + 1)[:-1]
    elif scheduler_name == "simple":
        indexes = (
            torch.arange(step_count, dtype=torch.float64)
            * (len(base_sigmas) - 1)
            / max(step_count - 1, 1)
        ).round().long()
        values = base_sigmas[indexes.clamp(0, len(base_sigmas) - 1)]
    elif scheduler_name == "ddim_uniform":
        indexes = (
            torch.linspace(0, len(base_sigmas) - 1, step_count)
            .round()
            .long()
        )
        values = base_sigmas[indexes]
    elif scheduler_name in {"beta", "beta57"}:
        alpha, beta = (0.6, 0.6) if scheduler_name == "beta" else (0.5, 0.7)
        quantiles = torch.linspace(
            0.001, 0.999, step_count, dtype=torch.float64
        )
        positions = torch.from_numpy(
            beta_distribution.ppf(
                quantiles.numpy(),
                alpha,
                beta,
            )
        )
        indexes = (
            positions * (len(base_sigmas) - 1)
        ).round().long().clamp(0, len(base_sigmas) - 1)
        values = base_sigmas[indexes]
    elif scheduler_name == "linear_quadratic":
        linear = torch.linspace(
            sigma_max,
            sigma_min,
            max(1, step_count // 2),
            dtype=torch.float64,
        )
        quadratic = torch.linspace(
            sigma_min,
            0.0,
            step_count - len(linear) + 1,
            dtype=torch.float64,
        )[1:].square()
        quadratic = (
            quadratic
            / max(float(quadratic[0]) if len(quadratic) else 1.0, 1e-8)
            * sigma_min
        )
        values = torch.cat((linear, quadratic))
        if len(values) < step_count:
            values = torch.cat(
                (values, torch.zeros(step_count - len(values)))
            )
        values = values[:step_count]
    elif scheduler_name == "kl_optimal":
        ramp = torch.linspace(0.0, 1.0, step_count, dtype=torch.float64)
        values = torch.tan(
            ramp * torch.atan(torch.tensor(sigma_min, dtype=torch.float64))
            + (1 - ramp)
            * torch.atan(torch.tensor(sigma_max, dtype=torch.float64))
        )
    else:
        values = torch.linspace(sigma_max, sigma_min, step_count)
    values = torch.nan_to_num(
        values.float(), nan=1e-5, posinf=1e-5, neginf=1e-5
    ).clamp(min=1e-5)
    return torch.cat((values, torch.zeros(1)))


def _timesteps_from_sigmas(sigmas: torch.Tensor, base_sigmas: torch.Tensor) -> torch.Tensor:
    """Map scheduled sigmas onto the model's training timestep grid."""
    if len(sigmas) < 2:
        raise ValueError("sigmas must contain at least two values")
    ascending = base_sigmas.flip(0).clamp(min=1e-10)
    log_base = ascending.log()
    values = sigmas[:-1].clamp(min=1e-10)
    log_values = values.log()
    high = torch.searchsorted(log_base, log_values, right=True).clamp(
        1, len(log_base) - 1
    )
    low = high - 1
    span = (log_base[high] - log_base[low]).clamp(min=1e-10)
    weight = ((log_values - log_base[low]) / span).clamp(0.0, 1.0)
    return (low + weight * (high - low)).to(torch.float32)


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
if not isinstance(latent_image, dict):
    raise TypeError("latent_image must be a LATENT wrapper")
if steps < 1 or steps > 200:
    raise ValueError("steps must be between 1 and 200")
if cfg < 0 or cfg > 50:
    raise ValueError("cfg must be between 0 and 50")
if start_at_step < 0 or end_at_step <= start_at_step:
    raise ValueError("end_at_step must be greater than start_at_step")

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
torch.manual_seed(int(noise_seed))
generator = torch.Generator(device=device).manual_seed(int(noise_seed))
prepare = globals().get("report_progress")
cancel = globals().get("raise_if_cancelled")
if callable(prepare):
    prepare(0.02, "building advanced sampler")

sampler_class = SAMPLER_CLASSES.get(sampler_name)
if sampler_class is None:
    raise ValueError(f"Unsupported sampler: {sampler_name}")
scheduler_config = dict(scheduler_config)
scheduler_config.update(
    {
        "num_train_timesteps": int(scheduler_config.get("num_train_timesteps", 1000)),
        "beta_start": float(scheduler_config.get("beta_start", 0.00085)),
        "beta_end": float(scheduler_config.get("beta_end", 0.012)),
        "beta_schedule": str(scheduler_config.get("beta_schedule", "scaled_linear")),
        "prediction_type": str(scheduler_config.get("prediction_type", "epsilon")),
        "timestep_spacing": str(scheduler_config.get("timestep_spacing", "leading")),
        "steps_offset": int(scheduler_config.get("steps_offset", 1)),
        "use_karras_sigmas": scheduler == "karras",
        "use_exponential_sigmas": scheduler == "exponential",
        "use_beta_sigmas": scheduler == "beta",
        "sigma_max": None,
        "sigma_min": None,
    }
)
if sampler_name in {
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
}:
    scheduler_config["algorithm_type"] = (
        "sde-dpmsolver++" if "_sde" in sampler_name else "dpmsolver++"
    )
    scheduler_config["solver_order"] = 3 if "3m" in sampler_name else 2
sampler_object = sampler_class.from_config(scheduler_config)
sampler_object.set_timesteps(int(steps), device=device)
timesteps = sampler_object.timesteps
native_schedules = {"normal", "karras", "exponential", "beta"}
if scheduler not in native_schedules:
    if not hasattr(sampler_object, "sigmas") or sampler_object.sigmas is None:
        raise ValueError(
            f"Scheduler '{scheduler}' is not supported by sampler '{sampler_name}'"
        )
    base_sigmas = _base_sigmas(scheduler_config)
    scheduled_sigmas = _scheduled_sigmas(
        base_sigmas,
        scheduler,
        len(timesteps),
    )
    scheduled_timesteps = _timesteps_from_sigmas(
        scheduled_sigmas, base_sigmas
    )
    if sampler_object.timesteps.dtype.is_floating_point:
        timesteps = scheduled_timesteps.to(
            device=device, dtype=sampler_object.timesteps.dtype
        )
    else:
        timesteps = scheduled_timesteps.round().to(
            device=device, dtype=sampler_object.timesteps.dtype
        )
    sampler_object.sigmas = scheduled_sigmas.to(device)
    sampler_object.timesteps = timesteps

start_index = min(int(start_at_step), len(timesteps))
end_index = min(max(int(end_at_step), start_index + 1), len(timesteps))
timesteps = timesteps[start_index:end_index]
full_sigmas = getattr(sampler_object, "sigmas", None)
if full_sigmas is not None:
    selected_sigmas = full_sigmas[start_index : end_index + 1].clone()
    if not return_with_leftover_noise:
        selected_sigmas[-1] = 0.0
    sampler_object.sigmas = selected_sigmas
else:
    selected_sigmas = None
sampler_object.timesteps = timesteps
if hasattr(sampler_object, "set_begin_index"):
    sampler_object.set_begin_index(0)
if hasattr(sampler_object, "_step_index"):
    sampler_object._step_index = None

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
    raise ValueError("latent_image wrapper is missing its output dimensions")
latents = latent.to(device=device, dtype=dtype).clone()
if add_noise:
    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if start_index > 0 and hasattr(sampler_object, "add_noise"):
        latents = sampler_object.add_noise(latents, noise, timesteps[0:1])
    else:
        if selected_sigmas is not None and len(selected_sigmas) > 0:
            initial_sigma = float(selected_sigmas[0])
            if start_index == 0:
                initial_sigma = float(
                    getattr(sampler_object, "init_noise_sigma", initial_sigma)
                )
        else:
            initial_sigma = 1.0
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
                f"advanced sampling {index + 1}/{total_steps}",
            )

latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)

latent = {
    "kind": "latent",
    "samples": latents.detach().cpu(),
    "width": width,
    "height": height,
    "batch_size": batch_size,
    "vae_scale_factor": int(latent_image.get("vae_scale_factor", 8)),
}

return latent: LATENT
