# /// script
# dependencies = ["torch>=2.2", "diffusers>=0.42.0", "numpy>=1.26", "torchsde>=0.2.6"]
# ///
from typing import Any, Literal
from contextlib import nullcontext

import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
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

FLOW_ARCHS = {
    "flux1",
    "flux2",
    "flux3",
    "qwen_image",
    "z_image",
    "krea2",
    "ideogram4",
    "fibo2",
    "anima",
    "moss_image",
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
adapter = model.get("sampler_adapter")
config = model.get("scheduler_config")
if (unet is None and not isinstance(adapter, dict)) or not isinstance(config, dict):
    raise ValueError(
        "model wrapper is missing its UNet, adapter or scheduler configuration"
    )
sampler_name = str(sampler.get("name", ""))
custom_sampler = sampler.get("sample")
flow_adapter = isinstance(adapter, dict) and arch in FLOW_ARCHS
scheduler_class = SCHEDULER_CLASSES.get(sampler_name)
if flow_adapter:
    if sampler_name not in {"euler", "heun"}:
        raise ValueError(
            "Flow-matching models support the euler and heun samplers"
        )
    scheduler_class = (
        FlowMatchEulerDiscreteScheduler
        if sampler_name == "euler"
        else FlowMatchHeunDiscreteScheduler
    )
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
if not flow_adapter:
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
positive_conditioning = None
negative_conditioning = None
text_conditioning = None
pooled_conditioning = None
if adapter is None:
    positive_conditioning = _conditioning(positive, arch, device, dtype)
    negative_conditioning = _conditioning(negative, arch, device, dtype)
    text_conditioning = torch.cat(
        [negative_conditioning["cond"], positive_conditioning["cond"]]
        if use_cfg
        else [positive_conditioning["cond"]],
        dim=0,
    )
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
inpaint = positive.get("inpaint") if isinstance(positive, dict) else None
if adapter is None and isinstance(positive, dict) and positive.get("ref_latents"):
    raise ValueError(
        "Reference images require a FLUX or Qwen edit-capable MODEL adapter"
    )
adapter_state = None
inpaint_channels = None
noise_mask = None
if isinstance(adapter, dict):
    references = (
        positive.get("ref_latents", [])
        if isinstance(positive, dict)
        else []
    )
    source_latent = samples
    if isinstance(inpaint, dict) and isinstance(
        inpaint.get("image_latents"), torch.Tensor
    ):
        source_latent = inpaint["image_latents"]
    adapter_state = adapter["prepare_latents"](
        source_latent,
        width,
        height,
        device,
        dtype,
        references=references,
        inpaint=inpaint,
        model=model,
    )
    latents = adapter_state["latents"].clone()
    noise_mask = adapter_state.get("noise_mask")
else:
    latents = samples.to(device=device, dtype=dtype).clone()
    if isinstance(inpaint, dict):
        mask = inpaint.get("mask")
        unet_channels = int(
            getattr(getattr(unet, "config", None), "in_channels", 4) or 4
        )
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device=device, dtype=dtype)
            noise_mask = torch.nn.functional.interpolate(
                mask,
                size=(latents.shape[2], latents.shape[3]),
                mode="area",
            )
        if unet_channels == 9:
            masked = inpaint.get("masked_latents")
            if noise_mask is None:
                raise TypeError("inpaint conditioning is missing its mask")
            if not isinstance(masked, torch.Tensor):
                raise TypeError("inpaint conditioning is missing masked latents")
            masked = masked.to(device=device, dtype=dtype)
            if masked.shape[1] != latents.shape[1]:
                raise ValueError(
                    "Inpaint masked latents do not match the model latent width"
                )
            latents = torch.zeros_like(latents)
            inpaint_channels = torch.cat([noise_mask, masked], dim=1)
            noise_mask = None
        else:
            image_latents = inpaint.get("image_latents")
            if isinstance(image_latents, torch.Tensor):
                image_latents = image_latents.to(device=device, dtype=dtype)
                if image_latents.shape != latents.shape:
                    raise ValueError(
                        "Inpaint pixels do not match the target latent shape"
                    )
                latents = image_latents.clone()
if add_noise:
    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if noise_mask is not None:
        latents = latents + noise_mask * noise * float(sigmas_tensor[0])
    elif sampler_object is not None and hasattr(sampler_object, "scale_noise"):
        latents = sampler_object.scale_noise(latents, timesteps[0:1], noise)
    elif sampler_object is not None and hasattr(sampler_object, "add_noise"):
        latents = sampler_object.add_noise(latents, noise, timesteps[0:1])
    else:
        latents = latents + float(sigmas_tensor[0]) * noise
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

resident_models = []
seen_models = set()
for resident_key in (
    "transformer",
    "unet",
    "unconditional_transformer",
    "text_conditioner",
):
    resident_value = model.get(resident_key)
    if resident_value is None or not hasattr(resident_value, "to"):
        continue
    if id(resident_value) in seen_models:
        continue
    seen_models.add(id(resident_value))
    resident_models.append(resident_value)
offload_model = bool(model.get("offload", False))
if offload_model:
    for resident_value in resident_models:
        resident_value.to(device)

try:
    from app.attention_dispatch import attention_context
except Exception:
    attention_context = None
attention_scope = nullcontext()
if attention_context is not None:
    attention_scope = attention_context(torch, model.get("attention"))

with attention_scope, torch.inference_mode():
    for index, timestep in enumerate(timesteps):
        if callable(cancel):
            cancel()
        if adapter_state is not None:
            next_timestep = (
                timesteps[index + 1]
                if index + 1 < len(timesteps)
                else None
            )
            noise_prediction = adapter["predict"](
                latents,
                adapter_state,
                timestep,
                positive,
                negative,
                float(cfg),
                next_timestep=next_timestep,
                guidance=model.get("guidance"),
                guidance_explicit=bool(model.get("guidance_explicit", False)),
                step_fraction=index / max(total_steps, 1),
            )
        else:
            model_input = torch.cat([latents] * 2) if use_cfg else latents
            if sampler_object is not None:
                model_input = sampler_object.scale_model_input(model_input, timestep)
            if inpaint_channels is not None:
                model_input = torch.cat(
                    [
                        model_input,
                        torch.cat([inpaint_channels] * 2)
                        if use_cfg
                        else inpaint_channels,
                    ],
                    dim=1,
                )
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
        if model.get("accumulation") == "fp16" and not torch.isfinite(
            noise_prediction
        ).all():
            raise FloatingPointError(
                "FP16 accumulation produced non-finite predictions; disable "
                "experimental FP16 accumulation or lower the precision risk"
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

if offload_model:
    for resident_value in resident_models:
        resident_value.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)
if adapter_state is not None:
    latents = adapter["finalize_latents"](latents, adapter_state)
    latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)

output = {
    "kind": "latent",
    "samples": latents.detach().cpu(),
    "width": width,
    "height": height,
    "batch_size": batch_size,
    "vae_scale_factor": int(
        model.get("vae_scale_factor", latent_image.get("vae_scale_factor", 8))
    ),
    "arch": str(model.get("arch", latent_image.get("arch", ""))),
    "latent_space": str(
        model.get("latent_space", latent_image.get("latent_space", "sd"))
    ),
    "latent_channels": int(
        model.get("latent_channels", latent_image.get("latent_channels", 4))
    ),
    "latent_packing": int(
        model.get("latent_packing", latent_image.get("latent_packing", 1))
    ),
    "scaling_factor": float(latent_image.get("scaling_factor", 0.18215)),
    "shift_factor": float(latent_image.get("shift_factor", 0.0) or 0.0),
    "latents_mean": latent_image.get("latents_mean"),
    "latents_std": latent_image.get("latents_std"),
}
denoised_output = {**output, "samples": output["samples"].clone()}

return output: LATENT, denoised_output: LATENT
