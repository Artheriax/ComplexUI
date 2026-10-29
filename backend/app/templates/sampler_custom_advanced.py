# /// script
# dependencies = [
#   "torch>=2.2",
#   "diffusers>=0.42.0",
#   "numpy>=1.26",
#   "torchsde>=0.2.6",
#   "PyWavelets>=1.6",
# ]
# ///
from typing import Any
from contextlib import nullcontext

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

noise: NOISE
guider: GUIDER
sampler: SAMPLER
sigmas: SIGMAS
latent_image: LATENT

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
adapter = model.get("sampler_adapter")
if not isinstance(model.get("unet"), torch.nn.Module) and not isinstance(adapter, dict):
    raise ValueError(
        f"guider model architecture '{model.get('arch', 'unknown')}' is not supported yet"
    )
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
noise_type = str(noise.get("noise_type", "gaussian"))
custom_noise_generator = noise.get("generate")
if not callable(custom_noise_generator) and noise_type not in NOISE_TYPES:
    raise ValueError(f"Unsupported noise type: {noise_type}")
noise_seed = int(noise.get("noise_seed", 0))

if not flow_adapter:
    config.update(
        {
            "num_train_timesteps": int(config.get("num_train_timesteps", 1000)),
            "beta_start": float(config.get("beta_start", 0.00085)),
            "beta_end": float(config.get("beta_end", 0.012)),
            "beta_schedule": str(
                config.get("beta_schedule", "scaled_linear")
            ),
            "prediction_type": str(config.get("prediction_type", "epsilon")),
            "timestep_spacing": str(
                config.get("timestep_spacing", "leading")
            ),
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
timesteps = None
if scheduler_class is not None:
    sampler_object = scheduler_class.from_config(config)
    sampler_object.set_timesteps(step_count, device=device)
    if (
        hasattr(sampler_object, "sigmas")
        and sampler_object.sigmas is not None
        ):
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

base_cfg = float(guider.get("cfg", 7.0))
start_percent = float(guider.get("start_percent", 0.0))
end_percent = float(guider.get("end_percent", 1.0))
use_cfg = base_cfg > 1.0
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
torch.manual_seed(noise_seed)
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
if noise_mask is not None:
    latents = latents + noise_mask * generated * float(sigmas_tensor[0])
elif sampler_object is not None and hasattr(sampler_object, "scale_noise"):
    latents = sampler_object.scale_noise(latents, timesteps[0:1], generated)
elif sampler_object is not None and hasattr(sampler_object, "add_noise"):
    latents = sampler_object.add_noise(latents, generated, timesteps[0:1])
else:
    initial_sigma = float(sigmas_tensor[0])
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
        step_fraction = index / max(total_steps - 1, 1)
        step_cfg = (
            base_cfg
            if start_percent <= step_fraction <= end_percent
            else 1.0
        )
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
                step_cfg,
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
                noise_prediction = unconditioned + step_cfg * (
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
            step_kwargs = {}
            if sampler_name in {"ddpm", "ddim"}:
                step_kwargs["generator"] = torch.Generator(
                    device=device
                ).manual_seed(noise_seed + index)
            elif adapter_state is None and sampler_name in {
                "dpmpp_2m_sde",
                "dpmpp_2m_sde_gpu",
                "dpmpp_3m_sde",
                "dpmpp_3m_sde_gpu",
            }:
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
denoised_output = {**output, "kind": "latent", "samples": output["samples"].clone()}

return output: LATENT, denoised_output: LATENT
