# /// script
# dependencies = ["torch>=2.2", "diffusers>=0.42.0", "numpy>=1.26", "scipy>=1.12", "torchsde>=0.2.6"]
# ///
from typing import Any, Literal

import math
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


def _flow_mu(
    config: dict[str, Any],
    arch: str,
    width: int,
    height: int,
    distilled: bool,
    steps: int,
):
    if arch != "fibo2" and not bool(
        config.get("use_dynamic_shifting", False)
    ):
        return None
    if arch == "krea2" and distilled:
        return 1.15
    scale = 8 if arch in {"z_image", "anima"} else 16
    image_seq_len = max(1, (height // scale) * (width // scale))
    if arch == "flux2":
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666
        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        slope = (m_200 - m_10) / 190.0
        intercept = m_200 - 200.0 * slope
        return float(slope * steps + intercept)
    base_seq = int(config.get("base_image_seq_len", 256))
    max_seq = int(config.get("max_image_seq_len", 4096))
    base_shift = float(config.get("base_shift", 0.5))
    max_shift = float(config.get("max_shift", 1.15))
    if max_seq <= base_seq:
        return max_shift
    slope = (max_shift - base_shift) / (max_seq - base_seq)
    return float(image_seq_len * slope + (base_shift - slope * base_seq))


def _linear_flow_sigmas(steps: int) -> torch.Tensor:
    """Linear flow schedule used by Anima and Fibo 2 inference."""
    count = max(int(steps), 1)
    return torch.cat(
        (
            torch.linspace(
                1.0,
                1.0 / count,
                count,
                dtype=torch.float64,
            ).float(),
            torch.zeros(1),
        )
    )


def _shift_flow_sigmas(values: torch.Tensor, mu: float) -> torch.Tensor:
    """Apply the exponential dynamic shift used by flow schedulers."""
    exponential = math.exp(float(mu))
    return exponential / (exponential + (1.0 / values - 1.0))


def _moss_sigmas(config: dict[str, Any], steps: int) -> torch.Tensor:
    """MOSS Image's shifted schedule snapped to its training grid."""
    count = int(config.get("num_train_timesteps", 1000) or 1000)
    shift = float(config.get("shift", 5.0) or 5.0)
    training_shift = float(config.get("training_shift", 1.0) or 1.0)

    def grid(size: int, scale: float) -> torch.Tensor:
        lower = scale / (count + scale - 1)
        values = torch.linspace(1.0, lower, size, dtype=torch.float64)
        return scale * values / (1.0 + (scale - 1.0) * values)

    sigmas = grid(max(int(steps), 1), shift)
    training = grid(count, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=count - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower]).abs()
        <= (training[upper] - sigmas).abs(),
        lower,
        upper,
    )
    snapped = torch.unique_consecutive(training[nearest])
    return torch.cat((snapped.float(), torch.zeros(1)))


def _ideogram4_sigmas(
    config: dict[str, Any],
    width: int,
    height: int,
    steps: int,
) -> torch.Tensor:
    """Ideogram 4's resolution-aware logit-normal flow schedule."""
    intervals = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    z = torch.special.ndtri(intervals)
    base_mu = float(config.get("base_mu", 0.0) or 0.0)
    mu = base_mu + 0.5 * math.log(max(width * height, 1) / (512 * 512))
    t_min = 1.0 / (1.0 + math.exp(0.5 * 18.0))
    t_max = 1.0 / (1.0 + math.exp(0.5 * -15.0))
    values = (1.0 - torch.special.expit(mu + 1.5 * z)).clamp(
        t_min,
        t_max,
    )
    sigmas = (1.0 - values).flip(0)[:-1]
    return torch.cat((sigmas.float(), torch.zeros(1)))


def _flow_base_sigmas(config: dict[str, Any]) -> torch.Tensor:
    """Flow-matching training sigma grid used by non-native schedules."""
    count = int(config.get("num_train_timesteps", 1000) or 1000)
    shift = float(config.get("shift", 1.0) or 1.0)
    values = torch.linspace(
        1.0,
        1.0 / max(count, 1),
        count,
        dtype=torch.float64,
    )
    if shift > 0 and shift != 1.0:
        values = shift * values / (1.0 + (shift - 1.0) * values)
    return values.float()


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
adapter = model.get("sampler_adapter")
scheduler_config = model.get("scheduler_config")
if unet is None and not isinstance(adapter, dict):
    raise ValueError(
        f"Architecture '{arch or 'unknown'}' is not supported by this sampler yet"
    )
if not isinstance(scheduler_config, dict):
    raise ValueError("model wrapper is missing its scheduler configuration")
latent = latent_image.get("samples")
if not isinstance(latent, torch.Tensor):
    raise TypeError("latent_image wrapper is missing its samples tensor")
schedule_width = int(latent_image.get("width", 0))
schedule_height = int(latent_image.get("height", 0))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(int(noise_seed))
generator = torch.Generator(device=device).manual_seed(int(noise_seed))
prepare = globals().get("report_progress")
cancel = globals().get("raise_if_cancelled")
if callable(prepare):
    prepare(0.02, "building advanced sampler")

flow_adapter = isinstance(adapter, dict) and arch in FLOW_ARCHS
sampler_class = SAMPLER_CLASSES.get(sampler_name)
if flow_adapter:
    if sampler_name not in {"euler", "heun"}:
        raise ValueError(
            "Flow-matching models support the euler and heun samplers"
        )
    sampler_class = (
        FlowMatchEulerDiscreteScheduler
        if sampler_name == "euler"
        else FlowMatchHeunDiscreteScheduler
    )
if sampler_class is None:
    raise ValueError(f"Unsupported sampler: {sampler_name}")
scheduler_config = dict(scheduler_config)
if not flow_adapter:
    scheduler_config.update(
        {
            "num_train_timesteps": int(
                scheduler_config.get("num_train_timesteps", 1000)
            ),
            "beta_start": float(scheduler_config.get("beta_start", 0.00085)),
            "beta_end": float(scheduler_config.get("beta_end", 0.012)),
            "beta_schedule": str(
                scheduler_config.get("beta_schedule", "scaled_linear")
            ),
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
if flow_adapter:
    flow_mu = _flow_mu(
        scheduler_config,
        arch,
        schedule_width,
        schedule_height,
        bool(model.get("is_distilled", False)),
        int(steps),
    )
    sampler_object.set_timesteps(
        int(steps),
        device=device,
        **({"mu": flow_mu} if flow_mu is not None else {}),
    )
else:
    sampler_object.set_timesteps(int(steps), device=device)
timesteps = sampler_object.timesteps
native_schedules = {"normal", "karras", "exponential", "beta"}
if not flow_adapter and scheduler not in native_schedules:
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
if flow_adapter and scheduler not in {"normal", "simple"}:
    flow_sigmas = _flow_base_sigmas(scheduler_config)
    scheduled_sigmas = _scheduled_sigmas(
        flow_sigmas,
        scheduler,
        len(timesteps),
    )
    flow_timesteps = (scheduled_sigmas[:-1] * 1000.0).to(
        device=device,
        dtype=sampler_object.timesteps.dtype,
    )
    sampler_object.sigmas = scheduled_sigmas.to(device)
    sampler_object.timesteps = flow_timesteps
    timesteps = flow_timesteps
if (
    flow_adapter
    and scheduler in {"normal", "simple"}
    and arch in {"anima", "fibo2"}
):
    scheduled_sigmas = _linear_flow_sigmas(int(steps))
    if arch == "fibo2" and flow_mu is not None:
        scheduled_sigmas = torch.cat(
            (
                _shift_flow_sigmas(scheduled_sigmas[:-1], flow_mu),
                torch.zeros(1),
            )
        )
    flow_timesteps = (scheduled_sigmas[:-1] * 1000.0).to(
        device=device,
        dtype=sampler_object.timesteps.dtype,
    )
    sampler_object.sigmas = scheduled_sigmas.to(device)
    sampler_object.timesteps = flow_timesteps
    timesteps = flow_timesteps
if (
    flow_adapter
    and scheduler in {"normal", "simple"}
    and arch == "moss_image"
):
    scheduled_sigmas = _moss_sigmas(scheduler_config, int(steps))
    flow_timesteps = (scheduled_sigmas[:-1] * 1000.0).to(
        device=device,
        dtype=sampler_object.timesteps.dtype,
    )
    sampler_object.sigmas = scheduled_sigmas.to(device)
    sampler_object.timesteps = flow_timesteps
    timesteps = flow_timesteps
if (
    flow_adapter
    and arch == "ideogram4"
    and scheduler in {"normal", "simple"}
):
    scheduled_sigmas = _ideogram4_sigmas(
        scheduler_config,
        schedule_width,
        schedule_height,
        int(steps),
    )
    flow_timesteps = (scheduled_sigmas[:-1] * 1000.0).to(
        device=device,
        dtype=sampler_object.timesteps.dtype,
    )
    sampler_object.sigmas = scheduled_sigmas.to(device)
    sampler_object.timesteps = flow_timesteps
    timesteps = flow_timesteps

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
    raise ValueError("latent_image wrapper is missing its output dimensions")
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
    source_latent = latent
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
    latents = latent.to(device=device, dtype=dtype).clone()
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
        latents = latents + noise_mask * noise * (
            float(selected_sigmas[0]) if selected_sigmas is not None else 1.0
        )
    elif start_index > 0 and hasattr(sampler_object, "add_noise"):
        latents = sampler_object.add_noise(latents, noise, timesteps[0:1])
    elif adapter_state is not None and hasattr(sampler_object, "scale_noise"):
        latents = sampler_object.scale_noise(latents, timesteps[0:1], noise)
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

if offload_model:
    for resident_value in resident_models:
        resident_value.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)
if adapter_state is not None:
    latents = adapter["finalize_latents"](latents, adapter_state)
    latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)

latent = {
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

return latent: LATENT
