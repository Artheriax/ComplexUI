# /// script
# dependencies = ["torch>=2.2", "scipy>=1.12"]
# ///
from typing import Any, Literal

import math

import torch
from scipy.stats import beta as beta_distribution

model: MODEL
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
steps: int = 25
denoise: float = 1.0


def _sigmas_from_config(config: dict[str, Any]) -> torch.Tensor:
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
        )
        betas = betas.square()
    elif beta_schedule == "squaredcos_cap_v2":
        steps_index = torch.arange(count, dtype=torch.float64)
        alphas = torch.cos(
            ((steps_index / count + 0.008) / 1.008) * torch.pi / 2
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
    mu = base_mu + 0.5 * math.log(
        max(width * height, 1) / (512 * 512)
    )
    t_min = 1.0 / (1.0 + math.exp(0.5 * 18.0))
    t_max = 1.0 / (1.0 + math.exp(0.5 * -15.0))
    values = (1.0 - torch.special.expit(mu + 1.5 * z)).clamp(
        t_min,
        t_max,
    )
    sigmas = (1.0 - values).flip(0)[:-1]
    return torch.cat((sigmas.float(), torch.zeros(1)))


def _flow_sigmas_from_config(
    config: dict[str, Any],
    mu: float | None = None,
) -> torch.Tensor:
    """Training sigma grid for flow-matching models (1.0 down to one step)."""
    count = int(config.get("num_train_timesteps", 1000) or 1000)
    shift = float(config.get("shift", 1.0) or 1.0)
    values = torch.linspace(
        1.0,
        1.0 / max(count, 1),
        count,
        dtype=torch.float64,
    )
    if mu is not None:
        if str(config.get("time_shift_type", "exponential")) == "linear":
            values = mu / (mu + (1.0 / values - 1.0))
        else:
            exponential = math.exp(mu)
            values = exponential / (
                exponential + (1.0 / values - 1.0)
            )
    elif shift > 0 and shift != 1.0:
        values = shift * values / (1.0 + (shift - 1.0) * values)
    return values.float()


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


def _schedule(base_sigmas: torch.Tensor, scheduler_name: str, step_count: int) -> torch.Tensor:
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
        indexes = torch.linspace(
            0,
            len(base_sigmas) - 1,
            step_count,
        ).round().long()
        values = base_sigmas[indexes]
    elif scheduler_name in {"beta", "beta57"}:
        alpha, beta = (0.6, 0.6) if scheduler_name == "beta" else (0.5, 0.7)
        quantiles = torch.linspace(
            0.001, 0.999, step_count, dtype=torch.float64
        )
        positions = torch.from_numpy(
            beta_distribution.ppf(quantiles.numpy(), alpha, beta)
        )
        indexes = (
            positions * (len(base_sigmas) - 1)
        ).round().long().clamp(0, len(base_sigmas) - 1)
        values = base_sigmas[indexes]
    elif scheduler_name == "linear_quadratic":
        linear_steps = max(1, step_count // 2)
        linear = torch.linspace(
            sigma_max,
            sigma_min,
            linear_steps,
            dtype=torch.float64,
        )
        quadratic_steps = step_count - linear_steps + 1
        quadratic = torch.linspace(
            0.0,
            1.0,
            quadratic_steps,
            dtype=torch.float64,
        )[1:].square()
        values = torch.cat(
            (
                linear,
                sigma_min * (1.0 - quadratic),
            )
        )[:step_count]
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


if not isinstance(model, dict):
    raise TypeError("model must be a MODEL wrapper")
config = model.get("scheduler_config")
if not isinstance(config, dict):
    raise ValueError("model wrapper is missing its scheduler configuration")
if steps < 1 or steps > 200:
    raise ValueError("steps must be between 1 and 200")
if denoise <= 0 or denoise > 1:
    raise ValueError("denoise must be greater than 0 and at most 1")

base_sigmas = _sigmas_from_config(config)
flow_adapter = (
    isinstance(model.get("sampler_adapter"), dict)
    and str(model.get("arch", ""))
    in {
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
)
arch = str(model.get("arch", ""))
if flow_adapter:
    recommended = model.get("recommended")
    if not isinstance(recommended, dict):
        recommended = {}
    flow_mu = _flow_mu(
        config,
        arch,
        int(recommended.get("width", 1024)),
        int(recommended.get("height", 1024)),
        bool(model.get("is_distilled", False)),
        int(steps),
    )
    if scheduler in {"normal", "simple"} and arch in {"anima", "fibo2"}:
        base_sigmas = _linear_flow_sigmas(int(steps))
        if arch == "fibo2" and flow_mu is not None:
            base_sigmas = torch.cat(
                (
                    _shift_flow_sigmas(base_sigmas[:-1], flow_mu),
                    torch.zeros(1),
                )
            )
        full_sigmas = base_sigmas
        timesteps = (full_sigmas[:-1] * 1000.0).to(torch.float32)
    elif scheduler in {"normal", "simple"} and arch == "moss_image":
        base_sigmas = _moss_sigmas(config, int(steps))
        full_sigmas = base_sigmas
        timesteps = (full_sigmas[:-1] * 1000.0).to(torch.float32)
    else:
        base_sigmas = _flow_sigmas_from_config(config, flow_mu)
        full_sigmas = _schedule(base_sigmas, scheduler, int(steps))
        timesteps = (full_sigmas[:-1] * 1000.0).to(torch.float32)
else:
    full_sigmas = _schedule(base_sigmas, scheduler, int(steps))
    timesteps = _timesteps_from_sigmas(full_sigmas, base_sigmas)
if (
    flow_adapter
    and arch == "ideogram4"
    and scheduler in {"normal", "simple"}
):
    recommended = model.get("recommended")
    if not isinstance(recommended, dict):
        recommended = {}
    full_sigmas = _ideogram4_sigmas(
        config,
        int(recommended.get("width", 2048)),
        int(recommended.get("height", 2048)),
        int(steps),
    )
    timesteps = (full_sigmas[:-1] * 1000.0).to(torch.float32)
if denoise < 1.0:
    kept_steps = max(1, int(round(int(steps) * float(denoise))))
    full_sigmas = full_sigmas[-(kept_steps + 1) :]
    timesteps = timesteps[-kept_steps:]
sigmas = {
    "kind": "sigmas",
    "sigmas": full_sigmas,
    "timesteps": timesteps,
    "steps": int(len(full_sigmas) - 1),
    "scheduler": scheduler,
    "denoise": float(denoise),
    "sigma_max": float(base_sigmas[0]),
    "sigma_min": float(base_sigmas[-1]),
    "arch": str(model.get("arch", "")),
}

return sigmas: SIGMAS
