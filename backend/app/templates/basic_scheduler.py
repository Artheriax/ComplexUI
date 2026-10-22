# /// script
# dependencies = ["torch>=2.2", "scipy>=1.12"]
# ///
from typing import Any, Literal

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
full_sigmas = _schedule(base_sigmas, scheduler, int(steps))
timesteps = _timesteps_from_sigmas(full_sigmas, base_sigmas)
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
