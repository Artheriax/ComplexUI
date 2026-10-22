from typing import Literal

sampler_name: Literal[
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

sampler = {
    "kind": "sampler",
    "name": sampler_name,
    "sample": None,
}

return sampler: SAMPLER
