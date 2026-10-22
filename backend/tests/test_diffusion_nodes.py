"""Focused tests for the core diffusion node templates and previews."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers import EulerDiscreteScheduler
from PIL import Image

from app.executor import NON_CACHEABLE_SOCKET_TYPES, build_output_preview
from app.schemas import SocketType
from app.security import get_security_registry
from app.transpiler import transpile_node

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


def _func(name: str):
    record = get_security_registry().get_template(name)
    return transpile_node(record.code, name).func


class _FakeUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.calls: list[dict] = []

    def forward(self, sample, timestep, encoder_hidden_states=None, added_cond_kwargs=None):
        self.calls.append(
            {
                "sample": sample.shape,
                "sample_std": float(sample.std()),
                "timestep": timestep.shape,
                "timestep_value": float(timestep.flatten()[0]),
                "encoder_hidden_states": encoder_hidden_states.shape,
                "added_cond_kwargs": added_cond_kwargs,
            }
        )
        return torch.zeros_like(sample)


class _FakeVAE:
    def __init__(self) -> None:
        self.tiling = False

    def decode(self, latents):
        return SimpleNamespace(
            sample=torch.zeros(
                latents.shape[0],
                3,
                latents.shape[2] * 8,
                latents.shape[3] * 8,
            )
        )

    def enable_tiling(self) -> None:
        self.tiling = True

    def disable_tiling(self) -> None:
        self.tiling = False


def _model(architecture: str = "sd15") -> tuple[dict, _FakeUNet]:
    unet = _FakeUNet()
    return (
        {
            "kind": "model",
            "arch": architecture,
            "name": "fake",
            "device": "cpu",
            "dtype": torch.float32,
            "unet": unet,
            "scheduler_config": dict(EulerDiscreteScheduler().config),
            "preset": "sd15",
            "recommended": {"steps": 25},
        },
        unet,
    )


def _conditioning(size: int = 4, pooled: bool = False) -> dict:
    value = {"kind": "conditioning", "cond": torch.zeros(1, 7, size)}
    if pooled:
        value["pooled"] = torch.zeros(1, 8)
    return value


def _latent(width: int = 32, height: int = 32) -> dict:
    return {
        "kind": "latent",
        "samples": torch.zeros(1, 4, height // 8, width // 8),
        "width": width,
        "height": height,
        "batch_size": 1,
        "vae_scale_factor": 8,
    }


class _FakeTokenizer:
    model_max_length = 77

    def __call__(self, text, **_kwargs):
        return SimpleNamespace(
            input_ids=torch.zeros(1, self.model_max_length, dtype=torch.long)
        )


class _FakeEncoderOutput:
    def __init__(self, last_hidden_state, pooled=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = (torch.zeros_like(last_hidden_state), last_hidden_state)
        self._pooled = last_hidden_state if pooled is None else pooled

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self._pooled


class _FakeTextEncoder:
    def __init__(self, last_hidden_state, pooled=None):
        self.output = _FakeEncoderOutput(last_hidden_state, pooled)

    def __call__(self, input_ids, output_hidden_states=True):
        return self.output


def test_resolution_validates_shared_dimensions() -> None:
    resolution = _func("resolution")
    assert resolution(width=512, height=768) == (512, 768)
    with pytest.raises(ValueError, match="multiples of 8"):
        resolution(width=513, height=768)
    with pytest.raises(ValueError, match="at least 8"):
        resolution(width=4, height=768)


def test_resolution_computes_aspect_ratios_and_megapixels() -> None:
    resolution = _func("resolution")
    assert resolution(
        width=512,
        height=512,
        aspect_ratio="16:9",
        orientation="landscape",
        megapixels=1.0,
        divisible_by=64,
    ) == (1344, 768)
    assert resolution(
        width=512,
        height=512,
        aspect_ratio="2:1",
        orientation="portrait",
        megapixels=1.0,
        divisible_by=8,
    ) == (704, 1416)
    assert resolution(
        width=768,
        height=768,
        aspect_ratio="4:3",
        orientation="landscape",
        megapixels=0.0,
        divisible_by=16,
    ) == (880, 672)


def test_resolution_rejects_invalid_resolution_controls() -> None:
    resolution = _func("resolution")
    with pytest.raises(ValueError, match="divisible_by"):
        resolution(width=512, height=512, divisible_by=10)
    with pytest.raises(ValueError, match="megapixels"):
        resolution(width=512, height=512, megapixels=-1)


def test_empty_latent_uses_checkpoint_scale_factor() -> None:
    empty = _func("empty_latent_image")
    (latent,) = empty(
        width=64,
        height=32,
        batch_size=2,
    )
    assert latent["samples"].shape == (2, 4, 4, 8)
    assert latent["width"] == 64
    assert latent["height"] == 32
    assert latent["batch_size"] == 2
    with pytest.raises(ValueError, match="multiples of 8"):
        empty(width=60, height=32, batch_size=1)


def test_empty_latent_computes_aspect_ratio() -> None:
    empty = _func("empty_latent_image")
    (latent,) = empty(
        width=512,
        height=512,
        batch_size=1,
        aspect_ratio="1:1",
        megapixels=0.25,
        divisible_by=32,
    )
    assert latent["width"] == 512
    assert latent["height"] == 512
    assert latent["samples"].shape == (1, 4, 64, 64)


def test_image_from_batch_selects_one_image() -> None:
    images = [Image.new("RGB", (2, 2), (index, 0, 0)) for index in range(3)]
    (selected,) = _func("image_from_batch")(images=images, index=2)
    assert selected is images[2]
    with pytest.raises(IndexError):
        _func("image_from_batch")(images=images, index=3)


def test_clip_text_encode_uses_last_hidden_state_and_sdxl_pooled() -> None:
    last = torch.ones(1, 77, 4)
    clip = {
        "arch": "sd15",
        "device": "cpu",
        "dtype": torch.float32,
        "text_encoder": _FakeTextEncoder(last),
        "tokenizer": _FakeTokenizer(),
    }
    (conditioning,) = _func("clip_text_encode")(clip=clip, text="hello")
    assert torch.equal(conditioning["cond"], last)
    assert conditioning["pooled"] is None

    last = torch.ones(1, 77, 4)
    last_2 = torch.full((1, 77, 4), 2.0)
    pooled = torch.full((1, 8), 7.0)
    clip = {
        "arch": "sdxl",
        "device": "cpu",
        "dtype": torch.float32,
        "text_encoder": _FakeTextEncoder(last),
        "tokenizer": _FakeTokenizer(),
        "text_encoder_2": _FakeTextEncoder(last_2, pooled),
        "tokenizer_2": _FakeTokenizer(),
    }
    (conditioning,) = _func("clip_text_encode")(clip=clip, text="hello")
    assert torch.equal(conditioning["cond"], torch.cat([last, last_2], dim=-1))
    assert torch.equal(conditioning["pooled"], pooled)


def test_ksampler_runs_cfg_with_a_fake_unet() -> None:
    model, unet = _model()
    conditioning = _conditioning()
    (result,) = _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=7,
        steps=2,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
    )

    assert result["samples"].shape == (1, 4, 4, 4)
    assert result["width"] == 32
    assert result["height"] == 32
    assert len(unet.calls) == 2
    assert unet.calls[0]["sample"] == (2, 4, 4, 4)
    assert unet.calls[0]["encoder_hidden_states"] == (2, 7, 4)
    assert unet.calls[0]["added_cond_kwargs"] is None


def test_ksampler_uses_real_sigma_scale_for_initial_noise() -> None:
    model, unet = _model()
    conditioning = _conditioning()
    _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(512, 512),
        seed=7,
        steps=4,
        cfg=1.0,
        sampler_name="euler",
        scheduler="karras",
    )

    assert 0.85 < unet.calls[0]["sample_std"] < 1.15


def test_ksampler_keeps_dpmpp_unit_initial_noise() -> None:
    model, unet = _model()
    conditioning = _conditioning()
    _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(512, 512),
        seed=7,
        steps=4,
        cfg=1.0,
        sampler_name="dpmpp_2m",
        scheduler="karras",
    )

    assert 0.85 < unet.calls[0]["sample_std"] < 1.15


def test_ksampler_builds_sdxl_added_conditioning() -> None:
    model, unet = _model("sdxl")
    positive = _conditioning(2048, pooled=True)
    negative = _conditioning(2048, pooled=True)
    (result,) = _func("ksampler")(
        model=model,
        positive=positive,
        negative=negative,
        latent_image=_latent(64, 64),
        seed=1,
        steps=1,
        cfg=2.0,
        sampler_name="euler",
        scheduler="karras",
    )

    added = unet.calls[0]["added_cond_kwargs"]
    assert result["samples"].shape == (1, 4, 8, 8)
    assert added["text_embeds"].shape == (2, 8)
    assert added["time_ids"].shape == (2, 6)
    assert unet.calls[0]["sample"] == (2, 4, 8, 8)


def test_vae_decode_returns_an_image_batch() -> None:
    vae_model = _FakeVAE()
    (image,) = _func("vae_decode")(
        vae={
            "vae": vae_model,
            "device": "cpu",
            "dtype": torch.float32,
            "scaling_factor": 0.18215,
        },
        samples=_latent(32, 24),
        tiled=True,
    )

    assert len(image) == 1
    assert image[0].size == (32, 24)
    assert vae_model.tiling is True


def test_diffusion_previews_cover_new_socket_types() -> None:
    batch = [Image.new("RGB", (64, 32), "red"), Image.new("RGB", (64, 32), "blue")]
    batch_preview = build_output_preview(SocketType.IMAGE_BATCH, batch)
    assert batch_preview["kind"] == "image_batch"
    assert batch_preview["count"] == 2
    assert batch_preview["width"] == 64
    assert batch_preview["thumbnail"].startswith("data:image/jpeg;base64,")

    model, _ = _model()
    model_preview = build_output_preview(SocketType.MODEL, model)
    assert model_preview["kind"] == "diffusion_model"
    assert model_preview["arch"] == "sd15"
    assert model_preview["parameters"] == 1
    assert model_preview["recommended"] == {"steps": 25}

    clip_preview = build_output_preview(
        SocketType.CLIP,
        {"arch": "sdxl", "text_encoder_2": object(), "device": "cpu"},
    )
    assert clip_preview == {
        "kind": "clip",
        "arch": "sdxl",
        "encoders": 2,
        "device": "cpu",
        "dtype": "",
    }

    vae_preview = build_output_preview(
        SocketType.VAE,
        {"arch": "sd15", "scaling_factor": 0.18215, "shift_factor": None},
    )
    assert vae_preview["kind"] == "vae"
    assert vae_preview["scaling_factor"] == 0.18215
    assert vae_preview["shift_factor"] is None

    conditioning = _conditioning(8, pooled=True)
    conditioning["text"] = "hello"
    conditioning_preview = build_output_preview(
        SocketType.CONDITIONING, conditioning
    )
    assert conditioning_preview["shape"] == [1, 7, 8]
    assert conditioning_preview["pooled_shape"] == [1, 8]
    assert conditioning_preview["text"] == "hello"

    latent_preview = build_output_preview(SocketType.LATENT, _latent())
    assert latent_preview["kind"] == "latent"
    assert latent_preview["shape"] == [1, 4, 4, 4]
    assert latent_preview["width"] == 32

    assert SocketType.MODEL in NON_CACHEABLE_SOCKET_TYPES
    assert SocketType.CLIP in NON_CACHEABLE_SOCKET_TYPES
    assert SocketType.VAE in NON_CACHEABLE_SOCKET_TYPES
    assert SocketType.LATENT not in NON_CACHEABLE_SOCKET_TYPES

def test_save_media_handles_single_and_multi_image_batches(tmp_path) -> None:
    save = _func("save_media")
    single = Image.new("RGB", (4, 4), "red")
    (path, saved) = save(media=[single], path=str(tmp_path / "single.png"))
    assert path == "outputs/single.png"
    assert isinstance(saved, Image.Image)
    assert (tmp_path / "single.png").is_file()

    batch = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    (path, saved) = save(media=batch, path=str(tmp_path / "batch.png"))
    assert path == "outputs/batch_000.png"
    assert isinstance(saved, list) and len(saved) == 2
    assert (tmp_path / "batch_000.png").is_file()
    assert (tmp_path / "batch_001.png").is_file()

@pytest.mark.parametrize(
    "sampler",
    [
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
    ],
)
def test_core_samplers_step_on_cpu(sampler: str) -> None:
    model, _ = _model()
    conditioning = _conditioning()
    (result,) = _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=3,
        steps=4,
        cfg=7.0,
        sampler_name=sampler,
        scheduler="karras",
        control_after_generated="fixed",
    )
    assert result["samples"].shape == (1, 4, 4, 4)
    assert torch.isfinite(result["samples"]).all()


@pytest.mark.parametrize(
    "scheduler",
    [
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
    ],
)
def test_ksampler_accepts_all_supported_schedules(scheduler: str) -> None:
    model, _ = _model()
    conditioning = _conditioning()
    (result,) = _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=3,
        steps=5,
        cfg=7.0,
        sampler_name="euler",
        scheduler=scheduler,
        control_after_generated="fixed",
    )
    assert result["samples"].shape == (1, 4, 4, 4)
    assert torch.isfinite(result["samples"]).all()


def test_basic_scheduler_builds_descending_sigmas() -> None:
    model, _ = _model()
    (sigmas,) = _func("basic_scheduler")(
        model=model,
        scheduler="karras",
        steps=4,
        denoise=1.0,
    )
    values = sigmas["sigmas"]
    assert sigmas["steps"] == 4
    assert len(values) == 5
    assert values[0] > values[-2] > 0
    assert values[-1] == 0

    preview = build_output_preview(SocketType.SIGMAS, sigmas)
    assert preview["kind"] == "sigmas"
    assert preview["steps"] == 4
    assert preview["scheduler"] == "karras"
    assert preview["sigmas"][0] > preview["sigmas"][-1]
    assert len(sigmas["timesteps"]) == 4
    assert torch.all(sigmas["timesteps"][:-1] >= sigmas["timesteps"][1:])


def test_basic_scheduler_timesteps_match_diffusers() -> None:
    model, _ = _model()
    (sigmas,) = _func("basic_scheduler")(
        model=model,
        scheduler="karras",
        steps=4,
        denoise=1.0,
    )
    config = dict(model["scheduler_config"])
    config.pop("_use_default_values", None)
    config["use_karras_sigmas"] = True
    scheduler = EulerDiscreteScheduler.from_config(config)
    scheduler.set_timesteps(4)

    assert torch.allclose(
        sigmas["sigmas"][:-1], scheduler.sigmas[:-1], atol=1e-4
    )
    assert torch.allclose(
        sigmas["timesteps"].float(), scheduler.timesteps.float(), atol=0.1
    )


@pytest.mark.parametrize(
    "scheduler",
    [
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
    ],
)
def test_basic_scheduler_accepts_all_supported_schedules(scheduler: str) -> None:
    model, _ = _model()
    (sigmas,) = _func("basic_scheduler")(
        model=model,
        scheduler=scheduler,
        steps=5,
        denoise=1.0,
    )
    values = sigmas["sigmas"]
    assert len(values) == 6
    assert torch.isfinite(values).all()
    assert values[0] > values[-2] > 0
    assert values[-1] == 0


def test_sampler_custom_advanced_uses_contracts() -> None:
    model, unet = _model()
    conditioning = _conditioning()
    guider = {
        "kind": "guider",
        "model": model,
        "positive": conditioning,
        "negative": conditioning,
        "cfg": 7.0,
    }
    sampler = {"kind": "sampler", "name": "euler", "sample": None}
    sigmas = {
        "kind": "sigmas",
        "sigmas": torch.tensor([1.0, 0.5, 0.0]),
        "steps": 2,
        "scheduler": "normal",
    }
    noise = {
        "kind": "noise",
        "noise_type": "gaussian",
        "noise_seed": 9,
        "alpha": 1.0,
        "k": 1.0,
        "scale": 0.1,
    }
    (output, denoised) = _func("sampler_custom_advanced")(
        guider=guider,
        sampler=sampler,
        sigmas=sigmas,
        noise=noise,
        latent_image=_latent(),
    )

    assert output["samples"].shape == (1, 4, 4, 4)
    assert denoised["samples"].shape == (1, 4, 4, 4)
    assert len(unet.calls) == 2


def test_sampler_custom_advanced_supports_custom_sampler() -> None:
    model, _ = _model()
    conditioning = _conditioning()
    guider = {
        "kind": "guider",
        "model": model,
        "positive": conditioning,
        "negative": conditioning,
        "cfg": 7.0,
    }
    calls = []

    def custom_sampler(**kwargs):
        calls.append(kwargs["step"])
        return kwargs["latents"]

    sampler = {
        "kind": "sampler",
        "name": "custom",
        "sample": custom_sampler,
    }
    sigmas = {
        "kind": "sigmas",
        "sigmas": torch.tensor([1.0, 0.0]),
        "steps": 1,
        "scheduler": "custom",
    }
    noise = {
        "kind": "noise",
        "noise_type": "uniform",
        "noise_seed": 4,
        "alpha": 1.0,
        "k": 1.0,
        "scale": 0.1,
    }
    (output, _) = _func("sampler_custom_advanced")(
        guider=guider,
        sampler=sampler,
        sigmas=sigmas,
        noise=noise,
        latent_image=_latent(),
    )

    assert output["samples"].shape == (1, 4, 4, 4)
    assert calls == [0]


def test_advanced_sampler_accepts_all_res4lyf_noise_types() -> None:
    model, _ = _model()
    conditioning = _conditioning()
    guider = {
        "kind": "guider",
        "model": model,
        "positive": conditioning,
        "negative": conditioning,
        "cfg": 7.0,
    }
    sigmas = {
        "kind": "sigmas",
        "sigmas": torch.tensor([1.0, 0.0]),
        "steps": 1,
        "scheduler": "custom",
    }
    for noise_type in (
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
    ):
        noise = {
            "kind": "noise",
            "noise_type": noise_type,
            "noise_seed": 2,
            "alpha": 1.0,
            "k": 1.0,
            "scale": 0.1,
        }
        (output, _) = _func("sampler_custom_advanced")(
            guider=guider,
            sampler={"kind": "sampler", "name": "euler", "sample": None},
            sigmas=sigmas,
            noise=noise,
            latent_image=_latent(),
        )
        assert output["samples"].shape == (1, 4, 4, 4), noise_type
        assert torch.isfinite(output["samples"]).all(), noise_type


def test_advanced_contract_previews() -> None:
    sampler = build_output_preview(
        SocketType.SAMPLER,
        {"kind": "sampler", "name": "euler", "sample": None},
    )
    assert sampler == {"kind": "sampler", "name": "euler", "custom": False}

    noise = build_output_preview(
        SocketType.NOISE,
        {
            "kind": "noise",
            "noise_type": "fractal",
            "noise_seed": 42,
            "alpha": 1.2,
            "k": 0.8,
            "scale": 0.1,
        },
    )
    assert noise["kind"] == "noise"
    assert noise["noise_type"] == "fractal"
    assert noise["noise_seed"] == 42

    model, _ = _model()
    guider = build_output_preview(
        SocketType.GUIDER,
        {
            "kind": "guider",
            "model": model,
            "positive": {"text": "hello"},
            "negative": {"text": "bad"},
            "cfg": 7.0,
        },
    )
    assert guider["kind"] == "guider"
    assert guider["cfg"] == 7.0
    assert guider["positive_text"] == "hello"
    assert guider["negative_text"] == "bad"

def test_advanced_sampler_supports_sde_variance_noise() -> None:
    model, _ = _model()
    conditioning = _conditioning()
    guider = {
        "kind": "guider",
        "model": model,
        "positive": conditioning,
        "negative": conditioning,
        "cfg": 7.0,
    }
    sigmas = {
        "kind": "sigmas",
        "sigmas": torch.tensor([1.0, 0.5, 0.0]),
        "steps": 2,
        "scheduler": "normal",
    }
    noise = {
        "kind": "noise",
        "noise_type": "fractal",
        "noise_seed": 5,
        "alpha": 1.0,
        "k": 1.0,
        "scale": 0.1,
    }
    (output, _) = _func("sampler_custom_advanced")(
        guider=guider,
        sampler={"kind": "sampler", "name": "dpmpp_2m_sde", "sample": None},
        sigmas=sigmas,
        noise=noise,
        latent_image=_latent(),
    )
    assert output["samples"].shape == (1, 4, 4, 4)
    assert torch.isfinite(output["samples"]).all()

class _FakeEncodeVAE:
    def __init__(self) -> None:
        self.tiling = False

    def encode(self, pixels):
        shape = pixels.shape
        latent = torch.zeros(shape[0], 4, shape[2] // 8, shape[3] // 8)
        return SimpleNamespace(
            latent_dist=SimpleNamespace(mode=lambda: latent)
        )

    def enable_tiling(self) -> None:
        self.tiling = True

    def disable_tiling(self) -> None:
        self.tiling = False


def test_image_scale_resizes_and_crops() -> None:
    scale = _func("image_scale")
    source = Image.new("RGB", (64, 32), "green")
    (resized,) = scale(
        image=source,
        width=32,
        height=32,
        method="lanczos",
        crop=False,
    )
    assert resized.size == (32, 32)

    (cropped,) = scale(
        image=source,
        width=16,
        height=16,
        method="nearest",
        crop=True,
    )
    assert cropped.size == (16, 16)


def test_vae_encode_returns_a_latent() -> None:
    vae_model = _FakeEncodeVAE()
    (latent,) = _func("vae_encode")(
        vae={
            "vae": vae_model,
            "device": "cpu",
            "dtype": torch.float32,
            "scaling_factor": 0.18215,
            "vae_scale_factor": 8,
        },
        pixels=Image.new("RGB", (32, 24), "blue"),
        tiled=True,
    )
    assert latent["samples"].shape == (1, 4, 3, 4)
    assert latent["width"] == 32
    assert latent["height"] == 24
    assert vae_model.tiling is True


def test_ksampler_denoise_slices_the_schedule() -> None:
    model, unet = _model()
    conditioning = _conditioning()
    (result,) = _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=3,
        steps=4,
        denoise=0.5,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
    )
    assert result["samples"].shape == (1, 4, 4, 4)
    assert len(unet.calls) == 2

def test_ksampler_advanced_respects_step_range() -> None:
    model, unet = _model()
    conditioning = _conditioning()
    (result,) = _func("ksampler_advanced")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        add_noise=True,
        noise_seed=4,
        steps=4,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        start_at_step=1,
        end_at_step=3,
        return_with_leftover_noise=False,
    )
    assert result["samples"].shape == (1, 4, 4, 4)
    assert len(unet.calls) == 2


def test_ksampler_advanced_can_return_leftover_noise() -> None:
    model, unet = _model()
    conditioning = _conditioning()
    (result,) = _func("ksampler_advanced")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        add_noise=False,
        noise_seed=4,
        steps=4,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        start_at_step=1,
        end_at_step=2,
        return_with_leftover_noise=True,
    )
    assert result["samples"].shape == (1, 4, 4, 4)
    assert len(unet.calls) == 1


def test_sampler_custom_uses_sampler_and_sigmas_contracts() -> None:
    model, unet = _model()
    conditioning = _conditioning()
    (output, denoised_output) = _func("sampler_custom")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        sampler={"kind": "sampler", "name": "euler", "sample": None},
        sigmas={
            "kind": "sigmas",
            "sigmas": torch.tensor([1.0, 0.5, 0.0]),
            "steps": 2,
        },
        latent_image=_latent(),
        add_noise=True,
        noise_seed=4,
        cfg=7.0,
    )
    assert output["samples"].shape == (1, 4, 4, 4)
    assert denoised_output["samples"].shape == (1, 4, 4, 4)
    assert len(unet.calls) == 2
