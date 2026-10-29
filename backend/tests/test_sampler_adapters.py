"""Architecture-aware sampler routing and Phase 3 conditioning nodes."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import EulerDiscreteScheduler
from PIL import Image

from app.security import get_security_registry
from app.transpiler import transpile_node

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

PIPELINES = {
    "flux1": "FluxKontextPipeline",
    "flux2": "Flux2Pipeline",
    "flux3": "Flux3DistilledModularPipeline",
    "qwen_image": "QwenImage21Pipeline",
    "z_image": "ZImagePipeline",
    "krea2": "Krea2Pipeline",
    "ideogram4": "Ideogram4Pipeline",
    "fibo2": "BriaFibo2Pipeline",
    "anima": "AnimaModularPipeline",
    "moss_image": "MOSSImagePipeline",
}

TRANSFORMERS = {
    "flux1": "FluxTransformer2DModel",
    "flux2": "Flux2Transformer2DModel",
    "flux3": "Flux3Transformer2DModel",
    "qwen_image": "QwenImage21Transformer2DModel",
    "z_image": "ZImageTransformer2DModel",
    "krea2": "Krea2Transformer2DModel",
    "ideogram4": "Ideogram4Transformer2DModel",
    "fibo2": "BriaFibo2Transformer2DModel",
    "anima": "CosmosTransformer3DModel",
    "moss_image": "MOSSImageTransformer",
}

IN_CHANNELS = {
    "flux1": 64,
    "flux2": 128,
    "flux3": 128,
    "qwen_image": 64,
    "z_image": 16,
    "krea2": 64,
    "ideogram4": 128,
    "fibo2": 32,
    "anima": 16,
    "moss_image": 128,
}

OUTPUT_SHAPES = {
    "flux1": (1, 16, 8, 8),
    "flux2": (1, 128, 4, 4),
    "flux3": (1, 128, 4, 4),
    "qwen_image": (1, 64, 4, 4),
    "z_image": (1, 16, 8, 8),
    "krea2": (1, 16, 8, 8),
    "ideogram4": (1, 128, 4, 4),
    "fibo2": (1, 128, 4, 4),
    "anima": (1, 16, 8, 8),
    "moss_image": (1, 128, 4, 4),
}


def _func(name: str):
    record = get_security_registry().get_template(name)
    return transpile_node(record.code, name).func


class _FakeTransformer(torch.nn.Module):
    def __init__(self, arch: str) -> None:
        super().__init__()
        self.arch = arch
        override = getattr(_FakeAutoModel, "in_channels_override", None)
        self.config = SimpleNamespace(
            in_channels=override or IN_CHANNELS[arch],
            guidance_embeds=bool(
                getattr(_FakeAutoModel, "guidance_embeds_override", False)
            ),
            patch_size=2,
            flow_map=(
                bool(getattr(_FakeAutoModel, "flow_map_override", False))
                if getattr(_FakeAutoModel, "flow_map_override", None) is not None
                else arch == "flux3"
            ),
        )
        self.hidden_lengths: list[int] = []
        self.hidden_channels: list[int] = []
        self.image_cond_lengths: list[int] = []
        self.guidance_values: list[float | None] = []
        self.timestep_values: list[float] = []
        self.skip_layers: list[object] = []
        self.context_latents: list[object] = []
        self.padding_masks: list[object] = []
        self.indicators: list[torch.Tensor] = []
        self.position_ids: list[torch.Tensor] = []
        self.target_timesteps: list[object] = []

    def forward(self, *args, **kwargs):
        if self.arch == "z_image":
            latents = args[0]
            self.hidden_lengths.append(int(latents[0].shape[-1]))
            return ([torch.zeros_like(item) for item in latents],)
        hidden = kwargs.get("hidden_states")
        if hidden is None:
            hidden = args[0]
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 1:
            timestep = args[1]
        if isinstance(timestep, torch.Tensor):
            self.timestep_values.append(float(timestep.reshape(-1)[0]))
        skip_layers = kwargs.get("skip_layers")
        if skip_layers is not None:
            self.skip_layers.append(skip_layers)
        if "context_latents" in kwargs:
            self.context_latents.append(kwargs["context_latents"])
        if "padding_mask" in kwargs:
            self.padding_masks.append(kwargs["padding_mask"])
        if "indicator" in kwargs:
            self.indicators.append(kwargs["indicator"])
        if "position_ids" in kwargs:
            self.position_ids.append(kwargs["position_ids"])
        if "target_timestep" in kwargs:
            self.target_timesteps.append(kwargs["target_timestep"])
        self.hidden_lengths.append(int(hidden.shape[1]))
        self.hidden_channels.append(int(hidden.shape[2]) if hidden.ndim == 3 else 0)
        guidance = kwargs.get("guidance")
        self.guidance_values.append(
            float(guidance.reshape(-1)[0]) if guidance is not None else None
        )
        cond = kwargs.get("image_cond_hidden_states")
        if cond is not None:
            self.image_cond_lengths.append(int(cond.shape[1]))
        return (torch.zeros_like(hidden),)


class _FakeAutoModel:
    arch = "flux1"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _FakeTransformer(cls.arch)


class _FakeScheduler:
    config = {
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": False,
    }

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def _write_repo(root: Path, arch: str, pipeline: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    index = {
        "_class_name": pipeline or PIPELINES[arch],
        "transformer": ["diffusers", TRANSFORMERS[arch]],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    if arch == "ideogram4":
        index["unconditional_transformer"] = [
            "diffusers",
            TRANSFORMERS[arch],
        ]
    (root / "model_index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    (root / "transformer").mkdir(exist_ok=True)
    (root / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": TRANSFORMERS[arch]}),
        encoding="utf-8",
    )
    if arch == "ideogram4":
        (root / "unconditional_transformer").mkdir(exist_ok=True)
        (root / "unconditional_transformer" / "config.json").write_text(
            json.dumps({"_class_name": TRANSFORMERS[arch]}),
            encoding="utf-8",
        )
    (root / "scheduler").mkdir(exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(_FakeScheduler.config),
        encoding="utf-8",
    )
    return root


def _load_model(
    tmp_path: Path,
    monkeypatch,
    arch: str,
    pipeline: str | None = None,
    in_channels: int | None = None,
    guidance_embeds: bool | None = None,
    flow_map: bool | None = None,
):
    loader = _func("load_diffusion_model")
    diffusers_module = loader.__globals__["diffusers"]
    _FakeAutoModel.arch = arch
    _FakeAutoModel.in_channels_override = in_channels
    _FakeAutoModel.guidance_embeds_override = guidance_embeds
    _FakeAutoModel.flow_map_override = flow_map
    monkeypatch.setattr(diffusers_module, "AutoModel", _FakeAutoModel)
    monkeypatch.setattr(diffusers_module, "SchedulerMixin", _FakeScheduler)
    repo = _write_repo(tmp_path / arch, arch, pipeline)
    (model,) = loader(
        unet_name=str(repo),
        type="auto",
        weight_dtype="fp32",
        allow_custom_code=True,
    )
    _FakeAutoModel.in_channels_override = None
    _FakeAutoModel.guidance_embeds_override = None
    _FakeAutoModel.flow_map_override = None
    return model


def _conditioning(arch: str) -> dict:
    if arch == "fibo2":
        cond = torch.zeros(1, 6, 4, 8)
    elif arch == "krea2":
        cond = torch.zeros(1, 4, 12, 8)
    else:
        cond = torch.zeros(1, 4, 8)
    return {
        "kind": "conditioning",
        "arch": arch,
        "cond": cond,
        "pooled": torch.zeros(1, 8) if arch == "flux1" else None,
        "attention_mask": torch.ones(1, 4),
        "text": "a red cube",
    }


def _latent(width: int = 64, height: int = 64, batch: int = 1) -> dict:
    return {
        "kind": "latent",
        "samples": torch.zeros(batch, 4, height // 8, width // 8),
        "width": width,
        "height": height,
        "batch_size": batch,
        "vae_scale_factor": 8,
    }


def _sampler_sigmas() -> dict:
    return {
        "sigmas": torch.tensor([1.0, 0.5, 0.0]),
        "timesteps": torch.tensor([1000.0, 500.0]),
    }


@pytest.mark.parametrize("arch", list(PIPELINES))
def test_ksampler_routes_through_architecture_adapter(
    tmp_path: Path, monkeypatch, arch: str
) -> None:
    model = _load_model(tmp_path, monkeypatch, arch)
    conditioning = _conditioning(arch)
    (result,) = _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=3,
        steps=2,
        denoise=1.0,
        cfg=1.0 if arch in {"flux2", "flux3"} else 4.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert tuple(result["samples"].shape) == OUTPUT_SHAPES[arch]
    assert result["arch"] == arch
    assert result["latent_channels"] > 0


@pytest.mark.parametrize("arch", ["flux1", "qwen_image", "krea2"])
def test_advanced_and_custom_samplers_share_adapter(
    tmp_path: Path, monkeypatch, arch: str
) -> None:
    model = _load_model(tmp_path, monkeypatch, arch)
    conditioning = _conditioning(arch)
    latent = _latent()
    expected = OUTPUT_SHAPES[arch]

    (advanced,) = _func("ksampler_advanced")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=latent,
        add_noise=True,
        noise_seed=4,
        steps=2,
        cfg=4.0,
        sampler_name="euler",
        scheduler="simple",
        start_at_step=0,
        end_at_step=10000,
        return_with_leftover_noise=False,
    )
    assert tuple(advanced["samples"].shape) == expected

    custom = _func("sampler_custom")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        sampler={"name": "euler", "sample": None},
        sigmas=_sampler_sigmas(),
        latent_image=latent,
        add_noise=True,
        noise_seed=5,
        cfg=4.0,
    )
    assert tuple(custom[0]["samples"].shape) == expected
    assert tuple(custom[1]["samples"].shape) == expected

    advanced_custom, _denoised = _func("sampler_custom_advanced")(
        noise={"noise_type": "gaussian", "noise_seed": 6},
        guider={
            "model": model,
            "positive": conditioning,
            "negative": conditioning,
            "cfg": 4.0,
        },
        sampler={"name": "euler", "sample": None},
        sigmas=_sampler_sigmas(),
        latent_image=latent,
    )
    assert tuple(advanced_custom["samples"].shape) == expected


@pytest.mark.parametrize("arch", ["flux1", "qwen_image", "z_image"])
def test_ksampler_adapter_expands_single_prompt_batch(
    tmp_path: Path, monkeypatch, arch: str
) -> None:
    model = _load_model(tmp_path, monkeypatch, arch)
    conditioning = _conditioning(arch)
    (result,) = _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(batch=2),
        seed=3,
        steps=1,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert result["batch_size"] == 2
    assert tuple(result["samples"].shape) == (
        2,
        *OUTPUT_SHAPES[arch][1:],
    )


def test_basic_scheduler_emits_flow_matching_sigmas(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "flux1")
    (sigmas,) = _func("basic_scheduler")(
        model=model,
        scheduler="simple",
        steps=4,
        denoise=1.0,
    )

    assert sigmas["arch"] == "flux1"
    assert sigmas["steps"] == 4
    assert tuple(sigmas["sigmas"].shape) == (5,)
    assert float(sigmas["sigmas"][0]) == pytest.approx(1.0)
    assert float(sigmas["sigmas"][-1]) == 0.0
    assert tuple(sigmas["timesteps"].shape) == (4,)
    assert float(sigmas["timesteps"][0]) == pytest.approx(1000.0)

    output, _denoised = _func("sampler_custom")(
        model=model,
        positive=_conditioning("flux1"),
        negative=_conditioning("flux1"),
        sampler={"name": "euler", "sample": None},
        sigmas=sigmas,
        latent_image=_latent(),
        add_noise=False,
        noise_seed=1,
        cfg=1.0,
    )
    assert tuple(output["samples"].shape) == OUTPUT_SHAPES["flux1"]


@pytest.mark.parametrize("scheduler", ["normal", "karras", "exponential", "beta"])
def test_flow_sampler_accepts_standard_schedules(
    tmp_path: Path, monkeypatch, scheduler: str
) -> None:
    model = _load_model(tmp_path, monkeypatch, "flux1")
    (result,) = _func("ksampler")(
        model=model,
        positive=_conditioning("flux1"),
        negative=_conditioning("flux1"),
        latent_image=_latent(),
        seed=3,
        steps=2,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler=scheduler,
    )

    assert tuple(result["samples"].shape) == OUTPUT_SHAPES["flux1"]


def test_flow_sampler_handles_dynamic_shift(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "flux2")
    model["scheduler_config"].update(
        {
            "use_dynamic_shifting": True,
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }
    )
    (result,) = _func("ksampler")(
        model=model,
        positive=_conditioning("flux2"),
        negative=_conditioning("flux2"),
        latent_image=_latent(1024, 1024),
        seed=3,
        steps=2,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert tuple(result["samples"].shape) == (1, 128, 64, 64)


def test_ksampler_rejects_unadapted_architecture() -> None:
    model = {
        "kind": "model",
        "arch": "fibo2",
        "device": "cpu",
        "dtype": torch.float32,
        "unet": None,
        "sampler_adapter": None,
        "scheduler_config": {},
    }

    with pytest.raises(ValueError, match="not supported by this sampler"):
        _func("ksampler")(
            model=model,
            positive={"kind": "conditioning", "cond": torch.zeros(1, 2, 2)},
            negative={"kind": "conditioning", "cond": torch.zeros(1, 2, 2)},
            latent_image=_latent(),
            seed=1,
            steps=1,
            denoise=1.0,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
        )


def test_conditioning_zero_out_zeros_tensors_and_preserves_references() -> None:
    reference = torch.ones(1, 16, 8, 8)
    conditioning = {
        "kind": "conditioning",
        "arch": "flux1",
        "cond": torch.ones(1, 4, 8),
        "pooled": torch.ones(1, 8),
        "attention_mask": torch.ones(1, 4),
        "ref_latents": [reference],
        "text": "blue",
    }

    (result,) = _func("conditioning_zero_out")(conditioning=conditioning)

    assert torch.count_nonzero(result["cond"]) == 0
    assert torch.count_nonzero(result["pooled"]) == 0
    assert result["attention_mask"] is conditioning["attention_mask"]
    assert result["ref_latents"] == [reference]
    assert result["text"] == ""


def test_encode_reference_images_resizes_and_records_latents() -> None:
    calls: list[tuple[int, int]] = []

    def encode(pixels):
        calls.append((int(pixels.shape[2]), int(pixels.shape[3])))
        return torch.ones(
            pixels.shape[0],
            16,
            pixels.shape[2] // 8,
            pixels.shape[3] // 8,
        )

    vae = {
        "kind": "vae",
        "arch": "flux1",
        "vae": object(),
        "device": "cpu",
        "dtype": torch.float32,
        "vae_scale_factor": 8,
        "latent_packing": 2,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "encode": encode,
    }
    text = {
        "kind": "conditioning",
        "arch": "flux1",
        "cond": torch.zeros(1, 4, 8),
        "pooled": torch.zeros(1, 8),
        "text": "edit",
    }

    (result,) = _func("encode_reference_images")(
        conditioning=text,
        vae=vae,
        image1=Image.new("RGB", (63, 65), "red"),
        image2=Image.new("RGB", (32, 32), "blue"),
        image3=None,
        image4=None,
        image5=None,
        max_pixels=16_000_000,
    )

    assert calls == [(64, 64), (32, 32)]
    assert result["ref_count"] == 2
    assert [tuple(ref.shape) for ref in result["ref_latents"]] == [
        (1, 16, 8, 8),
        (1, 16, 4, 4),
    ]
    assert result["ref_sizes"] == [(8, 8), (4, 4)]
    assert result["text"] == "edit"


def test_flux_kontext_sampler_consumes_reference_latents(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(
        tmp_path,
        monkeypatch,
        "flux1",
        pipeline="FluxKontextPipeline",
    )
    transformer = model["transformer"]
    text = _conditioning("flux1")
    vae = {
        "kind": "vae",
        "arch": "flux1",
        "vae": object(),
        "device": "cpu",
        "dtype": torch.float32,
        "vae_scale_factor": 8,
        "latent_packing": 1,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "encode": lambda pixels: torch.ones(
            pixels.shape[0],
            16,
            pixels.shape[2] // 8,
            pixels.shape[3] // 8,
        ),
    }
    (positive,) = _func("encode_reference_images")(
        conditioning=text,
        vae=vae,
        image1=Image.new("RGB", (64, 64), "red"),
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        max_pixels=16_000_000,
    )

    (result,) = _func("ksampler")(
        model=model,
        positive=positive,
        negative=text,
        latent_image=_latent(),
        seed=1,
        steps=1,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert tuple(result["samples"].shape) == OUTPUT_SHAPES["flux1"]
    assert transformer.hidden_lengths == [32]


def test_non_kontext_flux_rejects_reference_latents(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(
        tmp_path,
        monkeypatch,
        "flux1",
        pipeline="FluxPipeline",
    )
    positive = _conditioning("flux1")
    positive["ref_latents"] = [torch.ones(1, 16, 8, 8)]

    with pytest.raises(ValueError, match="does not accept reference images"):
        _func("ksampler")(
            model=model,
            positive=positive,
            negative=positive,
            latent_image=_latent(),
            seed=1,
            steps=1,
            denoise=1.0,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
        )


def _fake_vae(arch: str, channels: int, scale: int, packing: int) -> dict:
    def encode(pixels):
        height = pixels.shape[2] // (scale * packing)
        width = pixels.shape[3] // (scale * packing)
        return torch.ones(pixels.shape[0], channels, height, width)

    return {
        "kind": "vae",
        "arch": arch,
        "vae": object(),
        "device": "cpu",
        "dtype": torch.float32,
        "vae_scale_factor": scale,
        "latent_packing": packing,
        "scaling_factor": 0.18215,
        "shift_factor": 0.0,
        "encode": encode,
    }


def test_model_guidance_updates_model_and_rejects_unsupported() -> None:
    model = {
        "kind": "model",
        "arch": "flux1",
        "recommended": {"guidance": 4.75},
        "sampler_adapter": {"supports_guidance": True},
    }
    (guided,) = _func("model_guidance")(
        model=model,
        guidance_mode="architecture",
        guidance=3.0,
    )

    assert guided["guidance"] == pytest.approx(4.75)
    assert guided["guidance_source"] == "model_guidance_architecture"
    assert guided["guidance_explicit"] is True
    assert guided["arch"] == "flux1"

    (overridden,) = _func("model_guidance")(
        model=model,
        guidance_mode="override",
        guidance=4.25,
    )
    assert overridden["guidance"] == pytest.approx(4.25)
    assert overridden["guidance_source"] == "model_guidance_override"

    with pytest.raises(ValueError, match="does not use a guidance embedding"):
        _func("model_guidance")(
            model={
                "kind": "model",
                "arch": "z_image",
                "sampler_adapter": {"supports_guidance": False},
            },
            guidance_mode="override",
            guidance=4.0,
        )

    with pytest.raises(ValueError, match="does not use a guidance embedding"):
        _func("model_guidance")(
            model={"kind": "model", "arch": "sd15", "sampler_adapter": None},
            guidance_mode="override",
            guidance=4.0,
        )


def test_cfg_guider_extended_fields_and_guidance_override() -> None:
    conditioning = _conditioning("flux1")
    model = {
        "kind": "model",
        "arch": "flux1",
        "sampler_adapter": {"supports_guidance": True},
    }
    guider = _func("cfg_guider")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        cfg=6.0,
        guidance_mode="override",
        guidance=2.5,
        start_percent=0.2,
        end_percent=0.9,
    )[0]

    assert guider["cfg"] == 6.0
    assert guider["guidance_mode"] == "override"
    assert guider["guidance"] == pytest.approx(2.5)
    assert guider["model"]["guidance"] == pytest.approx(2.5)
    assert guider["model"]["guidance_explicit"] is True
    assert guider["start_percent"] == pytest.approx(0.2)
    assert guider["end_percent"] == pytest.approx(0.9)

    inherited = _func("cfg_guider")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        cfg=3.0,
        guidance_mode="inherit",
        guidance=9.0,
        start_percent=0.0,
        end_percent=1.0,
    )[0]
    assert inherited["guidance"] is None
    assert inherited["model"] is model

    with pytest.raises(ValueError, match="does not use a guidance embedding"):
        _func("cfg_guider")(
            model={
                "kind": "model",
                "arch": "z_image",
                "sampler_adapter": {"supports_guidance": False},
            },
            positive=conditioning,
            negative=conditioning,
            cfg=3.0,
            guidance_mode="override",
            guidance=2.0,
            start_percent=0.0,
            end_percent=1.0,
        )

    with pytest.raises(ValueError, match="less than end_percent"):
        _func("cfg_guider")(
            model=model,
            positive=conditioning,
            negative=conditioning,
            cfg=3.0,
            guidance_mode="inherit",
            guidance=3.0,
            start_percent=0.8,
            end_percent=0.8,
        )


@pytest.mark.parametrize("arch", ["flux1", "flux2"])
def test_explicit_model_guidance_combines_with_cfg(
    tmp_path: Path, monkeypatch, arch: str
) -> None:
    model = _load_model(
        tmp_path,
        monkeypatch,
        arch,
        guidance_embeds=True if arch == "flux1" else None,
    )
    assert model["capabilities"]["model_guidance"] is True
    (guided,) = _func("model_guidance")(
        model=model,
        guidance_mode="architecture",
        guidance=1.0,
    )
    conditioning = _conditioning(arch)

    (result,) = _func("ksampler")(
        model=guided,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=1,
        steps=1,
        denoise=1.0,
        cfg=4.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert tuple(result["samples"].shape) == OUTPUT_SHAPES[arch]
    transformer = guided["transformer"]
    assert len(transformer.guidance_values) == 2
    assert all(
        value is not None and value == pytest.approx(guided["guidance"])
        for value in transformer.guidance_values
    )


STANDARD_FLOW_ARCHS = [
    "flux1",
    "flux2",
    "qwen_image",
    "z_image",
    "krea2",
]
STANDARD_SAMPLERS = [
    "ksampler",
    "ksampler_advanced",
    "sampler_custom",
    "sampler_custom_advanced",
]


def _guided_model(
    tmp_path: Path,
    monkeypatch,
    arch: str,
) -> dict:
    model = _load_model(
        tmp_path,
        monkeypatch,
        arch,
        guidance_embeds=True if arch == "flux1" else None,
    )
    if arch in {"flux1", "flux2"}:
        (model,) = _func("model_guidance")(
            model=model,
            guidance_mode="architecture",
            guidance=1.0,
        )
    return model


@pytest.mark.parametrize("arch", STANDARD_FLOW_ARCHS)
@pytest.mark.parametrize("sampler_node", STANDARD_SAMPLERS)
def test_standard_family_guidance_sampler_matrix(
    tmp_path: Path,
    monkeypatch,
    arch: str,
    sampler_node: str,
) -> None:
    model = _guided_model(tmp_path, monkeypatch, arch)
    conditioning = _conditioning(arch)
    latent = _latent()
    cfg = 1.0 if arch in {"flux1", "flux2"} else 4.0

    if sampler_node == "ksampler":
        (result,) = _func(sampler_node)(
            model=model,
            positive=conditioning,
            negative=conditioning,
            latent_image=latent,
            seed=2,
            steps=1,
            denoise=1.0,
            cfg=cfg,
            sampler_name="euler",
            scheduler="simple",
        )
    elif sampler_node == "ksampler_advanced":
        (result,) = _func(sampler_node)(
            model=model,
            positive=conditioning,
            negative=conditioning,
            latent_image=latent,
            add_noise=True,
            noise_seed=2,
            steps=1,
            cfg=cfg,
            sampler_name="euler",
            scheduler="simple",
            start_at_step=0,
            end_at_step=10000,
            return_with_leftover_noise=False,
        )
    elif sampler_node == "sampler_custom":
        output, _ = _func(sampler_node)(
            model=model,
            positive=conditioning,
            negative=conditioning,
            sampler={"name": "euler", "sample": None},
            sigmas=_sampler_sigmas(),
            latent_image=latent,
            add_noise=True,
            noise_seed=2,
            cfg=cfg,
        )
        result = output
    else:
        output, _ = _func(sampler_node)(
            noise={
                "kind": "noise",
                "noise_type": "gaussian",
                "noise_seed": 2,
                "alpha": 1.0,
                "k": 1.0,
                "scale": 0.1,
            },
            guider={
                "kind": "guider",
                "model": model,
                "positive": conditioning,
                "negative": conditioning,
                "cfg": cfg,
                "guidance_mode": "inherit",
                "guidance": None,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
            sampler={"name": "euler", "sample": None},
            sigmas=_sampler_sigmas(),
            latent_image=latent,
        )
        result = output

    assert tuple(result["samples"].shape) == OUTPUT_SHAPES[arch]


def test_cfg_guider_interval_applies_only_inside_window() -> None:
    class FakeUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(in_channels=4)

        def forward(self, sample, timestep, **kwargs):
            negative = torch.zeros_like(sample[:1])
            positive = torch.ones_like(sample[:1])
            return torch.cat([negative, positive], dim=0)

    seen: list[float] = []

    def custom_sampler(**kwargs):
        seen.append(float(kwargs["noise_prediction"].mean()))
        return kwargs["latents"]

    model = {
        "kind": "model",
        "arch": "sd15",
        "device": "cpu",
        "dtype": torch.float32,
        "unet": FakeUNet(),
        "sampler_adapter": None,
        "scheduler_config": dict(EulerDiscreteScheduler().config),
    }
    conditioning = {
        "kind": "conditioning",
        "arch": "sd15",
        "cond": torch.zeros(1, 77, 768),
    }
    output, _ = _func("sampler_custom_advanced")(
        noise={
            "kind": "noise",
            "noise_type": "gaussian",
            "noise_seed": 1,
            "alpha": 1.0,
            "k": 1.0,
            "scale": 0.1,
        },
        guider={
            "kind": "guider",
            "model": model,
            "positive": conditioning,
            "negative": conditioning,
            "cfg": 7.0,
            "guidance_mode": "inherit",
            "guidance": None,
            "start_percent": 0.5,
            "end_percent": 1.0,
        },
        sampler={"name": "euler", "sample": custom_sampler},
        sigmas=_sampler_sigmas(),
        latent_image=_latent(),
    )

    assert seen == [pytest.approx(1.0), pytest.approx(7.0)]
    assert output["samples"].shape == (1, 4, 8, 8)


@pytest.mark.parametrize("arch", ["sd15", "sdxl"])
@pytest.mark.parametrize("sampler_node", STANDARD_SAMPLERS)
def test_standard_sd_family_sampler_matrix(
    arch: str,
    sampler_node: str,
) -> None:
    class FakeUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(in_channels=4)

        def forward(self, sample, timestep, **kwargs):
            return torch.zeros_like(sample[:, :4])

    model = {
        "kind": "model",
        "arch": arch,
        "device": "cpu",
        "dtype": torch.float32,
        "unet": FakeUNet(),
        "sampler_adapter": None,
        "scheduler_config": dict(EulerDiscreteScheduler().config),
        "recommended": {"guidance": 7.0},
    }
    conditioning = {
        "kind": "conditioning",
        "arch": arch,
        "cond": torch.zeros(1, 77, 2048 if arch == "sdxl" else 768),
        "pooled": torch.zeros(1, 1280) if arch == "sdxl" else None,
    }
    latent = _latent()

    if sampler_node == "ksampler":
        (result,) = _func(sampler_node)(
            model=model,
            positive=conditioning,
            negative=conditioning,
            latent_image=latent,
            seed=2,
            steps=1,
            denoise=1.0,
            cfg=7.0,
            sampler_name="euler",
            scheduler="simple",
        )
    elif sampler_node == "ksampler_advanced":
        (result,) = _func(sampler_node)(
            model=model,
            positive=conditioning,
            negative=conditioning,
            latent_image=latent,
            add_noise=True,
            noise_seed=2,
            steps=1,
            cfg=7.0,
            sampler_name="euler",
            scheduler="simple",
            start_at_step=0,
            end_at_step=10000,
            return_with_leftover_noise=False,
        )
    elif sampler_node == "sampler_custom":
        output, _ = _func(sampler_node)(
            model=model,
            positive=conditioning,
            negative=conditioning,
            sampler={"name": "euler", "sample": None},
            sigmas=_sampler_sigmas(),
            latent_image=latent,
            add_noise=True,
            noise_seed=2,
            cfg=7.0,
        )
        result = output
    else:
        output, _ = _func(sampler_node)(
            noise={
                "kind": "noise",
                "noise_type": "gaussian",
                "noise_seed": 2,
                "alpha": 1.0,
                "k": 1.0,
                "scale": 0.1,
            },
            guider={
                "kind": "guider",
                "model": model,
                "positive": conditioning,
                "negative": conditioning,
                "cfg": 7.0,
                "guidance_mode": "inherit",
                "guidance": None,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
            sampler={"name": "euler", "sample": None},
            sigmas=_sampler_sigmas(),
            latent_image=latent,
        )
        result = output

    assert result["samples"].shape == (1, 4, 8, 8)


@pytest.mark.parametrize(
    "arch",
    ["sd15", "sdxl", *STANDARD_FLOW_ARCHS],
)
def test_basic_scheduler_standard_family_matrix(
    tmp_path: Path, monkeypatch, arch: str
) -> None:
    if arch in PIPELINES:
        model = _load_model(tmp_path, monkeypatch, arch)
    else:
        model = {
            "kind": "model",
            "arch": arch,
            "device": "cpu",
            "dtype": torch.float32,
            "unet": torch.nn.Identity(),
            "sampler_adapter": None,
            "scheduler_config": dict(EulerDiscreteScheduler().config),
        }
    (sigmas,) = _func("basic_scheduler")(
        model=model,
        scheduler="simple",
        steps=4,
        denoise=1.0,
    )

    assert sigmas["arch"] == arch
    assert sigmas["steps"] == 4
    assert len(sigmas["sigmas"]) == 5
    assert len(sigmas["timesteps"]) == 4
    assert float(sigmas["sigmas"][0]) > float(sigmas["sigmas"][-1])


@pytest.mark.parametrize("arch", ["ideogram4", "fibo2", "anima", "moss_image"])
def test_basic_scheduler_complex_family_matrix(
    tmp_path: Path, monkeypatch, arch: str
) -> None:
    model = _load_model(tmp_path, monkeypatch, arch)
    (sigmas,) = _func("basic_scheduler")(
        model=model,
        scheduler="simple",
        steps=4,
        denoise=1.0,
    )

    assert sigmas["arch"] == arch
    assert sigmas["steps"] == 4
    assert len(sigmas["sigmas"]) == 5
    assert len(sigmas["timesteps"]) == 4
    assert float(sigmas["sigmas"][0]) > float(sigmas["sigmas"][-1])
    if arch == "anima":
        assert sigmas["sigmas"].tolist() == pytest.approx(
            [1.0, 0.75, 0.5, 0.25, 0.0]
        )
    elif arch == "moss_image":
        assert float(sigmas["sigmas"][0]) > 0.99
        assert float(sigmas["sigmas"][-1]) == 0.0


def test_standard_sampler_contract_nodes() -> None:
    (sampler,) = _func("sampler_select")(sampler_name="heun")
    assert sampler == {"kind": "sampler", "name": "heun", "sample": None}

    (noise,) = _func("noise_generator")(
        noise_type="perlin",
        noise_seed=7,
        alpha=1.5,
        k=1.0,
        scale=0.2,
    )
    assert noise["noise_type"] == "perlin"
    assert noise["noise_seed"] == 7
    assert noise["alpha"] == pytest.approx(1.5)

    sigmas = {
        "kind": "sigmas",
        "sigmas": torch.tensor([1.0, 0.5, 0.0]),
        "timesteps": torch.tensor([1000.0, 500.0]),
    }
    (preview,) = _func("preview_sigmas")(sigmas=sigmas)
    assert preview is sigmas


@pytest.mark.parametrize("arch", ["ideogram4", "fibo2", "anima", "moss_image"])
@pytest.mark.parametrize("sampler_node", ["ksampler", "ksampler_advanced"])
def test_complex_family_sampler_matrix(
    tmp_path: Path,
    monkeypatch,
    arch: str,
    sampler_node: str,
) -> None:
    model = _load_model(tmp_path, monkeypatch, arch)
    conditioning = _conditioning(arch)
    latent = _latent()
    cfg = {
        "ideogram4": 7.0,
        "fibo2": 5.0,
        "anima": 4.0,
        "moss_image": 3.0,
    }[arch]

    if sampler_node == "ksampler":
        (result,) = _func(sampler_node)(
            model=model,
            positive=conditioning,
            negative=conditioning,
            latent_image=latent,
            seed=7,
            steps=2,
            denoise=1.0,
            cfg=cfg,
            sampler_name="euler",
            scheduler="simple",
        )
    else:
        (result,) = _func(sampler_node)(
            model=model,
            positive=conditioning,
            negative=conditioning,
            latent_image=latent,
            add_noise=True,
            noise_seed=7,
            steps=2,
            cfg=cfg,
            sampler_name="euler",
            scheduler="simple",
            start_at_step=0,
            end_at_step=10000,
            return_with_leftover_noise=False,
        )

    assert result["arch"] == arch
    assert tuple(result["samples"].shape) == OUTPUT_SHAPES[arch]
    assert model["sampler_supported"] is True
    assert model["capabilities"]["model_guidance"] is False

    transformer = model["transformer"]
    first_timestep = {
        "ideogram4": 0.0,
        "fibo2": 1.0,
        "anima": 1.0,
        "moss_image": 1000.0,
    }[arch]
    assert transformer.timestep_values
    assert transformer.timestep_values[0] == pytest.approx(
        first_timestep,
        abs=2e-3,
    )


def test_ideogram4_dual_branch_layout_and_timestep(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "ideogram4")
    conditioning = _conditioning("ideogram4")
    (result,) = _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=8,
        steps=2,
        denoise=1.0,
        cfg=7.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert tuple(result["samples"].shape) == OUTPUT_SHAPES["ideogram4"]
    transformer = model["transformer"]
    unconditional = model["unconditional_transformer"]
    assert transformer.position_ids[0].shape == (1, 20, 3)
    assert transformer.indicators[0][0, :4].tolist() == [3, 3, 3, 3]
    assert transformer.indicators[0][0, 4:].tolist() == [2] * 16
    assert len(transformer.timestep_values) == 2
    assert transformer.timestep_values[0] == pytest.approx(0.0, abs=2e-3)
    assert 0.85 < transformer.timestep_values[1] < 0.95
    assert len(unconditional.timestep_values) == 2
    assert unconditional.timestep_values[0] == pytest.approx(0.0, abs=2e-3)
    assert 0.85 < unconditional.timestep_values[1] < 0.95


def test_fibo2_context_latents_from_references_and_mask(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "fibo2")
    conditioning = _conditioning("fibo2")
    conditioning["ref_latents"] = [torch.ones(1, 128, 4, 4)]
    conditioning["inpaint"] = {
        "mask": torch.ones(1, 1, 64, 64),
        "image_latents": torch.ones(1, 128, 4, 4),
        "masked_latents": torch.ones(1, 128, 4, 4),
    }

    (result,) = _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=9,
        steps=2,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert tuple(result["samples"].shape) == OUTPUT_SHAPES["fibo2"]
    contexts = model["transformer"].context_latents[0]
    assert contexts is not None
    assert len(contexts) == 2
    assert tuple(contexts[0].shape) == (1, 32, 8, 8)
    assert tuple(contexts[1].shape) == (1, 32, 8, 8)


def test_anima_padding_mask_and_sigma_timestep(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "anima")
    conditioning = _conditioning("anima")
    (result,) = _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=10,
        steps=2,
        denoise=1.0,
        cfg=4.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert tuple(result["samples"].shape) == OUTPUT_SHAPES["anima"]
    transformer = model["transformer"]
    assert transformer.padding_masks[0].shape == (1, 1, 64, 64)
    assert transformer.timestep_values[0] == pytest.approx(1.0)


def test_moss_uses_raw_timestep_and_negative_skip_layers(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "moss_image")
    conditioning = _conditioning("moss_image")
    (guided,) = _func("moss_guidance")(
        model=model,
        uncond_skip_layers="22, 24",
        slg_layers="24",
        slg_scale=2.0,
        slg_start=0.0,
        slg_stop=1.0,
    )
    (result,) = _func("ksampler")(
        model=guided,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=11,
        steps=2,
        denoise=1.0,
        cfg=3.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert tuple(result["samples"].shape) == OUTPUT_SHAPES["moss_image"]
    transformer = guided["transformer"]
    assert transformer.timestep_values[0] == pytest.approx(1000.0)
    assert (22, 24) in transformer.skip_layers
    assert (24,) in transformer.skip_layers


def test_moss_slg_runs_without_cfg(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "moss_image")
    conditioning = _conditioning("moss_image")
    (guided,) = _func("moss_guidance")(
        model=model,
        uncond_skip_layers="",
        slg_layers="24",
        slg_scale=2.5,
        slg_start=0.0,
        slg_stop=1.0,
    )
    _func("ksampler")(
        model=guided,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=13,
        steps=2,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )
    assert guided["transformer"].skip_layers == [(24,), (24,)]


def test_moss_guidance_uses_step_fraction_window(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "moss_image")
    conditioning = _conditioning("moss_image")
    (guided,) = _func("moss_guidance")(
        model=model,
        uncond_skip_layers="22",
        slg_layers="",
        slg_scale=0.0,
        slg_start=0.5,
        slg_stop=1.0,
    )
    _func("ksampler")(
        model=guided,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=15,
        steps=2,
        denoise=1.0,
        cfg=3.0,
        sampler_name="euler",
        scheduler="simple",
    )
    transformer = guided["transformer"]
    assert transformer.skip_layers == [(22,)]


def test_moss_guidance_rejects_out_of_range_layers() -> None:
    model = {
        "kind": "model",
        "arch": "moss_image",
        "transformer": SimpleNamespace(layers=[None] * 4),
    }
    with pytest.raises(ValueError, match="outside the model's 4 blocks"):
        _func("moss_guidance")(
            model=model,
            uncond_skip_layers="4",
            slg_layers="",
            slg_scale=0.0,
            slg_start=0.0,
            slg_stop=1.0,
        )


def test_moss_guidance_validates_configuration() -> None:
    model = {"kind": "model", "arch": "moss_image"}
    (guided,) = _func("moss_guidance")(
        model=model,
        uncond_skip_layers="22",
        slg_layers="",
        slg_scale=0.0,
        slg_start=0.0,
        slg_stop=1.0,
    )
    assert guided["moss_guidance"] == {
        "uncond_skip_layers": (22,),
        "slg_layers": (),
        "slg_scale": 0.0,
        "slg_start": 0.0,
        "slg_stop": 1.0,
    }

    with pytest.raises(ValueError, match="only applies to MOSS"):
        _func("moss_guidance")(
            model={"kind": "model", "arch": "flux1"},
            uncond_skip_layers="22",
            slg_layers="",
            slg_scale=0.0,
            slg_start=0.0,
            slg_stop=1.0,
        )
    with pytest.raises(ValueError, match="slg_layers must be set"):
        _func("moss_guidance")(
            model=model,
            uncond_skip_layers="22",
            slg_layers="",
            slg_scale=1.0,
            slg_start=0.0,
            slg_stop=1.0,
        )
    with pytest.raises(ValueError, match="less than slg_stop"):
        _func("moss_guidance")(
            model=model,
            uncond_skip_layers="22",
            slg_layers="",
            slg_scale=0.0,
            slg_start=0.8,
            slg_stop=0.8,
        )


def test_flux3_target_timestep_follows_flow_map_config(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "flux3", flow_map=True)
    conditioning = _conditioning("flux3")
    _func("ksampler")(
        model=model,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=12,
        steps=2,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )
    assert model["transformer"].target_timesteps

    plain = _load_model(tmp_path, monkeypatch, "flux3", flow_map=False)
    _func("ksampler")(
        model=plain,
        positive=conditioning,
        negative=conditioning,
        latent_image=_latent(),
        seed=12,
        steps=2,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )
    assert plain["transformer"].target_timesteps == []


def test_basic_scheduler_uses_ideogram4_logit_normal_schedule(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "ideogram4")
    (sigmas,) = _func("basic_scheduler")(
        model=model,
        scheduler="simple",
        steps=4,
        denoise=1.0,
    )

    values = sigmas["sigmas"]
    assert len(values) == 5
    assert float(values[-1]) == 0.0
    assert 0.99 < float(values[0]) < 1.0
    assert all(
        float(values[index]) > float(values[index + 1])
        for index in range(len(values) - 2)
    )
    assert len(sigmas["timesteps"]) == 4


@pytest.mark.parametrize("arch", ["ideogram4", "fibo2", "anima", "moss_image"])
@pytest.mark.parametrize(
    "sampler_node",
    ["sampler_custom", "sampler_custom_advanced"],
)
def test_complex_family_custom_sampler_matrix(
    tmp_path: Path,
    monkeypatch,
    arch: str,
    sampler_node: str,
) -> None:
    model = _load_model(tmp_path, monkeypatch, arch)
    conditioning = _conditioning(arch)
    latent = _latent()
    cfg = {
        "ideogram4": 7.0,
        "fibo2": 5.0,
        "anima": 4.0,
        "moss_image": 3.0,
    }[arch]

    if sampler_node == "sampler_custom":
        output, _ = _func(sampler_node)(
            model=model,
            positive=conditioning,
            negative=conditioning,
            sampler={"name": "euler", "sample": None},
            sigmas=_sampler_sigmas(),
            latent_image=latent,
            add_noise=True,
            noise_seed=8,
            cfg=cfg,
        )
        result = output
    else:
        output, _ = _func(sampler_node)(
            noise={
                "kind": "noise",
                "noise_type": "gaussian",
                "noise_seed": 8,
                "alpha": 1.0,
                "k": 1.0,
                "scale": 0.1,
            },
            guider={
                "kind": "guider",
                "model": model,
                "positive": conditioning,
                "negative": conditioning,
                "cfg": cfg,
                "guidance_mode": "inherit",
                "guidance": None,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
            sampler={"name": "euler", "sample": None},
            sigmas=_sampler_sigmas(),
            latent_image=latent,
        )
        result = output

    assert result["arch"] == arch
    assert tuple(result["samples"].shape) == OUTPUT_SHAPES[arch]


def test_fibo2_rejects_more_than_five_references(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "fibo2")
    conditioning = _conditioning("fibo2")
    conditioning["ref_latents"] = [
        torch.ones(1, 128, 4, 4) for _ in range(6)
    ]
    with pytest.raises(ValueError, match="at most five reference"):
        _func("ksampler")(
            model=model,
            positive=conditioning,
            negative=conditioning,
            latent_image=_latent(),
            seed=14,
            steps=1,
            denoise=1.0,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
        )


def test_load_image_returns_alpha_or_luminance_mask(tmp_path: Path) -> None:
    alpha_path = tmp_path / "alpha.png"
    source = Image.new("RGBA", (4, 4), (10, 20, 30, 128))
    source.putpixel((0, 0), (0, 0, 0, 0))
    source.save(alpha_path)

    image, mask = _func("load_image")(
        path=str(alpha_path),
        max_pixels=1_000,
    )
    assert image.mode == "RGB"
    assert image.size == (4, 4)
    assert mask.mode == "RGB"
    assert mask.size == (4, 4)
    assert mask.getpixel((0, 0)) == (0, 0, 0)
    assert mask.getpixel((1, 1)) == (128, 128, 128)

    gray_path = tmp_path / "gray.png"
    gray = Image.new("L", (4, 4), 0)
    gray.putpixel((0, 0), 255)
    gray.save(gray_path)
    _, gray_mask = _func("load_image")(
        path=str(gray_path),
        max_pixels=1_000,
    )
    assert gray_mask.getpixel((0, 0)) == (255, 255, 255)
    assert gray_mask.getpixel((1, 1)) == (0, 0, 0)


def test_encode_reference_images_accepts_ten_inputs() -> None:
    vae = _fake_vae("flux2", 128, 8, 2)
    images = {
        f"image{index}": Image.new("RGB", (32, 32), (index, 0, 0))
        for index in range(1, 11)
    }
    (result,) = _func("encode_reference_images")(
        conditioning=_conditioning("flux2"),
        vae=vae,
        **images,
        max_pixels=16_000_000,
    )

    assert result["ref_count"] == 10
    assert len(result["ref_latents"]) == 10
    assert len(result["ref_sizes"]) == 10


def test_inpaint_conditioner_accepts_segmentation_masks() -> None:
    source = Image.new("L", (64, 64), 0)
    source.paste(255, (16, 16, 48, 48))
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    masks = {
        "width": 64,
        "height": 64,
        "masks": [
            {
                "label": "object",
                "mask": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        ],
    }
    positive = {
        "kind": "conditioning",
        "arch": "flux1",
        "cond": torch.zeros(1, 4, 8),
        "text": "replace the object",
    }
    negative = dict(positive, text="blurry")
    vae = _fake_vae("flux1", 16, 8, 1)

    positive_result, negative_result = _func("inpaint_conditioner")(
        positive=positive,
        negative=negative,
        vae=vae,
        pixels=Image.new("RGB", (64, 64), "gray"),
        mask=None,
        masks=masks,
        grow_mask_by=0,
    )

    inpaint = positive_result["inpaint"]
    assert inpaint["mask"].shape == (1, 1, 64, 64)
    assert float(inpaint["mask"][0, 0, 32, 32]) == 1.0
    assert float(inpaint["mask"][0, 0, 0, 0]) == 0.0
    assert negative_result["inpaint"]["mask"].shape == (1, 1, 64, 64)

    with pytest.raises(ValueError, match="either mask or masks"):
        _func("inpaint_conditioner")(
            positive=positive,
            negative=negative,
            vae=vae,
            pixels=Image.new("RGB", (64, 64), "gray"),
            mask=None,
            masks=None,
            grow_mask_by=0,
        )


def test_make_mask_builds_a_binary_mask_with_expected_geometry() -> None:
    mask = _func("make_mask")(
        width=64,
        height=64,
        shape="ellipse",
        margin=8,
        feather=0.0,
    )[0]

    assert isinstance(mask, Image.Image)
    assert mask.size == (64, 64)
    assert mask.getpixel((32, 32)) == (255, 255, 255)
    assert mask.getpixel((0, 0)) == (0, 0, 0)


def test_inpaint_conditioner_encodes_pixels_mask_and_masked_latents() -> None:
    positive = {
        "kind": "conditioning",
        "arch": "flux1",
        "cond": torch.zeros(1, 4, 8),
        "text": "replace the sky",
    }
    negative = {
        "kind": "conditioning",
        "arch": "flux1",
        "cond": torch.zeros(1, 4, 8),
        "text": "blurry",
    }
    vae = _fake_vae("flux1", 16, 8, 1)
    (positive_result, negative_result) = _func("inpaint_conditioner")(
        positive=positive,
        negative=negative,
        vae=vae,
        pixels=Image.new("RGB", (64, 64), "gray"),
        mask=Image.new("L", (64, 64), 255),
        grow_mask_by=0,
    )

    inpaint = positive_result["inpaint"]
    assert inpaint["mask"].shape == (1, 1, 64, 64)
    assert inpaint["image_latents"].shape == (1, 16, 8, 8)
    assert inpaint["masked_latents"].shape == (1, 16, 8, 8)
    assert positive_result["text"] == "replace the sky"
    assert negative_result["inpaint"]["mask"].shape == (1, 1, 64, 64)


def test_flux2_multi_reference_tokens_join_the_sequence(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "flux2")
    vae = _fake_vae("flux2", 128, 8, 2)
    (positive,) = _func("encode_reference_images")(
        conditioning=_conditioning("flux2"),
        vae=vae,
        image1=Image.new("RGB", (32, 32), "red"),
        image2=Image.new("RGB", (32, 32), "blue"),
        image3=None,
        image4=None,
        image5=None,
        max_pixels=16_000_000,
    )

    (result,) = _func("ksampler")(
        model=model,
        positive=positive,
        negative=_conditioning("flux2"),
        latent_image=_latent(),
        seed=1,
        steps=1,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert result["samples"].shape == OUTPUT_SHAPES["flux2"]
    assert model["transformer"].hidden_lengths == [24]


def test_qwen_multi_reference_tokens_and_shapes_are_forwarded(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "qwen_image")
    vae = _fake_vae("qwen_image", 64, 16, 1)
    (positive,) = _func("encode_reference_images")(
        conditioning=_conditioning("qwen_image"),
        vae=vae,
        image1=Image.new("RGB", (32, 32), "red"),
        image2=Image.new("RGB", (32, 32), "blue"),
        image3=None,
        image4=None,
        image5=None,
        max_pixels=16_000_000,
    )

    (result,) = _func("ksampler")(
        model=model,
        positive=positive,
        negative=_conditioning("qwen_image"),
        latent_image=_latent(),
        seed=1,
        steps=1,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert result["samples"].shape == OUTPUT_SHAPES["qwen_image"]
    assert model["transformer"].hidden_lengths == [24]


def test_flux3_references_use_separate_conditioning_stream(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "flux3")
    vae = _fake_vae("flux3", 128, 8, 2)
    (positive,) = _func("encode_reference_images")(
        conditioning=_conditioning("flux3"),
        vae=vae,
        image1=Image.new("RGB", (32, 32), "red"),
        image2=Image.new("RGB", (32, 32), "blue"),
        image3=None,
        image4=None,
        image5=None,
        max_pixels=16_000_000,
    )

    (result,) = _func("ksampler")(
        model=model,
        positive=positive,
        negative=_conditioning("flux3"),
        latent_image=_latent(),
        seed=1,
        steps=1,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert result["samples"].shape == OUTPUT_SHAPES["flux3"]
    assert model["transformer"].hidden_lengths == [16]
    assert model["transformer"].image_cond_lengths == [8]


def _inpaint_payload() -> dict:
    return {
        "kind": "inpaint",
        "mask": torch.ones(1, 1, 64, 64),
        "image_latents": torch.ones(1, 16, 8, 8),
        "masked_latents": torch.ones(1, 16, 8, 8),
        "width": 64,
        "height": 64,
    }


@pytest.mark.parametrize(
    ("arch", "noise_mask_shape"),
    [("flux1", (1, 16, 1)), ("z_image", (1, 1, 8, 8))],
)
def test_inpaint_masks_match_adapter_latent_layouts(
    tmp_path: Path,
    monkeypatch,
    arch: str,
    noise_mask_shape: tuple[int, ...],
) -> None:
    model = _load_model(tmp_path, monkeypatch, arch)
    inpaint = _inpaint_payload()
    adapter = model["sampler_adapter"]
    state = adapter["prepare_latents"](
        inpaint["image_latents"],
        64,
        64,
        torch.device("cpu"),
        torch.float32,
        inpaint=inpaint,
    )

    assert state["noise_mask"].shape == noise_mask_shape
    conditioning = _conditioning(arch)
    positive = dict(conditioning)
    positive["inpaint"] = inpaint
    (result,) = _func("ksampler")(
        model=model,
        positive=positive,
        negative=conditioning,
        latent_image=_latent(),
        seed=1,
        steps=1,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )
    assert tuple(result["samples"].shape) == OUTPUT_SHAPES[arch]


def test_ksampler_advanced_combines_references_and_inpainting(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(tmp_path, monkeypatch, "flux2")
    vae = _fake_vae("flux2", 128, 8, 2)
    positive, negative = _func("inpaint_conditioner")(
        positive=_conditioning("flux2"),
        negative=_conditioning("flux2"),
        vae=vae,
        pixels=Image.new("RGB", (64, 64), "gray"),
        mask=Image.new("L", (64, 64), 255),
        grow_mask_by=0,
    )
    (positive,) = _func("encode_reference_images")(
        conditioning=positive,
        vae=vae,
        image1=Image.new("RGB", (32, 32), "red"),
        image2=Image.new("RGB", (32, 32), "blue"),
        max_pixels=16_000_000,
    )

    (result,) = _func("ksampler_advanced")(
        model=model,
        positive=positive,
        negative=negative,
        latent_image=_latent(),
        add_noise=True,
        noise_seed=3,
        steps=1,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert tuple(result["samples"].shape) == OUTPUT_SHAPES["flux2"]
    assert positive["ref_count"] == 2
    assert "inpaint" in positive


def test_flux_fill_concatenates_mask_and_masked_latents(
    tmp_path: Path, monkeypatch
) -> None:
    model = _load_model(
        tmp_path,
        monkeypatch,
        "flux1",
        pipeline="FluxFillPipeline",
        in_channels=384,
    )
    conditioning = _conditioning("flux1")
    inpaint = {
        "kind": "inpaint",
        "mask": torch.ones(1, 1, 64, 64),
        "image_latents": torch.ones(1, 16, 8, 8),
        "masked_latents": torch.ones(1, 16, 8, 8),
        "width": 64,
        "height": 64,
    }
    positive = dict(conditioning)
    positive["inpaint"] = inpaint

    (result,) = _func("ksampler")(
        model=model,
        positive=positive,
        negative=conditioning,
        latent_image=_latent(),
        seed=1,
        steps=1,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
    )

    assert result["samples"].shape == OUTPUT_SHAPES["flux1"]
    assert model["transformer"].hidden_channels == [384]


@pytest.mark.parametrize("node_name", ["ksampler", "ksampler_advanced"])
def test_sd_inpaint_uses_nine_channel_unet_input(node_name: str) -> None:
    scheduler_config = dict(EulerDiscreteScheduler().config)

    class FakeUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(in_channels=9)
            self.channels_seen = []

        def forward(self, sample, timestep, **kwargs):
            self.channels_seen.append(int(sample.shape[1]))
            return torch.zeros_like(sample[:, :4])

    unet = FakeUNet()
    model = {
        "kind": "model",
        "arch": "sd15",
        "device": "cpu",
        "dtype": torch.float32,
        "unet": unet,
        "sampler_adapter": None,
        "scheduler_config": scheduler_config,
    }
    conditioning = {
        "kind": "conditioning",
        "arch": "sd15",
        "cond": torch.zeros(1, 77, 768),
    }
    inpaint = {
        "kind": "inpaint",
        "mask": torch.ones(1, 1, 64, 64),
        "image_latents": torch.ones(1, 4, 8, 8),
        "masked_latents": torch.ones(1, 4, 8, 8),
        "width": 64,
        "height": 64,
    }
    positive = dict(conditioning)
    positive["inpaint"] = inpaint

    kwargs = {
        "model": model,
        "positive": positive,
        "negative": conditioning,
        "latent_image": _latent(),
        "steps": 1,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "normal",
    }
    if node_name == "ksampler":
        kwargs.update(seed=1, denoise=1.0)
    else:
        kwargs.update(
            add_noise=True,
            noise_seed=1,
            start_at_step=0,
            end_at_step=10000,
            return_with_leftover_noise=False,
        )

    (result,) = _func(node_name)(**kwargs)

    assert unet.channels_seen == [9]
    assert result["samples"].shape == (1, 4, 8, 8)
