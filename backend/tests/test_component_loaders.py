"""Fake-component tests for the Phase 2 loaders."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from app.architectures import ARCHITECTURES
from app.security import get_security_registry
from app.transpiler import transpile_node


def _func(name: str):
    record = get_security_registry().get_template(name)
    return transpile_node(record.code, name).func


class _FakeTextEncoder(torch.nn.Module):
    """Small deterministic text encoder with configurable hidden width."""

    width = 8

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        output_hidden_states=True,
        use_cache=False,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids are required")
        batch, seq = input_ids.shape
        hidden = torch.full((batch, seq, self.width), 0.125, dtype=torch.float32)
        hidden_states = [hidden + index * 0.0 for index in range(40)]
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=hidden[:, 0, :],
            text_embeds=hidden[:, 0, :],
            hidden_states=hidden_states,
        )


class _FakeTextConditioner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def forward(
        self,
        source_hidden_states,
        target_input_ids,
        target_attention_mask=None,
        source_attention_mask=None,
    ):
        return torch.zeros(
            source_hidden_states.shape[0],
            target_input_ids.shape[1],
            4,
            dtype=source_hidden_states.dtype,
            device=source_hidden_states.device,
        )


class _FakeTokenizer:
    model_max_length = 512

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(
        self,
        text,
        padding=None,
        max_length=None,
        truncation=None,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(text, str):
            text = [text]
        length = min(int(max_length or 8), 8)
        batch = len(text)
        return SimpleNamespace(
            input_ids=torch.ones(batch, length, dtype=torch.long),
            attention_mask=torch.ones(batch, length, dtype=torch.long),
        )

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ):
        if isinstance(messages, dict):
            messages = [messages]
        content = ""
        for message in messages:
            value = message.get("content", "")
            if isinstance(value, list):
                value = " ".join(
                    str(part.get("text", ""))
                    for part in value
                    if isinstance(part, dict)
                )
            content += str(value)
        return content


class _FakeComponent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))


class _FakeScheduler:
    config = {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "prediction_type": "epsilon",
    }

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class _FakeSingleFileComponent:
    @classmethod
    def from_single_file(cls, *args, **kwargs):
        return _FakeComponent()


class _FakeAutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _FakeComponent()


_TEXT_CLASS_NAMES = {
    "CLIPTextModel",
    "CLIPTextModelWithProjection",
    "T5EncoderModel",
    "Mistral3ForConditionalGeneration",
    "Qwen3ForCausalLM",
    "Qwen3VLForConditionalGeneration",
    "Qwen3Model",
    "Qwen3VLModel",
    "CustomMossTextEncoder",
}


def _patch_text_module(monkeypatch, module) -> None:
    if module is None:
        return
    for name in _TEXT_CLASS_NAMES:
        monkeypatch.setattr(module, name, _FakeTextEncoder, raising=False)
    monkeypatch.setattr(module, "AutoModel", _FakeTextEncoder, raising=False)
    monkeypatch.setattr(module, "AutoTokenizer", _FakeTokenizer, raising=False)
    monkeypatch.setattr(module, "AutoProcessor", _FakeTokenizer, raising=False)


def _patch_diffusers_module(monkeypatch, module) -> None:
    if module is None:
        return
    monkeypatch.setattr(module, "AutoModel", _FakeAutoModel, raising=False)
    monkeypatch.setattr(module, "SchedulerMixin", _FakeScheduler, raising=False)
    monkeypatch.setattr(
        module,
        "AnimaTextConditioner",
        _FakeTextConditioner,
        raising=False,
    )


def _fake_vae_class(kind: str):
    class FakeVAE(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __init__(self) -> None:
            super().__init__()
            if kind == "flux2":
                latent_channels = 8
                config = SimpleNamespace(
                    latent_channels=latent_channels,
                    block_out_channels=(128, 256, 512, 512),
                    scaling_factor=1.0,
                    shift_factor=0.0,
                    batch_norm_eps=1e-5,
                )
                self.bn = SimpleNamespace(
                    running_mean=torch.zeros(latent_channels * 4),
                    running_var=torch.ones(latent_channels * 4),
                )
            elif kind in {"qwen", "qwen21", "wan"}:
                latent_channels = 32 if kind == "qwen21" else 16
                config = SimpleNamespace(
                    z_dim=latent_channels,
                    block_out_channels=(128, 256, 512, 512),
                    scaling_factor=1.0,
                    shift_factor=0.0,
                    latents_mean=[0.0] * latent_channels,
                    latents_std=[1.0] * latent_channels,
                )
                if kind == "qwen21":
                    config.in_channels = 4
            else:
                latent_channels = 4
                config = SimpleNamespace(
                    latent_channels=latent_channels,
                    block_out_channels=(128, 256, 512, 512),
                    scaling_factor=0.18215,
                    shift_factor=0.0,
                )
            self.config = config
            self.kind = kind
            self.tiling = False
            self.latent_channels = latent_channels

        def encode(self, pixels):
            batch, _, height, width = pixels.shape
            if self.kind in {"qwen", "qwen21", "wan"}:
                latents = torch.zeros(
                    batch,
                    self.latent_channels,
                    1,
                    max(1, height // 16),
                    max(1, width // 16),
                )
            else:
                latents = torch.zeros(
                    batch,
                    self.latent_channels,
                    max(1, height // 8),
                    max(1, width // 8),
                )
            return SimpleNamespace(
                latent_dist=SimpleNamespace(mode=lambda: latents)
            )

        def decode(self, latents, return_dict=False):
            if latents.ndim == 5:
                height = latents.shape[-2] * 16
                width = latents.shape[-1] * 16
                sample = torch.zeros(latents.shape[0], 3, 1, height, width)
            else:
                height = latents.shape[-2] * 8
                width = latents.shape[-1] * 8
                sample = torch.zeros(latents.shape[0], 3, height, width)
            return (sample,)

        def enable_tiling(self) -> None:
            self.tiling = True

        def disable_tiling(self) -> None:
            self.tiling = False

    return FakeVAE


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_repo(
    root: Path,
    pipeline_class: str,
    components: list[tuple[str, str]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    index: dict[str, object] = {"_class_name": pipeline_class}
    for name, class_name in components:
        index[name] = ["diffusers", class_name]
        component = root / name
        if name in {"tokenizer", "tokenizer_2", "processor", "t5_tokenizer"}:
            component.mkdir(parents=True, exist_ok=True)
            _write_json(component / "tokenizer_config.json", {"class": class_name})
        else:
            component.mkdir(parents=True, exist_ok=True)
            _write_json(component / "config.json", {"_class_name": class_name})
    _write_json(root / "model_index.json", index)
    return root


TEXT_COMPONENTS: dict[str, list[tuple[str, str]]] = {
    "sd15": [("text_encoder", "CLIPTextModel"), ("tokenizer", "CLIPTokenizer")],
    "sdxl": [
        ("text_encoder", "CLIPTextModel"),
        ("tokenizer", "CLIPTokenizer"),
        ("text_encoder_2", "CLIPTextModelWithProjection"),
        ("tokenizer_2", "CLIPTokenizer"),
    ],
    "flux1": [
        ("text_encoder", "CLIPTextModel"),
        ("tokenizer", "CLIPTokenizer"),
        ("text_encoder_2", "T5EncoderModel"),
        ("tokenizer_2", "T5TokenizerFast"),
    ],
    "flux2": [
        ("text_encoder", "Mistral3ForConditionalGeneration"),
        ("processor", "AutoProcessor"),
    ],
    "flux3": [
        ("text_encoder", "Qwen3VLForConditionalGeneration"),
        ("processor", "AutoProcessor"),
    ],
    "qwen_image": [
        ("text_encoder", "Qwen3VLForConditionalGeneration"),
        ("processor", "AutoProcessor"),
    ],
    "z_image": [("text_encoder", "Qwen3Model"), ("tokenizer", "Qwen2Tokenizer")],
    "krea2": [("text_encoder", "Qwen3VLModel"), ("tokenizer", "AutoTokenizer")],
    "ideogram4": [
        ("text_encoder", "Qwen3VLModel"),
        ("tokenizer", "AutoTokenizer"),
    ],
    "fibo2": [
        ("text_encoder", "Qwen3VLForConditionalGeneration"),
        ("processor", "AutoProcessor"),
    ],
    "anima": [
        ("text_encoder", "Qwen3Model"),
        ("tokenizer", "Qwen2Tokenizer"),
        ("t5_tokenizer", "T5TokenizerFast"),
        ("text_conditioner", "AnimaTextConditioner"),
    ],
    "moss_image": [
        ("text_encoder", "CustomMossTextEncoder"),
        ("processor", "AutoProcessor"),
    ],
}

PIPELINE_CLASSES = {
    "sd15": "StableDiffusionPipeline",
    "sdxl": "StableDiffusionXLImg2ImgPipeline",
    "flux1": "FluxKontextPipeline",
    "flux2": "Flux2Pipeline",
    "flux3": "Flux3ModularPipeline",
    "qwen_image": "QwenImage21Pipeline",
    "z_image": "ZImagePipeline",
    "krea2": "Krea2Pipeline",
    "ideogram4": "Ideogram4Pipeline",
    "fibo2": "BriaFibo2Pipeline",
    "anima": "AnimaModularPipeline",
    "moss_image": "MOSSImagePipeline",
}

TRANSFORMER_CLASSES = {
    "sd15": "UNet2DConditionModel",
    "sdxl": "UNet2DConditionModel",
    "flux1": "FluxTransformer2DModel",
    "flux2": "Flux2Transformer2DModel",
    "flux3": "Flux3Transformer2DModel",
    "qwen_image": "QwenImage21Transformer2DModel",
    "z_image": "ZImageTransformer2DModel",
    "krea2": "Krea2Transformer2DModel",
    "ideogram4": "Ideogram4Transformer2DModel",
    "fibo2": "BriaFibo2Transformer2DModel",
    "anima": "CosmosTransformer3DModel",
    "moss_image": "CustomMossTransformer",
}

VAE_CASES = {
    "sd15": ("StableDiffusionPipeline", "AutoencoderKL", "standard"),
    "flux2": ("Flux2Pipeline", "AutoencoderKLFlux2", "flux2"),
    "krea2": ("Krea2Pipeline", "AutoencoderKLQwenImage", "qwen"),
    "qwen_image": ("QwenImage21Pipeline", "AutoencoderKLQwenImage21", "qwen21"),
    "wan": ("WanPipeline", "AutoencoderKLWan", "wan"),
}


@pytest.fixture(autouse=True)
def patch_transformers(monkeypatch):
    import importlib

    _patch_text_module(monkeypatch, importlib.import_module("transformers"))
    yield


@pytest.fixture(autouse=True)
def patch_diffusers(monkeypatch):
    import importlib

    _patch_diffusers_module(monkeypatch, importlib.import_module("diffusers"))
    yield


def test_diffusion_model_loads_full_repo_for_every_family(
    tmp_path: Path, monkeypatch
) -> None:
    loader = _func("load_diffusion_model")
    _patch_diffusers_module(monkeypatch, loader.__globals__.get("diffusers"))
    for arch, spec in ARCHITECTURES.items():
        component = "unet" if arch in {"sd15", "sdxl"} else "transformer"
        repo = _write_repo(
            tmp_path / arch,
            PIPELINE_CLASSES[arch],
            [
                (component, TRANSFORMER_CLASSES[arch]),
                ("scheduler", "FlowMatchEulerDiscreteScheduler"),
            ],
        )
        _write_json(
            repo / "scheduler" / "scheduler_config.json",
            {"num_train_timesteps": 1000},
        )
        (model,) = loader(
            unet_name=str(repo),
            type="auto",
            weight_dtype="fp32",
            allow_custom_code=True,
        )
        assert model["arch"] == arch
        assert model["transformer"] is not None
        assert model["component_paths"][component] == component
        assert model["recommended"]["steps"] == spec.steps
        assert model["capabilities"]["guidance"] == spec.guidance
        if arch in {"sd15", "sdxl"}:
            assert model["unet"] is model["transformer"]
        else:
            assert model["unet"] is None


def test_load_clip_complex_family_feature_shapes(
    tmp_path: Path, monkeypatch
) -> None:
    loader = _func("load_clip")
    _patch_text_module(monkeypatch, loader.__globals__.get("transformers"))
    expected = {
        "ideogram4": (1, 32, 104),
        "fibo2": (1, 6, 8, 24),
        "anima": (1, 8, 4),
        "moss_image": (1, 8, 8),
    }
    for arch, shape in expected.items():
        repo = _write_repo(
            tmp_path / f"complex-{arch}",
            PIPELINE_CLASSES[arch],
            TEXT_COMPONENTS[arch],
        )
        (clip,) = loader(
            clip_name=str(repo),
            type="auto",
            device="cpu",
            max_sequence_length=32,
            allow_custom_code=arch == "moss_image",
        )
        conditioning = clip["encode"]("a structured caption")
        assert shape == tuple(conditioning["cond"].shape)
        assert conditioning["attention_mask"].shape[0] == 1
        assert conditioning["encoders"]["text"]["hidden"].shape == shape


def test_load_clip_fibo2_empty_prompt_uses_null_caption(
    tmp_path: Path, monkeypatch
) -> None:
    loader = _func("load_clip")
    _patch_text_module(monkeypatch, loader.__globals__.get("transformers"))
    repo = _write_repo(
        tmp_path / "fibo-null",
        PIPELINE_CLASSES["fibo2"],
        TEXT_COMPONENTS["fibo2"],
    )
    (clip,) = loader(
        clip_name=str(repo),
        type="auto",
        device="cpu",
        max_sequence_length=32,
    )
    conditioning = clip["encode"]("")
    assert conditioning["text"] == ""
    assert conditioning["cond"].shape[0] == 1
    assert conditioning["cond"].shape[1] == 6


def test_diffusion_model_loads_component_folder_and_single_file(
    tmp_path: Path, monkeypatch
) -> None:
    loader = _func("load_diffusion_model")
    _patch_diffusers_module(monkeypatch, loader.__globals__.get("diffusers"))
    component = tmp_path / "flux-transformer"
    component.mkdir()
    _write_json(component / "config.json", {"_class_name": "FluxTransformer2DModel"})
    (model,) = loader(
        unet_name=str(component),
        type="auto",
        weight_dtype="fp32",
        allow_custom_code=False,
    )
    assert model["arch"] == "flux1"
    assert model["component_paths"]["component"] == component.name

    single = tmp_path / "flux-transformer.safetensors"
    single.write_bytes(b"weights")
    _write_json(tmp_path / "config.json", {"_class_name": "FluxTransformer2DModel"})
    monkeypatch.setattr(
        loader.__globals__["diffusers"],
        "FluxTransformer2DModel",
        _FakeSingleFileComponent,
        raising=False,
    )
    (single_model,) = loader(
        unet_name=str(single),
        type="auto",
        weight_dtype="fp32",
        allow_custom_code=False,
    )
    assert single_model["arch"] == "flux1"
    assert single_model["transformer"] is not None


def test_diffusion_model_loads_repo_custom_component(
    tmp_path: Path, monkeypatch
) -> None:
    loader = _func("load_diffusion_model")
    diffusers_module = loader.__globals__["diffusers"]
    _patch_diffusers_module(monkeypatch, diffusers_module)

    class _BrokenAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("custom pipeline components are not registered")

    class _BrokenScheduler:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("custom scheduler requires the repository file")

    monkeypatch.setattr(diffusers_module, "AutoModel", _BrokenAutoModel)
    monkeypatch.setattr(diffusers_module, "SchedulerMixin", _BrokenScheduler)
    repo = tmp_path / "moss"
    transformer = repo / "transformer"
    transformer.mkdir(parents=True)
    _write_json(
        transformer / "config.json",
        {"_class_name": "MOSSImageTransformer"},
    )
    scheduler = repo / "scheduler"
    scheduler.mkdir(parents=True)
    _write_json(
        scheduler / "scheduler_config.json",
        {
            "_class_name": "MOSSImageScheduler",
            "num_train_timesteps": 1000,
            "shift": 5.0,
            "training_shift": 1.0,
        },
    )
    (repo / "pipeline_moss_image.py").write_text(
        "import torch\n"
        "\n"
        "class MOSSImageTransformer(torch.nn.Module):\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, path, **kwargs):\n"
        "        model = cls()\n"
        "        model.config = type(\n"
        "            'Config', (), {'in_channels': 128, 'guidance_embeds': False}\n"
        "        )()\n"
        "        return model\n"
        "\n"
        "    def forward(self, *args, **kwargs):\n"
        "        raise RuntimeError('not executed')\n",
        encoding="utf-8",
    )
    _write_json(
        repo / "model_index.json",
        {
            "_class_name": "MOSSImagePipeline",
            "transformer": ["pipeline_moss_image", "MOSSImageTransformer"],
            "scheduler": ["pipeline_moss_image", "MOSSImageScheduler"],
        },
    )

    (model,) = loader(
        unet_name=str(repo),
        type="auto",
        weight_dtype="fp32",
        allow_custom_code=True,
    )

    assert model["arch"] == "moss_image"
    assert model["transformer"].__class__.__name__ == "MOSSImageTransformer"
    assert model["scheduler_config"]["shift"] == 5.0


def test_load_clip_supports_every_family(tmp_path: Path, monkeypatch) -> None:
    loader = _func("load_clip")
    _patch_text_module(monkeypatch, loader.__globals__.get("transformers"))
    for arch, components in TEXT_COMPONENTS.items():
        repo = _write_repo(tmp_path / f"clip-{arch}", PIPELINE_CLASSES[arch], components)
        (clip,) = loader(
            clip_name=str(repo),
            type="auto",
            device="cpu",
            max_sequence_length=32,
            allow_custom_code=arch == "moss_image",
        )
        assert clip["arch"] == arch
        assert clip["text_encoder"] is not None
        conditioning = clip["encode"]("a red cube")
        assert conditioning["arch"] == arch
        assert isinstance(conditioning["cond"], torch.Tensor)
        assert conditioning["cond"].shape[0] == 1
        if arch == "flux1":
            assert set(conditioning["encoders"]) == {"clip", "t5"}
        if arch in {"sdxl", "flux1"}:
            assert clip["text_encoder_2"] is not None
        if arch == "anima":
            assert conditioning["t5_input_ids"] is not None
    moss_repo = _write_repo(
        tmp_path / "clip-moss-denied",
        PIPELINE_CLASSES["moss_image"],
        TEXT_COMPONENTS["moss_image"],
    )
    with pytest.raises(ValueError, match="allow_custom_code"):
        loader(
            clip_name=str(moss_repo),
            type="auto",
            device="cpu",
            max_sequence_length=32,
            allow_custom_code=False,
        )


def _write_dual_component(
    root: Path,
    name: str,
    class_name: str,
    tokenizer_name: str | None,
) -> Path:
    component = root / name
    component.mkdir(parents=True)
    _write_json(component / "config.json", {"_class_name": class_name})
    if tokenizer_name is not None:
        tokenizer = root / tokenizer_name
        tokenizer.mkdir(parents=True, exist_ok=True)
        _write_json(tokenizer / "tokenizer_config.json", {"class": "CLIPTokenizer"})
    return component


def test_load_clip_dual_combines_sdxl_and_flux_pairs(
    tmp_path: Path, monkeypatch
) -> None:
    loader = _func("load_clip_dual")
    _patch_text_module(monkeypatch, loader.__globals__.get("transformers"))
    sdxl = tmp_path / "sdxl"
    first = _write_dual_component(
        sdxl, "text_encoder", "CLIPTextModel", "tokenizer"
    )
    second = _write_dual_component(
        sdxl, "text_encoder_2", "CLIPTextModelWithProjection", "tokenizer_2"
    )
    (clip,) = loader(
        clip_name1=str(first),
        clip_name2=str(second),
        type="auto",
        device="cpu",
        max_sequence_length=32,
        allow_custom_code=False,
    )
    assert clip["arch"] == "sdxl"
    conditioning = clip["encode"]("a red cube")
    assert conditioning["cond"].shape[-1] == 16
    assert conditioning["pooled"] is not None

    flux = tmp_path / "flux"
    clip_encoder = _write_dual_component(
        flux, "text_encoder", "CLIPTextModel", "tokenizer"
    )
    t5_encoder = _write_dual_component(
        flux, "text_encoder_2", "T5EncoderModel", "tokenizer_2"
    )
    (flux_clip,) = loader(
        clip_name1=str(clip_encoder),
        clip_name2=str(t5_encoder),
        type="auto",
        device="cpu",
        max_sequence_length=32,
        allow_custom_code=False,
    )
    assert flux_clip["arch"] == "flux1"
    flux_conditioning = flux_clip["encode"]("a red cube")
    assert set(flux_conditioning["encoders"]) == {"clip", "t5"}

    with pytest.raises(ValueError, match="single text encoder"):
        loader(
            clip_name1=str(clip_encoder),
            clip_name2=str(t5_encoder),
            type="flux2",
            device="cpu",
            max_sequence_length=32,
            allow_custom_code=False,
        )


def test_load_vae_supports_registered_classes(
    tmp_path: Path, monkeypatch
) -> None:
    loader = _func("load_vae")
    module = loader.__globals__["diffusers"]
    _patch_diffusers_module(monkeypatch, module)
    for arch, (pipeline_class, vae_class, kind) in VAE_CASES.items():
        monkeypatch.setattr(
            module,
            vae_class,
            _fake_vae_class(kind),
            raising=False,
        )
        repo = _write_repo(
            tmp_path / f"vae-{arch}",
            pipeline_class,
            [("vae", vae_class)],
        )
        (vae,) = loader(
            vae_name=str(repo),
            weight_dtype="fp32",
            allow_custom_code=False,
        )
        assert vae["arch"] == arch
        assert vae["latent_space"] in {"sd", "flux2", "qwen", "wan"}
        pixels = torch.zeros(1, 3, 32, 32)
        latents = vae["encode"](pixels, tiled=False)
        assert isinstance(latents, torch.Tensor)
        decoded = vae["decode"](latents, tiled=False)
        assert isinstance(decoded, torch.Tensor)
        assert decoded.shape[1] == 3


def test_load_diffusion_model_rejects_fp8_and_tracks_residency(
    tmp_path: Path, monkeypatch
) -> None:
    loader = _func("load_diffusion_model")
    _patch_diffusers_module(monkeypatch, loader.__globals__["diffusers"])
    repo = _write_repo(
        tmp_path / "sd15-residency",
        PIPELINE_CLASSES["sd15"],
        [("unet", TRANSFORMER_CLASSES["sd15"])],
    )
    monkeypatch.setenv("COMPLEXUI_MEMORY_PROFILE", "low")
    monkeypatch.setenv("COMPLEXUI_ATTENTION", "sdpa-math")

    with pytest.raises(ValueError, match="FP8"):
        loader(
            unet_name=str(repo),
            type="auto",
            weight_dtype="fp8_e4m3fn",
            allow_custom_code=False,
        )

    (model,) = loader(
        unet_name=str(repo),
        type="auto",
        weight_dtype="fp32",
        allow_custom_code=False,
    )
    assert model["memory_profile"] == "low"
    assert model["offload"] is True
    assert model["resident_device"] == "cpu"
    assert model["storage_dtype"] == "fp32"
    assert model["compute_dtype"] == str(torch.float32)
    assert model["attention"]["backend"] == "sdpa-math"
    assert model["channels_last"] is True
    assert model["load_time_ms"] >= 0


def test_load_vae_tracks_compute_dtype_and_residency(
    tmp_path: Path, monkeypatch
) -> None:
    loader = _func("load_vae")
    module = loader.__globals__["diffusers"]
    _patch_diffusers_module(monkeypatch, module)
    monkeypatch.setenv("COMPLEXUI_MEMORY_PROFILE", "low")
    pipeline_class, vae_class, kind = VAE_CASES["sd15"]
    monkeypatch.setattr(module, vae_class, _fake_vae_class(kind), raising=False)
    repo = _write_repo(
        tmp_path / "vae-residency",
        pipeline_class,
        [("vae", vae_class)],
    )

    (vae,) = loader(
        vae_name=str(repo),
        weight_dtype="fp16",
        compute_dtype="fp32",
        allow_custom_code=False,
    )

    assert vae["storage_dtype"] == "fp16"
    assert vae["compute_dtype"] == str(torch.float32)
    assert vae["memory_profile"] == "low"
    assert vae["offload"] is True
    assert vae["resident_device"] == "cpu"
    assert vae["channels_last"] is True
    assert vae["load_time_ms"] >= 0


def test_load_checkpoint_exposes_adapter_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    import struct

    fake_vae = _fake_vae_class("standard")()
    fake_unet = _FakeComponent()
    fake_text_encoder = _FakeTextEncoder()

    class FakePipe:
        def __init__(self) -> None:
            self.unet = fake_unet
            self.vae = fake_vae
            self.text_encoder = fake_text_encoder
            self.tokenizer = _FakeTokenizer()
            self.text_encoder_2 = _FakeTextEncoder()
            self.tokenizer_2 = _FakeTokenizer()
            self.scheduler = SimpleNamespace(
                config={
                    "num_train_timesteps": 1000,
                    "beta_start": 0.00085,
                    "beta_end": 0.012,
                    "beta_schedule": "scaled_linear",
                    "prediction_type": "epsilon",
                }
            )
            self.vae_scale_factor = 8
            self.safety_checker = object()

        def to(self, device):
            return self

        def set_progress_bar_config(self, **kwargs):
            return None

    class FakePipeline:
        @classmethod
        def from_single_file(cls, *args, **kwargs):
            return FakePipe()

    loader = _func("load_checkpoint")
    monkeypatch.setitem(
        loader.__globals__,
        "StableDiffusionXLPipeline",
        FakePipeline,
    )
    payload = json.dumps({"conditioner.embedders.1.model": {}}).encode("utf-8")
    checkpoint = tmp_path / "pony_test.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", len(payload)) + payload)
    (model, clip, vae) = loader(checkpoint_name=str(checkpoint))
    assert model["arch"] == "sdxl"
    assert model["preset"] == "sdxl_pony"
    assert model["recommended"]["steps"] == 30
    assert model["capabilities"]["negative_prompt"] == "required"
    assert clip["text_encoder_2"] is not None
    assert vae["latent_space"] == "sd"
    latents = vae["encode"](torch.zeros(1, 3, 32, 32))
    decoded = vae["decode"](latents)
    assert decoded.shape[1] == 3


def test_clip_text_encode_delegates_to_wrapper_callable() -> None:
    calls = []

    def encode(text):
        calls.append(text)
        return {
            "kind": "conditioning",
            "arch": "flux1",
            "cond": torch.ones(1, 7, 8),
            "pooled": torch.ones(1, 8),
            "text": text,
        }

    (conditioning,) = _func("clip_text_encode")(
        clip={"arch": "flux1", "encode": encode},
        text="hello",
    )

    assert calls == ["hello"]
    assert conditioning["arch"] == "flux1"
    assert tuple(conditioning["cond"].shape) == (1, 7, 8)


def test_vae_encode_decode_delegate_to_wrapper_callables() -> None:
    calls = []

    def encode(pixels, **kwargs):
        calls.append(("encode", tuple(pixels.shape), kwargs))
        return torch.zeros(1, 8, 2, 2)

    def decode(samples, **kwargs):
        calls.append(("decode", tuple(samples.shape), kwargs))
        return torch.zeros(samples.shape[0], 3, 32, 32)

    vae = {
        "arch": "flux2",
        "vae": _FakeComponent(),
        "device": "cpu",
        "dtype": torch.float32,
        "vae_scale_factor": 8,
        "latent_space": "flux2",
        "latent_channels": 8,
        "latent_packing": 2,
        "scaling_factor": 1.0,
        "shift_factor": 0.0,
        "encode": encode,
        "decode": decode,
    }
    image = Image.new("RGB", (32, 32), "red")
    (latent,) = _func("vae_encode")(
        pixels=image,
        vae=vae,
        tiled=False,
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
    )
    assert latent["arch"] == "flux2"
    assert latent["latent_space"] == "flux2"
    assert latent["latent_packing"] == 2
    assert latent["latent_channels"] == 8
    (images,) = _func("vae_decode")(
        samples=latent,
        vae=vae,
        tiled=False,
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
    )
    assert len(images) == 1
    assert images[0].size == (32, 32)
    assert [call[0] for call in calls] == ["encode", "decode"]
