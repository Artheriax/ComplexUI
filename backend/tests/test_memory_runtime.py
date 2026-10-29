"""Memory-profile, offload and VAE fallback contracts."""

from __future__ import annotations

import pytest
import torch

from app.security import get_security_registry
from app.transpiler import transpile_node


def _func(name: str):
    record = get_security_registry().get_template(name)
    return transpile_node(record.code, name).func


class _RecordingModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_calls: list[tuple[tuple, dict]] = []
        self.slice_value = None

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)

    def set_attention_slice(self, value) -> None:
        self.slice_value = value

    def forward(self, value):
        return value


def _model(transformer: _RecordingModule | None = None, **overrides) -> dict:
    component = transformer or _RecordingModule()
    value = {
        "kind": "model",
        "arch": "sd15",
        "device": "cpu",
        "dtype": torch.float32,
        "transformer": component,
        "unet": component,
    }
    value.update(overrides)
    return value


def test_memory_profile_low_enables_model_offload() -> None:
    (model,) = _func("model_memory")(
        model=_model(),
        profile="low",
        offload="server",
        attention_slicing=False,
        channels_last="off",
    )

    assert model["memory_profile"] == "low"
    assert model["offload"] is True
    assert model["memory"]["profile"] == "low"


def test_memory_profile_resident_disables_offload() -> None:
    (model,) = _func("model_memory")(
        model=_model(),
        profile="resident",
        offload="server",
        attention_slicing=False,
        channels_last="off",
    )

    assert model["offload"] is False


def test_channels_last_and_attention_slicing() -> None:
    transformer = _RecordingModule()
    (model,) = _func("model_memory")(
        model=_model(transformer),
        profile="server",
        offload="none",
        attention_slicing=True,
        channels_last="on",
    )

    assert model["channels_last"] is True
    assert model["attention_slicing"] is True
    assert transformer.slice_value == "auto"
    assert any(
        kwargs.get("memory_format") is torch.channels_last
        for _args, kwargs in transformer.to_calls
    )


def test_offload_after_compile_is_rejected() -> None:
    with pytest.raises(ValueError, match="compiled"):
        _func("model_memory")(
            model=_model(compiled=True),
            profile="low",
            offload="model",
            attention_slicing=False,
            channels_last="off",
        )


def test_sequential_offload_uses_diffusers_group_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import diffusers.hooks as hooks

    calls: list[dict] = []

    def fake_apply_group_offloading(module, **kwargs):
        calls.append({"module": module, **kwargs})

    monkeypatch.setattr(hooks, "apply_group_offloading", fake_apply_group_offloading)
    transformer = _RecordingModule()
    (model,) = _func("model_memory")(
        model=_model(transformer),
        profile="resident",
        offload="sequential",
        group_size=2,
        attention_slicing=False,
        channels_last="off",
    )

    assert model["offload"] is False
    assert model["offload_mode"] == "sequential"
    assert calls and calls[0]["module"] is transformer
    assert calls[0]["offload_type"] == "leaf_level"
    assert calls[0]["use_stream"] is False
    assert calls[0]["num_blocks_per_group"] is None


@pytest.mark.parametrize(
    ("offload", "group_size", "expected_blocks"),
    [("block", 2, 1), ("group", 3, 3)],
)
def test_block_and_group_offload_use_diffusers_hooks(
    offload: str,
    group_size: int,
    expected_blocks: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import diffusers.hooks as hooks

    calls: list[dict] = []

    def fake_apply_group_offloading(module, **kwargs):
        calls.append({"module": module, **kwargs})

    monkeypatch.setattr(hooks, "apply_group_offloading", fake_apply_group_offloading)
    transformer = _RecordingModule()
    (model,) = _func("model_memory")(
        model=_model(transformer),
        profile="resident",
        offload=offload,
        group_size=group_size,
        attention_slicing=False,
        channels_last="off",
    )

    assert model["offload_mode"] == offload
    assert model["offload_group_size"] == expected_blocks
    assert calls[0]["offload_type"] == "block_level"
    assert calls[0]["num_blocks_per_group"] == expected_blocks
    assert calls[0]["use_stream"] is False


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="group offload integration needs CUDA"
)
def test_real_group_offloading_forward_pass() -> None:
    from diffusers import UNet2DConditionModel
    from diffusers.hooks import apply_group_offloading

    torch.manual_seed(0)
    model = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=8,
    ).to("cuda", dtype=torch.float16)
    model.eval()
    apply_group_offloading(
        model,
        onload_device="cuda",
        offload_device="cpu",
        offload_type="block_level",
        num_blocks_per_group=1,
        non_blocking=True,
        use_stream=True,
        record_stream=False,
        low_cpu_mem_usage=False,
    )
    sample = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.float16)
    timestep = torch.tensor([500], device="cuda")
    conditioning = torch.randn(1, 8, 32, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        output = model(
            sample,
            timestep,
            encoder_hidden_states=conditioning,
        ).sample

    assert output.shape == sample.shape


def _conditioning() -> dict:
    return {"kind": "conditioning", "cond": torch.zeros(1, 4, 4)}


def _latent(batch: int = 1) -> dict:
    return {
        "kind": "latent",
        "samples": torch.zeros(batch, 4, 4, 4),
        "width": 32,
        "height": 32,
        "batch_size": batch,
        "vae_scale_factor": 8,
    }


class _SamplerUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.to_calls: list[object] = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)

    def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
        return torch.zeros_like(sample)


def test_sampler_moves_offloaded_model_and_releases() -> None:
    unet = _SamplerUNet()
    model = {
        "kind": "model",
        "arch": "sd15",
        "device": "cpu",
        "dtype": torch.float32,
        "unet": unet,
        "transformer": unet,
        "scheduler_config": {"num_train_timesteps": 1000},
        "offload": True,
    }
    (result,) = _func("ksampler")(
        model=model,
        positive=_conditioning(),
        negative=_conditioning(),
        latent_image=_latent(),
        seed=1,
        steps=1,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="normal",
        control_after_generated="fixed",
    )

    assert result["samples"].shape[0] == 1
    assert len(unet.to_calls) >= 2
    assert unet.to_calls[-1][0] == ("cpu",)


def test_sampler_enters_model_attention_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    from app import attention_dispatch

    seen: list[object] = []

    @contextmanager
    def fake_context(_torch, selection):
        seen.append(selection)
        yield

    monkeypatch.setattr(attention_dispatch, "attention_context", fake_context)
    unet = _SamplerUNet()
    selection = {"backend": "sdpa-math", "device": "cpu"}
    model = {
        "kind": "model",
        "arch": "sd15",
        "device": "cpu",
        "dtype": torch.float32,
        "unet": unet,
        "transformer": unet,
        "scheduler_config": {"num_train_timesteps": 1000},
        "attention": selection,
    }

    _func("ksampler")(
        model=model,
        positive=_conditioning(),
        negative=_conditioning(),
        latent_image=_latent(),
        seed=1,
        steps=1,
        denoise=1.0,
        cfg=1.0,
        sampler_name="euler",
        scheduler="normal",
        control_after_generated="fixed",
    )

    assert seen == [selection]


def test_sampler_rejects_nonfinite_fp16_accumulation() -> None:
    unet = _SamplerUNet()

    def nan_forward(sample, timestep, encoder_hidden_states=None, **kwargs):
        return torch.full_like(sample, float("nan"))

    unet.forward = nan_forward
    model = {
        "kind": "model",
        "arch": "sd15",
        "device": "cpu",
        "dtype": torch.float32,
        "unet": unet,
        "transformer": unet,
        "scheduler_config": {"num_train_timesteps": 1000},
        "accumulation": "fp16",
    }

    with pytest.raises(FloatingPointError, match="non-finite"):
        _func("ksampler")(
            model=model,
            positive=_conditioning(),
            negative=_conditioning(),
            latent_image=_latent(),
            seed=1,
            steps=1,
            denoise=1.0,
            cfg=1.0,
            sampler_name="euler",
            scheduler="normal",
            control_after_generated="fixed",
        )


class _FakeVAE:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.calls = 0
        self.tiled = False
        self.to_calls: list[str] = []
        self.tile_sample_min_size = None
        self.tile_overlap_factor = None

    def to(self, device):
        self.to_calls.append(str(device))
        return self

    def enable_tiling(self) -> None:
        self.tiled = True

    def disable_tiling(self) -> None:
        self.tiled = False

    def decode(self, latents):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("CUDA out of memory")
        return type("Output", (), {"sample": torch.zeros(latents.shape[0], 3, 32, 32)})()


def _vae_wrapper(model: _FakeVAE, **overrides) -> dict:
    value = {
        "kind": "vae",
        "arch": "sd15",
        "vae": model,
        "device": "cpu",
        "dtype": torch.float32,
        "vae_scale_factor": 8,
        "scaling_factor": 1.0,
    }
    value.update(overrides)
    return value


def test_vae_decode_slices_batch() -> None:
    vae = _FakeVAE()
    image = _func("vae_decode")(
        samples=_latent(batch=2),
        vae=_vae_wrapper(vae),
        tiled=False,
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
        sliced=True,
        auto_tiled_retry=False,
    )[0]

    assert len(image) == 2
    assert vae.calls == 2


def test_vae_decode_retries_tiled_once_on_oom() -> None:
    vae = _FakeVAE(fail_first=True)
    image = _func("vae_decode")(
        samples=_latent(),
        vae=_vae_wrapper(vae),
        tiled=False,
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
        sliced=False,
        auto_tiled_retry=True,
    )[0]

    assert len(image) == 1
    assert vae.calls == 2
    assert vae.tiled is True


def test_vae_decode_clamps_overlap() -> None:
    vae = _FakeVAE()
    _func("vae_decode")(
        samples=_latent(),
        vae=_vae_wrapper(vae),
        tiled=True,
        tile_size=64,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
        sliced=False,
        auto_tiled_retry=False,
    )

    assert vae.tile_sample_min_size == 64
    assert vae.tile_overlap_factor == 16 / 64


def test_offloaded_vae_moves_for_decode_and_releases() -> None:
    vae = _FakeVAE()
    _func("vae_decode")(
        samples=_latent(),
        vae=_vae_wrapper(vae, offload=True, resident_device="cpu"),
        tiled=False,
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
        sliced=False,
        auto_tiled_retry=False,
    )

    assert vae.to_calls[0] == "cpu"
    assert vae.to_calls[-1] == "cpu"
