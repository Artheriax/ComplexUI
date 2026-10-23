"""Image and latent resize/upscale node tests."""

from __future__ import annotations

import pytest
from PIL import Image

from app.security import get_security_registry
from app.transpiler import transpile_node


def _func(name: str):
    record = get_security_registry().get_template(name)
    return transpile_node(record.code, name).func


def test_resize_by_side_keeps_aspect_and_respects_no_upscale() -> None:
    resize = _func("resize_by_side")
    source = Image.new("RGB", (200, 100), "red")
    (result,) = resize(
        image=source,
        side="longer",
        side_resolution=160,
        round_to=0,
        aspect_ratio="keep",
        allow_upscale=True,
        resample="lanczos",
    )
    assert result.size == (160, 80)

    (capped,) = resize(
        image=Image.new("RGB", (100, 50)),
        side="longer",
        side_resolution=400,
        round_to=0,
        aspect_ratio="keep",
        allow_upscale=False,
    )
    assert capped.size == (100, 50)


def test_resize_by_side_crops_to_requested_aspect() -> None:
    resize = _func("resize_by_side")
    source = Image.new("RGB", (200, 100), "red")
    (center,) = resize(
        image=source,
        side="longer",
        side_resolution=96,
        round_to=0,
        aspect_ratio="1:1",
        crop_from="center",
        allow_upscale=True,
    )
    assert center.size == (96, 96)

    (top_left,) = resize(
        image=source,
        side="longer",
        side_resolution=64,
        round_to=0,
        aspect_ratio="1:1",
        crop_from="top-left",
        allow_upscale=True,
    )
    assert top_left.size == (64, 64)


def test_upscale_image_by_multiplier() -> None:
    upscale = _func("upscale_image_by")
    (result,) = upscale(
        image=Image.new("RGB", (10, 8)),
        scale_by=2.5,
        resample="nearest",
    )
    assert result.size == (25, 20)


def test_resize_image_by_megapixels() -> None:
    resize = _func("resize_image_by_megapixels")
    (result,) = resize(
        image=Image.new("RGB", (200, 100)),
        megapixels=0.04,
        round_to=0,
        allow_upscale=True,
    )
    assert result.width * result.height == pytest.approx(40_000, rel=0.01)
    assert result.width / result.height == pytest.approx(2.0, rel=0.01)


def _latent(samples):
    return {
        "kind": "latent",
        "samples": samples,
        "width": samples.shape[-1] * 8,
        "height": samples.shape[-2] * 8,
        "batch_size": samples.shape[0],
        "vae_scale_factor": 8,
    }


def test_upscale_latent_fixed_dimensions() -> None:
    torch = pytest.importorskip("torch")
    upscale = _func("upscale_latent")
    latent = _latent(torch.zeros((1, 4, 8, 8)))
    (result,) = upscale(
        samples=latent,
        width=128,
        height=128,
        method="nearest",
    )
    assert tuple(result["samples"].shape) == (1, 4, 16, 16)
    assert result["width"] == 128
    assert result["height"] == 128


def test_upscale_latent_by_multiplier() -> None:
    torch = pytest.importorskip("torch")
    upscale = _func("upscale_latent_by")
    latent = _latent(torch.zeros((2, 4, 8, 6)))
    (result,) = upscale(samples=latent, scale_by=2, method="nearest")
    assert tuple(result["samples"].shape) == (2, 4, 16, 12)
    assert result["width"] == 96
    assert result["height"] == 128


def test_resize_latent_by_megapixels() -> None:
    torch = pytest.importorskip("torch")
    resize = _func("resize_latent_by_megapixels")
    latent = _latent(torch.zeros((1, 4, 8, 16)))
    (result,) = resize(
        samples=latent,
        megapixels=0.02,
        round_to=0,
        allow_upscale=True,
        method="nearest",
    )
    assert result["width"] * result["height"] == pytest.approx(
        20_000, rel=0.1
    )
    assert result["width"] / result["height"] == pytest.approx(2.0, rel=0.1)


def test_resize_image_mode_dimensions_and_crop() -> None:
    resize = _func("resize_image")
    (exact,) = resize(
        image=Image.new("RGB", (200, 100)),
        mode="By Dimensions",
        width=120,
        height=80,
        crop=False,
        resample="nearest",
    )
    assert exact.size == (120, 80)

    (cropped,) = resize(
        image=Image.new("RGB", (200, 100)),
        mode="By Dimensions",
        width=64,
        height=64,
        crop=True,
        resample="nearest",
    )
    assert cropped.size == (64, 64)


def test_resize_image_side_defaults_to_upscaling() -> None:
    resize = _func("resize_image")
    (result,) = resize(
        image=Image.new("RGB", (100, 50)),
        mode="By Side",
        side="longer",
        side_resolution=400,
        round_to=0,
        aspect_ratio="keep",
    )
    assert result.size == (400, 200)

    (capped,) = resize(
        image=Image.new("RGB", (100, 50)),
        mode="By Side",
        side="longer",
        side_resolution=400,
        round_to=0,
        aspect_ratio="keep",
        allow_upscale=False,
    )
    assert capped.size == (100, 50)


def test_resize_image_mode_megapixels_and_multiplier() -> None:
    resize = _func("resize_image")
    (preset,) = resize(
        image=Image.new("RGB", (200, 100)),
        mode="By Megapixels",
        megapixel_preset="0.36 MP - Small",
        round_to=0,
    )
    assert preset.width * preset.height == pytest.approx(
        360_000, rel=0.01
    )
    assert preset.width / preset.height == pytest.approx(2.0, rel=0.01)

    (custom,) = resize(
        image=Image.new("RGB", (200, 100)),
        mode="By Megapixels",
        megapixel_preset="Custom",
        custom_megapixels=0.04,
        round_to=0,
    )
    assert custom.width * custom.height == pytest.approx(
        40_000, rel=0.01
    )

    (scaled,) = resize(
        image=Image.new("RGB", (10, 8)),
        mode="By Multiplier",
        scale_by=2.5,
        resample="nearest",
    )
    assert scaled.size == (25, 20)


def test_resize_latent_mode_dimensions() -> None:
    torch = pytest.importorskip("torch")
    resize = _func("resize_latent")
    latent = _latent(torch.zeros((1, 4, 8, 8)))
    (result,) = resize(
        samples=latent,
        mode="By Dimensions",
        width=128,
        height=64,
        crop=False,
        method="nearest",
    )
    assert tuple(result["samples"].shape) == (1, 4, 8, 16)
    assert result["width"] == 128
    assert result["height"] == 64


def test_resize_latent_mode_megapixels_and_multiplier() -> None:
    torch = pytest.importorskip("torch")
    resize = _func("resize_latent")
    latent = _latent(torch.zeros((1, 4, 8, 16)))

    (custom,) = resize(
        samples=latent,
        mode="By Megapixels",
        megapixel_preset="Custom",
        custom_megapixels=0.02,
        round_to=0,
        allow_upscale=True,
        method="nearest",
    )
    assert custom["width"] * custom["height"] == pytest.approx(
        20_000, rel=0.1
    )
    assert custom["width"] / custom["height"] == pytest.approx(2.0, rel=0.1)

    (scaled,) = resize(
        samples=latent,
        mode="By Multiplier",
        scale_by=2,
        method="nearest",
    )
    assert tuple(scaled["samples"].shape) == (1, 4, 16, 32)
    assert scaled["width"] == 256
    assert scaled["height"] == 128


def test_preview_mask_is_a_pass_through() -> None:
    preview = _func("preview_mask")
    masks = {
        "kind": "masks",
        "masks": [],
        "count": 0,
        "thumbnail": None,
    }
    (result,) = preview(masks=masks)
    assert result is masks
