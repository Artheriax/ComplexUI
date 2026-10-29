"""Architecture registry and template-option contract tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.architectures import (
    ARCHITECTURE_LITERALS,
    ARCHITECTURES,
    canonical_architecture,
)
from app.transpiler import _rewrite_returns, parse_node

TEMPLATES = Path(__file__).resolve().parents[1] / "app" / "templates"


def _template(name: str) -> str:
    return (TEMPLATES / f"{name}.py").read_text(encoding="utf-8")


def _type_options(name: str) -> list[str]:
    schema = parse_node(_template(name), name)
    widget = next(widget for widget in schema.widgets if widget.name == "type")
    return [str(option) for option in widget.options]


def _map_constant(name: str, constant: str) -> dict[str, str]:
    rewritten, _, errors = _rewrite_returns(_template(name))
    if errors:
        raise AssertionError(f"{name}: could not rewrite typed returns")
    tree = ast.parse(rewritten)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ]
        if constant in targets:
            value = ast.literal_eval(node.value)
            return {str(key): str(item) for key, item in value.items()}
    raise AssertionError(f"{name}.{constant} not found")


def _dict_constant(name: str, constant: str) -> dict[str, dict]:
    rewritten, _, errors = _rewrite_returns(_template(name))
    if errors:
        raise AssertionError(f"{name}: could not rewrite typed returns")
    tree = ast.parse(rewritten)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ]
        if constant in targets:
            value = ast.literal_eval(node.value)
            return {
                str(key): item
                for key, item in value.items()
                if isinstance(item, dict)
            }
    raise AssertionError(f"{name}.{constant} not found")


def test_architecture_aliases_resolve_to_canonical_ids() -> None:
    for spec in ARCHITECTURES.values():
        assert canonical_architecture(spec.id) == spec.id
        for alias in spec.aliases:
            assert canonical_architecture(alias) == spec.id


@pytest.mark.parametrize(
    "template",
    ["load_diffusion_model", "load_clip", "load_clip_dual"],
)
def test_template_architecture_options_match_registry(template: str) -> None:
    assert _type_options(template) == list(ARCHITECTURE_LITERALS)


@pytest.mark.parametrize(
    "template",
    ["load_diffusion_model", "load_clip", "load_clip_dual"],
)
def test_template_pipeline_maps_match_registry(template: str) -> None:
    pipeline_map = _map_constant(template, "PIPELINE_ARCH")
    for spec in ARCHITECTURES.values():
        for class_name in spec.pipeline_classes:
            if class_name not in pipeline_map:
                continue
            assert pipeline_map[class_name] == spec.id


@pytest.mark.parametrize("template", ["load_diffusion_model", "load_clip"])
def test_full_pipeline_maps_cover_registry(template: str) -> None:
    pipeline_map = _map_constant(template, "PIPELINE_ARCH")
    for spec in ARCHITECTURES.values():
        for class_name in spec.pipeline_classes:
            assert pipeline_map.get(class_name) == spec.id


def test_vae_maps_cover_registry_classes() -> None:
    class_map = _map_constant("load_vae", "VAE_CLASS_ARCH")
    defaults = _map_constant("load_vae", "DEFAULT_VAE_CLASS")
    mapped = set(class_map) | set(defaults.values())
    for spec in ARCHITECTURES.values():
        for class_name in spec.vae_classes:
            assert class_name in mapped


def test_diffusion_arch_info_matches_registry() -> None:
    info = _dict_constant("load_diffusion_model", "ARCH_INFO")
    assert set(info) == set(ARCHITECTURES)
    for arch, spec in ARCHITECTURES.items():
        entry = info[arch]
        assert entry["label"] == spec.label
        assert entry["width"] == spec.default_width
        assert entry["height"] == spec.default_height
        assert entry["steps"] == spec.steps
        assert entry["cfg"] == spec.cfg
        assert entry["sampler"] == spec.sampler
        assert entry["scheduler"] == spec.scheduler
        assert entry["latent_space"] == spec.latent_space
        assert entry["latent_channels"] == spec.latent_channels
        assert entry["latent_packing"] == spec.latent_packing
        assert entry["vae_scale_factor"] == spec.vae_scale_factor
        assert entry["negative_prompt"] == spec.negative_prompt
        assert entry["guidance"] == spec.guidance
        assert entry.get("guidance_default", entry["cfg"]) == (
            spec.cfg if spec.guidance_default is None else spec.guidance_default
        )
        assert spec.recommended()["guidance"] == (
            spec.cfg if spec.guidance_default is None else spec.guidance_default
        )
        assert spec.capabilities()["guidance_default"] == (
            spec.cfg if spec.guidance_default is None else spec.guidance_default
        )
        assert entry["max_references"] == spec.max_references
        assert entry["text_to_image"] == spec.supports_text_to_image
        assert entry["img2img"] == spec.supports_img2img
        assert entry["inpaint"] == spec.supports_inpaint
        assert entry["edit"] == spec.supports_edit


def test_diffusion_model_component_classes_match_registry() -> None:
    class_map = _map_constant("load_diffusion_model", "CLASS_ARCH")
    for spec in ARCHITECTURES.values():
        for class_name in spec.transformer_classes:
            if class_name == "UNet2DConditionModel":
                continue
            if class_name not in class_map:
                continue
            assert class_map[class_name] == spec.id
