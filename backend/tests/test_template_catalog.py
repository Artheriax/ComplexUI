"""Catalog metadata coverage for the official node templates."""

from __future__ import annotations

from app.security import get_security_registry
from app.template_catalog import CATEGORIES, TEMPLATE_META, template_meta


def test_every_official_template_has_metadata() -> None:
    names = {record.name for record in get_security_registry().templates()}
    assert set(TEMPLATE_META) == names


def test_metadata_entries_are_complete_and_well_formed() -> None:
    for name, meta in TEMPLATE_META.items():
        assert meta.title.strip(), f"{name}: empty title"
        assert meta.description.strip(), f"{name}: empty description"
        assert meta.category in CATEGORIES, f"{name}: unknown category"
        assert len(meta.description) <= 90, f"{name}: description too long"


def test_categories_are_unique_and_ordered() -> None:
    assert len(CATEGORIES) == len(set(CATEGORIES))


def test_template_meta_falls_back_for_unknown_names() -> None:
    meta = template_meta("custom_magic_node")
    assert meta.title == "Custom Magic Node"
    assert meta.category == "Other"
    assert meta.hidden is False


def test_only_example_dataset_and_synthetic_nodes_are_hidden() -> None:
    hidden = {name for name, meta in TEMPLATE_META.items() if meta.hidden}
    assert hidden == {
        "load_iris",
        "load_wine",
        "load_breast_cancer",
        "load_diabetes",
        "load_digits",
        "make_classification_dataset",
        "make_messy_dataset",
        "tensor_dataset",
    }
    assert len(hidden) < len(TEMPLATE_META)
