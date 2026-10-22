"""User-created node templates stored in the writable workspace."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.security import SecurityRegistry, get_security_registry
from app.user_templates import (
    InvalidUserTemplate,
    UserTemplateConflict,
    UserTemplateStore,
    get_user_template_store,
)


@pytest.fixture(autouse=True)
def _clear_template_caches():
    get_security_registry.cache_clear()
    get_user_template_store.cache_clear()
    yield
    get_security_registry.cache_clear()
    get_user_template_store.cache_clear()


CODE = "value: int = 3\nreturn value: INT\n"


def test_user_template_store_writes_and_reloads(tmp_path: Path) -> None:
    store = UserTemplateStore(tmp_path / "templates")
    record = store.save(
        name="my_value",
        title="My Value",
        category="Custom",
        description="A tiny value node",
        code=CODE,
    )
    assert record.name == "my_value"
    assert record.code == CODE
    assert (tmp_path / "templates" / "my_value.py").is_file()
    assert store.list_templates()[0].title == "My Value"

    with pytest.raises(UserTemplateConflict):
        store.save(
            name="my_value",
            title="My Value",
            category="Custom",
            description="",
            code=CODE,
        )

    updated = store.save(
        name="my_value",
        title="Updated",
        category="Custom",
        description="",
        code=CODE,
        overwrite=True,
    )
    assert updated.title == "Updated"
    assert store.get("my_value").description == ""


def test_user_templates_are_trusted_by_security_registry() -> None:
    store = get_user_template_store()
    store.save(
        name="trusted_value",
        title="Trusted Value",
        category="Custom",
        description="",
        code=CODE,
    )
    registry = SecurityRegistry()
    assert registry.is_user_template("trusted_value")
    assert registry.is_official(CODE)
    assert registry.is_trusted("node-1", CODE)


def test_user_template_api_roundtrip() -> None:
    client = TestClient(app)
    payload = {
        "name": "api_value",
        "title": "API Value",
        "category": "Custom",
        "description": "Saved through the API",
        "code": CODE,
    }
    response = client.post("/api/templates", json=payload)
    assert response.status_code == 201, response.text
    template = response.json()
    assert template["source"] == "user"
    assert template["name"] == "api_value"

    listed = client.get("/api/templates")
    assert listed.status_code == 200
    assert any(item["name"] == "api_value" for item in listed.json())

    rejected = client.post(
        "/api/templates",
        json={**payload, "code": "return ???: INT\n"},
    )
    assert rejected.status_code == 422

    bundled = client.post(
        "/api/templates",
        json={**payload, "name": "int_value", "overwrite": True},
    )
    assert bundled.status_code == 409

    deleted = client.delete("/api/templates/api_value")
    assert deleted.status_code == 200
    assert client.get("/api/templates").json()

    missing = client.delete("/api/templates/api_value")
    assert missing.status_code == 403


def test_invalid_template_names_are_rejected(tmp_path: Path) -> None:
    store = UserTemplateStore(tmp_path)
    with pytest.raises(InvalidUserTemplate):
        store.save(
            name="Not Valid",
            title="Bad",
            category="Custom",
            description="",
            code=CODE,
        )
