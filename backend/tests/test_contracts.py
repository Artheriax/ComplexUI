"""Cross-language contract checks for frontend/backend mirrors."""

from __future__ import annotations

import re
from pathlib import Path

from app.schemas import SocketType
from app.security import code_hash

FRONTEND_GRAPH = Path(__file__).resolve().parents[2] / "frontend/src/types/graph.ts"


def _frontend_socket_names() -> set[str]:
    source = FRONTEND_GRAPH.read_text(encoding="utf-8")
    match = re.search(
        r"export const SOCKET_TYPE_NAMES = new Set<SocketType>\(\[(.*?)\]\);",
        source,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError("Could not find SOCKET_TYPE_NAMES in frontend graph types")
    return set(re.findall(r'"([A-Z_]+)"', match.group(1)))


def test_frontend_socket_types_match_backend_enum() -> None:
    assert _frontend_socket_names() == {member.name for member in SocketType}


def test_frontend_and_backend_hash_vectors_match() -> None:
    assert code_hash("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert code_hash("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_code_hash_normalizes_line_endings() -> None:
    assert code_hash("x = 1\r\ny = 2\r\n") == code_hash("x = 1\ny = 2\n")
