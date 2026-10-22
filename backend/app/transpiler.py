"""ComplexUI node-code transpiler.

Turns a raw, flat Python snippet (the node DSL) into:

* a :class:`~app.schemas.NodeSchema` describing input sockets, widgets,
  output sockets, PEP 723 dependencies and syntax errors, and
* (via :func:`transpile_node`) a compiled callable wrapped in
  ``def __node_exec_<node_id>(...)`` inside an isolated namespace.

Every source rewrite preserves the physical line count of the original
snippet, so tracebacks raised by the wrapped callable can be mapped back to
the user's code with the single ``line_offset`` exposed by
:class:`TranspileResult` (``raw_line = traceback_line - line_offset``).

All top-level returns that reference output variables are rewritten into a
uniform container tuple, so the callable always returns a tuple with exactly
one entry per declared output socket (or ``None`` when a node declares no
outputs).
"""

from __future__ import annotations

import ast
import keyword
import re
import token
import tokenize
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from typing import Any

from .schemas import (
    InputSocket,
    NodeSchema,
    OutputSocket,
    SyntaxErrorInfo,
    WidgetParam,
    normalize_socket_type,
    widget_choice_allowed,
)

__all__ = [
    "TranspileError",
    "TranspileResult",
    "parse_node",
    "transpile_node",
]

_PEP723_BLOCK = re.compile(
    r"^[ \t]*# /// script[ \t]*\n(?P<body>(?:[^\n]*\n)*?)[ \t]*# ///[ \t]*(?:\n|$)",
    re.MULTILINE,
)

_MODEL_TYPE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

_RETURN_ITEM_HINT = (
    "Return items must be plain variable names optionally annotated with a "
    "socket type, e.g. `return model: ML_MODEL`"
)


class TranspileError(Exception):
    """Raised by :func:`transpile_node` when the node code cannot be compiled."""

    def __init__(self, schema: NodeSchema) -> None:
        details = "; ".join(error.message for error in schema.syntax_errors)
        super().__init__(f"Node code failed to transpile: {details}")
        self.schema = schema


@dataclass(frozen=True, slots=True)
class TranspileResult:
    """Outcome of compiling a node snippet."""

    schema: NodeSchema
    func: Callable[..., Any]
    func_name: str
    source: str
    line_offset: int


@dataclass(frozen=True, slots=True)
class _ReturnItem:
    name: str
    type_name: str


@dataclass(slots=True)
class _Bindings:
    inputs: list[InputSocket]
    widgets: list[WidgetParam]
    hoisted_imports: list[str]
    blank_spans: list[tuple[int, int]]
    errors: list[SyntaxErrorInfo]


def parse_node(code: str, node_id: str = "") -> NodeSchema:
    """Statically parse node code into a :class:`NodeSchema`.

    This never executes or imports anything from the snippet and never
    raises: all problems are reported through ``schema.syntax_errors``.
    """
    return _analyze(code, node_id).schema


def transpile_node(code: str, node_id: str = "node") -> TranspileResult:
    """Compile node code into an executable callable.

    Raises :class:`TranspileError` (carrying the offending schema) when the
    snippet has syntax errors.
    """
    analysis = _analyze(code, node_id)
    if analysis.schema.syntax_errors:
        raise TranspileError(analysis.schema)
    compiled = compile(analysis.source, f"<node:{node_id}>", "exec")
    namespace: dict[str, Any] = {}
    exec(compiled, namespace)  # noqa: S102 - isolated namespace by design
    func = namespace.get(analysis.func_name)
    if not callable(func):
        raise TranspileError(analysis.schema)
    return TranspileResult(
        schema=analysis.schema,
        func=func,
        func_name=analysis.func_name,
        source=analysis.source,
        line_offset=analysis.line_offset,
    )


@dataclass(slots=True)
class _Analysis:
    schema: NodeSchema
    source: str
    func_name: str
    line_offset: int


def _analyze(code: str, node_id: str) -> _Analysis:
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    dependencies, errors = _extract_dependencies(normalized)
    rewritten, outputs, return_errors = _rewrite_returns(normalized)
    errors.extend(return_errors)
    bindings = _extract_bindings(rewritten)
    errors.extend(bindings.errors)
    source, func_name, line_offset = _wrap(
        rewritten,
        node_id,
        bindings.inputs,
        bindings.widgets,
        bindings.hoisted_imports,
        bindings.blank_spans,
    )
    schema = NodeSchema(
        node_id=node_id,
        inputs=bindings.inputs,
        widgets=bindings.widgets,
        outputs=outputs,
        dependencies=dependencies,
        syntax_errors=errors,
    )
    return _Analysis(
        schema=schema, source=source, func_name=func_name, line_offset=line_offset
    )


def _extract_dependencies(code: str) -> tuple[list[str], list[SyntaxErrorInfo]]:
    errors: list[SyntaxErrorInfo] = []
    match = _PEP723_BLOCK.search(code)
    if match is None:
        return [], errors
    start_line = code.count("\n", 0, match.start()) + 1
    cleaned_lines: list[str] = []
    for line in match.group("body").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            stripped = stripped[1:]
            if stripped.startswith(" "):
                stripped = stripped[1:]
        cleaned_lines.append(stripped)
    try:
        data = tomllib.loads("\n".join(cleaned_lines))
    except tomllib.TOMLDecodeError as exc:
        errors.append(
            SyntaxErrorInfo(
                line=start_line,
                column=0,
                message=f"Invalid PEP 723 TOML header: {exc}",
            )
        )
        return [], errors
    dependencies = data.get("dependencies", [])
    if not isinstance(dependencies, list) or not all(
        isinstance(dep, str) for dep in dependencies
    ):
        errors.append(
            SyntaxErrorInfo(
                line=start_line,
                column=0,
                message="PEP 723 header 'dependencies' must be a list of strings",
            )
        )
        return [], errors
    return list(dependencies), errors


def _rewrite_returns(
    code: str,
) -> tuple[str, list[OutputSocket], list[SyntaxErrorInfo]]:
    errors: list[SyntaxErrorInfo] = []
    try:
        tokens = list(tokenize.generate_tokens(StringIO(code).readline))
    except tokenize.TokenError as exc:
        message = str(exc.args[0]) if exc.args else "Could not tokenize code"
        position = exc.args[1] if len(exc.args) > 1 else (1, 0)
        errors.append(
            SyntaxErrorInfo(line=position[0], column=position[1], message=message)
        )
        return code, [], errors
    except (IndentationError, SyntaxError) as exc:
        errors.append(
            SyntaxErrorInfo(
                line=getattr(exc, "lineno", None) or 1,
                column=max((getattr(exc, "offset", None) or 1) - 1, 0),
                message=getattr(exc, "msg", None) or str(exc),
            )
        )
        return code, [], errors

    lines = code.split("\n")
    outputs: list[OutputSocket] = []
    seen_outputs: set[str] = set()
    replacements: list[tuple[int, int, list[str]]] = []

    for ret_index in _find_top_level_returns(tokens):
        ret_token = tokens[ret_index]
        start_row = ret_token.start[0]
        stmt, end_row = _collect_statement(tokens, ret_index)
        if not stmt:
            continue
        items, error = _parse_return_items(stmt, start_row, ret_token.start[1])
        if error is not None:
            errors.append(error)
            continue
        if len(items) == 1 and items[0].name == "None" and not items[0].type_name:
            continue
        invalid: str | None = None
        statement_names: set[str] = set()
        for item in items:
            if (
                not item.name.isidentifier()
                or keyword.iskeyword(item.name)
                or item.name in seen_outputs
                or item.name in statement_names
            ):
                invalid = item.name
                break
            statement_names.add(item.name)
        if invalid is not None:
            errors.append(
                SyntaxErrorInfo(
                    line=start_row,
                    column=ret_token.start[1],
                    message=f"Invalid or duplicate return variable name '{invalid}'",
                )
            )
            continue
        for item in items:
            seen_outputs.add(item.name)
            outputs.append(
                OutputSocket(
                    name=item.name,
                    data_type=normalize_socket_type(item.type_name or "ANY"),
                )
            )
        span = end_row - start_row + 1
        comments = " ".join(
            tok.string for tok in stmt if tok.type == token.COMMENT
        )
        names = [item.name for item in items]
        prefix = lines[start_row - 1][: ret_token.start[1]]
        if span == 1:
            body = ", ".join(names) + ("," if len(names) == 1 else "")
            replacement = f"return ({body})"
            if comments:
                replacement += "  " + comments
            replacement_lines = [prefix + replacement]
        else:
            head = f"return ({', '.join(names)},"
            if comments:
                head += "  " + comments
            replacement_lines = [prefix + head, *["    #"] * (span - 2), ")"]
        if len(replacement_lines) != span:
            errors.append(
                SyntaxErrorInfo(
                    line=start_row,
                    column=ret_token.start[1],
                    message=(
                        "Could not rewrite typed return while preserving "
                        "line numbers"
                    ),
                )
            )
            continue
        replacements.append((start_row, end_row, replacement_lines))

    for start_row, end_row, replacement_lines in sorted(replacements, reverse=True):
        lines[start_row - 1 : end_row] = replacement_lines
    return "\n".join(lines), outputs, errors


def _find_top_level_returns(tokens: list[tokenize.TokenInfo]) -> list[int]:
    """Indices of ``return`` tokens that belong to the flat script body.

    Returns nested inside helper ``def``/``class`` bodies are left untouched.
    Control-flow blocks (``if``/``for``/``while``/``with``) at module level do
    not count as nested scopes.
    """
    indices: list[int] = []
    indent_stack: list[bool] = []
    pending_scope_row: int | None = None
    at_line_start = True
    for index, tok in enumerate(tokens):
        if tok.type == token.INDENT:
            indent_stack.append(pending_scope_row is not None)
            pending_scope_row = None
            at_line_start = False
            continue
        if tok.type == token.DEDENT:
            if indent_stack:
                indent_stack.pop()
            continue
        if tok.type == token.NEWLINE:
            at_line_start = True
            continue
        if tok.type in (token.NL, token.COMMENT, token.ENCODING):
            continue
        if tok.type == token.ENDMARKER:
            break
        if pending_scope_row is not None and tok.start[0] > pending_scope_row:
            pending_scope_row = None
        if at_line_start:
            if tok.type == token.NAME and tok.string == "async":
                continue
            if tok.type == token.NAME and tok.string in ("def", "class"):
                pending_scope_row = tok.start[0]
            at_line_start = False
        if (
            tok.type == token.NAME
            and tok.string == "return"
            and not any(indent_stack)
            and pending_scope_row is None
        ):
            indices.append(index)
    return indices


def _collect_statement(
    tokens: list[tokenize.TokenInfo], ret_index: int
) -> tuple[list[tokenize.TokenInfo], int]:
    """Tokens of the logical line starting at ``ret_index`` and its last row."""
    stmt: list[tokenize.TokenInfo] = []
    start_row = tokens[ret_index].start[0]
    end_row = start_row
    index = ret_index + 1
    while index < len(tokens):
        tok = tokens[index]
        if tok.type in (token.NEWLINE, token.ENDMARKER):
            break
        if tok.type in (token.NL, token.ENCODING):
            index += 1
            continue
        stmt.append(tok)
        end_row = max(end_row, tok.start[0])
        index += 1
    return stmt, end_row


def _parse_return_items(
    stmt: list[tokenize.TokenInfo], line: int, column: int
) -> tuple[list[_ReturnItem], SyntaxErrorInfo | None]:
    tokens = [tok for tok in stmt if tok.type != token.COMMENT]
    if not tokens:
        return [], None
    tokens = _strip_outer_parens(tokens)
    if not tokens:
        return [], None
    segments = _split_top_level_commas(tokens)
    if segments is None:
        return [], SyntaxErrorInfo(
            line=line,
            column=column,
            message=(
                "Semicolon-separated statements are not supported in node code"
            ),
        )
    items: list[_ReturnItem] = []
    for segment in segments:
        if not segment:
            return [], SyntaxErrorInfo(
                line=line, column=column, message=_RETURN_ITEM_HINT
            )
        if len(segment) == 1 and segment[0].type == token.NAME:
            items.append(_ReturnItem(segment[0].string, ""))
            continue
        if (
            len(segment) >= 3
            and segment[0].type == token.NAME
            and segment[1].type == token.OP
            and segment[1].string == ":"
        ):
            parts: list[str] = []
            expect_name = True
            valid = True
            for tok in segment[2:]:
                if expect_name:
                    if tok.type != token.NAME:
                        valid = False
                        break
                    parts.append(tok.string)
                    expect_name = False
                else:
                    if not (tok.type == token.OP and tok.string == "."):
                        valid = False
                        break
                    expect_name = True
            if valid and parts and not expect_name:
                items.append(_ReturnItem(segment[0].string, ".".join(parts)))
                continue
        return [], SyntaxErrorInfo(
            line=line, column=column, message=_RETURN_ITEM_HINT
        )
    return items, None


def _strip_outer_parens(
    tokens: list[tokenize.TokenInfo],
) -> list[tokenize.TokenInfo]:
    while (
        len(tokens) >= 2
        and tokens[0].type == token.OP
        and tokens[0].string == "("
    ):
        depth = 0
        close_index: int | None = None
        for index, tok in enumerate(tokens):
            if tok.type == token.OP and tok.string in "([{":
                depth += 1
            elif tok.type == token.OP and tok.string in ")]}":
                depth -= 1
                if depth == 0:
                    close_index = index
                    break
        if close_index == len(tokens) - 1:
            tokens = tokens[1:-1]
        else:
            break
    return tokens


def _split_top_level_commas(
    tokens: list[tokenize.TokenInfo],
) -> list[list[tokenize.TokenInfo]] | None:
    segments: list[list[tokenize.TokenInfo]] = []
    current: list[tokenize.TokenInfo] = []
    depth = 0
    for tok in tokens:
        if tok.type == token.OP and tok.string in "([{":
            depth += 1
        elif tok.type == token.OP and tok.string in ")]}":
            depth -= 1
            if depth < 0:
                return None
        elif tok.type == token.OP and depth == 0:
            if tok.string == ",":
                segments.append(current)
                current = []
                continue
            if tok.string == ";":
                return None
        current.append(tok)
    segments.append(current)
    if segments and not segments[-1]:
        segments.pop()
    return segments


def _annotation_text(node: ast.expr) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ast.unparse(node)


def _is_model_selector_annotation(annotation: ast.expr) -> bool:
    text = _annotation_text(annotation).strip().strip("'\"")
    return text in {"ModelSelector", "complexui.ModelSelector"}


def _model_selector_type(value: ast.expr | None) -> str:
    if not isinstance(value, ast.Call):
        raise ValueError(
            "ModelSelector widgets must use "
            'ModelSelector(type="...") as their default'
        )
    target = value.func
    if isinstance(target, ast.Name):
        name = target.id
    elif isinstance(target, ast.Attribute):
        name = target.attr
    else:
        name = ""
    if name != "ModelSelector":
        raise ValueError("ModelSelector default must call ModelSelector(type=...)")
    if value.args:
        raise ValueError("ModelSelector accepts only a type keyword argument")
    selected: ast.expr | None = None
    for keyword_item in value.keywords:
        if keyword_item.arg == "type":
            selected = keyword_item.value
        else:
            raise ValueError("ModelSelector accepts only a type keyword argument")
    if selected is None:
        raise ValueError("ModelSelector requires a type=... argument")
    try:
        model_type = ast.literal_eval(selected)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError("ModelSelector type must be a string literal") from exc
    if not isinstance(model_type, str):
        raise ValueError("ModelSelector type must be a string literal")
    cleaned = model_type.strip()
    if not _MODEL_TYPE_PATTERN.fullmatch(cleaned):
        raise ValueError(
            "ModelSelector type must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}"
        )
    return cleaned


def _literal_subscript(annotation: ast.expr) -> ast.Subscript | None:
    candidate = annotation
    if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
        try:
            candidate = ast.parse(candidate.value, mode="eval").body
        except SyntaxError:
            return None
    if not isinstance(candidate, ast.Subscript):
        return None
    base = candidate.value
    if isinstance(base, ast.Name):
        return candidate if base.id == "Literal" else None
    if isinstance(base, ast.Attribute):
        return candidate if base.attr == "Literal" else None
    return None


def _literal_widget_options(annotation: ast.Subscript) -> tuple[list[Any], str]:
    elements = (
        annotation.slice.elts
        if isinstance(annotation.slice, ast.Tuple)
        else [annotation.slice]
    )
    options: list[Any] = []
    kinds: set[str] = set()
    for element in elements:
        try:
            option = ast.literal_eval(element)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ValueError(
                "Literal widget options must be literal values"
            ) from exc
        if isinstance(option, bool):
            kind = "bool"
        elif isinstance(option, int):
            kind = "int"
        elif isinstance(option, float):
            kind = "float"
        elif isinstance(option, str):
            kind = "str"
        else:
            raise ValueError(
                "Literal widget options must be strings, integers, "
                "floats or booleans"
            )
        options.append(option)
        kinds.add(kind)
    if not options:
        raise ValueError("Literal widgets need at least one option")
    if kinds <= {"int", "float"}:
        data_type = "float" if "float" in kinds else "int"
    elif len(kinds) == 1:
        data_type = next(iter(kinds))
    else:
        raise ValueError(
            "Literal widget options must share one scalar type"
        )
    return options, data_type


def _extract_bindings(source: str) -> _Bindings:
    inputs: list[InputSocket] = []
    widgets: list[WidgetParam] = []
    hoisted_imports: list[str] = []
    blank_spans: list[tuple[int, int]] = []
    errors: list[SyntaxErrorInfo] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        errors.append(
            SyntaxErrorInfo(
                line=exc.lineno or 1,
                column=max((exc.offset or 1) - 1, 0),
                message=exc.msg,
            )
        )
        return _Bindings(inputs, widgets, hoisted_imports, blank_spans, errors)

    lines = source.split("\n")
    seen: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end_line = node.end_lineno or node.lineno
            hoisted_imports.append(
                "\n".join(lines[node.lineno - 1 : end_line])
            )
            blank_spans.append((node.lineno, end_line))
            continue
        if not isinstance(node, ast.AnnAssign):
            continue
        blank_spans.append((node.lineno, node.end_lineno or node.lineno))
        if not isinstance(node.target, ast.Name):
            errors.append(
                SyntaxErrorInfo(
                    line=node.lineno,
                    column=node.col_offset,
                    message=(
                        "Node inputs/widgets must be simple names, "
                        "e.g. 'x: DATASET'"
                    ),
                )
            )
            continue
        name = node.target.id
        if name in seen:
            errors.append(
                SyntaxErrorInfo(
                    line=node.lineno,
                    column=node.col_offset,
                    message=(
                        f"Duplicate declaration of '{name}' "
                        f"(already declared as a {seen[name]})"
                    ),
                )
            )
            continue
        type_text = _annotation_text(node.annotation)
        model_selector = _is_model_selector_annotation(node.annotation)
        if node.value is None:
            if model_selector:
                errors.append(
                    SyntaxErrorInfo(
                        line=node.lineno,
                        column=node.col_offset,
                        message=(
                            "ModelSelector widgets require "
                            'ModelSelector(type="...") as their default'
                        ),
                    )
                )
                continue
            seen[name] = "input socket"
            inputs.append(
                InputSocket(name=name, data_type=normalize_socket_type(type_text))
            )
            continue
        if model_selector:
            try:
                model_type = _model_selector_type(node.value)
            except ValueError as exc:
                errors.append(
                    SyntaxErrorInfo(
                        line=node.lineno,
                        column=node.col_offset,
                        message=str(exc),
                    )
                )
                continue
            seen[name] = "widget"
            widgets.append(
                WidgetParam(
                    name=name,
                    data_type="ModelSelector",
                    default="",
                    model_type=model_type,
                )
            )
            continue
        options: list[Any] = []
        literal = _literal_subscript(node.annotation)
        if literal is not None:
            try:
                options, type_text = _literal_widget_options(literal)
            except ValueError as exc:
                errors.append(
                    SyntaxErrorInfo(
                        line=node.lineno,
                        column=node.col_offset,
                        message=str(exc),
                    )
                )
                continue
        try:
            default = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError) as exc:
            errors.append(
                SyntaxErrorInfo(
                    line=node.lineno,
                    column=node.col_offset,
                    message=(
                        f"Widget default for '{name}' must be a literal "
                        f"value: {exc}"
                    ),
                )
            )
            continue
        if options and not widget_choice_allowed(default, options):
            errors.append(
                SyntaxErrorInfo(
                    line=node.lineno,
                    column=node.col_offset,
                    message=(
                        f"Widget default for '{name}' must be one of "
                        f"{options!r}"
                    ),
                )
            )
            continue
        seen[name] = "widget"
        widgets.append(
            WidgetParam(
                name=name,
                data_type=type_text,
                default=default,
                options=options,
            )
        )
    return _Bindings(inputs, widgets, hoisted_imports, blank_spans, errors)


def _multiline_string_rows(source: str) -> set[int]:
    """Rows that are continuation lines of a multi-line string literal."""
    rows: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(StringIO(source).readline):
            is_string_token = tok.type == token.STRING or token.tok_name[
                tok.type
            ].startswith("FSTRING")
            if is_string_token and tok.end[0] > tok.start[0]:
                rows.update(range(tok.start[0] + 1, tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return set()
    return rows


def _sanitize_name(node_id: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_]", "_", node_id).strip("_")
    if not clean or clean[0].isdigit() or keyword.iskeyword(clean):
        clean = f"node_{clean}"
    return clean


def _wrap(
    source: str,
    node_id: str,
    inputs: list[InputSocket],
    widgets: list[WidgetParam],
    hoisted_imports: list[str],
    blank_spans: list[tuple[int, int]],
) -> tuple[str, str, int]:
    body_lines = source.split("\n")
    for start, end in blank_spans:
        for index in range(start - 1, min(end, len(body_lines))):
            body_lines[index] = ""
    string_rows = _multiline_string_rows(source)
    indented = [
        f"    {line}" if line.strip() and (index + 1) not in string_rows else line
        for index, line in enumerate(body_lines)
    ]

    func_name = f"__node_exec_{_sanitize_name(node_id)}"
    params = [socket.name for socket in inputs] + [
        f"{widget.name}={widget.default!r}" for widget in widgets
    ]
    lines: list[str] = []
    for chunk in hoisted_imports:
        lines.extend(chunk.split("\n"))
    header_lines = len(lines)
    lines.append(f"def {func_name}({', '.join(params)}):")
    if any(line.strip() for line in indented):
        lines.extend(indented)
    else:
        lines.append("    pass")
    return "\n".join(lines), func_name, header_lines + 1
