"""ComplexUI operator launcher.

Modes:
    dev       backend (auto-reload) + Vite dev server (default)
    backend   backend only
    frontend  Vite dev server only
    prod      build the frontend, then serve everything from the backend
    check     full verification gate (pytest, typecheck, vitest, build)

Examples:
    .\\start.cmd
    .\\start.cmd prod
    .\\start.cmd check
    .\\start.cmd dev --backend-port 8010 --open
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import IO

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = REPO_ROOT / "frontend"

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"
MAGENTA = "\x1b[35m"

TAGS = {"backend": (CYAN, "[backend] "), "frontend": (MAGENTA, "[frontend]")}
HEALTH_TIMEOUT_SECONDS = 45.0
POLL_INTERVAL_SECONDS = 0.2


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _enable_windows_ansi() -> bool:
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


class ConsoleRenderer:
    """Write pumped logs while keeping one progress line at the bottom."""

    def __init__(
        self,
        stream: IO[str] | None = None,
        *,
        enabled: bool | None = None,
        ansi: bool | None = None,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = is_tty if enabled is None else enabled
        self.ansi = (
            self.enabled
            and (_enable_windows_ansi() if ansi is None else ansi)
        )
        self._lock = threading.RLock()
        self._progress: str | None = None
        self._drawn_width = 0

    def _erase(self) -> None:
        if self._drawn_width == 0:
            return
        if self.ansi:
            self.stream.write("\r\x1b[2K")
        else:
            self.stream.write("\r" + (" " * self._drawn_width) + "\r")
        self._drawn_width = 0

    def _draw(self) -> None:
        if not self.enabled or not self._progress:
            return
        self.stream.write(self._progress)
        self.stream.flush()
        self._drawn_width = len(_ANSI_RE.sub("", self._progress))

    def write(self, text: str) -> None:
        with self._lock:
            self._erase()
            self.stream.write(text)
            if not text.endswith("\n"):
                self.stream.write("\n")
            self.stream.flush()
            self._draw()

    def update_progress(self, text: str) -> None:
        with self._lock:
            self._progress = text or None
            if not self.enabled:
                if text:
                    self.stream.write(text + "\n")
                    self.stream.flush()
                return
            self._erase()
            self._draw()

    def clear_progress(self) -> None:
        with self._lock:
            self._erase()
            self._progress = None


class Theme:
    """Tiny ANSI helper honoring --no-color / NO_COLOR / tty state."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.console: ConsoleRenderer | None = None

    def paint(self, text: str, color: str) -> str:
        return f"{color}{text}{RESET}" if self.enabled else text

    def tag(self, name: str) -> str:
        color, label = TAGS[name]
        return self.paint(label, color)

    def _write(self, text: str) -> None:
        if self.console is not None:
            self.console.write(text + "\n")
        else:
            print(text)

    def info(self, message: str) -> None:
        self._write(f"{self.paint('[complexui]', BOLD)} {message}")

    def warn(self, message: str) -> None:
        self._write(
            f"{self.paint('[complexui]', BOLD)} {self.paint(message, YELLOW)}"
        )

    def fail(self, message: str) -> None:
        self._write(
            f"{self.paint('[complexui]', BOLD)} {self.paint(message, RED)}"
        )

    def ok(self, message: str) -> None:
        self._write(
            f"{self.paint('[complexui]', BOLD)} {self.paint(message, GREEN)}"
        )


def _resolve_color(explicit_off: bool) -> bool:
    if explicit_off or "NO_COLOR" in os.environ:
        return False
    force_color = os.environ.get("FORCE_COLOR")
    if force_color is not None:
        return force_color.strip().lower() not in {"0", "false", "no", "off"}
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    return sys.stdout.isatty()


def _default_log_format() -> str:
    value = os.environ.get("COMPLEXUI_LOG_FORMAT", "text").strip().lower()
    return value if value in {"text", "json"} else "text"


def _default_accelerator() -> str:
    value = os.environ.get("COMPLEXUI_ACCELERATOR", "auto").strip().lower()
    return value if value in {"auto", "cpu", "cuda"} else "auto"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="launch.py",
        description="ComplexUI launcher: dev servers, production mode, checks.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="dev",
        choices=("dev", "backend", "frontend", "prod", "check"),
        help="what to run (default: dev)",
    )
    parser.add_argument(
        "--backend-port", type=int, default=8000, help="backend port (default 8000)"
    )
    parser.add_argument(
        "--frontend-port", type=int, default=5173, help="Vite port (default 5173)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind address (default 127.0.0.1)"
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="allow non-loopback access (disables Host/Origin protection)",
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "cuda"),
        default=_default_accelerator(),
        help=(
            "PyTorch accelerator target (default: auto; CUDA uses the cu130 index)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("COMPLEXUI_LOG_LEVEL", "INFO"),
        help="backend console log level (default INFO)",
    )
    parser.add_argument(
        "--log-format",
        choices=("text", "json"),
        default=_default_log_format(),
        help="backend log format (default text; json emits one object per line)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="directory for daily .txt logs (default backend/workspace/logs)",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="disable persistent .txt logs and keep console output only",
    )
    parser.add_argument(
        "--no-reload", action="store_true", help="disable uvicorn auto-reload"
    )
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    parser.add_argument(
        "--open", action="store_true", help="open the UI in a browser once ready"
    )
    args = parser.parse_args(argv)
    for flag in ("backend_port", "frontend_port"):
        value = getattr(args, flag)
        if not 1 <= value <= 65535:
            parser.error(f"--{flag.replace('_', '-')} must be 1-65535")
    if not is_loopback_host(args.host) and not args.allow_network:
        parser.error(
            f"--host {args.host} exposes the API beyond loopback; "
            "add --allow-network to confirm the risk"
        )
    return args


def is_loopback_host(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1", "[::1]"}


def require_tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        print(
            f"[complexui] '{name}' was not found on PATH. "
            "Install it and retry (see README.md).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return found


def build_backend_command(args: argparse.Namespace, *, reload: bool) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.backend_port),
    ]
    if reload:
        command.append("--reload")
    return command


def build_frontend_command(npm: str, args: argparse.Namespace) -> list[str]:
    command = [
        npm,
        "run",
        "dev",
        "--",
        "--port",
        str(args.frontend_port),
        "--strictPort",
    ]
    if not is_loopback_host(args.host):
        command.extend(["--host", args.host])
    return command


def backend_environment(args: argparse.Namespace, *, serve_dist: bool) -> dict[str, str]:
    environment = os.environ.copy()
    environment["COMPLEXUI_LOG_LEVEL"] = args.log_level
    environment["COMPLEXUI_LOG_FORMAT"] = args.log_format
    environment["COMPLEXUI_ACCELERATOR"] = args.accelerator
    if args.no_log_file:
        environment["COMPLEXUI_LOG_DIR"] = "0"
    elif args.log_dir:
        environment["COMPLEXUI_LOG_DIR"] = str(Path(args.log_dir).expanduser())
    if args.allow_network and not is_loopback_host(args.host):
        environment.setdefault("COMPLEXUI_ALLOWED_HOSTS", "*")
    if serve_dist:
        environment["COMPLEXUI_SERVE_DIST"] = str(FRONTEND_DIR / "dist")
    return environment


def spawn(
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    theme: Theme,
    *,
    prefix: bool = True,
    console: ConsoleRenderer | None = None,
):
    kwargs: dict[str, object] = {}
    if os.name != "nt":
        kwargs["start_new_session"] = True
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **kwargs,
    )
    tag = theme.tag(name)

    def pump() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            text = f"{tag} {line}" if prefix else line
            if console is None:
                sys.stdout.write(text)
                sys.stdout.flush()
                continue
            if "[progress]" in text:
                console.update_progress(text.rstrip("\n"))
                continue
            if any(
                marker in text
                for marker in ("[start]", "[done ]", "[error]", "run ")
            ):
                console.clear_progress()
            console.write(text)

    threading.Thread(target=pump, name=f"{name}-pump", daemon=True).start()
    theme.info(f"started {name}: {' '.join(command)}")
    return process


def terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def shutdown_all(processes: dict[str, subprocess.Popen]) -> None:
    for process in processes.values():
        terminate_tree(process)
    for process in processes.values():
        if process.stdout is not None:
            process.stdout.close()


def wait_for_http(url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(POLL_INTERVAL_SECONDS)
    return False


def run_streamed(label: str, command: list[str], cwd: Path, env: dict[str, str] | None) -> int:
    print()
    print(f"=== {label} ===")
    sys.stdout.flush()
    completed = subprocess.run(command, cwd=str(cwd), env=env)
    return completed.returncode


def ensure_frontend_dependencies(npm: str, theme: Theme) -> int:
    """Run ``npm install`` once when ``frontend/node_modules`` is missing."""
    if (FRONTEND_DIR / "node_modules").is_dir():
        return 0
    theme.info("frontend dependencies missing; running `npm install` (one-time)...")
    code = run_streamed("npm install", [npm, "install"], FRONTEND_DIR, None)
    if code != 0:
        theme.fail(f"`npm install` failed (exit code {code})")
        return code
    theme.ok("frontend dependencies installed")
    return 0


def run_check(args: argparse.Namespace, theme: Theme, npm: str) -> int:
    require_tool("uv")
    steps: list[tuple[str, list[str], Path]] = [
        ("backend lint", ["uv", "run", "ruff", "check", "."], BACKEND_DIR),
        ("backend tests", ["uv", "run", "pytest", "-q"], BACKEND_DIR),
        ("frontend lint", [npm, "run", "lint"], FRONTEND_DIR),
        ("frontend typecheck", [npm, "run", "typecheck"], FRONTEND_DIR),
        ("frontend tests", [npm, "run", "test"], FRONTEND_DIR),
        ("frontend build", [npm, "run", "build"], FRONTEND_DIR),
    ]
    for label, command, cwd in steps:
        theme.info(f"check: {label}")
        code = run_streamed(label, command, cwd, None)
        if code != 0:
            theme.fail(f"{label} failed (exit code {code})")
            return code
    theme.ok("all checks passed")
    return 0


def warn_port_mismatches(args: argparse.Namespace, theme: Theme) -> None:
    if args.backend_port != 8000:
        theme.warn(
            "frontend/vite.config.ts proxies /api and /ws to port 8000; "
            f"--backend-port {args.backend_port} requires updating the proxy to match"
        )
    if args.frontend_port != 5173:
        theme.warn(
            "backend CORS allows only http://localhost:5173; "
            f"--frontend-port {args.frontend_port} requires updating the CORS list"
        )


def supervise(processes: dict[str, subprocess.Popen]) -> int:
    while True:
        for process in processes.values():
            code = process.poll()
            if code is not None:
                return code
        time.sleep(POLL_INTERVAL_SECONDS)


def probe_host(host: str) -> str:
    if host in ("0.0.0.0", "::", ""):
        return "127.0.0.1"
    return host


def _run_version(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip().splitlines()
    return output[0].strip() if output else None


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _fetch_json(url: str) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return payload if isinstance(payload, dict) else None


def print_banner(
    args: argparse.Namespace,
    theme: Theme,
    health: dict[str, object] | None,
    metrics: dict[str, object] | None,
) -> None:
    health = health or {}
    branch = _git_value("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    commit = _git_value("rev-parse", "--short", "HEAD") or "unknown"
    dirty = _git_value("status", "--porcelain") is not None
    remote = _git_value("remote", "get-url", "origin") or "no remote"
    version = str(health.get("version", "0.1.0"))
    accelerator = str(health.get("effective_accelerator", args.accelerator))
    workspace = Path(
        os.environ.get(
            "COMPLEXUI_WORKSPACE", str(BACKEND_DIR / "workspace")
        )
    ).expanduser()
    inputs = Path(
        os.environ.get("COMPLEXUI_INPUT_DIR", str(REPO_ROOT / "inputs"))
    ).expanduser()
    outputs = Path(
        os.environ.get("COMPLEXUI_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
    ).expanduser()
    models = str(health.get("models_root", REPO_ROOT / "models"))
    if args.no_log_file:
        log_dir: Path | None = None
    elif args.log_dir:
        log_dir = Path(args.log_dir).expanduser()
    else:
        configured_logs = os.environ.get("COMPLEXUI_LOG_DIR")
        log_dir = (
            Path(configured_logs).expanduser()
            if configured_logs and configured_logs != "0"
            else workspace / "logs"
        )
    limits = health.get("limits")
    limits = limits if isinstance(limits, dict) else {}
    torch = health.get("torch")
    gpus = metrics.get("gpus") if isinstance(metrics, dict) else None
    gpu_text = ""
    if isinstance(gpus, list) and gpus:
        parts = []
        for gpu in gpus[:2]:
            if not isinstance(gpu, dict):
                continue
            name = str(gpu.get("name", "GPU"))
            vram = gpu.get("vram_total_bytes")
            if isinstance(vram, (int, float)) and vram > 0:
                name += f" ({vram / (1024 * 1024):.0f} MiB)"
            parts.append(name)
        gpu_text = ", ".join(parts)
    if not gpu_text:
        gpu_text = "none detected"
    python_version = platform.python_version()
    uv = _run_version(["uv", "--version"]) or "unavailable"
    node = _run_version(["node", "--version"]) or "unavailable"
    npm = _run_version(["npm", "--version"]) or "unavailable"
    url_host = probe_host(args.host)
    backend_url = f"http://{url_host}:{args.backend_port}"
    ui_url = (
        f"http://localhost:{args.frontend_port}"
        if args.mode == "dev"
        else backend_url
    )
    divider = theme.paint("-" * 78, DIM)
    print(divider)
    print(
        f"{theme.paint('ComplexUI', BOLD + CYAN)} {version}  "
        f"{theme.paint('mode', DIM)}={args.mode}  "
        f"{theme.paint('git', DIM)}={branch}@{commit}"
        f"{' (dirty)' if dirty else ''}"
    )
    print(f"{theme.paint('git remote', DIM)} {remote}")
    print(
        f"{theme.paint('runtime', DIM)} python {python_version} · {uv} · "
        f"node {node} · npm {npm}"
    )
    print(
        f"{theme.paint('platform', DIM)} {platform.platform()} · "
        f"{theme.paint('venv', DIM)} {sys.prefix}"
    )
    print(
        f"{theme.paint('urls', DIM)} backend {backend_url} · UI {ui_url}"
    )
    print(
        f"{theme.paint('paths', DIM)} workspace {workspace} · "
        f"inputs {inputs} · outputs {outputs} · models {models}"
    )
    print(
        f"{theme.paint('hardware', DIM)} accelerator {accelerator} · "
        f"torch {torch or 'not installed'} · CUDA "
        f"{'yes' if health.get('torch_cuda') else 'no'} · GPU {gpu_text}"
    )
    print(
        f"{theme.paint('limits', DIM)} node "
        f"{limits.get('node_timeout_seconds', '?')}s · run "
        f"{limits.get('run_timeout_seconds', '?')}s · memory "
        f"{limits.get('max_memory_mb', '?')} MiB"
    )
    print(
        f"{theme.paint('logs', DIM)} console {args.log_level.upper()}/"
        f"{args.log_format} · files "
        f"{log_dir if log_dir is not None else 'disabled'} · runs "
        f"{log_dir / 'runs' if log_dir is not None else 'disabled'}"
    )
    print(divider)


def launch_servers(
    args: argparse.Namespace,
    theme: Theme,
    npm: str,
    *,
    with_frontend: bool,
    reload: bool,
    serve_dist: bool,
) -> int:
    processes: dict[str, subprocess.Popen] = {}
    console = ConsoleRenderer()
    theme.console = console
    theme.info(
        f"mode={'dev' if with_frontend else ('prod' if serve_dist else 'backend')} "
        f"backend=http://{args.host}:{args.backend_port} "
        f"log-level={args.log_level.upper()} log-format={args.log_format}"
    )
    try:
        processes["backend"] = spawn(
            "backend",
            build_backend_command(args, reload=reload),
            BACKEND_DIR,
            backend_environment(args, serve_dist=serve_dist),
            theme,
            prefix=args.log_format != "json",
            console=console,
        )
        if with_frontend:
            processes["frontend"] = spawn(
                "frontend",
                build_frontend_command(npm, args),
                FRONTEND_DIR,
                os.environ.copy(),
                theme,
                console=console,
            )

        health_url = f"http://{probe_host(args.host)}:{args.backend_port}/api/health"
        if not wait_for_http(health_url, HEALTH_TIMEOUT_SECONDS):
            theme.fail(f"backend did not become ready at {health_url}")
            return 1
        theme.ok(f"backend ready at {health_url}")

        ui_url = f"http://{probe_host(args.host)}:{args.backend_port}"
        if with_frontend:
            ui_url = f"http://localhost:{args.frontend_port}"
            if not wait_for_http(ui_url, HEALTH_TIMEOUT_SECONDS):
                theme.fail(f"frontend did not become ready at {ui_url}")
                return 1
            theme.ok(f"frontend ready at {ui_url}")

        health = _fetch_json(health_url)
        metrics = _fetch_json(
            f"http://{probe_host(args.host)}:{args.backend_port}/api/system/metrics"
        )
        print_banner(args, theme, health, metrics)
        theme.info(f"UI: {theme.paint(ui_url, BOLD)}  (Ctrl+C to stop)")
        if args.open:
            webbrowser.open(ui_url)
        return supervise(processes)
    except KeyboardInterrupt:
        print()
        theme.info("shutting down...")
        return 0
    finally:
        shutdown_all(processes)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    theme = Theme(_resolve_color(args.no_color))

    if (
        args.allow_network
        and not is_loopback_host(args.host)
        and args.mode in {"dev", "backend", "prod"}
    ):
        theme.warn(
            "NETWORK MODE: the API executes Python code without "
            "authentication and is reachable beyond loopback. "
            "Use only on trusted networks."
        )

    npm = ""
    if args.mode in ("dev", "frontend", "prod", "check"):
        npm = require_tool("npm")
    if args.mode in ("dev", "backend", "prod", "check"):
        require_tool("uv")

    if npm:
        code = ensure_frontend_dependencies(npm, theme)
        if code != 0:
            return code

    if args.mode == "check":
        return run_check(args, theme, npm)

    if args.mode == "frontend":
        console = ConsoleRenderer()
        theme.console = console
        theme.info(f"frontend-only mode on http://localhost:{args.frontend_port}")
        warn_port_mismatches(args, theme)
        processes = {
            "frontend": spawn(
                "frontend",
                build_frontend_command(npm, args),
                FRONTEND_DIR,
                os.environ.copy(),
                theme,
                console=console,
            )
        }
        try:
            return supervise(processes)
        except KeyboardInterrupt:
            print()
            theme.info("shutting down...")
            return 0
        finally:
            shutdown_all(processes)

    if args.mode == "prod":
        warn_port_mismatches(args, theme)
        code = run_streamed(
            "frontend build", [npm, "run", "build"], FRONTEND_DIR, None
        )
        if code != 0:
            theme.fail(f"frontend build failed (exit code {code})")
            return code
        theme.ok("frontend bundle built into frontend/dist")
        return launch_servers(
            args,
            theme,
            npm,
            with_frontend=False,
            reload=False,
            serve_dist=True,
        )

    if args.mode == "backend":
        return launch_servers(
            args,
            theme,
            npm,
            with_frontend=False,
            reload=not args.no_reload,
            serve_dist=False,
        )

    warn_port_mismatches(args, theme)
    return launch_servers(
        args,
        theme,
        npm,
        with_frontend=True,
        reload=not args.no_reload,
        serve_dist=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
