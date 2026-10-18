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
import os
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


class Theme:
    """Tiny ANSI helper honoring --no-color / NO_COLOR / tty state."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def paint(self, text: str, color: str) -> str:
        return f"{color}{text}{RESET}" if self.enabled else text

    def tag(self, name: str) -> str:
        color, label = TAGS[name]
        return self.paint(label, color)

    def info(self, message: str) -> None:
        print(f"{self.paint('[complexui]', BOLD)} {message}")

    def warn(self, message: str) -> None:
        print(f"{self.paint('[complexui]', BOLD)} {self.paint(message, YELLOW)}")

    def fail(self, message: str) -> None:
        print(f"{self.paint('[complexui]', BOLD)} {self.paint(message, RED)}")

    def ok(self, message: str) -> None:
        print(f"{self.paint('[complexui]', BOLD)} {self.paint(message, GREEN)}")


def _resolve_color(explicit_off: bool) -> bool:
    if explicit_off or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


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
        "--log-level",
        default=os.environ.get("COMPLEXUI_LOG_LEVEL", "INFO"),
        help="backend console log level (default INFO)",
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
    return args


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
    return [
        npm,
        "run",
        "dev",
        "--",
        "--port",
        str(args.frontend_port),
        "--strictPort",
    ]


def backend_environment(args: argparse.Namespace, *, serve_dist: bool) -> dict[str, str]:
    environment = os.environ.copy()
    environment["COMPLEXUI_LOG_LEVEL"] = args.log_level
    if serve_dist:
        environment["COMPLEXUI_SERVE_DIST"] = str(FRONTEND_DIR / "dist")
    return environment


def spawn(name: str, command: list[str], cwd: Path, env: dict[str, str], theme: Theme):
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
            sys.stdout.write(f"{tag} {line}")
            sys.stdout.flush()

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
        ("backend tests", ["uv", "run", "pytest", "-q"], BACKEND_DIR),
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
        for name, process in processes.items():
            code = process.poll()
            if code is not None:
                return code
        time.sleep(POLL_INTERVAL_SECONDS)


def probe_host(host: str) -> str:
    if host in ("0.0.0.0", "::", ""):
        return "127.0.0.1"
    return host


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
    theme.info(
        f"mode={'dev' if with_frontend else ('prod' if serve_dist else 'backend')} "
        f"backend=http://{args.host}:{args.backend_port} "
        f"log-level={args.log_level.upper()}"
    )
    try:
        processes["backend"] = spawn(
            "backend",
            build_backend_command(args, reload=reload),
            BACKEND_DIR,
            backend_environment(args, serve_dist=serve_dist),
            theme,
        )
        if with_frontend:
            processes["frontend"] = spawn(
                "frontend",
                build_frontend_command(npm, args),
                FRONTEND_DIR,
                os.environ.copy(),
                theme,
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
        theme.info(f"frontend-only mode on http://localhost:{args.frontend_port}")
        warn_port_mismatches(args, theme)
        processes = {
            "frontend": spawn(
                "frontend",
                build_frontend_command(npm, args),
                FRONTEND_DIR,
                os.environ.copy(),
                theme,
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
