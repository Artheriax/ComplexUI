"""Process-isolated execution host with wall-clock and memory guards."""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing.connection import _ConnectionBase
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

from .execution_worker import run_worker
from .executor import GraphPlan, cache_snapshot, transferable_cache
from .schemas import WorkflowGraph
from .security import get_security_registry

DEFAULT_NODE_TIMEOUT_SECONDS = 600.0
DEFAULT_RUN_TIMEOUT_SECONDS = 3_600.0
DEFAULT_MAX_MEMORY_MB = 12_288
ENV_NODE_TIMEOUT = "COMPLEXUI_NODE_TIMEOUT_SECONDS"
ENV_RUN_TIMEOUT = "COMPLEXUI_RUN_TIMEOUT_SECONDS"
ENV_MAX_MEMORY_MB = "COMPLEXUI_MAX_MEMORY_MB"


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    node_timeout_seconds: float
    run_timeout_seconds: float
    max_memory_bytes: int


@dataclass(slots=True)
class WorkerHandle:
    process: BaseProcess
    connection: _ConnectionBase[Any, Any]
    stop_event: Any


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def execution_limits() -> ExecutionLimits:
    """Read per-run node/run/memory limits from the environment."""
    return ExecutionLimits(
        node_timeout_seconds=_env_float(ENV_NODE_TIMEOUT, DEFAULT_NODE_TIMEOUT_SECONDS),
        run_timeout_seconds=_env_float(ENV_RUN_TIMEOUT, DEFAULT_RUN_TIMEOUT_SECONDS),
        max_memory_bytes=_env_int(ENV_MAX_MEMORY_MB, DEFAULT_MAX_MEMORY_MB) * 1024 * 1024,
    )


def start_worker(
    graph: WorkflowGraph,
    plan: GraphPlan,
    memory_limit_bytes: int,
) -> WorkerHandle:
    """Spawn a disposable process with the current cache and approvals."""
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    stop_event = context.Event()
    process = context.Process(
        target=run_worker,
        args=(
            graph,
            plan,
            get_security_registry().approved_pairs(),
            transferable_cache(cache_snapshot()),
            stop_event,
            child_connection,
            memory_limit_bytes,
        ),
        daemon=True,
        name="complexui-execution",
    )
    try:
        process.start()
    except Exception:
        parent_connection.close()
        child_connection.close()
        raise
    child_connection.close()
    return WorkerHandle(
        process=process,
        connection=parent_connection,
        stop_event=stop_event,
    )


def stop_worker(handle: WorkerHandle, *, terminate: bool) -> None:
    """Stop or cooperatively finish a worker process and its descendants."""
    if terminate and handle.process.is_alive():
        if sys.platform == "win32":
            try:
                subprocess.run(  # noqa: S603 - PID is numeric
                    [
                        "taskkill",
                        "/PID",
                        str(handle.process.pid),
                        "/T",
                        "/F",
                    ],
                    capture_output=True,
                    check=False,
                    timeout=5,
                )
            except (OSError, subprocess.SubprocessError):
                handle.process.terminate()
        else:
            handle.process.terminate()
    handle.process.join(timeout=5)
    if handle.process.is_alive():
        handle.process.kill()
        handle.process.join(timeout=5)
    try:
        handle.connection.close()
    except OSError:
        pass


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


@lru_cache(maxsize=1)
def _windows_memory_api():
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.OpenProcess.argtypes = [
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_ulong,
    ]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    return kernel32, psapi


def process_memory_bytes(pid: int) -> int | None:
    """Best-effort resident memory for one process, or ``None`` if unavailable."""
    try:
        if sys.platform == "win32":
            kernel32, psapi = _windows_memory_api()
            handle = kernel32.OpenProcess(0x1000, 0, pid)
            if not handle:
                return None
            try:
                counters = _ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    return None
                return int(counters.WorkingSetSize)
            finally:
                kernel32.CloseHandle(handle)
        if sys.platform.startswith("linux"):
            statm = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8")
            resident_pages = int(statm.split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        output = subprocess.run(  # noqa: S603 - fixed command, PID is numeric
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            check=False,
            timeout=2,
        )
        if output.returncode != 0:
            return None
        return int(output.stdout.decode("utf-8", "replace").strip()) * 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


__all__ = [
    "DEFAULT_MAX_MEMORY_MB",
    "DEFAULT_NODE_TIMEOUT_SECONDS",
    "DEFAULT_RUN_TIMEOUT_SECONDS",
    "ENV_MAX_MEMORY_MB",
    "ENV_NODE_TIMEOUT",
    "ENV_RUN_TIMEOUT",
    "ExecutionLimits",
    "WorkerHandle",
    "execution_limits",
    "process_memory_bytes",
    "start_worker",
    "stop_worker",
]

