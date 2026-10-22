import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type {
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
} from "react";

import { useSystemMetrics } from "../hooks/useSystemMetrics";
import {
  clampPercent,
  formatBytes,
  formatPercent,
  formatTemperature,
} from "../lib/systemMetrics";
import { useSettingsStore } from "../store/useSettingsStore";
import type { MonitorPosition } from "../store/useSettingsStore";
import { CloseIcon, GripIcon } from "./icons";

const MONITOR_WIDTH = 260;
const ESTIMATED_HEIGHT = 172;
const VIEWPORT_MARGIN = 8;
const KEYBOARD_STEP = 16;

function viewportSize(): { width: number; height: number } {
  return {
    width: typeof window === "undefined" ? 1024 : window.innerWidth,
    height: typeof window === "undefined" ? 768 : window.innerHeight,
  };
}

function defaultPosition(): MonitorPosition {
  const viewport = viewportSize();
  return {
    x: Math.max(VIEWPORT_MARGIN, viewport.width / 2 - MONITOR_WIDTH / 2),
    y: Math.max(VIEWPORT_MARGIN, viewport.height - ESTIMATED_HEIGHT - 24),
  };
}

function clampPosition(
  position: MonitorPosition,
  size?: { width: number; height: number },
): MonitorPosition {
  const viewport = viewportSize();
  const width = size?.width ?? MONITOR_WIDTH;
  const height = size?.height ?? ESTIMATED_HEIGHT;
  const maxX = Math.max(
    VIEWPORT_MARGIN,
    viewport.width - width - VIEWPORT_MARGIN,
  );
  const maxY = Math.max(
    VIEWPORT_MARGIN,
    viewport.height - height - VIEWPORT_MARGIN,
  );
  return {
    x: Math.min(maxX, Math.max(VIEWPORT_MARGIN, position.x)),
    y: Math.min(maxY, Math.max(VIEWPORT_MARGIN, position.y)),
  };
}

interface DragState {
  pointerId: number;
  startX: number;
  startY: number;
  origin: MonitorPosition;
}

function MetricBar({
  label,
  percent,
  detail,
  tone,
  unavailable = false,
}: {
  label: string;
  percent: number | null;
  detail: string;
  tone: string;
  unavailable?: boolean;
}) {
  const value = clampPercent(percent);
  return (
    <div className={unavailable ? "opacity-45" : undefined}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[9px] font-semibold uppercase tracking-wider text-zinc-400">
          {label}
        </span>
        <span className="truncate font-mono text-[10px] tabular-nums text-zinc-200">
          {formatPercent(percent)}
          <span className="text-zinc-500"> · {detail}</span>
        </span>
      </div>
      <div
        role="progressbar"
        aria-label={`${label} utilization`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value)}
        aria-valuetext={`${label} ${formatPercent(percent)}, ${detail}`}
        className="mt-0.5 h-1.5 overflow-hidden rounded-full bg-zinc-700/45"
      >
        <div
          className={`h-full rounded-full ${tone} transition-[width] duration-300`}
          style={{ width: `${unavailable ? 0 : value}%` }}
        />
      </div>
    </div>
  );
}

export function SystemMonitor() {
  const intervalMs = useSettingsStore(
    (state) => state.settings.systemMonitorIntervalMs,
  );
  const storedPosition = useSettingsStore(
    (state) => state.settings.systemMonitorPosition,
  );
  const updateSetting = useSettingsStore((state) => state.updateSetting);
  const { metrics, error, updating } = useSystemMetrics(true, intervalMs);

  const rootRef = useRef<HTMLDivElement>(null);
  const positionRef = useRef<MonitorPosition | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [position, setPosition] = useState<MonitorPosition>(() =>
    clampPosition(storedPosition ?? defaultPosition()),
  );
  const [dragging, setDragging] = useState(false);

  const applyPosition = useCallback(
    (next: MonitorPosition, persist: boolean) => {
      const rect = rootRef.current?.getBoundingClientRect();
      const clamped = clampPosition(next, rect);
      const current = positionRef.current;
      const unchanged =
        current !== null && current.x === clamped.x && current.y === clamped.y;
      if (!unchanged) {
        positionRef.current = clamped;
        setPosition(clamped);
      }
      if (persist) {
        updateSetting("systemMonitorPosition", clamped);
      }
    },
    [updateSetting],
  );

  useLayoutEffect(() => {
    applyPosition(position, false);
  }, [applyPosition, position]);

  useEffect(() => {
    applyPosition(storedPosition ?? defaultPosition(), false);
  }, [applyPosition, storedPosition]);

  useEffect(() => {
    const onResize = () => {
      const current = positionRef.current;
      if (current) {
        applyPosition(current, false);
      }
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [applyPosition]);

  useEffect(() => {
    if (!dragging) {
      return;
    }
    const onPointerMove = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag || event.pointerId !== drag.pointerId) {
        return;
      }
      applyPosition(
        {
          x: drag.origin.x + event.clientX - drag.startX,
          y: drag.origin.y + event.clientY - drag.startY,
        },
        false,
      );
    };
    const onPointerUp = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag || event.pointerId !== drag.pointerId) {
        return;
      }
      dragRef.current = null;
      setDragging(false);
      const current = positionRef.current;
      if (current) {
        applyPosition(current, true);
      }
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
    };
  }, [applyPosition, dragging]);

  const startDrag = (event: ReactPointerEvent<HTMLElement>) => {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    const rect = rootRef.current?.getBoundingClientRect();
    const origin = clampPosition(positionRef.current ?? position, rect);
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      origin,
    };
    setDragging(true);
  };

  const moveWithKeyboard = (
    event: ReactKeyboardEvent<HTMLButtonElement>,
  ) => {
    const deltas: Record<string, [number, number]> = {
      ArrowLeft: [-KEYBOARD_STEP, 0],
      ArrowRight: [KEYBOARD_STEP, 0],
      ArrowUp: [0, -KEYBOARD_STEP],
      ArrowDown: [0, KEYBOARD_STEP],
    };
    const delta = deltas[event.key];
    if (!delta) {
      return;
    }
    event.preventDefault();
    const current = positionRef.current ?? position;
    applyPosition(
      { x: current.x + delta[0], y: current.y + delta[1] },
      true,
    );
  };

  const gpu = metrics?.gpus[0] ?? null;
  const cpu = metrics?.cpu ?? null;
  const memory = metrics?.memory ?? null;
  const vramPercent =
    gpu?.vram_used_bytes != null && gpu.vram_total_bytes != null
      ? (gpu.vram_used_bytes / gpu.vram_total_bytes) * 100
      : null;
  const stale = Boolean(error && metrics);

  const cpuDetail =
    cpu?.cores != null ? `${cpu.cores} cores` : "N/A";
  const gpuDetail = gpu ? formatTemperature(gpu.temperature_c) : "N/A";
  const vramDetail =
    gpu?.vram_used_bytes != null && gpu.vram_total_bytes != null
      ? `${formatBytes(gpu.vram_used_bytes)} / ${formatBytes(gpu.vram_total_bytes)}`
      : "N/A";
  const ramDetail = memory
    ? `${formatBytes(memory.used_bytes)} / ${formatBytes(memory.total_bytes)}`
    : "N/A";

  return (
    <div
      ref={rootRef}
      role="region"
      aria-label="System monitor"
      data-testid="system-monitor"
      className={`fixed z-30 w-[260px] select-none text-zinc-200 drop-shadow-[0_1px_3px_rgba(0,0,0,0.9)] ${
        stale ? "opacity-70" : ""
      }`}
      style={{ left: position.x, top: position.y }}
    >
      <div
        className={`flex touch-none items-center gap-1.5 pb-1 ${
          dragging ? "cursor-grabbing" : "cursor-move"
        }`}
        onPointerDown={startDrag}
      >
        <button
          type="button"
          aria-label="Move system monitor"
          title="Drag to move · arrow keys to nudge"
          onKeyDown={moveWithKeyboard}
          className="-ml-1 shrink-0 rounded p-0.5 text-zinc-500 hover:bg-zinc-800/60 hover:text-cyan-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/70"
        >
          <GripIcon className="h-3 w-3" />
        </button>
        <span
          aria-hidden="true"
          className={`h-1.5 w-1.5 rounded-full ${
            error ? "bg-red-400" : updating ? "bg-amber-300" : "bg-emerald-400"
          }`}
        />
        <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-zinc-300">
          System
        </span>
        <span
          className="min-w-0 flex-1 truncate text-right text-[9px] text-zinc-500"
          title={gpu?.name ?? "No NVIDIA GPU detected"}
        >
          {gpu?.name ?? "No NVIDIA GPU"}
        </span>
        <button
          type="button"
          aria-label="Hide system monitor"
          title="Hide system monitor (re-enable in Settings)"
          onPointerDown={(event) => event.stopPropagation()}
          onClick={() => updateSetting("showSystemMonitor", false)}
          className="shrink-0 rounded p-0.5 text-zinc-500 hover:bg-zinc-800/60 hover:text-zinc-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/70"
        >
          <CloseIcon className="h-3 w-3" />
        </button>
      </div>

      {error && !metrics && (
        <p className="pb-1 text-[9px] text-amber-300/90">
          System metrics unavailable
        </p>
      )}

      <div className="space-y-1.5">
        <MetricBar
          label="CPU"
          percent={cpu?.utilization ?? null}
          detail={cpuDetail}
          tone="bg-cyan-400"
          unavailable={!cpu}
        />
        <MetricBar
          label="GPU"
          percent={gpu?.utilization ?? null}
          detail={gpuDetail}
          tone="bg-violet-400"
          unavailable={!gpu}
        />
        <MetricBar
          label="VRAM"
          percent={vramPercent}
          detail={vramDetail}
          tone="bg-amber-400"
          unavailable={!gpu}
        />
        <MetricBar
          label="RAM"
          percent={memory?.utilization ?? null}
          detail={ramDetail}
          tone="bg-emerald-400"
          unavailable={!memory}
        />
      </div>
    </div>
  );
}
