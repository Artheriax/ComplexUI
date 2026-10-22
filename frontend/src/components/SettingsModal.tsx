import { useState } from "react";
import type { ReactNode } from "react";

import { useDialogA11y } from "../hooks/useDialogA11y";
import { useHealthInfo } from "../hooks/useHealthInfo";
import { useGraphStore } from "../store/useGraphStore";
import type {
  CanvasBackgroundVariant,
  EdgeColorMode,
  EdgeLineStyle,
  HandleSize,
  SystemMonitorIntervalMs,
} from "../store/useSettingsStore";
import {
  SYSTEM_MONITOR_INTERVALS,
  useSettingsStore,
} from "../store/useSettingsStore";
import {
  CloseIcon,
  ResetIcon,
  SettingsIcon,
  WarningIcon,
} from "./icons";

type SettingsCategory =
  | "connections"
  | "canvas"
  | "system"
  | "execution"
  | "general";

const CATEGORIES: { id: SettingsCategory; label: string; description: string }[] = [
  {
    id: "connections",
    label: "Connections",
    description: "How sockets and connection lines look and behave.",
  },
  {
    id: "canvas",
    label: "Canvas",
    description: "Background, grid and minimap preferences.",
  },
  {
    id: "system",
    label: "System",
    description: "Live system monitor and backend runtime diagnostics.",
  },
  {
    id: "execution",
    label: "Execution",
    description: "Run behavior, safety prompts and backend limits.",
  },
  {
    id: "general",
    label: "General",
    description: "Reset preferences and storage information.",
  },
];

const MONITOR_INTERVAL_OPTIONS: {
  value: SystemMonitorIntervalMs;
  label: string;
}[] = SYSTEM_MONITOR_INTERVALS.map((value) => ({
  value,
  label: `${value / 1000}s`,
}));

const ACCENT_COLORS = [
  { value: "#22d3ee", label: "Cyan" },
  { value: "#34d399", label: "Emerald" },
  { value: "#fbbf24", label: "Amber" },
  { value: "#fb7185", label: "Rose" },
  { value: "#a78bfa", label: "Violet" },
  { value: "#38bdf8", label: "Sky" },
  { value: "#a1a1aa", label: "Zinc" },
];

const EDGE_STYLE_OPTIONS: {
  value: EdgeLineStyle;
  label: string;
  preview: ReactNode;
}[] = [
  { value: "bezier", label: "Bezier", preview: <EdgeStylePreview style="bezier" /> },
  {
    value: "smoothstep",
    label: "Smooth step",
    preview: <EdgeStylePreview style="smoothstep" />,
  },
  { value: "step", label: "Step", preview: <EdgeStylePreview style="step" /> },
  {
    value: "straight",
    label: "Straight",
    preview: <EdgeStylePreview style="straight" />,
  },
];

const COLOR_MODE_OPTIONS: { value: EdgeColorMode; label: string }[] = [
  { value: "socket", label: "Socket type" },
  { value: "single", label: "Single color" },
];

const HANDLE_SIZE_OPTIONS: { value: HandleSize; label: string }[] = [
  { value: "compact", label: "Compact" },
  { value: "comfortable", label: "Comfortable" },
  { value: "large", label: "Large" },
];

const BACKGROUND_OPTIONS: { value: CanvasBackgroundVariant; label: string }[] = [
  { value: "dots", label: "Dots" },
  { value: "lines", label: "Lines" },
  { value: "cross", label: "Cross" },
  { value: "none", label: "None" },
];

function EdgeStylePreview({ style }: { style: EdgeLineStyle }) {
  const path =
    style === "bezier"
      ? "M2 18 C 18 18, 18 4, 34 4"
      : style === "smoothstep"
        ? "M2 18 C 8 18, 8 4, 14 4 H 22 C 28 4, 28 18, 34 18"
        : style === "step"
          ? "M2 18 H 18 V 4 H 34"
          : "M2 18 L 34 4";
  return (
    <svg
      viewBox="0 0 36 22"
      className="h-5 w-8 shrink-0"
      fill="none"
      aria-hidden="true"
    >
      <path
        d={path}
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SettingRow({
  label,
  description,
  children,
}: {
  label: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-zinc-800/70 py-3 last:border-b-0">
      <div className="min-w-0 max-w-sm">
        <div className="text-xs font-medium text-zinc-200">{label}</div>
        <p className="mt-0.5 text-[11px] leading-snug text-zinc-500">{description}</p>
      </div>
      <div className="flex flex-wrap items-center justify-end gap-1.5">{children}</div>
    </div>
  );
}

function ToggleSwitch({
  checked,
  label,
  disabled = false,
  onChange,
}: {
  checked: boolean;
  label: string;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative h-5 w-9 shrink-0 rounded-full border transition-colors ${
        checked
          ? "border-cyan-500 bg-cyan-600/80"
          : "border-zinc-600 bg-zinc-800"
      } disabled:cursor-not-allowed disabled:opacity-40`}
    >
      <span
        aria-hidden="true"
        className={`absolute left-0.5 top-0.5 h-3.5 w-3.5 rounded-full bg-white shadow transition-transform ${
          checked ? "translate-x-4" : "translate-x-0"
        }`}
      />
    </button>
  );
}

function ToggleRow({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <SettingRow label={label} description={description}>
      <ToggleSwitch checked={checked} label={label} onChange={onChange} />
    </SettingRow>
  );
}

function SegmentedRow<T extends string | number>({
  label,
  description,
  value,
  options,
  onChange,
}: {
  label: string;
  description: string;
  value: T;
  options: { value: T; label: string; preview?: ReactNode }[];
  onChange: (value: T) => void;
}) {
  return (
    <SettingRow label={label} description={description}>
      <div
        role="radiogroup"
        aria-label={label}
        className="flex flex-wrap justify-end gap-1"
      >
        {options.map((option) => {
          const active = option.value === value;
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onChange(option.value)}
              className={`flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition-colors ${
                active
                  ? "border-cyan-600 bg-cyan-950/50 text-cyan-200"
                  : "border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:bg-zinc-800/60 hover:text-zinc-200"
              }`}
            >
              {option.preview}
              <span>{option.label}</span>
            </button>
          );
        })}
      </div>
    </SettingRow>
  );
}

function SliderRow({
  label,
  description,
  value,
  min,
  max,
  step,
  formatValue,
  onChange,
}: {
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step: number;
  formatValue: (value: number) => string;
  onChange: (value: number) => void;
}) {
  return (
    <SettingRow label={label} description={description}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        aria-valuetext={formatValue(value)}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-36 cursor-pointer accent-cyan-500"
      />
      <span className="w-14 text-right font-mono text-[10px] text-zinc-400">
        {formatValue(value)}
      </span>
    </SettingRow>
  );
}

function AccentColorRow({
  value,
  disabled,
  onChange,
}: {
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <SettingRow
      label="Accent color"
      description="Used when connection lines use a single color."
    >
      <div
        className={`flex flex-wrap justify-end gap-1.5 ${
          disabled ? "pointer-events-none opacity-40" : ""
        }`}
        role="radiogroup"
        aria-label="Accent color"
        aria-disabled={disabled}
      >
        {ACCENT_COLORS.map((color) => (
          <button
            key={color.value}
            type="button"
            role="radio"
            aria-checked={value === color.value}
            aria-label={color.label}
            title={color.label}
            onClick={() => onChange(color.value)}
            className={`h-5 w-5 rounded-full border-2 transition-transform hover:scale-110 ${
              value === color.value
                ? "border-white ring-2 ring-cyan-500/60"
                : "border-zinc-950"
            }`}
            style={{ backgroundColor: color.value }}
          />
        ))}
      </div>
    </SettingRow>
  );
}

function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "—";
  }
  return value > 0 ? `${Math.round(value)}s` : "disabled";
}

function RuntimeDiagnostics({
  health,
  loading,
  error,
  backendConnected,
  onRefresh,
}: {
  health: {
    accelerator: string;
    effective_accelerator: string;
    torch: string | null;
    torch_cuda: boolean;
    nvidia: boolean;
  } | null;
  loading: boolean;
  error: string | null;
  backendConnected: boolean;
  onRefresh: () => void;
}) {
  const rows = [
    ["Backend", backendConnected ? "connected" : "offline"],
    [
      "Accelerator",
      health
        ? `${health.accelerator} → ${health.effective_accelerator}`
        : "—",
    ],
    ["Torch", health?.torch ?? (health ? "not installed" : "—")],
    [
      "CUDA build",
      health ? (health.torch_cuda ? "yes" : "no") : "—",
    ],
    [
      "NVIDIA GPU",
      health ? (health.nvidia ? "detected" : "not detected") : "—",
    ],
  ];
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-3">
      <div className="flex items-center gap-2">
        <div className="text-xs font-medium text-zinc-200">Runtime</div>
        <button
          type="button"
          title="Refresh runtime information"
          aria-label="Refresh runtime information"
          onClick={onRefresh}
          disabled={loading}
          className="ml-auto flex items-center gap-1 rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-wait"
        >
          <ResetIcon className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>
      {error && (
        <div className="mt-2 flex items-start gap-1.5 rounded border border-red-900/60 bg-red-950/40 p-1.5 text-[10px] text-red-300">
          <WarningIcon className="mt-0.5 h-3 w-3 shrink-0" />
          <span>{error}</span>
        </div>
      )}
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[10px]">
        {rows.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-zinc-500">{label}</dt>
            <dd className="min-w-0 truncate text-right font-mono text-zinc-300">
              {value}
            </dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-[10px] leading-relaxed text-zinc-600">
        Accelerator changes take effect on restart; torch installs use the
        configured CPU or cu130 wheel index.
      </p>
    </div>
  );
}

function RunLimits({
  limits,
}: {
  limits: {
    node_timeout_seconds: number;
    run_timeout_seconds: number;
    max_memory_mb: number;
  } | null;
}) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-3">
      <div className="text-xs font-medium text-zinc-200">Run limits</div>
      <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
        Read-only values from the backend environment and launcher.
      </p>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[10px]">
        <dt className="text-zinc-500">Node timeout</dt>
        <dd className="text-right font-mono text-zinc-300">
          {formatSeconds(limits?.node_timeout_seconds)}
        </dd>
        <dt className="text-zinc-500">Run timeout</dt>
        <dd className="text-right font-mono text-zinc-300">
          {formatSeconds(limits?.run_timeout_seconds)}
        </dd>
        <dt className="text-zinc-500">Memory limit</dt>
        <dd className="text-right font-mono text-zinc-300">
          {limits && limits.max_memory_mb > 0
            ? `${limits.max_memory_mb} MB`
            : limits
              ? "disabled"
              : "—"}
        </dd>
      </dl>
    </div>
  );
}

export function SettingsModal() {
  const settings = useSettingsStore((state) => state.settings);
  const updateSetting = useSettingsStore((state) => state.updateSetting);
  const resetSettings = useSettingsStore((state) => state.resetSettings);
  const closeSettings = useSettingsStore((state) => state.closeSettings);
  const [category, setCategory] = useState<SettingsCategory>("connections");
  const dialogRef = useDialogA11y<HTMLDivElement>(closeSettings);
  const activeCategory = CATEGORIES.find((item) => item.id === category) ?? CATEGORIES[0];
  const healthState = useHealthInfo();
  const backendConnected = useGraphStore((state) => state.wsConnected);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-modal-title"
        tabIndex={-1}
        className="flex max-h-[min(680px,90vh)] w-full max-w-3xl flex-col overflow-hidden rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl outline-none"
      >
        <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
          <SettingsIcon className="h-4 w-4 text-cyan-400" />
          <span id="settings-modal-title" className="text-sm font-semibold text-zinc-100">
            Settings
          </span>
          <span className="hidden text-[11px] text-zinc-500 sm:inline">
            Runtime, execution, canvas and connection preferences
          </span>
          <button
            type="button"
            className="ml-auto rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
            title="Close settings"
            aria-label="Close settings"
            onClick={closeSettings}
          >
            <CloseIcon />
          </button>
        </div>

        <div className="flex min-h-0 flex-1">
          <nav
            role="tablist"
            aria-label="Settings categories"
            aria-orientation="vertical"
            className="w-40 shrink-0 border-r border-zinc-800 bg-zinc-950/40 p-2"
          >
            {CATEGORIES.map((item) => (
              <button
                key={item.id}
                type="button"
                role="tab"
                id={`settings-tab-${item.id}`}
                aria-selected={category === item.id}
                aria-controls={`settings-panel-${item.id}`}
                onClick={() => setCategory(item.id)}
                className={`mb-1 w-full rounded-md px-2.5 py-2 text-left text-xs transition-colors ${
                  category === item.id
                    ? "bg-zinc-800 font-medium text-cyan-200"
                    : "text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200"
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>

          <section
            role="tabpanel"
            id={`settings-panel-${category}`}
            aria-labelledby={`settings-tab-${category}`}
            tabIndex={0}
            className="min-w-0 flex-1 overflow-y-auto p-4 outline-none"
          >
            <div className="mb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
                {activeCategory.label}
              </h3>
              <p className="mt-0.5 text-[11px] text-zinc-500">
                {activeCategory.description}
              </p>
            </div>

            {category === "connections" && (
              <div>
                <ToggleRow
                  label="Show connection lines"
                  description="Hide lines while keeping every connection intact."
                  checked={settings.showEdges}
                  onChange={(value) => updateSetting("showEdges", value)}
                />
                <ToggleRow
                  label="Node picker on connection drop"
                  description="Search for a compatible node when a connection is dropped on empty canvas."
                  checked={settings.connectionDropSearch}
                  onChange={(value) =>
                    updateSetting("connectionDropSearch", value)
                  }
                />
                <SegmentedRow
                  label="Line style"
                  description="Choose how connection lines are drawn."
                  value={settings.edgeStyle}
                  options={EDGE_STYLE_OPTIONS}
                  onChange={(value) => updateSetting("edgeStyle", value)}
                />
                <SliderRow
                  label="Line thickness"
                  description="Stroke width for normal and in-progress connections."
                  value={settings.edgeWidth}
                  min={1}
                  max={6}
                  step={0.5}
                  formatValue={(value) => `${value.toFixed(1)} px`}
                  onChange={(value) => updateSetting("edgeWidth", value)}
                />
                <ToggleRow
                  label="Animated flow"
                  description="Animate dashes along each connection line."
                  checked={settings.edgeAnimated}
                  onChange={(value) => updateSetting("edgeAnimated", value)}
                />
                <ToggleRow
                  label="Arrowheads"
                  description="Show the direction of data flow at the target socket."
                  checked={settings.showArrowheads}
                  onChange={(value) => updateSetting("showArrowheads", value)}
                />
                <SegmentedRow
                  label="Line color"
                  description="Match the source socket type or use one accent color."
                  value={settings.edgeColorMode}
                  options={COLOR_MODE_OPTIONS}
                  onChange={(value) => updateSetting("edgeColorMode", value)}
                />
                <AccentColorRow
                  value={settings.accentColor}
                  disabled={settings.edgeColorMode !== "single"}
                  onChange={(value) => updateSetting("accentColor", value)}
                />
                <SegmentedRow
                  label="Handle hit area"
                  description="Size of the clickable area around each socket dot."
                  value={settings.handleSize}
                  options={HANDLE_SIZE_OPTIONS}
                  onChange={(value) => updateSetting("handleSize", value)}
                />
              </div>
            )}

            {category === "canvas" && (
              <div>
                <SegmentedRow
                  label="Background pattern"
                  description="Choose the canvas backdrop or turn it off."
                  value={settings.backgroundVariant}
                  options={BACKGROUND_OPTIONS}
                  onChange={(value) => updateSetting("backgroundVariant", value)}
                />
                <SliderRow
                  label="Grid spacing"
                  description="Spacing for the background pattern and snap grid."
                  value={settings.gridSpacing}
                  min={10}
                  max={40}
                  step={2}
                  formatValue={(value) => `${value} px`}
                  onChange={(value) => updateSetting("gridSpacing", value)}
                />
                <ToggleRow
                  label="Snap to grid"
                  description="Align dragged nodes to the grid spacing."
                  checked={settings.snapToGrid}
                  onChange={(value) => updateSetting("snapToGrid", value)}
                />
                <ToggleRow
                  label="Show minimap"
                  description="Display the minimap in the bottom-right corner."
                  checked={settings.showMiniMap}
                  onChange={(value) => updateSetting("showMiniMap", value)}
                />
              </div>
            )}

            {category === "system" && (
              <div className="space-y-4">
                <div>
                  <ToggleRow
                    label="Show system monitor"
                    description="Display the floating CPU, GPU, VRAM and RAM monitor on the canvas."
                    checked={settings.showSystemMonitor}
                    onChange={(value) =>
                      updateSetting("showSystemMonitor", value)
                    }
                  />
                  <SegmentedRow
                    label="Update interval"
                    description="How often the monitor samples system metrics."
                    value={settings.systemMonitorIntervalMs}
                    options={MONITOR_INTERVAL_OPTIONS}
                    onChange={(value) =>
                      updateSetting("systemMonitorIntervalMs", value)
                    }
                  />
                  <SettingRow
                    label="Monitor position"
                    description="Reset the floating monitor to its default bottom-center position."
                  >
                    <button
                      type="button"
                      onClick={() =>
                        updateSetting("systemMonitorPosition", null)
                      }
                      disabled={settings.systemMonitorPosition === null}
                      className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 transition-colors hover:border-zinc-600 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      <ResetIcon />
                      Reset position
                    </button>
                  </SettingRow>
                </div>
                <RuntimeDiagnostics
                  health={healthState.health}
                  loading={healthState.loading}
                  error={healthState.error}
                  backendConnected={backendConnected}
                  onRefresh={() => void healthState.refresh()}
                />
              </div>
            )}

            {category === "execution" && (
              <div className="space-y-4">
                <div>
                  <ToggleRow
                    label="Open Execution panel on run"
                    description="Show the console automatically when a workflow starts."
                    checked={settings.autoOpenConsole}
                    onChange={(value) =>
                      updateSetting("autoOpenConsole", value)
                    }
                  />
                  <ToggleRow
                    label="Confirm canvas clearing"
                    description="Ask before the Clear action removes every node."
                    checked={settings.confirmClearCanvas}
                    onChange={(value) =>
                      updateSetting("confirmClearCanvas", value)
                    }
                  />
                </div>
                <RunLimits limits={healthState.health?.limits ?? null} />
              </div>
            )}

            {category === "general" && (
              <div className="space-y-4">
                <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-3">
                  <div className="text-xs font-medium text-zinc-200">
                    Reset preferences
                  </div>
                  <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
                    Restore all connection, canvas, system monitor and execution
                    settings to their defaults. Your workflow is not affected.
                  </p>
                  <button
                    type="button"
                    onClick={resetSettings}
                    className="mt-3 flex items-center gap-1.5 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 transition-colors hover:border-red-800 hover:bg-red-950/40 hover:text-red-200"
                  >
                    <ResetIcon /> Reset all settings
                  </button>
                </div>
                <p className="text-[11px] leading-relaxed text-zinc-600">
                  Preferences are stored locally in this browser and apply
                  instantly. Workflow files never include these display settings.
                </p>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
