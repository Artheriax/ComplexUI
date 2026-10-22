import { create } from "zustand";
import { persist } from "zustand/middleware";

export type EdgeLineStyle = "bezier" | "smoothstep" | "step" | "straight";
export type EdgeColorMode = "socket" | "single";
export type HandleSize = "compact" | "comfortable" | "large";
export type CanvasBackgroundVariant = "dots" | "lines" | "cross" | "none";
export type SystemMonitorIntervalMs = 1000 | 2000 | 5000;

export interface MonitorPosition {
  x: number;
  y: number;
}

export interface AppSettings {
  showEdges: boolean;
  connectionDropSearch: boolean;
  edgeStyle: EdgeLineStyle;
  edgeWidth: number;
  edgeColorMode: EdgeColorMode;
  accentColor: string;
  edgeAnimated: boolean;
  showArrowheads: boolean;
  handleSize: HandleSize;
  backgroundVariant: CanvasBackgroundVariant;
  gridSpacing: number;
  snapToGrid: boolean;
  showMiniMap: boolean;
  showSystemMonitor: boolean;
  systemMonitorIntervalMs: SystemMonitorIntervalMs;
  systemMonitorPosition: MonitorPosition | null;
  autoOpenConsole: boolean;
  confirmClearCanvas: boolean;
}

export const SETTINGS_STORAGE_KEY = "complexui.settings.v1";

export const DEFAULT_SETTINGS: AppSettings = {
  showEdges: true,
  connectionDropSearch: true,
  edgeStyle: "bezier",
  edgeWidth: 2.5,
  edgeColorMode: "socket",
  accentColor: "#22d3ee",
  edgeAnimated: false,
  showArrowheads: false,
  handleSize: "comfortable",
  backgroundVariant: "dots",
  gridSpacing: 20,
  snapToGrid: false,
  showMiniMap: true,
  showSystemMonitor: true,
  systemMonitorIntervalMs: 2000,
  systemMonitorPosition: null,
  autoOpenConsole: true,
  confirmClearCanvas: true,
};

export const SYSTEM_MONITOR_INTERVALS: readonly SystemMonitorIntervalMs[] = [
  1000,
  2000,
  5000,
];

export const HANDLE_SIZE_VALUES: Record<HandleSize, { hit: number; dot: number }> = {
  compact: { hit: 14, dot: 10 },
  comfortable: { hit: 18, dot: 10 },
  large: { hit: 24, dot: 12 },
};

const EDGE_STYLES: readonly EdgeLineStyle[] = [
  "bezier",
  "smoothstep",
  "step",
  "straight",
];
const EDGE_COLOR_MODES: readonly EdgeColorMode[] = ["socket", "single"];
const HANDLE_SIZES: readonly HandleSize[] = ["compact", "comfortable", "large"];
const BACKGROUND_VARIANTS: readonly CanvasBackgroundVariant[] = [
  "dots",
  "lines",
  "cross",
  "none",
];

function pickEnum<T extends string>(
  value: unknown,
  allowed: readonly T[],
  fallback: T,
): T {
  return typeof value === "string" && (allowed as readonly string[]).includes(value)
    ? (value as T)
    : fallback;
}

function pickBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function pickNumber(
  value: unknown,
  fallback: number,
  min: number,
  max: number,
): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(max, Math.max(min, value))
    : fallback;
}

function pickColor(value: unknown, fallback: string): string {
  return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value)
    ? value
    : fallback;
}

function pickMonitorPosition(value: unknown): MonitorPosition | null {
  if (typeof value !== "object" || value === null) {
    return null;
  }
  const record = value as Record<string, unknown>;
  if (
    typeof record.x !== "number" ||
    !Number.isFinite(record.x) ||
    typeof record.y !== "number" ||
    !Number.isFinite(record.y)
  ) {
    return null;
  }
  return {
    x: Math.min(100_000, Math.max(0, record.x)),
    y: Math.min(100_000, Math.max(0, record.y)),
  };
}

function pickMonitorInterval(value: unknown): SystemMonitorIntervalMs {
  return SYSTEM_MONITOR_INTERVALS.includes(value as SystemMonitorIntervalMs)
    ? (value as SystemMonitorIntervalMs)
    : DEFAULT_SETTINGS.systemMonitorIntervalMs;
}

export function sanitiseSettings(value: unknown): AppSettings {
  if (typeof value !== "object" || value === null) {
    return { ...DEFAULT_SETTINGS };
  }
  const record = value as Record<string, unknown>;
  return {
    showEdges: pickBoolean(record.showEdges, DEFAULT_SETTINGS.showEdges),
    connectionDropSearch: pickBoolean(
      record.connectionDropSearch,
      DEFAULT_SETTINGS.connectionDropSearch,
    ),
    edgeStyle: pickEnum(record.edgeStyle, EDGE_STYLES, DEFAULT_SETTINGS.edgeStyle),
    edgeWidth: pickNumber(record.edgeWidth, DEFAULT_SETTINGS.edgeWidth, 1, 6),
    edgeColorMode: pickEnum(
      record.edgeColorMode,
      EDGE_COLOR_MODES,
      DEFAULT_SETTINGS.edgeColorMode,
    ),
    accentColor: pickColor(record.accentColor, DEFAULT_SETTINGS.accentColor),
    edgeAnimated: pickBoolean(
      record.edgeAnimated,
      DEFAULT_SETTINGS.edgeAnimated,
    ),
    showArrowheads: pickBoolean(
      record.showArrowheads,
      DEFAULT_SETTINGS.showArrowheads,
    ),
    handleSize: pickEnum(
      record.handleSize,
      HANDLE_SIZES,
      DEFAULT_SETTINGS.handleSize,
    ),
    backgroundVariant: pickEnum(
      record.backgroundVariant,
      BACKGROUND_VARIANTS,
      DEFAULT_SETTINGS.backgroundVariant,
    ),
    gridSpacing: pickNumber(
      record.gridSpacing,
      DEFAULT_SETTINGS.gridSpacing,
      10,
      40,
    ),
    snapToGrid: pickBoolean(record.snapToGrid, DEFAULT_SETTINGS.snapToGrid),
    showMiniMap: pickBoolean(record.showMiniMap, DEFAULT_SETTINGS.showMiniMap),
    showSystemMonitor: pickBoolean(
      record.showSystemMonitor,
      DEFAULT_SETTINGS.showSystemMonitor,
    ),
    systemMonitorIntervalMs: pickMonitorInterval(
      record.systemMonitorIntervalMs,
    ),
    systemMonitorPosition: pickMonitorPosition(record.systemMonitorPosition),
    autoOpenConsole: pickBoolean(
      record.autoOpenConsole,
      DEFAULT_SETTINGS.autoOpenConsole,
    ),
    confirmClearCanvas: pickBoolean(
      record.confirmClearCanvas,
      DEFAULT_SETTINGS.confirmClearCanvas,
    ),
  };
}

interface SettingsStore {
  settings: AppSettings;
  isOpen: boolean;
  updateSetting: <K extends keyof AppSettings>(
    key: K,
    value: AppSettings[K],
  ) => void;
  resetSettings: () => void;
  openSettings: () => void;
  closeSettings: () => void;
}

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      settings: { ...DEFAULT_SETTINGS },
      isOpen: false,
      updateSetting: (key, value) =>
        set((state) => ({
          settings: sanitiseSettings({ ...state.settings, [key]: value }),
        })),
      resetSettings: () => set({ settings: { ...DEFAULT_SETTINGS } }),
      openSettings: () => set({ isOpen: true }),
      closeSettings: () => set({ isOpen: false }),
    }),
    {
      name: SETTINGS_STORAGE_KEY,
      partialize: (state) => ({ settings: state.settings }),
      merge: (persisted, current) => {
        const persistedSettings =
          typeof persisted === "object" && persisted !== null
            ? (persisted as { settings?: unknown }).settings
            : undefined;
        return { ...current, settings: sanitiseSettings(persistedSettings) };
      },
    },
  ),
);
