import { create } from "zustand";

export type SidebarPanel = "nodes" | "workflows" | "templates" | "datasets";

interface UiStore {
  sidebarPanel: SidebarPanel | null;
  consoleOpen: boolean;
  shortcutsOpen: boolean;
  toggleSidebarPanel: (panel: SidebarPanel) => void;
  closeSidebarPanel: () => void;
  setConsoleOpen: (open: boolean) => void;
  toggleConsole: () => void;
  openShortcuts: () => void;
  closeShortcuts: () => void;
}

export const useUiStore = create<UiStore>((set) => ({
  sidebarPanel: null,
  consoleOpen: false,
  shortcutsOpen: false,
  toggleSidebarPanel: (panel) =>
    set((state) => ({
      sidebarPanel: state.sidebarPanel === panel ? null : panel,
    })),
  closeSidebarPanel: () => set({ sidebarPanel: null }),
  setConsoleOpen: (open) => set({ consoleOpen: open }),
  toggleConsole: () => set((state) => ({ consoleOpen: !state.consoleOpen })),
  openShortcuts: () => set({ shortcutsOpen: true }),
  closeShortcuts: () => set({ shortcutsOpen: false }),
}));
