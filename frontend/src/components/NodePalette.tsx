import type { ReactNode } from "react";

import { useSettingsStore } from "../store/useSettingsStore";
import type { SidebarPanel } from "../store/useUiStore";
import { useUiStore } from "../store/useUiStore";
import { AssetLibrary } from "./AssetLibrary";
import { NodeLibrary } from "./NodeLibrary";
import { ProjectLibrary } from "./ProjectLibrary";
import { WorkflowLibrary } from "./WorkflowLibrary";
import {
  DatabaseIcon,
  FolderIcon,
  GridIcon,
  KeyboardIcon,
  SettingsIcon,
  SubgraphIcon,
  TerminalIcon,
} from "./icons";

interface RailItem {
  id: SidebarPanel;
  label: string;
  icon: ReactNode;
}

const TOP_ITEMS: RailItem[] = [
  { id: "nodes", label: "Nodes", icon: <GridIcon className="h-4 w-4" /> },
  {
    id: "workflows",
    label: "Workflows",
    icon: <FolderIcon className="h-4 w-4" />,
  },
  {
    id: "templates",
    label: "Templates",
    icon: <SubgraphIcon className="h-4 w-4" />,
  },
  {
    id: "datasets",
    label: "Assets",
    icon: <DatabaseIcon className="h-4 w-4" />,
  },
];

const railButton =
  "flex h-9 w-9 items-center justify-center rounded-md border transition-colors";

function RailButton({
  active,
  label,
  icon,
  onClick,
}: {
  active: boolean;
  label: string;
  icon: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-expanded={active}
      onClick={onClick}
      className={`${railButton} ${
        active
          ? "border-cyan-700 bg-cyan-950/60 text-cyan-200"
          : "border-transparent text-zinc-500 hover:border-zinc-700 hover:bg-zinc-800/70 hover:text-zinc-200"
      }`}
    >
      {icon}
    </button>
  );
}

export function NodePalette() {
  const sidebarPanel = useUiStore((state) => state.sidebarPanel);
  const toggleSidebarPanel = useUiStore((state) => state.toggleSidebarPanel);
  const consoleOpen = useUiStore((state) => state.consoleOpen);
  const toggleConsole = useUiStore((state) => state.toggleConsole);
  const openShortcuts = useUiStore((state) => state.openShortcuts);
  const openSettings = useSettingsStore((state) => state.openSettings);

  return (
    <aside className="flex shrink-0 border-r border-zinc-800 bg-zinc-900/40">
      <nav
        aria-label="Workspace panels"
        className="flex w-12 shrink-0 flex-col items-center gap-1 border-r border-zinc-800 bg-zinc-950/60 py-2"
      >
        {TOP_ITEMS.map((item) => (
          <RailButton
            key={item.id}
            active={sidebarPanel === item.id}
            label={item.label}
            icon={item.icon}
            onClick={() => toggleSidebarPanel(item.id)}
          />
        ))}
        <div className="mt-auto flex flex-col items-center gap-1">
          <RailButton
            active={consoleOpen}
            label="Console"
            icon={<TerminalIcon className="h-4 w-4" />}
            onClick={toggleConsole}
          />
          <RailButton
            active={false}
            label="Keyboard shortcuts"
            icon={<KeyboardIcon className="h-4 w-4" />}
            onClick={openShortcuts}
          />
          <RailButton
            active={false}
            label="Settings"
            icon={<SettingsIcon className="h-4 w-4" />}
            onClick={openSettings}
          />
        </div>
      </nav>
      {sidebarPanel !== null && (
        <section
          id={`sidebar-panel-${sidebarPanel}`}
          aria-label={TOP_ITEMS.find((item) => item.id === sidebarPanel)?.label}
          className="flex w-60 min-w-0 flex-col border-r border-zinc-800"
        >
          {sidebarPanel === "nodes" ? (
            <NodeLibrary />
          ) : sidebarPanel === "workflows" ? (
            <ProjectLibrary />
          ) : sidebarPanel === "templates" ? (
            <WorkflowLibrary />
          ) : (
            <AssetLibrary />
          )}
        </section>
      )}
    </aside>
  );
}

