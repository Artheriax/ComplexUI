import { Suspense, lazy, useEffect, useRef, useState } from "react";

import { Canvas } from "./components/Canvas";
import { ExecutionPanel } from "./components/ExecutionPanel";
import { NodePalette } from "./components/NodePalette";
import { SaveProjectDialog } from "./components/SaveProjectDialog";
import { SettingsModal } from "./components/SettingsModal";
import { ShortcutsDialog } from "./components/ShortcutsDialog";
import { WindowTabs } from "./components/WindowTabs";
import { SpinnerIcon } from "./components/icons";
import { useExecutionSocket } from "./hooks/useExecutionSocket";
import { useGraphShortcuts } from "./hooks/useGraphShortcuts";
import { useWorkflowAutosave } from "./hooks/useWorkflowAutosave";
import { useGraphStore } from "./store/useGraphStore";
import { useSettingsStore } from "./store/useSettingsStore";
import { useUiStore } from "./store/useUiStore";

const CodeModal = lazy(async () => {
  const module = await import("./components/CodeModal");
  return { default: module.CodeModal };
});

const DiffModal = lazy(async () => {
  const module = await import("./components/DiffModal");
  return { default: module.DiffModal };
});

function ModalLoading() {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60">
      <div className="flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-3 text-xs text-zinc-300">
        <SpinnerIcon className="h-4 w-4 animate-spin text-cyan-400" />
        Loading editor…
      </div>
    </div>
  );
}

export default function App() {
  const loadTemplates = useGraphStore((state) => state.loadTemplates);
  const editingNodeId = useGraphStore((state) => state.editingNodeId);
  const hasPendingApproval = useGraphStore(
    (state) => state.pendingApproval !== null,
  );
  const saveDialogOpen = useGraphStore((state) => state.saveDialogOpen);
  const activeWindowId = useGraphStore((state) => state.activeWindowId);
  const settingsOpen = useSettingsStore((state) => state.isOpen);
  const consoleOpen = useUiStore((state) => state.consoleOpen);
  const shortcutsOpen = useUiStore((state) => state.shortcutsOpen);

  const [ready, setReady] = useState(false);
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) {
      return;
    }
    startedRef.current = true;
    void (async () => {
      try {
        await Promise.all([
          loadTemplates(),
          useGraphStore.getState().loadWorkflowTemplates(),
          useGraphStore.getState().loadProjects(),
        ]);
        await useGraphStore.getState().restoreFromStorage();
      } finally {
        setReady(true);
      }
    })();
  }, [loadTemplates]);

  useExecutionSocket();
  useWorkflowAutosave();
  useGraphShortcuts();

  if (!ready) {
    return (
      <div className="flex h-screen items-center justify-center bg-zinc-950">
        <div className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 text-xs text-zinc-300">
          <SpinnerIcon className="h-4 w-4 animate-spin text-cyan-400" />
          Restoring workspace…
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col">
      <WindowTabs />
      <div className="flex min-h-0 flex-1">
        <NodePalette />
        <main aria-label="Workflow canvas" className="relative min-w-0 flex-1">
          <Canvas key={activeWindowId} />
        </main>
        {consoleOpen && <ExecutionPanel />}
      </div>
      {editingNodeId && (
        <Suspense fallback={<ModalLoading />}>
          <CodeModal key={editingNodeId} nodeId={editingNodeId} />
        </Suspense>
      )}
      {hasPendingApproval && (
        <Suspense fallback={<ModalLoading />}>
          <DiffModal />
        </Suspense>
      )}
      {settingsOpen && <SettingsModal />}
      {shortcutsOpen && <ShortcutsDialog />}
      {saveDialogOpen && <SaveProjectDialog />}
    </div>
  );
}
