import { Suspense, lazy, useEffect } from "react";

import { Canvas } from "./components/Canvas";
import { ExecutionPanel } from "./components/ExecutionPanel";
import { NodePalette } from "./components/NodePalette";
import { Toolbar } from "./components/Toolbar";
import { SpinnerIcon } from "./components/icons";
import { useExecutionSocket } from "./hooks/useExecutionSocket";
import { useGraphShortcuts } from "./hooks/useGraphShortcuts";
import { useWorkflowAutosave } from "./hooks/useWorkflowAutosave";
import { useGraphStore } from "./store/useGraphStore";

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

  useEffect(() => {
    void (async () => {
      await Promise.all([
        loadTemplates(),
        useGraphStore.getState().loadWorkflowTemplates(),
      ]);
      await useGraphStore.getState().restoreFromStorage();
    })();
  }, [loadTemplates]);

  useExecutionSocket();
  useWorkflowAutosave();
  useGraphShortcuts();

  return (
    <div className="flex h-screen flex-col">
      <Toolbar />
      <div className="flex min-h-0 flex-1">
        <NodePalette />
        <main aria-label="Workflow canvas" className="relative min-w-0 flex-1">
          <Canvas />
        </main>
        <ExecutionPanel />
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
    </div>
  );
}
