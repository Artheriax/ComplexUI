import { useRef, useState } from "react";
import type { ChangeEvent } from "react";

import { exportWorkflow } from "../lib/api";
import {
  downloadTextFile,
  downloadWorkflow,
  readWorkflowFile,
} from "../lib/persistence";
import { useGraphStore } from "../store/useGraphStore";
import { useUiStore } from "../store/useUiStore";
import type { ExportFormat } from "../types/graph";

function reportError(error: unknown, errorType: string): void {
  useGraphStore.setState((state) => ({
    execution: {
      ...state.execution,
      status: "idle",
      error: {
        message: error instanceof Error ? error.message : String(error),
        errorType,
        nodeId: null,
        line: null,
        traceback: "",
      },
    },
  }));
  useUiStore.getState().setConsoleOpen(true);
}

export function useWorkflowFileActions() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [exporting, setExporting] = useState(false);

  const save = (): void => {
    useGraphStore.getState().requestSave();
  };

  const exportJson = (): void => {
    const state = useGraphStore.getState();
    downloadWorkflow(state.getWorkflowDocument(state.projectName));
  };

  const exportArtifact = async (format: ExportFormat): Promise<void> => {
    setExporting(true);
    try {
      const state = useGraphStore.getState();
      const exported = await exportWorkflow(
        state.getExecutionGraph().graph,
        format,
      );
      downloadTextFile(
        exported.filename,
        exported.content,
        exported.media_type,
      );
    } catch (error) {
      reportError(error, "Export");
    } finally {
      setExporting(false);
    }
  };

  const importFile = async (file: File): Promise<void> => {
    try {
      const saved = await readWorkflowFile(file);
      await useGraphStore.getState().openWorkflowInWindow(saved);
    } catch (error) {
      reportError(error, "Import");
    }
  };

  const onFileInputChange = (event: ChangeEvent<HTMLInputElement>): void => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) {
      void importFile(file);
    }
  };

  const openImport = (): void => {
    fileInputRef.current?.click();
  };

  const clearCanvas = (): void => {
    const state = useGraphStore.getState();
    if (
      state.nodes.length === 0 &&
      state.edges.length === 0 &&
      Object.keys(state.subgraphs).length === 0
    ) {
      return;
    }
    const message =
      state.scopeStack.length > 0
        ? "Clear this nested subgraph canvas? You can undo this with Ctrl+Z."
        : "Clear the canvas? Autosave will also be overwritten. You can undo this with Ctrl+Z.";
    if (window.confirm(message)) {
      state.clearCanvas();
    }
  };

  return {
    fileInputRef,
    exporting,
    save,
    exportJson,
    exportArtifact,
    importFile,
    onFileInputChange,
    openImport,
    clearCanvas,
  };
}
