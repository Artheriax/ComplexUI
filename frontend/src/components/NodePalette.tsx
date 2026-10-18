import type { DragEvent } from "react";

import { useGraphStore } from "../store/useGraphStore";
import { SOCKET_COLORS } from "../types/graph";
import type { TemplateInfo } from "../types/graph";
import { DRAG_MIME } from "./Canvas";
import { PlusIcon, SpinnerIcon, WarningIcon } from "./icons";

const TEMPLATE_BLURBS: Record<string, string> = {
  load_iris: "Polars DataFrame with 150 iris rows",
  train_test_split: "Split into train/test Datasets",
  knn_train: "KNeighborsClassifier (n_neighbors)",
  evaluate_metrics: "Accuracy + classification report",
  tensor_dataset: "Synthetic 2-class tensors (torch)",
  mlp_train: "Train a small MLP, returns loss curve",
  evaluate_torch: "Accuracy + loss on tensor data",
};

function TemplateCard({ template }: { template: TemplateInfo }) {
  const addNodeFromTemplate = useGraphStore((state) => state.addNodeFromTemplate);

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.setData(DRAG_MIME, template.name);
    event.dataTransfer.effectAllowed = "move";
  };

  const onClick = () => {
    const nodeCount = useGraphStore.getState().nodes.length;
    addNodeFromTemplate(template, {
      x: 80 + (nodeCount % 5) * 70,
      y: 80 + (nodeCount % 5) * 60,
    });
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      onClick={onClick}
      className="cursor-grab rounded-lg border border-zinc-800 bg-zinc-900 p-2.5 transition-colors hover:border-cyan-700 hover:bg-zinc-800/70 active:cursor-grabbing"
    >
      <div className="flex items-center gap-1.5">
        <span className="text-xs font-semibold text-zinc-100">{template.title}</span>
      </div>
      <p className="mt-0.5 text-[11px] leading-snug text-zinc-500">
        {TEMPLATE_BLURBS[template.name] ?? "ML building block"}
      </p>
      <div className="mt-1.5 flex flex-wrap items-center gap-1">
        {template.node_schema.inputs.map((socket) => (
          <span
            key={`in-${socket.name}`}
            className="rounded px-1 py-px text-[9px]"
            style={{ backgroundColor: `${SOCKET_COLORS[socket.data_type]}22`, color: SOCKET_COLORS[socket.data_type] }}
          >
            {socket.name} in
          </span>
        ))}
        {template.node_schema.outputs.map((socket) => (
          <span
            key={`out-${socket.name}`}
            className="rounded px-1 py-px text-[9px]"
            style={{ backgroundColor: `${SOCKET_COLORS[socket.data_type]}22`, color: SOCKET_COLORS[socket.data_type] }}
          >
            {socket.name} out
          </span>
        ))}
      </div>
    </div>
  );
}

export function NodePalette() {
  const templates = useGraphStore((state) => state.templates);
  const templatesLoaded = useGraphStore((state) => state.templatesLoaded);
  const templatesError = useGraphStore((state) => state.templatesError);

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/40">
      <div className="border-b border-zinc-800 px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
        Node Library
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto p-2.5">
        <button
          type="button"
          onClick={() => {
            const count = useGraphStore.getState().nodes.length;
            useGraphStore.getState().addScratchNode({
              x: 80 + (count % 5) * 70,
              y: 80 + (count % 5) * 60,
            });
          }}
          className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed border-zinc-600 px-2.5 py-2 text-xs text-zinc-300 transition-colors hover:border-cyan-600 hover:bg-zinc-800/60 hover:text-cyan-200"
        >
          <PlusIcon /> New scratch node
        </button>
        {!templatesLoaded && !templatesError && (
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <SpinnerIcon className="h-3.5 w-3.5 animate-spin" /> Loading templates…
          </div>
        )}
        {templatesError && (
          <div className="flex items-start gap-2 rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300">
            <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{templatesError}</span>
          </div>
        )}
        {templates.map((template) => (
          <TemplateCard key={template.name} template={template} />
        ))}
      </div>
      <div className="border-t border-zinc-800 p-2.5 text-[10px] leading-relaxed text-zinc-600">
        Drag a template onto the canvas, connect sockets, then hit Run.
      </div>
    </aside>
  );
}
