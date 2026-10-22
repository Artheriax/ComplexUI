import type { AppEdge, NodeSchema, WidgetParam } from "../types/graph";
import { defaultWidgetValues } from "../types/graph";

export function widgetValueAllowed(
  widget: WidgetParam,
  value: unknown,
): boolean {
  const options = widget.options ?? [];
  if (options.length > 0 && !options.some((option) => option === value)) {
    return false;
  }
  switch (widget.data_type) {
    case "int":
      return typeof value === "number" && Number.isInteger(value);
    case "float":
      return typeof value === "number" && Number.isFinite(value);
    case "str":
      return typeof value === "string";
    case "bool":
      return typeof value === "boolean";
    default:
      return true;
  }
}

export function mergeWidgetValues(
  schema: NodeSchema,
  previous: Record<string, unknown>,
): Record<string, unknown> {
  const next = defaultWidgetValues(schema);
  for (const widget of schema.widgets) {
    if (
      previous[widget.name] !== undefined &&
      widgetValueAllowed(widget, previous[widget.name])
    ) {
      next[widget.name] = previous[widget.name];
    }
  }
  return next;
}

export function reconcileEdges(
  edges: AppEdge[],
  nodeId: string,
  schema: NodeSchema,
): AppEdge[] {
  const targets = new Set([
    ...schema.inputs.map((socket) => socket.name),
    ...schema.widgets.map((widget) => widget.name),
  ]);
  const outputs = new Set(schema.outputs.map((socket) => socket.name));
  return edges.filter((edge) => {
    if (edge.source === nodeId && !outputs.has(edge.sourceHandle ?? "")) {
      return false;
    }
    if (edge.target === nodeId && !targets.has(edge.targetHandle ?? "")) {
      return false;
    }
    return true;
  });
}
