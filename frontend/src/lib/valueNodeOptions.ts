import type {
  AppEdge,
  AppNode,
  WidgetChoice,
  WidgetParam,
} from "../types/graph";

export const VALUE_NODE_TEMPLATES = new Set([
  "int_value",
  "float_value",
  "string_value",
  "bool_value",
]);

function sameChoice(left: WidgetChoice, right: WidgetChoice): boolean {
  return typeof left === typeof right && Object.is(left, right);
}

export function isOfficialValueNode(data: AppNode["data"]): boolean {
  return Boolean(
    data.templateName &&
      VALUE_NODE_TEMPLATES.has(data.templateName) &&
      data.templateCode !== null &&
      data.code === data.templateCode,
  );
}

function targetWidget(
  node: AppNode | undefined,
  handle: string | null | undefined,
): WidgetParam | null {
  if (!node || !handle) {
    return null;
  }
  return (
    node.data.schema.widgets.find((widget) => widget.name === handle) ?? null
  );
}

export function deriveValueNodeOptions(
  nodeId: string,
  data: AppNode["data"],
  nodes: AppNode[],
  edges: AppEdge[],
): WidgetChoice[] | null {
  if (!isOfficialValueNode(data)) {
    return null;
  }
  const nodesById = new Map(nodes.map((candidate) => [candidate.id, candidate]));
  const constrained = edges
    .filter((edge) => edge.source === nodeId)
    .map((edge) =>
      targetWidget(nodesById.get(edge.target), edge.targetHandle),
    )
    .filter(
      (widget): widget is WidgetParam =>
        widget !== null && Boolean(widget.options && widget.options.length > 0),
    );
  if (constrained.length === 0) {
    return null;
  }

  let options = [...(constrained[0].options ?? [])];
  for (const widget of constrained.slice(1)) {
    options = options.filter((option) =>
      (widget.options ?? []).some((candidate) => sameChoice(option, candidate)),
    );
  }
  return options;
}

export function normalizeValueChoice(
  value: unknown,
  options: WidgetChoice[] | null,
): unknown {
  if (!options || options.length === 0) {
    return value;
  }
  return options.some((option) => sameChoice(option, value as WidgetChoice))
    ? value
    : options[0];
}

export function normalizeValueNodeValues(
  nodes: AppNode[],
  edges: AppEdge[],
): AppNode[] {
  return nodes.map((node) => {
    const options = deriveValueNodeOptions(node.id, node.data, nodes, edges);
    if (!options || options.length === 0) {
      return node;
    }
    const current = node.data.widgetValues.value;
    const next = normalizeValueChoice(current, options);
    if (sameChoice(next as WidgetChoice, current as WidgetChoice)) {
      return node;
    }
    return {
      ...node,
      data: {
        ...node.data,
        widgetValues: { ...node.data.widgetValues, value: next },
      },
    };
  });
}
