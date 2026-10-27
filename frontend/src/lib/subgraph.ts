import type { XYPosition } from "@xyflow/react";

import type {
  AppEdge,
  AppNode,
  InputSocket,
  OutputSocket,
  SocketType,
  SubgraphDefinition,
  SubgraphMap,
} from "../types/graph";
import {
  isFrameNode,
  isSubgraphInstanceNode,
  isSubgraphManagerNode,
  SUBGRAPH_INPUT_NODE_ID,
  SUBGRAPH_OUTPUT_NODE_ID,
  widgetSocketType,
} from "../types/graph";
import type { WorkflowGraphPayload } from "../types/graph";
import { emptyNodeSchema } from "./graphEditing";

const FALLBACK_WIDTH = 250;
const MAX_FLAT_NODES = 200;
const MAX_FLAT_EDGES = 1_000;
const MAX_FLAT_ID = 200;

export class SubgraphError extends Error {}

export function localNodeId(path: string, nodeId: string): string {
  return path ? `sub:${path}/${nodeId}` : nodeId;
}

export function createSubgraphManagerNodes(
  definition: SubgraphDefinition,
): AppNode[] {
  return [
    {
      id: SUBGRAPH_INPUT_NODE_ID,
      type: "complex",
      position: definition.inputPosition,
      selected: false,
      data: {
        label: "Inputs",
        templateName: null,
        templateCode: null,
        code: "",
        schema: emptyNodeSchema(SUBGRAPH_INPUT_NODE_ID, [], definition.inputs),
        widgetValues: {},
        isTrusted: true,
        color: "sky",
        subgraphRole: "input",
      },
    },
    {
      id: SUBGRAPH_OUTPUT_NODE_ID,
      type: "complex",
      position: definition.outputPosition,
      selected: false,
      data: {
        label: "Outputs",
        templateName: null,
        templateCode: null,
        code: "",
        schema: emptyNodeSchema(SUBGRAPH_OUTPUT_NODE_ID, definition.outputs, []),
        widgetValues: {},
        isTrusted: true,
        color: "amber",
        subgraphRole: "output",
      },
    },
  ];
}

export function createSubgraphInstanceNode(
  id: string,
  definition: SubgraphDefinition,
  position: XYPosition,
  label = definition.name,
): AppNode {
  return {
    id,
    type: "complex",
    position,
    selected: true,
    data: {
      label,
      templateName: null,
      templateCode: null,
      code: "",
      schema: emptyNodeSchema(id, definition.inputs, definition.outputs),
      widgetValues: {},
      isTrusted: true,
      color: "default",
      isSubgraph: true,
      subgraphId: definition.id,
    },
  };
}

export function stripSubgraphManagers(nodes: AppNode[]): AppNode[] {
  return nodes.filter((node) => !isSubgraphManagerNode(node));
}

export function subgraphDependsOn(
  subgraphs: SubgraphMap,
  definitionId: string,
  targetId: string,
  seen = new Set<string>(),
): boolean {
  if (definitionId === targetId) {
    return true;
  }
  if (seen.has(definitionId)) {
    return false;
  }
  seen.add(definitionId);
  const definition = subgraphs[definitionId];
  if (!definition) {
    return false;
  }
  return definition.nodes.some(
    (node) =>
      isSubgraphInstanceNode(node) &&
      node.data.subgraphId !== undefined &&
      subgraphDependsOn(subgraphs, node.data.subgraphId, targetId, seen),
  );
}

interface FlatEndpoint {
  nodeId: string;
  handle: string;
}

interface GraphContext {
  path: string;
  nodesById: Map<string, AppNode>;
  edges: AppEdge[];
  stack: string[];
}

export interface FlattenedWorkflow {
  graph: WorkflowGraphPayload;
  labels: Record<string, string>;
}

function joinPath(path: string, nodeId: string): string {
  return path ? `${path}/${nodeId}` : nodeId;
}

function definitionFor(
  node: AppNode,
  subgraphs: SubgraphMap,
): SubgraphDefinition {
  const definition = subgraphs[node.data.subgraphId ?? ""];
  if (!definition) {
    throw new SubgraphError(
      `Subgraph '${node.data.label}' references missing definition '${node.data.subgraphId ?? ""}'.`,
    );
  }
  return definition;
}

function childContext(
  definition: SubgraphDefinition,
  parent: GraphContext,
  instanceId: string,
): GraphContext {
  const instancePath = joinPath(parent.path, instanceId);
  return {
    path: instancePath,
    nodesById: new Map(definition.nodes.map((node) => [node.id, node])),
    edges: definition.edges,
    stack: [...parent.stack, definition.id],
  };
}

function resolveSource(
  node: AppNode,
  handle: string,
  context: GraphContext,
  subgraphs: SubgraphMap,
): FlatEndpoint[] {
  if (isFrameNode(node) || isSubgraphManagerNode(node)) {
    return [];
  }
  if (!isSubgraphInstanceNode(node)) {
    return [{ nodeId: localNodeId(context.path, node.id), handle }];
  }
  const definition = definitionFor(node, subgraphs);
  if (context.stack.includes(definition.id)) {
    throw new SubgraphError(
      `Subgraph cycle detected at '${definition.name}'.`,
    );
  }
  const child = childContext(definition, context, node.id);
  return child.edges
    .filter(
      (edge) =>
        edge.target === SUBGRAPH_OUTPUT_NODE_ID &&
        (edge.targetHandle ?? "") === handle,
    )
    .flatMap((edge) => {
      const source = child.nodesById.get(edge.source);
      return source
        ? resolveSource(
            source,
            edge.sourceHandle ?? "",
            child,
            subgraphs,
          )
        : [];
    });
}

function resolveTarget(
  node: AppNode,
  handle: string,
  context: GraphContext,
  subgraphs: SubgraphMap,
): FlatEndpoint[] {
  if (isFrameNode(node) || isSubgraphManagerNode(node)) {
    return [];
  }
  if (!isSubgraphInstanceNode(node)) {
    return [{ nodeId: localNodeId(context.path, node.id), handle }];
  }
  const definition = definitionFor(node, subgraphs);
  if (context.stack.includes(definition.id)) {
    throw new SubgraphError(
      `Subgraph cycle detected at '${definition.name}'.`,
    );
  }
  const child = childContext(definition, context, node.id);
  return child.edges
    .filter(
      (edge) =>
        edge.source === SUBGRAPH_INPUT_NODE_ID &&
        (edge.sourceHandle ?? "") === handle,
    )
    .flatMap((edge) => {
      const target = child.nodesById.get(edge.target);
      return target
        ? resolveTarget(
            target,
            edge.targetHandle ?? "",
            child,
            subgraphs,
          )
        : [];
    });
}

export function flattenWorkflow(
  nodes: AppNode[],
  edges: AppEdge[],
  subgraphs: SubgraphMap,
): FlattenedWorkflow {
  const graph: WorkflowGraphPayload = { nodes: [], edges: [] };
  const labels: Record<string, string> = {};

  const expand = (
    graphNodes: AppNode[],
    graphEdges: AppEdge[],
    path: string,
    stack: string[],
  ): void => {
    const nodesById = new Map(graphNodes.map((node) => [node.id, node]));
    const context: GraphContext = {
      path,
      nodesById,
      edges: graphEdges,
      stack,
    };

    for (const node of graphNodes) {
      if (
        isFrameNode(node) ||
        isSubgraphManagerNode(node) ||
        isSubgraphInstanceNode(node)
      ) {
        continue;
      }
      const flatId = localNodeId(path, node.id);
      if (flatId.length > MAX_FLAT_ID) {
        throw new SubgraphError(
          "Nested subgraph node ids exceed the 200-character limit.",
        );
      }
      graph.nodes.push({
        id: flatId,
        code: node.data.code,
        widget_values: node.data.widgetValues,
        ...(path ? { approval_id: node.id } : {}),
      });
      labels[flatId] = node.data.label;
      if (graph.nodes.length > MAX_FLAT_NODES) {
        throw new SubgraphError(
          `Expanded workflow has too many nodes (${graph.nodes.length} > ${MAX_FLAT_NODES}).`,
        );
      }
    }

    for (const node of graphNodes) {
      if (!isSubgraphInstanceNode(node)) {
        continue;
      }
      const definition = definitionFor(node, subgraphs);
      if (stack.includes(definition.id)) {
        throw new SubgraphError(
          `Subgraph cycle detected at '${definition.name}'.`,
        );
      }
      expand(
        definition.nodes,
        definition.edges,
        joinPath(path, node.id),
        [...stack, definition.id],
      );
    }

    for (const edge of graphEdges) {
      if (
        edge.source === SUBGRAPH_INPUT_NODE_ID ||
        edge.target === SUBGRAPH_OUTPUT_NODE_ID
      ) {
        continue;
      }
      const source = nodesById.get(edge.source);
      const target = nodesById.get(edge.target);
      if (!source || !target) {
        continue;
      }
      const sources = resolveSource(
        source,
        edge.sourceHandle ?? "",
        context,
        subgraphs,
      );
      const targets = resolveTarget(
        target,
        edge.targetHandle ?? "",
        context,
        subgraphs,
      );
      for (const sourceEndpoint of sources) {
        for (const targetEndpoint of targets) {
          graph.edges.push({
            source_node_id: sourceEndpoint.nodeId,
            source_handle: sourceEndpoint.handle,
            target_node_id: targetEndpoint.nodeId,
            target_handle: targetEndpoint.handle,
          });
          if (graph.edges.length > MAX_FLAT_EDGES) {
            throw new SubgraphError(
              `Expanded workflow has too many edges (${graph.edges.length} > ${MAX_FLAT_EDGES}).`,
            );
          }
        }
      }
    }
  };

  expand(nodes, edges, "", []);
  return { graph, labels };
}

interface BoundaryBinding {
  port: OutputSocket;
  nodeId: string;
  handle: string;
}

export interface BuiltSubgraph {
  definition: SubgraphDefinition;
  instance: AppNode;
  nodes: AppNode[];
  edges: AppEdge[];
}

export function uniqueSubgraphPortName(
  base: string,
  used: Set<string>,
): string {
  const cleaned = base.trim() || "port";
  let candidate = cleaned;
  let index = 2;
  while (used.has(candidate)) {
    candidate = `${cleaned}_${index}`;
    index += 1;
  }
  used.add(candidate);
  return candidate;
}

export function syncSubgraphInstanceSchemas(
  nodes: AppNode[],
  subgraphs: SubgraphMap,
): AppNode[] {
  return nodes.map((node) => {
    if (!isSubgraphInstanceNode(node)) {
      return node;
    }
    const definition = subgraphs[node.data.subgraphId ?? ""];
    if (!definition) {
      return node;
    }
    return {
      ...node,
      data: {
        ...node.data,
        schema: emptyNodeSchema(node.id, definition.inputs, definition.outputs),
      },
    };
  });
}

export function syncSubgraphDefinitions(subgraphs: SubgraphMap): SubgraphMap {
  return Object.fromEntries(
    Object.entries(subgraphs).map(([id, definition]) => [
      id,
      {
        ...definition,
        nodes: syncSubgraphInstanceSchemas(definition.nodes, subgraphs),
      },
    ]),
  );
}

function edgeId(
  source: string,
  sourceHandle: string,
  target: string,
  targetHandle: string,
): string {
  return `${source}:${sourceHandle}->${target}:${targetHandle}`;
}

export function buildSubgraphFromSelection(options: {
  definitionId: string;
  instanceId: string;
  name: string;
  nodes: AppNode[];
  edges: AppEdge[];
  selectedIds: Set<string>;
}): BuiltSubgraph {
  const { definitionId, instanceId, name, nodes, edges, selectedIds } =
    options;
  const selectedNodes = nodes.filter((node) => selectedIds.has(node.id));
  if (selectedNodes.length === 0) {
    throw new SubgraphError("Select at least one node to create a subgraph.");
  }
  if (
    selectedNodes.some(
      (node) => isFrameNode(node) || isSubgraphManagerNode(node),
    )
  ) {
    throw new SubgraphError(
      "Frames and shared interface nodes cannot be nested in a subgraph.",
    );
  }
  const minX = Math.min(...selectedNodes.map((node) => node.position.x));
  const minY = Math.min(...selectedNodes.map((node) => node.position.y));
  const maxX = Math.max(
    ...selectedNodes.map(
      (node) =>
        node.position.x +
        (node.width ?? node.measured?.width ?? FALLBACK_WIDTH),
    ),
  );

  const inputBindings: { port: InputSocket; nodeId: string; handle: string }[] =
    [];
  const outputBindings: BoundaryBinding[] = [];
  const inputByTarget = new Map<string, { port: InputSocket; handle: string }>();
  const outputBySource = new Map<string, BoundaryBinding>();
  const usedInputs = new Set<string>();
  const usedOutputs = new Set<string>();

  for (const edge of edges) {
    const sourceSelected = selectedIds.has(edge.source);
    const targetSelected = selectedIds.has(edge.target);
    if (sourceSelected === targetSelected) {
      continue;
    }
    if (targetSelected) {
      const target = nodes.find((node) => node.id === edge.target);
      if (!target) {
        continue;
      }
      const handle = edge.targetHandle ?? "";
      const key = `${edge.target}\u0000${handle}`;
      if (inputByTarget.has(key)) {
        continue;
      }
      const socket = target.data.schema.inputs.find(
        (candidate) => candidate.name === handle,
      );
      const widget = target.data.schema.widgets.find(
        (candidate) => candidate.name === handle,
      );
      const portName = uniqueSubgraphPortName(handle, usedInputs);
      const port: InputSocket = {
        name: portName,
        data_type: socket
          ? socket.data_type
          : widget
            ? widgetSocketType(widget.data_type)
            : "ANY",
      };
      const binding = { port, nodeId: edge.target, handle };
      inputBindings.push(binding);
      inputByTarget.set(key, { port, handle });
    } else if (sourceSelected) {
      const source = nodes.find((node) => node.id === edge.source);
      if (!source) {
        continue;
      }
      const handle = edge.sourceHandle ?? "";
      const key = `${edge.source}\u0000${handle}`;
      if (outputBySource.has(key)) {
        continue;
      }
      const socket = source.data.schema.outputs.find(
        (candidate) => candidate.name === handle,
      );
      const portName = uniqueSubgraphPortName(handle, usedOutputs);
      const binding: BoundaryBinding = {
        port: {
          name: portName,
          data_type: (socket?.data_type ?? "ANY") as SocketType,
        },
        nodeId: edge.source,
        handle,
      };
      outputBindings.push(binding);
      outputBySource.set(key, binding);
    }
  }

  const nextNodes: AppNode[] = [];
  const nextEdges: AppEdge[] = [];
  for (const edge of edges) {
    const sourceSelected = selectedIds.has(edge.source);
    const targetSelected = selectedIds.has(edge.target);
    if (sourceSelected && targetSelected) {
      continue;
    }
    if (sourceSelected) {
      const outgoing = outputBySource.get(
        `${edge.source}\u0000${edge.sourceHandle ?? ""}`,
      );
      if (!outgoing) {
        continue;
      }
      const sourceHandle = outgoing.port.name;
      const targetHandle = edge.targetHandle ?? "";
      nextEdges.push({
        ...edge,
        id: edgeId(instanceId, sourceHandle, edge.target, targetHandle),
        source: instanceId,
        sourceHandle,
      });
      continue;
    }
    if (targetSelected) {
      const incoming = inputByTarget.get(
        `${edge.target}\u0000${edge.targetHandle ?? ""}`,
      );
      if (!incoming) {
        continue;
      }
      const sourceHandle = edge.sourceHandle ?? "";
      const targetHandle = incoming.port.name;
      nextEdges.push({
        ...edge,
        id: edgeId(edge.source, sourceHandle, instanceId, targetHandle),
        target: instanceId,
        targetHandle,
      });
      continue;
    }
    nextEdges.push(edge);
  }

  const definitionNodes = selectedNodes.map((node) => ({
    ...node,
    selected: false,
    position: {
      x: node.position.x - minX,
      y: node.position.y - minY,
    },
    data: { ...node.data },
  }));
  const definitionEdges: AppEdge[] = edges
    .filter(
      (edge) => selectedIds.has(edge.source) && selectedIds.has(edge.target),
    )
    .map((edge) => ({ ...edge, selected: false }));
  for (const binding of inputBindings) {
    definitionEdges.push({
      id: edgeId(
        SUBGRAPH_INPUT_NODE_ID,
        binding.port.name,
        binding.nodeId,
        binding.handle,
      ),
      source: SUBGRAPH_INPUT_NODE_ID,
      sourceHandle: binding.port.name,
      target: binding.nodeId,
      targetHandle: binding.handle,
    });
  }
  for (const binding of outputBindings) {
    definitionEdges.push({
      id: edgeId(
        binding.nodeId,
        binding.handle,
        SUBGRAPH_OUTPUT_NODE_ID,
        binding.port.name,
      ),
      source: binding.nodeId,
      sourceHandle: binding.handle,
      target: SUBGRAPH_OUTPUT_NODE_ID,
      targetHandle: binding.port.name,
    });
  }

  const definition: SubgraphDefinition = {
    id: definitionId,
    name,
    inputs: inputBindings.map((binding) => binding.port),
    outputs: outputBindings.map((binding) => binding.port),
    nodes: definitionNodes,
    edges: definitionEdges,
    inputPosition: { x: -260, y: 0 },
    outputPosition: { x: maxX - minX + 220, y: 0 },
  };
  const instance = createSubgraphInstanceNode(
    instanceId,
    definition,
    { x: minX, y: minY },
    name,
  );
  for (const node of nodes) {
    if (selectedIds.has(node.id)) {
      continue;
    }
    nextNodes.push(node.selected ? { ...node, selected: false } : node);
  }
  nextNodes.push(instance);
  return {
    definition,
    instance,
    nodes: nextNodes,
    edges: nextEdges,
  };
}
