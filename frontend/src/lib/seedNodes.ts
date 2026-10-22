import type { AppNode, ExecutionEvent } from "../types/graph";

export function advanceSeedNodes(
  nodes: AppNode[],
  event: ExecutionEvent,
): AppNode[] {
  if (event.type !== "NODE_COMPLETED" || event.data.cached === true) {
    return nodes;
  }
  const node = nodes.find((candidate) => candidate.id === event.node_id);
  if (!node) {
    return nodes;
  }
  const values = node.data.widgetValues;
  const seedKey =
    typeof values.seed === "number"
      ? "seed"
      : typeof values.noise_seed === "number"
        ? "noise_seed"
        : null;
  const seed = seedKey ? values[seedKey] : undefined;
  const control =
    values.control_after_generated ?? values.control_after_generate;
  if (seedKey === null || typeof seed !== "number" || !Number.isFinite(seed)) {
    return nodes;
  }
  let nextSeed: number;
  if (control === "increment") {
    nextSeed = seed + 1;
  } else if (control === "decrement") {
    nextSeed = seed - 1;
  } else if (control === "randomize") {
    nextSeed = Math.floor(Math.random() * 4_294_967_295);
  } else {
    return nodes;
  }
  if (nextSeed === seed) {
    return nodes;
  }
  return nodes.map((candidate) =>
    candidate.id === node.id
      ? {
          ...candidate,
          data: {
            ...candidate.data,
            widgetValues: { ...values, [seedKey]: nextSeed },
          },
        }
      : candidate,
  );
}
