import { ConnectionLineType, MarkerType } from "@xyflow/react";
import { expect, test } from "vitest";

import {
  CONNECTION_LINE_TYPE_BY_STYLE,
  buildDisplayEdges,
} from "./edgePresentation";
import { DEFAULT_SETTINGS } from "../store/useSettingsStore";
import type { AppEdge, AppNode, OutputSocket } from "../types/graph";
import { SOCKET_COLORS } from "../types/graph";

function fakeNode(id: string, outputs: OutputSocket[]): AppNode {
  return {
    id,
    type: "complex",
    position: { x: 0, y: 0 },
    data: {
      label: id,
      templateName: null,
      templateCode: null,
      code: "",
      schema: {
        node_id: id,
        inputs: [],
        widgets: [],
        outputs,
        dependencies: [],
        syntax_errors: [],
      },
      widgetValues: {},
      isTrusted: true,
      color: "default",
    },
  };
}

const loadNode = fakeNode("load", [{ name: "df", data_type: "DATAFRAME" }]);
const trainNode = fakeNode("train", [{ name: "model", data_type: "ML_MODEL" }]);
const edges: AppEdge[] = [
  {
    id: "load->train",
    source: "load",
    sourceHandle: "df",
    target: "train",
    targetHandle: "data",
  },
  {
    id: "missing->train",
    source: "missing",
    sourceHandle: "ghost",
    target: "train",
    targetHandle: "data",
  },
];

test("colors edges by source socket type and applies thickness", () => {
  const presented = buildDisplayEdges(edges, [loadNode, trainNode], {
    ...DEFAULT_SETTINGS,
  });

  expect(presented[0]?.type).toBe("default");
  expect(presented[0]?.style).toMatchObject({
    stroke: SOCKET_COLORS.DATAFRAME,
    strokeWidth: 2.5,
  });
  expect(presented[0]?.hidden).toBe(false);
  expect(presented[0]?.animated).toBe(false);
  expect(presented[0]?.markerEnd).toBeUndefined();
  expect(presented[1]?.style).toMatchObject({
    stroke: SOCKET_COLORS.ANY,
  });
});

test("applies the single accent color, line style, animation and arrowheads", () => {
  const presented = buildDisplayEdges(edges, [loadNode, trainNode], {
    ...DEFAULT_SETTINGS,
    showEdges: false,
    edgeStyle: "straight",
    edgeWidth: 4,
    edgeColorMode: "single",
    accentColor: "#fbbf24",
    edgeAnimated: true,
    showArrowheads: true,
  });

  expect(presented[0]?.type).toBe("straight");
  expect(presented[0]?.style).toMatchObject({
    stroke: "#fbbf24",
    strokeWidth: 4,
  });
  expect(presented[0]?.hidden).toBe(true);
  expect(presented[0]?.animated).toBe(true);
  expect(presented[0]?.markerEnd).toMatchObject({
    type: MarkerType.ArrowClosed,
    color: "#fbbf24",
  });
});

test("maps line styles to connection preview styles", () => {
  expect(CONNECTION_LINE_TYPE_BY_STYLE.bezier).toBe(ConnectionLineType.Bezier);
  expect(CONNECTION_LINE_TYPE_BY_STYLE.smoothstep).toBe(
    ConnectionLineType.SmoothStep,
  );
  expect(CONNECTION_LINE_TYPE_BY_STYLE.step).toBe(ConnectionLineType.Step);
  expect(CONNECTION_LINE_TYPE_BY_STYLE.straight).toBe(
    ConnectionLineType.Straight,
  );
});
