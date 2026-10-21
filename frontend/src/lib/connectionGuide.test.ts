import { expect, test } from "vitest";

import type { ConnectionGuide } from "../store/useConnectionGuideStore";
import { classifyHandle } from "./connectionGuide";

const sourceGuide: ConnectionGuide = {
  mode: "connect",
  direction: "source",
  dataType: "IMAGE",
  originNodeId: "source",
  originHandleId: "image",
};
const targetGuide: ConnectionGuide = {
  mode: "connect",
  direction: "target",
  dataType: "DATAFRAME",
  originNodeId: "target",
  originHandleId: "df",
};
const bundleGuide: ConnectionGuide = {
  mode: "bundle",
  direction: "source",
  dataType: "IMAGE",
  originNodeId: "source",
  originHandleId: "image",
};

test("source drags highlight only compatible free inputs", () => {
  const common = {
    guide: sourceGuide,
    kind: "target" as const,
    nodeId: "other",
    handleId: "input",
  };

  expect(classifyHandle({ ...common, dataType: "MEDIA" })).toBe("compatible");
  expect(classifyHandle({ ...common, dataType: "IMAGE" })).toBe("compatible");
  expect(classifyHandle({ ...common, dataType: "DATAFRAME" })).toBe(
    "incompatible",
  );
  expect(classifyHandle({ ...common, dataType: "MEDIA", occupied: true })).toBe(
    "incompatible",
  );
  expect(
    classifyHandle({
      ...common,
      nodeId: "source",
      dataType: "IMAGE",
    }),
  ).toBe("incompatible");
});

test("target drags highlight only compatible outputs", () => {
  expect(
    classifyHandle({
      guide: targetGuide,
      kind: "source",
      dataType: "DATAFRAME",
      nodeId: "other",
      handleId: "out",
    }),
  ).toBe("compatible");
  expect(
    classifyHandle({
      guide: targetGuide,
      kind: "source",
      dataType: "IMAGE",
      nodeId: "other",
      handleId: "out",
    }),
  ).toBe("incompatible");
  expect(
    classifyHandle({
      guide: targetGuide,
      kind: "target",
      dataType: "DATAFRAME",
      nodeId: "other",
      handleId: "in",
    }),
  ).toBe("incompatible");
});

test("bundle drags highlight only matching output types", () => {
  expect(
    classifyHandle({
      guide: bundleGuide,
      kind: "source",
      dataType: "IMAGE",
      nodeId: "other",
      handleId: "image",
    }),
  ).toBe("compatible");
  expect(
    classifyHandle({
      guide: bundleGuide,
      kind: "source",
      dataType: "MEDIA",
      nodeId: "other",
      handleId: "media",
    }),
  ).toBe("incompatible");
  expect(
    classifyHandle({
      guide: bundleGuide,
      kind: "target",
      dataType: "IMAGE",
      nodeId: "other",
      handleId: "image",
    }),
  ).toBe("incompatible");
});

test("an edge drag treats its current target as available", () => {
  const guide: ConnectionGuide = {
    ...sourceGuide,
    ignoredTargetNodeId: "target",
    ignoredTargetHandleId: "input",
  };

  expect(
    classifyHandle({
      guide,
      kind: "target",
      dataType: "IMAGE",
      nodeId: "target",
      handleId: "input",
      occupied: true,
    }),
  ).toBe("compatible");
  expect(
    classifyHandle({
      guide,
      kind: "target",
      dataType: "IMAGE",
      nodeId: "other",
      handleId: "input",
      occupied: true,
    }),
  ).toBe("incompatible");
});

test("the origin handle stays active", () => {
  expect(
    classifyHandle({
      guide: sourceGuide,
      kind: "source",
      dataType: "IMAGE",
      nodeId: "source",
      handleId: "image",
    }),
  ).toBe("active");
});
