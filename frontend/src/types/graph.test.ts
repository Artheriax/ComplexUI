import { describe, expect, it } from "vitest";

import {
  SOCKET_COLORS,
  SOCKET_LABELS,
  SOCKET_TYPE_NAMES,
} from "./graph";

describe("socket registry", () => {
  it("covers every socket type used by colors and labels", () => {
    expect(new Set(Object.keys(SOCKET_COLORS))).toEqual(SOCKET_TYPE_NAMES);
    expect(new Set(Object.keys(SOCKET_LABELS))).toEqual(SOCKET_TYPE_NAMES);
  });
});
