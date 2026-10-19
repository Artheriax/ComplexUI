// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { parseCode } from "../lib/api";
import { useGraphStore } from "../store/useGraphStore";
import type { NodeSchema, TemplateInfo } from "../types/graph";
import { CodeModal } from "./CodeModal";

vi.mock("@monaco-editor/react", async () => {
  const React = await import("react");
  return {
    default: ({
      value,
      onChange,
    }: {
      value?: string;
      onChange?: (value: string) => void;
    }) =>
      React.createElement("textarea", {
        value,
        onChange: (event: { target: { value: string } }) =>
          onChange?.(event.target.value),
      }),
  };
});

vi.mock("../lib/monaco", () => ({}));

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, parseCode: vi.fn() };
});

const mockedParse = vi.mocked(parseCode);

const schema: NodeSchema = {
  node_id: "",
  inputs: [],
  widgets: [{ name: "k", data_type: "int", default: 5 }],
  outputs: [{ name: "k", data_type: "ANY" }],
  dependencies: [],
  syntax_errors: [],
};

const template: TemplateInfo = {
  name: "demo",
  title: "Demo Node",
  category: "Preprocessing",
  description: "Demo template",
  hidden: false,
  code: "# demo\nk: int = 5\nreturn k: ANY\n",
  code_hash: "demo-hash",
  node_schema: schema,
};

const EDITED = "# edited\nk: int = 7\nreturn k: ANY\n";

beforeEach(() => {
  vi.useFakeTimers();
  mockedParse.mockReset();
  vi.stubGlobal("crypto", {
    randomUUID: () => `node-${Math.random().toString(16).slice(2)}`,
    subtle: { digest: async () => new ArrayBuffer(32) },
  });
  useGraphStore.setState({
    templates: [template],
    templatesLoaded: true,
    nodes: [],
    edges: [],
    past: [],
    future: [],
    pendingApproval: null,
    approvedHashes: new Set<string>(),
    editingNodeId: null,
    execution: {
      status: "idle",
      runId: null,
      order: [],
      nodeStates: {},
      previews: {},
      richOutputs: {},
      cached: {},
      error: null,
    },
  });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function openEditor(): { id: string; save: HTMLButtonElement } {
  const id = useGraphStore
    .getState()
    .addNodeFromTemplate(template, { x: 0, y: 0 });
  useGraphStore.setState({ editingNodeId: id });
  render(createElement(CodeModal, { nodeId: id }));
  const save = screen.getByRole("button", {
    name: "Save & apply",
  }) as HTMLButtonElement;
  return { id, save };
}

describe("CodeModal", () => {
  it("keeps save disabled until the debounced parse catches up", async () => {
    const { id, save } = openEditor();
    mockedParse.mockResolvedValue({ ...schema, node_id: id });

    expect(save.disabled).toBe(true);

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: EDITED },
    });

    expect(save.disabled).toBe(true);

    await act(async () => {
      vi.advanceTimersByTime(400);
    });

    expect(mockedParse).toHaveBeenCalledWith(EDITED, id);
    expect(save.disabled).toBe(false);
  });

  it("keeps save disabled when the edited code has syntax errors", async () => {
    const { id, save } = openEditor();
    mockedParse.mockResolvedValue({
      ...schema,
      node_id: id,
      syntax_errors: [{ line: 1, column: 0, message: "broken syntax" }],
    });

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: EDITED },
    });
    await act(async () => {
      vi.advanceTimersByTime(400);
    });

    expect(screen.getByText(/broken syntax/)).toBeTruthy();
    expect(save.disabled).toBe(true);
  });

  it("exposes dialog semantics and closes on Escape", () => {
    openEditor();

    expect(screen.getByRole("dialog", { name: "Demo Node" })).toBeTruthy();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(useGraphStore.getState().editingNodeId).toBeNull();
  });

  it("saves the edited code with the freshly parsed schema", async () => {    const { id, save } = openEditor();
    const freshSchema = { ...schema, node_id: id };
    mockedParse.mockResolvedValue(freshSchema);

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: EDITED },
    });
    await act(async () => {
      vi.advanceTimersByTime(400);
    });
    vi.useRealTimers();

    await act(async () => {
      fireEvent.click(save);
    });

    const pending = useGraphStore.getState().pendingApproval;
    expect(pending?.nodeId).toBe(id);
    expect(pending?.code).toBe(EDITED);
    expect(pending?.schema).toEqual(freshSchema);
  });
});
