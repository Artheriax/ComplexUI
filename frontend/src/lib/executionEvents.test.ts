import { describe, expect, it } from "vitest";

import {
  idleExecution,
  pruneExecution,
  reduceExecutionEvent,
  reduceRunStatus,
} from "./executionEvents";

describe("execution event reducer", () => {
  it("starts a run with its order and run id", () => {
    const execution = reduceExecutionEvent(idleExecution(), {
      type: "EXECUTION_STARTED",
      node_id: "",
      data: { order: ["a", "b"], run_id: "run-1" },
    });

    expect(execution.status).toBe("running");
    expect(execution.runId).toBe("run-1");
    expect(execution.order).toEqual(["a", "b"]);
  });

  it("clamps progress and records completion payloads", () => {
    const running = reduceExecutionEvent(idleExecution(), {
      type: "EXECUTION_STARTED",
      node_id: "",
      data: { run_id: "run-1" },
    });
    const progressed = reduceExecutionEvent(running, {
      type: "NODE_PROGRESS",
      node_id: "a",
      data: { progress: 1.4, message: "working" },
    });
    expect(progressed.nodeStates.a).toEqual({
      status: "running",
      progress: 1,
      progressMessage: "working",
    });

    const completed = reduceExecutionEvent(progressed, {
      type: "NODE_COMPLETED",
      node_id: "a",
      data: {
        cached: true,
        outputs: { result: { rows: 1, cols: 1, head: [] } },
        rich_outputs: [{ kind: "text", source: "stdout", text: "ok", truncated: false }],
      },
    });
    expect(completed.status).toBe("running");
    expect(completed.cached.a).toBe(true);
    expect(completed.nodeStates.a?.status).toBe("completed");
    expect(completed.richOutputs.a).toHaveLength(1);
  });

  it("records node and run failures", () => {
    const nodeError = reduceExecutionEvent(idleExecution(), {
      type: "NODE_ERROR",
      node_id: "a",
      data: { line: 3, message: "boom", error_type: "ValueError" },
    });
    expect(nodeError.nodeStates.a?.error).toEqual({
      line: 3,
      message: "boom",
      errorType: "ValueError",
      traceback: "",
    });

    const failed = reduceExecutionEvent(nodeError, {
      type: "EXECUTION_FAILED",
      node_id: "a",
      data: { node_id: "a", line: 3, message: "boom", error_type: "ValueError" },
    });
    expect(failed.status).toBe("failed");
    expect(failed.error?.nodeId).toBe("a");
  });

  it("clears running nodes when a run is cancelled", () => {
    const running = reduceExecutionEvent(idleExecution(), {
      type: "NODE_STARTED",
      node_id: "a",
      data: {},
    });
    const cancelled = reduceExecutionEvent(running, {
      type: "EXECUTION_CANCELLED",
      node_id: "",
      data: {},
    });

    expect(cancelled.status).toBe("cancelled");
    expect(cancelled.nodeStates.a).toEqual({ status: "idle" });
  });
});

describe("execution helpers", () => {
  it("prunes execution state for removed nodes", () => {
    const execution = reduceExecutionEvent(idleExecution(), {
      type: "NODE_COMPLETED",
      node_id: "a",
      data: { outputs: { result: 1 } },
    });
    const pruned = pruneExecution(execution, new Set(["a"]));

    expect(pruned.nodeStates).toEqual({});
    expect(pruned.previews).toEqual({});
  });

  it("marks a lost run as failed on status resync", () => {
    const running = reduceExecutionEvent(idleExecution(), {
      type: "EXECUTION_STARTED",
      node_id: "",
      data: { run_id: "run-1" },
    });
    const lost = reduceRunStatus(running, { status: "idle", run_id: null });

    expect(lost.status).toBe("failed");
    expect(lost.error?.errorType).toBe("Disconnected");
  });
});
