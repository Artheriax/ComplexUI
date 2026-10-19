// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fetchRunStatus } from "../lib/api";
import { useGraphStore } from "../store/useGraphStore";
import { useExecutionSocket } from "./useExecutionSocket";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, fetchRunStatus: vi.fn() };
});

const mockedStatus = vi.mocked(fetchRunStatus);

class FakeSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeSocket[] = [];

  readyState = FakeSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((message: { data: string }) => void) | null = null;

  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }

  close(): void {
    if (this.readyState === FakeSocket.CLOSED) {
      return;
    }
    this.readyState = FakeSocket.CLOSED;
    this.onclose?.();
  }
}

beforeEach(() => {
  FakeSocket.instances = [];
  mockedStatus.mockReset();
  mockedStatus.mockResolvedValue({ status: "idle", run_id: null });
  vi.stubGlobal("WebSocket", FakeSocket);
  useGraphStore.setState({
    wsConnected: false,
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
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useExecutionSocket", () => {
  it("resyncs the run status when the socket opens", async () => {
    mockedStatus.mockResolvedValue({ status: "running", run_id: "srv-9" });
    const { unmount } = renderHook(() => useExecutionSocket());

    await act(async () => {
      FakeSocket.instances[0]?.onopen?.();
    });

    expect(mockedStatus).toHaveBeenCalledTimes(1);
    expect(useGraphStore.getState().wsConnected).toBe(true);
    expect(useGraphStore.getState().execution.status).toBe("running");
    expect(useGraphStore.getState().execution.runId).toBe("srv-9");
    unmount();
  });

  it("reconnects with backoff after the socket closes", () => {
    vi.useFakeTimers();
    const { unmount } = renderHook(() => useExecutionSocket());
    expect(FakeSocket.instances).toHaveLength(1);

    act(() => {
      const socket = FakeSocket.instances[0]!;
      socket.readyState = FakeSocket.OPEN;
      socket.onclose?.();
    });
    expect(useGraphStore.getState().wsConnected).toBe(false);

    act(() => {
      vi.advanceTimersByTime(1_000);
    });
    expect(FakeSocket.instances).toHaveLength(2);
    unmount();
  });

  it("stops reconnecting after unmount", () => {
    vi.useFakeTimers();
    const { unmount } = renderHook(() => useExecutionSocket());

    unmount();
    act(() => {
      vi.advanceTimersByTime(30_000);
    });

    expect(FakeSocket.instances).toHaveLength(1);
  });
});
