// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SystemMetrics } from "../lib/api";
import { fetchSystemMetrics } from "../lib/api";
import { useSystemMetrics } from "./useSystemMetrics";

vi.mock("../lib/api", () => ({
  fetchSystemMetrics: vi.fn(),
}));

const metrics: SystemMetrics = {
  cpu: { utilization: 12, cores: 8 },
  memory: { total_bytes: 100, used_bytes: 40, utilization: 40 },
  gpus: [],
  timestamp: "2026-01-01T00:00:00Z",
};

const mockedFetch = vi.mocked(fetchSystemMetrics);

beforeEach(() => {
  vi.useFakeTimers();
  mockedFetch.mockReset();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("useSystemMetrics", () => {
  it("keeps one request in flight across visibility changes", async () => {
    let resolveFirst: ((value: SystemMetrics) => void) | undefined;
    mockedFetch.mockImplementationOnce(
      () =>
        new Promise<SystemMetrics>((resolve) => {
          resolveFirst = resolve;
        }),
    );

    const { result } = renderHook(() => useSystemMetrics(true, 1000));
    expect(mockedFetch).toHaveBeenCalledTimes(1);

    act(() => {
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(mockedFetch).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveFirst?.(metrics);
    });
    expect(result.current.metrics).toEqual(metrics);
    expect(result.current.updating).toBe(false);
  });

  it("polls once per interval after a request settles", async () => {
    mockedFetch.mockResolvedValue(metrics);
    renderHook(() => useSystemMetrics(true, 1000));

    await act(async () => {
      await Promise.resolve();
    });
    expect(mockedFetch).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });
    expect(mockedFetch).toHaveBeenCalledTimes(2);
  });

  it("does not poll while disabled", () => {
    renderHook(() => useSystemMetrics(false, 1000));
    expect(mockedFetch).not.toHaveBeenCalled();
  });
});
