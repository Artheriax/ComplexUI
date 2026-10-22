import { useEffect, useState } from "react";

import { fetchSystemMetrics } from "../lib/api";
import type { SystemMetrics } from "../lib/api";

export interface SystemMetricsState {
  metrics: SystemMetrics | null;
  error: string | null;
  updating: boolean;
}

function errorMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/** Poll system metrics while the monitor is visible, pausing in hidden tabs. */
export function useSystemMetrics(
  enabled: boolean,
  intervalMs: number,
): SystemMetricsState {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updating, setUpdating] = useState(false);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    let disposed = false;
    let timer: number | undefined;
    let controller: AbortController | null = null;

    function schedule(delay: number): void {
      if (!disposed) {
        timer = window.setTimeout(() => {
          void poll();
        }, delay);
      }
    }

    async function poll(): Promise<void> {
      if (disposed) {
        return;
      }
      if (document.visibilityState === "hidden") {
        schedule(intervalMs);
        return;
      }
      controller = new AbortController();
      setUpdating(true);
      try {
        const next = await fetchSystemMetrics(controller.signal);
        if (!disposed) {
          setMetrics(next);
          setError(null);
        }
      } catch (cause) {
        if (
          disposed ||
          (cause instanceof DOMException && cause.name === "AbortError")
        ) {
          return;
        }
        setError(errorMessage(cause));
      } finally {
        if (!disposed) {
          setUpdating(false);
          schedule(intervalMs);
        }
      }
    }

    const onVisibilityChange = () => {
      if (document.visibilityState !== "visible") {
        return;
      }
      window.clearTimeout(timer);
      void poll();
    };

    void poll();
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      disposed = true;
      controller?.abort();
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [enabled, intervalMs]);

  return { metrics, error, updating };
}
