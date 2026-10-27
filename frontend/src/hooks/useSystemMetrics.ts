import { useEffect, useState } from "react";

import { fetchSystemMetrics } from "../lib/api";
import type { SystemMetrics } from "../lib/api";
import { errorMessage } from "../lib/errors";

export interface SystemMetricsState {
  metrics: SystemMetrics | null;
  error: string | null;
  updating: boolean;
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
    let sequence = 0;
    let inFlight = false;

    function schedule(delay: number): void {
      window.clearTimeout(timer);
      if (!disposed) {
        timer = window.setTimeout(() => {
          void poll();
        }, delay);
      }
    }

    function cancelInFlight(): void {
      if (!inFlight) {
        return;
      }
      sequence += 1;
      controller?.abort();
      controller = null;
      inFlight = false;
      setUpdating(false);
    }

    async function poll(): Promise<void> {
      if (disposed || inFlight) {
        return;
      }
      if (document.visibilityState !== "visible") {
        window.clearTimeout(timer);
        return;
      }
      const current = ++sequence;
      inFlight = true;
      controller = new AbortController();
      setUpdating(true);
      try {
        const next = await fetchSystemMetrics(controller.signal);
        if (!disposed && current === sequence) {
          setMetrics(next);
          setError(null);
        }
      } catch (cause) {
        if (
          disposed ||
          current !== sequence ||
          (cause instanceof DOMException && cause.name === "AbortError")
        ) {
          return;
        }
        setError(errorMessage(cause));
      } finally {
        if (current === sequence) {
          inFlight = false;
          controller = null;
          if (!disposed) {
            setUpdating(false);
            schedule(intervalMs);
          }
        }
      }
    }

    const onVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        window.clearTimeout(timer);
        cancelInFlight();
        return;
      }
      window.clearTimeout(timer);
      void poll();
    };

    void poll();
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      disposed = true;
      sequence += 1;
      controller?.abort();
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [enabled, intervalMs]);

  return { metrics, error, updating };
}
