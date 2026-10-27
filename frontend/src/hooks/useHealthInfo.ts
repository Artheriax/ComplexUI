import { useCallback, useEffect, useRef, useState } from "react";

import { fetchHealth } from "../lib/api";
import type { HealthInfo } from "../lib/api";
import { errorMessage } from "../lib/errors";

export interface HealthInfoState {
  health: HealthInfo | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

/** Load backend runtime diagnostics once on mount, with manual refresh. */
export function useHealthInfo(enabled = true): HealthInfoState {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    if (!enabled) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const next = await fetchHealth();
      if (mounted.current) {
        setHealth(next);
      }
    } catch (cause) {
      if (mounted.current) {
        setError(errorMessage(cause));
      }
    } finally {
      if (mounted.current) {
        setLoading(false);
      }
    }
  }, [enabled]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refresh();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  return { health, loading, error, refresh };
}
