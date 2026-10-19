import { useEffect } from "react";

import { fetchRunStatus } from "../lib/api";
import { useGraphStore } from "../store/useGraphStore";
import type { ExecutionEvent } from "../types/graph";

const MAX_RECONNECT_DELAY_MS = 10_000;

export function executionSocketUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/execution`;
}

/**
 * Connects to the backend execution telemetry socket and feeds every event
 * into the graph store, reconnecting with exponential backoff.
 */
export function useExecutionSocket(): void {
  useEffect(() => {
    let socket: WebSocket | null = null;
    let disposed = false;
    let attempts = 0;
    let reconnectTimer: number | undefined;

    const connect = () => {
      socket = new WebSocket(executionSocketUrl());

      socket.onopen = () => {
        attempts = 0;
        useGraphStore.getState().setWsConnected(true);
        void fetchRunStatus()
          .then((status) => {
            if (!disposed) {
              useGraphStore.getState().syncRunStatus(status);
            }
          })
          .catch(() => {
            // The socket is open but REST may lag during a restart; the next
            // reconnect will resync again.
          });
      };

      socket.onmessage = (message: MessageEvent<string>) => {
        try {
          const event = JSON.parse(message.data) as ExecutionEvent;
          useGraphStore.getState().applyExecutionEvent(event);
        } catch {
          // Ignore frames that are not valid execution events.
        }
      };

      socket.onclose = () => {
        useGraphStore.getState().setWsConnected(false);
        if (disposed) {
          return;
        }
        attempts += 1;
        const delay = Math.min(500 * 2 ** attempts, MAX_RECONNECT_DELAY_MS);
        reconnectTimer = window.setTimeout(connect, delay);
      };

      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    return () => {
      disposed = true;
      window.clearTimeout(reconnectTimer);
      if (!socket) {
        return;
      }
      if (socket.readyState === WebSocket.CONNECTING) {
        socket.onopen = () => socket?.close();
        socket.onclose = null;
      } else {
        socket.close();
      }
    };
  }, []);
}
