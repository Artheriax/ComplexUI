import { useEffect, useRef, useState } from "react";

import {
  openSavedOutput,
  revealSavedOutput,
  savedMediaRawUrl,
} from "../lib/api";
import { copyImageFromUrl } from "../lib/clipboard";

export type MediaActionState =
  | { kind: "idle" }
  | { kind: "working" }
  | { kind: "success"; message: string }
  | { kind: "error"; message: string };

export function useMediaActions(
  path: string | null,
  kind: "image" | "video" | null,
) {
  const [state, setState] = useState<MediaActionState>({ kind: "idle" });
  const timerRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
    },
    [],
  );

  const announce = (message: string) => {
    setState({ kind: "success", message });
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
    }
    timerRef.current = window.setTimeout(
      () => setState({ kind: "idle" }),
      1800,
    );
  };

  const run = async (message: string, task: () => Promise<unknown>) => {
    setState({ kind: "working" });
    try {
      await task();
      announce(message);
    } catch (cause) {
      setState({
        kind: "error",
        message: cause instanceof Error ? cause.message : String(cause),
      });
    }
  };

  const copy = () => {
    if (path === null || kind !== "image") {
      return;
    }
    void run("Copied image", () =>
      copyImageFromUrl(savedMediaRawUrl(path)),
    );
  };

  const open = () => {
    if (path === null) {
      return;
    }
    void run("Opened", () => openSavedOutput(path));
  };

  const reveal = () => {
    if (path === null) {
      return;
    }
    void run("Shown in folder", () => revealSavedOutput(path));
  };

  return {
    state,
    busy: state.kind === "working",
    copy,
    open,
    reveal,
  };
}
