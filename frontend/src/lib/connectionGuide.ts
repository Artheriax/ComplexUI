import type { SocketType } from "../types/graph";
import { socketTypesCompatible } from "./socketCompatibility";
import type { ConnectionGuide } from "../store/useConnectionGuideStore";

export type HandleGuideState =
  | "neutral"
  | "active"
  | "compatible"
  | "incompatible";

export function classifyHandle({
  guide,
  kind,
  dataType,
  nodeId,
  handleId,
  occupied = false,
}: {
  guide: ConnectionGuide | null;
  kind: "source" | "target";
  dataType: SocketType;
  nodeId: string;
  handleId: string;
  occupied?: boolean;
}): HandleGuideState {
  if (!guide) {
    return "neutral";
  }
  const isOrigin =
    nodeId === guide.originNodeId && handleId === guide.originHandleId;
  const isIgnoredTarget =
    nodeId === guide.ignoredTargetNodeId &&
    handleId === guide.ignoredTargetHandleId;
  if (guide.mode === "bundle") {
    if (isOrigin) {
      return "active";
    }
    if (kind === "target") {
      return "incompatible";
    }
    return dataType === guide.dataType ? "compatible" : "incompatible";
  }
  if (isOrigin) {
    return "active";
  }
  if (guide.direction === "source") {
    if (kind === "source") {
      return "incompatible";
    }
    if ((occupied && !isIgnoredTarget) || nodeId === guide.originNodeId) {
      return "incompatible";
    }
    return socketTypesCompatible(guide.dataType, dataType)
      ? "compatible"
      : "incompatible";
  }
  if (kind === "target") {
    return "incompatible";
  }
  if (nodeId === guide.originNodeId) {
    return "incompatible";
  }
  return socketTypesCompatible(dataType, guide.dataType)
    ? "compatible"
    : "incompatible";
}

export function handleGuideClass(state: HandleGuideState): string {
  if (state === "compatible") {
    return "cu-handle-compatible";
  }
  if (state === "incompatible") {
    return "cu-handle-incompatible";
  }
  if (state === "active") {
    return "cu-handle-active";
  }
  return "";
}
