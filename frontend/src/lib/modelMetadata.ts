import type { ModelMemoryEstimate, ModelQuantizationInfo } from "./api";

const GIB = 1024 ** 3;

export function formatModelSize(bytes: number): string {
  if (bytes <= 0) {
    return "0 B";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  if (bytes < GIB) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(bytes / GIB).toFixed(2)} GB`;
}

export function formatParameterCount(value: number | null | undefined): string | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2)}B`;
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return String(value);
}

export function quantizationTag(info: ModelQuantizationInfo | null | undefined): string | null {
  if (!info || info.format === "none") {
    return null;
  }
  return info.label || info.format;
}

export function quantizationDetail(
  info: ModelQuantizationInfo | null | undefined,
): string | null {
  if (!info || info.format === "none") {
    return null;
  }
  const details: string[] = [];
  if (info.variant && !info.label.includes(info.variant)) {
    details.push(info.variant);
  }
  if (info.weight_bits) {
    details.push(`${info.weight_bits}-bit`);
  }
  if (info.activation) {
    details.push(info.activation);
  }
  if (info.runtime && info.runtime !== "native") {
    details.push(info.runtime);
  }
  return details.length > 0 ? details.join(" · ") : null;
}

export function memoryTag(memory: ModelMemoryEstimate | null | undefined): string | null {
  if (!memory) {
    return null;
  }
  if (memory.unknown) {
    return "memory unknown";
  }
  const resident = formatModelSize(memory.resident_estimate_bytes);
  return `~${resident} resident`;
}

export function memoryDetail(memory: ModelMemoryEstimate | null | undefined): string | null {
  if (!memory) {
    return null;
  }
  const parts = [
    `${formatModelSize(memory.weights_bytes)} weights`,
    `${formatModelSize(memory.offload_ram_estimate_bytes)} offload RAM`,
    `${memory.confidence} confidence`,
  ];
  return `estimate: ${parts.join(" · ")}`;
}
