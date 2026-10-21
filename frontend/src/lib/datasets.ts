import type { AssetKind } from "./api";

export const TABLE_ACCEPT = [
  ".csv",
  ".tsv",
  ".parquet",
  ".xlsx",
  ".xlsm",
  ".ndjson",
  ".jsonl",
].join(",");

export const IMAGE_ACCEPT = [
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".bmp",
  ".gif",
  ".tif",
  ".tiff",
].join(",");

export const VIDEO_ACCEPT = [".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"].join(
  ",",
);

export const ASSET_ACCEPT = [TABLE_ACCEPT, IMAGE_ACCEPT, VIDEO_ACCEPT].join(",");

export const DATASET_ACCEPT = ASSET_ACCEPT;

export function assetAccept(kind: AssetKind | null | undefined): string {
  if (kind === "image") {
    return IMAGE_ACCEPT;
  }
  if (kind === "video") {
    return VIDEO_ACCEPT;
  }
  if (kind === "table") {
    return TABLE_ACCEPT;
  }
  return ASSET_ACCEPT;
}

export function formatDatasetSize(bytes: number): string {
  if (bytes < 1_024) {
    return `${bytes} B`;
  }
  if (bytes < 1_024 * 1_024) {
    return `${(bytes / 1_024).toFixed(1)} KB`;
  }
  return `${(bytes / (1_024 * 1_024)).toFixed(1)} MB`;
}

export function formatDatasetDate(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

export function datasetFileName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}
