export const DATASET_ACCEPT = [
  ".csv",
  ".tsv",
  ".parquet",
  ".xlsx",
  ".xlsm",
  ".ndjson",
  ".jsonl",
].join(",");

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
