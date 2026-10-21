import type {
  ClassificationsPreview,
  DataFramePreview,
  DatasetPreview,
  DetectionsPreview,
  ImagePreview,
  MasksPreview,
  ModelPreview,
  PreviewValue,
  TensorPreview,
  VideoPreview,
  VisionModelPreview,
} from "../types/graph";

function isDataFramePreview(value: PreviewValue): value is DataFramePreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "rows" in value &&
    "cols" in value &&
    "head" in value
  );
}

function isDatasetPreview(value: PreviewValue): value is DatasetPreview {
  return typeof value === "object" && value !== null && "X_shape" in value;
}

function isTensorPreview(value: PreviewValue): value is TensorPreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "shape" in value &&
    "dtype" in value &&
    "sample" in value
  );
}

function isModelPreview(value: PreviewValue): value is ModelPreview {
  return typeof value === "object" && value !== null && "class" in value;
}

function isImagePreview(value: PreviewValue): value is ImagePreview {
  return typeof value === "object" && value !== null && "kind" in value && value.kind === "image";
}

function isVideoPreview(value: PreviewValue): value is VideoPreview {
  return typeof value === "object" && value !== null && "kind" in value && value.kind === "video";
}

function isVisionModelPreview(value: PreviewValue): value is VisionModelPreview {
  return (
    typeof value === "object" && value !== null && "kind" in value && value.kind === "vision_model"
  );
}

function isDetectionsPreview(value: PreviewValue): value is DetectionsPreview {
  return typeof value === "object" && value !== null && "kind" in value && value.kind === "detections";
}

function isMasksPreview(value: PreviewValue): value is MasksPreview {
  return typeof value === "object" && value !== null && "kind" in value && value.kind === "masks";
}

function isClassificationsPreview(
  value: PreviewValue,
): value is ClassificationsPreview {
  return (
    typeof value === "object" && value !== null && "kind" in value && value.kind === "classifications"
  );
}

function ItemList({ items }: { items: unknown[] }) {
  return (
    <pre className="max-h-32 overflow-auto whitespace-pre-wrap text-[10px] text-zinc-400">
      {JSON.stringify(items, null, 2)}
    </pre>
  );
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

function DataFrameView({
  preview,
  compact,
}: {
  preview: DataFramePreview;
  compact: boolean;
}) {
  const columns = preview.head[0] ? Object.keys(preview.head[0]) : [];
  return (
    <div>
      <div className="mb-1 text-[10px] text-zinc-500">
        {preview.rows} rows x {preview.cols} cols — first {preview.head.length}
      </div>
      <div
        className={`overflow-auto rounded border border-zinc-800 ${
          compact ? "max-h-40" : ""
        }`}
      >
        <table className="w-full border-collapse text-[10px]">
          <thead className="sticky top-0">
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className="border-b border-zinc-800 bg-zinc-900 px-1.5 py-0.5 text-left font-medium text-zinc-400"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.head.map((row, index) => (
              <tr key={index} className="odd:bg-zinc-900/40">
                {columns.map((column) => (
                  <td
                    key={column}
                    className={`px-1.5 py-0.5 text-zinc-300 ${
                      compact ? "max-w-32 truncate" : ""
                    }`}
                    title={compact ? String(row[column]) : undefined}
                  >
                    {String(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function PreviewValueView({
  value,
  compact = false,
  fill = false,
}: {
  value: PreviewValue;
  compact?: boolean;
  fill?: boolean;
}) {
  if (isDataFramePreview(value)) {
    return <DataFrameView preview={value} compact={compact} />;
  }
  if (isDatasetPreview(value)) {
    return (
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-zinc-300">
        <span>X {JSON.stringify(value.X_shape)}</span>
        <span>y {JSON.stringify(value.y_shape)}</span>
        <span className="text-zinc-500">{value.dtype}</span>
      </div>
    );
  }
  if (isTensorPreview(value)) {
    return (
      <div className="text-[10px] text-zinc-300">
        <div className="flex flex-wrap gap-x-3">
          <span>shape {JSON.stringify(value.shape)}</span>
          <span className="text-zinc-500">{value.dtype}</span>
          <span className="text-zinc-500">{value.device}</span>
        </div>
        {value.sample.length > 0 && (
          <div className="mt-0.5 font-mono text-zinc-400">
            [{value.sample.map((item) => item.toFixed(4)).join(", ")}
            {value.sample.length >= 5 ? ", …" : ""}]
          </div>
        )}
      </div>
    );
  }
  if (isImagePreview(value)) {
    return (
      <div
        className={
          fill
            ? "flex h-full min-h-0 items-center gap-2 overflow-hidden"
            : `flex items-center gap-2 ${compact ? "max-h-40" : ""}`
        }
      >
        {value.thumbnail ? (
          <img
            src={value.thumbnail}
            alt="Node output"
            className={
              fill
                ? "h-full min-h-0 w-full flex-1 rounded border border-zinc-800 object-contain"
                : "max-h-40 max-w-full rounded border border-zinc-800 object-contain"
            }
          />
        ) : (
          <div className="rounded border border-dashed border-zinc-700 px-3 py-4 text-[10px] text-zinc-500">
            Image preview unavailable
          </div>
        )}
        <div className="shrink-0 text-[10px] text-zinc-400">
          <div>
            {value.width}x{value.height}
          </div>
          <div className="text-zinc-500">{value.mode}</div>
        </div>
      </div>
    );
  }
  if (isVideoPreview(value)) {
    return (
      <div
        className={
          fill
            ? "flex h-full min-h-0 flex-1 flex-col overflow-hidden text-[10px] text-zinc-300"
            : "text-[10px] text-zinc-300"
        }
      >
        {value.url ? (
          <video
            src={value.url}
            poster={value.poster ?? undefined}
            controls
            preload="metadata"
            className={
              fill
                ? "h-full min-h-0 w-full flex-1 rounded border border-zinc-800 object-contain"
                : "max-h-40 max-w-full rounded border border-zinc-800"
            }
          >
            Your browser does not support video playback.
          </video>
        ) : (
          <div className="rounded border border-dashed border-zinc-700 px-3 py-4 text-zinc-500">
            Video preview unavailable
          </div>
        )}
        <div className="mt-1 flex shrink-0 flex-wrap gap-x-3 text-zinc-500">
          {value.width !== null && value.height !== null && (
            <span>
              {value.width}x{value.height}
            </span>
          )}
          {value.fps !== null && <span>{formatNumber(value.fps)} fps</span>}
          {value.frames !== null && <span>{value.frames} frames</span>}
          {value.duration !== null && (
            <span>{formatNumber(value.duration)} s</span>
          )}
        </div>
        {value.path && <div className="truncate text-zinc-600">{value.path}</div>}
      </div>
    );
  }
  if (isVisionModelPreview(value)) {
    const isLocal = value.source === "local";
    return (
      <div className="text-[10px] text-zinc-300">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
          <span
            className={`rounded px-1 py-px text-[9px] font-medium ${
              isLocal
                ? "bg-emerald-950/60 text-emerald-300"
                : "bg-zinc-800 text-zinc-400"
            }`}
          >
            {isLocal ? "local" : "built-in"}
          </span>
          <span className="text-amber-300">{value.backend}</span>
          <span className="text-zinc-500">{value.task}</span>
          {value.format && (
            <span className="text-zinc-600">{value.format}</span>
          )}
        </div>
        <div className="truncate text-zinc-400">{value.model}</div>
        {isLocal && value.path ? (
          <div className="mt-0.5 truncate font-mono text-zinc-500" title={value.path}>
            {value.path}
          </div>
        ) : (
          <div className="text-zinc-600">
            weights {JSON.stringify(value.weights)} · device{" "}
            {JSON.stringify(value.device)}
          </div>
        )}
        {isLocal && value.fingerprint && (
          <div className="truncate font-mono text-[9px] text-zinc-700">
            fingerprint {value.fingerprint}
          </div>
        )}
      </div>
    );
  }
  if (isMasksPreview(value)) {
    return (
      <div className="flex items-center gap-2">
        {value.thumbnail ? (
          <img
            src={value.thumbnail}
            alt="Segmentation masks"
            className="max-h-32 max-w-full rounded border border-zinc-800 object-contain"
          />
        ) : (
          <span className="text-[10px] text-zinc-500">{value.count} masks</span>
        )}
        {value.thumbnail && (
          <span className="text-[10px] text-zinc-400">{value.count} masks</span>
        )}
      </div>
    );
  }
  if (isDetectionsPreview(value)) {
    return (
      <div>
        <div className="mb-1 text-[10px] text-zinc-400">
          {value.count} detections
        </div>
        {value.items.length > 0 && <ItemList items={value.items} />}
      </div>
    );
  }
  if (isClassificationsPreview(value)) {
    return (
      <div>
        <div className="mb-1 text-[10px] text-zinc-400">
          {value.items.length} classifications
        </div>
        {value.items.length > 0 && <ItemList items={value.items} />}
      </div>
    );
  }
  if (isModelPreview(value)) {
    const params =
      value.params !== undefined
        ? Object.entries(value.params)
            .slice(0, 6)
            .map(([key, paramValue]) => `${key}=${JSON.stringify(paramValue)}`)
            .join(", ")
        : value.parameters !== undefined
          ? `${value.parameters} parameters · ${value.layers} tensors`
          : "";
    return (
      <div className="text-[10px] text-zinc-300">
        <span className="text-amber-300">{value.class}</span>
        {params && <div className="mt-0.5 text-zinc-500">{params}</div>}
      </div>
    );
  }
  if (typeof value === "object" && value !== null) {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <dl className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
        {entries.map(([key, entryValue]) => (
          <div key={key} className="col-span-2 flex justify-between gap-2">
            <dt className="text-zinc-500">{key}</dt>
            <dd className="text-zinc-200">
              {typeof entryValue === "number"
                ? formatNumber(entryValue)
                : typeof entryValue === "string"
                  ? entryValue
                  : JSON.stringify(entryValue)}
            </dd>
          </div>
        ))}
      </dl>
    );
  }
  return <pre className="whitespace-pre-wrap text-[10px] text-zinc-300">{String(value)}</pre>;
}
