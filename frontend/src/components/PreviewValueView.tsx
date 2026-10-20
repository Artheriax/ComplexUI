import type {
  DataFramePreview,
  DatasetPreview,
  ModelPreview,
  PreviewValue,
  TensorPreview,
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
}: {
  value: PreviewValue;
  compact?: boolean;
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
