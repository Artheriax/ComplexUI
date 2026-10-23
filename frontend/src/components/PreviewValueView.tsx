import type {
  BatchPreview,
  ClassificationsPreview,
  ClipPreview,
  ConditioningPreview,
  DataFramePreview,
  DatasetPreview,
  DetectionsPreview,
  DiffusionModelPreview,
  GuiderPreview,
  ImageBatchPreview,
  ImagePreview,
  LatentPreview,
  MasksPreview,
  NoisePreview,
  SamplerPreview,
  SigmasPreview,
  ModelPreview,
  PreviewValue,
  TensorPreview,
  VaePreview,
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

function isBatchPreview(value: PreviewValue): value is BatchPreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "batch"
  );
}

function isImageBatchPreview(value: PreviewValue): value is ImageBatchPreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "image_batch"
  );
}

function isSigmasPreview(value: PreviewValue): value is SigmasPreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "sigmas"
  );
}

function isSamplerPreview(value: PreviewValue): value is SamplerPreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "sampler"
  );
}

function isNoisePreview(value: PreviewValue): value is NoisePreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "noise"
  );
}

function isGuiderPreview(value: PreviewValue): value is GuiderPreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "guider"
  );
}

function isVideoPreview(value: PreviewValue): value is VideoPreview {
  return typeof value === "object" && value !== null && "kind" in value && value.kind === "video";
}

function isDiffusionModelPreview(
  value: PreviewValue,
): value is DiffusionModelPreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "diffusion_model"
  );
}

function isClipPreview(value: PreviewValue): value is ClipPreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "clip"
  );
}

function isVaePreview(value: PreviewValue): value is VaePreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "vae"
  );
}

function isConditioningPreview(
  value: PreviewValue,
): value is ConditioningPreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "conditioning"
  );
}

function isLatentPreview(value: PreviewValue): value is LatentPreview {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "latent"
  );
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

function SigmasView({
  preview,
  compact,
}: {
  preview: SigmasPreview;
  compact: boolean;
}) {
  const values = preview.sigmas;
  if (values.length < 2) {
    return (
      <div className="text-[10px] text-zinc-500">Sigma schedule unavailable</div>
    );
  }
  const width = 240;
  const height = compact ? 44 : 72;
  const max = Math.max(...values, 1e-6);
  const min = Math.min(...values);
  const span = Math.max(max - min, 1e-6);
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = height - ((value - min) / span) * (height - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full rounded border border-zinc-800 bg-zinc-950/70"
        role="img"
        aria-label="Sigma schedule"
      >
        <polyline
          points={points}
          fill="none"
          stroke="#22d3ee"
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div className="mt-0.5 flex flex-wrap gap-x-2 text-[10px] text-zinc-500">
        <span>{preview.steps} steps</span>
        <span className="text-cyan-300">{preview.scheduler}</span>
        {preview.sigma_max !== null && preview.sigma_min !== null && (
          <span>
            {formatNumber(preview.sigma_max)} → {formatNumber(preview.sigma_min)}
          </span>
        )}
      </div>
    </div>
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
  if (isBatchPreview(value)) {
    return (
      <div className="rounded border border-dashed border-zinc-700 px-3 py-4 text-[10px] text-zinc-400">
        {value.count} items
      </div>
    );
  }
  if (isImageBatchPreview(value)) {
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
            alt="Image batch"
            className={
              fill
                ? "h-full min-h-0 w-full flex-1 rounded border border-zinc-800 object-contain"
                : "max-h-40 max-w-full rounded border border-zinc-800 object-contain"
            }
          />
        ) : (
          <div className="rounded border border-dashed border-zinc-700 px-3 py-4 text-[10px] text-zinc-500">
            Batch preview unavailable
          </div>
        )}
        <div className="shrink-0 text-[10px] text-zinc-400">
          <div>{value.count} images</div>
          <div className="text-zinc-500">
            {value.width ?? "?"}x{value.height ?? "?"}
          </div>
        </div>
      </div>
    );
  }
  if (isDiffusionModelPreview(value)) {
    return (
      <div className="text-[10px] text-zinc-300">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
          <span className="rounded bg-amber-950/60 px-1 py-px text-[9px] font-medium text-amber-300">
            {value.arch || "diffusion"}
          </span>
          {value.preset && (
            <span className="text-zinc-500">{value.preset}</span>
          )}
          {value.dtype && <span className="text-zinc-600">{value.dtype}</span>}
          {value.device && (
            <span className="text-zinc-600">{value.device}</span>
          )}
        </div>
        <div className="truncate text-zinc-400">{value.name}</div>
        <div className="mt-0.5 flex flex-wrap gap-x-3 text-zinc-500">
          <span>{value.class}</span>
          {value.parameters !== null && (
            <span>{value.parameters.toLocaleString()} params</span>
          )}
          {value.source !== "local" && <span>{value.source}</span>}
        </div>
        {value.recommended && Object.keys(value.recommended).length > 0 && (
          <div className="mt-0.5 text-zinc-600">
            {Object.entries(value.recommended)
              .map(([key, entry]) => `${key}=${String(entry)}`)
              .join(" · ")}
          </div>
        )}
      </div>
    );
  }
  if (isClipPreview(value)) {
    return (
      <div className="text-[10px] text-zinc-300">
        <div className="flex flex-wrap gap-x-2">
          <span className="text-sky-300">CLIP {value.arch}</span>
          <span className="text-zinc-500">{value.encoders} encoder(s)</span>
          {value.dtype && <span className="text-zinc-600">{value.dtype}</span>}
          {value.device && <span className="text-zinc-600">{value.device}</span>}
        </div>
      </div>
    );
  }
  if (isVaePreview(value)) {
    return (
      <div className="text-[10px] text-zinc-300">
        <div className="flex flex-wrap gap-x-2">
          <span className="text-teal-300">VAE {value.arch}</span>
          {value.scaling_factor !== null && (
            <span className="text-zinc-500">
              scale {formatNumber(value.scaling_factor)}
            </span>
          )}
          {value.dtype && <span className="text-zinc-600">{value.dtype}</span>}
          {value.device && <span className="text-zinc-600">{value.device}</span>}
        </div>
      </div>
    );
  }
  if (isConditioningPreview(value)) {
    return (
      <div className="text-[10px] text-zinc-300">
        <div className="text-violet-300">
          {value.arch} conditioning {JSON.stringify(value.shape)}
        </div>
        {value.pooled_shape && (
          <div className="text-zinc-500">
            pooled {JSON.stringify(value.pooled_shape)}
          </div>
        )}
        {value.text && <div className="truncate text-zinc-500">{value.text}</div>}
      </div>
    );
  }
  if (isLatentPreview(value)) {
    return (
      <div className="text-[10px] text-zinc-300">
        <div className="text-fuchsia-300">
          latent {JSON.stringify(value.shape)}
        </div>
        <div className="flex flex-wrap gap-x-3 text-zinc-500">
          {value.width !== null && value.height !== null && (
            <span>
              {value.width}x{value.height}
            </span>
          )}
          {value.batch_size !== null && <span>batch {value.batch_size}</span>}
          {value.dtype && <span>{value.dtype}</span>}
        </div>
      </div>
    );
  }
  if (isSigmasPreview(value)) {
    return <SigmasView preview={value} compact={compact} />;
  }
  if (isSamplerPreview(value)) {
    return (
      <div className="text-[10px] text-zinc-300">
        <span className="text-amber-300">{value.name || "custom"}</span>
        {value.custom && <span className="ml-2 text-zinc-500">custom contract</span>}
      </div>
    );
  }
  if (isNoisePreview(value)) {
    return (
      <div className="text-[10px] text-zinc-300">
        <div className="text-lime-300">{value.noise_type}</div>
        <div className="flex flex-wrap gap-x-2 text-zinc-500">
          {value.noise_seed !== null && (
            <span>noise seed {value.noise_seed}</span>
          )}
          {value.noise_type === "fractal" && (
            <>
              <span>alpha {formatNumber(value.alpha)}</span>
              <span>k {formatNumber(value.k)}</span>
              <span>scale {formatNumber(value.scale)}</span>
            </>
          )}
        </div>
      </div>
    );
  }
  if (isGuiderPreview(value)) {
    return (
      <div className="text-[10px] text-zinc-300">
        <div className="flex flex-wrap gap-x-2">
          <span className="text-rose-300">CFG {String(value.cfg ?? "?")}</span>
          {value.arch && <span className="text-zinc-500">{value.arch}</span>}
        </div>
        {value.model && <div className="truncate text-zinc-400">{value.model}</div>}
        {value.positive_text && (
          <div className="truncate text-zinc-500">+ {value.positive_text}</div>
        )}
        {value.negative_text && (
          <div className="truncate text-zinc-600">− {value.negative_text}</div>
        )}
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
