import { useRef, useState } from "react";
import type { ReactNode } from "react";

import { pickFolder } from "../lib/api";
import {
  previewSavePath,
  SAVE_DISABLED_TOKENS,
  SAVE_FILENAME_TOKENS,
} from "../lib/outputNames";
import type { PreviewValue, WidgetParam } from "../types/graph";
import { WidgetField } from "./WidgetField";
import { SettingsIcon, WarningIcon } from "./icons";

const PRIMARY_FORMATS = ["png", "jpg", "webp"];

interface SaveMediaFieldsProps {
  widgets: WidgetParam[];
  values: Record<string, unknown>;
  connected: Set<string>;
  onChange: (name: string, value: unknown) => void;
  renderHandle: (
    name: string,
    widget: WidgetParam,
    connected: boolean,
  ) => ReactNode;
  upstreamPreview: PreviewValue | null;
}

function isAbsoluteFolder(value: string): boolean {
  return (
    /^[A-Za-z]:[\\/]/.test(value) ||
    /^\\\\/.test(value) ||
    /^\//.test(value)
  );
}

function mediaFacts(preview: PreviewValue | null): {
  width?: number;
  height?: number;
  batchCount?: number;
  videoSuffix?: string;
} {
  if (preview === null || typeof preview !== "object") {
    return {};
  }
  const record = preview as Record<string, unknown>;
  const kind = record.kind;
  if (kind === "image") {
    return {
      width: typeof record.width === "number" ? record.width : undefined,
      height: typeof record.height === "number" ? record.height : undefined,
    };
  }
  if (kind === "image_batch") {
    return {
      width: typeof record.width === "number" ? record.width : undefined,
      height: typeof record.height === "number" ? record.height : undefined,
      batchCount:
        typeof record.count === "number" ? record.count : undefined,
    };
  }
  if (kind === "video") {
    const path = typeof record.path === "string" ? record.path : "";
    const match = /\.([A-Za-z0-9]{2,5})$/.exec(path);
    return {
      width: typeof record.width === "number" ? record.width : undefined,
      height: typeof record.height === "number" ? record.height : undefined,
      videoSuffix: match ? `.${match[1].toLowerCase()}` : ".mp4",
    };
  }
  return {};
}

function tokenValue(value: unknown): string {
  return value === undefined || value === null ? "" : String(value);
}

export function SaveMediaFields({
  widgets,
  values,
  connected,
  onChange,
  renderHandle,
  upstreamPreview,
}: SaveMediaFieldsProps) {
  const byName = new Map(widgets.map((widget) => [widget.name, widget]));
  const folderWidget = byName.get("folder");
  const filenameWidget = byName.get("filename");
  const formatWidget = byName.get("format");
  const modelWidget = byName.get("model_name");
  const qualityWidget = byName.get("quality");
  const compressWidget = byName.get("compress_level");
  const optimizeWidget = byName.get("optimize");
  const filenameRef = useRef<HTMLInputElement>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [browseState, setBrowseState] = useState<
    "idle" | "waiting" | "error"
  >("idle");
  const [browseError, setBrowseError] = useState<string | null>(null);

  if (
    folderWidget === undefined ||
    filenameWidget === undefined ||
    formatWidget === undefined ||
    modelWidget === undefined ||
    qualityWidget === undefined ||
    compressWidget === undefined ||
    optimizeWidget === undefined
  ) {
    return null;
  }

  const folder = tokenValue(values.folder);
  const filename = tokenValue(values.filename);
  const format = tokenValue(values.format) || "png";
  const modelName = tokenValue(values.model_name) || "image";
  const facts = mediaFacts(upstreamPreview);
  const preview = previewSavePath({
    folder,
    filename,
    format,
    modelName,
    width: facts.width,
    height: facts.height,
    batchCount: facts.batchCount,
    videoSuffix: facts.videoSuffix,
  });
  const formats = formatWidget.options ?? ["png", "jpg", "webp"];
  const primaryFormats = PRIMARY_FORMATS.filter((candidate) =>
    formats.includes(candidate),
  );
  const extraFormats = formats.filter(
    (candidate) => !PRIMARY_FORMATS.includes(String(candidate)),
  );

  const insertFilenameToken = (token: string) => {
    const input = filenameRef.current;
    const start = input?.selectionStart ?? filename.length;
    const end = input?.selectionEnd ?? start;
    const insertion = `%${token}%`;
    const next = `${filename.slice(0, start)}${insertion}${filename.slice(end)}`;
    onChange("filename", next);
    const cursor = start + insertion.length;
    window.requestAnimationFrame(() => {
      input?.focus();
      input?.setSelectionRange(cursor, cursor);
    });
  };

  const insertDateFolder = () => {
    const separator = folder && !/[\\/]$/.test(folder) ? "/" : "";
    onChange("folder", `${folder}${separator}%date%`);
  };

  const browse = async () => {
    setBrowseState("waiting");
    setBrowseError(null);
    try {
      const initial = isAbsoluteFolder(folder) && !folder.includes("%")
        ? folder
        : undefined;
      const selected = await pickFolder(initial);
      if (selected !== null) {
        onChange("folder", selected);
      }
      setBrowseState("idle");
    } catch (cause) {
      setBrowseState("error");
      setBrowseError(cause instanceof Error ? cause.message : String(cause));
    }
  };

  const reset = () => {
    onChange("folder", folderWidget.default);
    onChange("filename", filenameWidget.default);
    onChange("format", formatWidget.default);
    onChange("quality", qualityWidget.default);
    onChange("compress_level", compressWidget.default);
    onChange("optimize", optimizeWidget.default);
  };

  return (
    <div className="space-y-2 px-2 py-1">
      <div className="relative pl-3">
        {renderHandle("folder", folderWidget, connected.has("folder"))}
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-zinc-400">Folder</span>
          <div className="flex min-w-0 flex-1 items-center gap-1">
            <input
              type="text"
              aria-label="Save folder"
              className="nodrag min-w-0 flex-1 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono text-[10px] text-zinc-100 outline-none focus:border-cyan-500 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-500"
              value={folder}
              disabled={connected.has("folder")}
              placeholder="%date%"
              onChange={(event) => onChange("folder", event.target.value)}
            />
            <button
              type="button"
              className="nodrag shrink-0 rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-300 hover:bg-zinc-800 disabled:cursor-wait disabled:text-zinc-600"
              disabled={browseState === "waiting"}
              title="Browse for a folder"
              onClick={() => void browse()}
            >
              {browseState === "waiting" ? "…" : "Browse"}
            </button>
          </div>
        </div>
        <p className="mt-0.5 text-[9px] leading-relaxed text-zinc-600">
          Empty uses <span className="font-mono">outputs</span>. Relative paths
          stay under outputs; Browse can pick any folder.
        </p>
        {browseState === "waiting" && (
          <p role="status" className="mt-0.5 text-[9px] text-cyan-400">
            Waiting for the folder dialog…
          </p>
        )}
        {browseState === "error" && browseError !== null && (
          <p role="alert" className="mt-0.5 flex items-start gap-1 text-[9px] text-red-400">
            <WarningIcon className="mt-px h-3 w-3 shrink-0" />
            <span>{browseError}</span>
          </p>
        )}
      </div>

      <div className="relative pl-3">
        {renderHandle("filename", filenameWidget, connected.has("filename"))}
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-zinc-400">Filename</span>
          <input
            ref={filenameRef}
            type="text"
            aria-label="Filename template"
            className="nodrag min-w-0 flex-1 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 font-mono text-[10px] text-zinc-100 outline-none focus:border-cyan-500 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-zinc-900 disabled:text-zinc-500"
            value={filename}
            disabled={connected.has("filename")}
            placeholder="%model%_%counter%"
            onChange={(event) => onChange("filename", event.target.value)}
          />
        </div>
        <p className="mt-0.5 text-[9px] text-zinc-600">
          Use <span className="font-mono">/</span> in the filename to create
          subfolders.
        </p>
        <div className="mt-1 flex flex-wrap gap-1">
          {SAVE_FILENAME_TOKENS.map((item) => (
            <button
              key={item.token}
              type="button"
              className="nodrag rounded border border-zinc-700 px-1 py-0.5 text-[9px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
              disabled={connected.has("filename")}
              onClick={() => insertFilenameToken(item.token)}
            >
              + {item.label}
            </button>
          ))}
          {SAVE_DISABLED_TOKENS.map((item) => (
            <button
              key={item.token}
              type="button"
              className="nodrag cursor-not-allowed rounded border border-zinc-800 px-1 py-0.5 text-[9px] text-zinc-700"
              disabled
              title="Requires metadata support"
            >
              + {item.label}
            </button>
          ))}
          <button
            type="button"
            className="nodrag rounded border border-zinc-700 px-1 py-0.5 text-[9px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
            disabled={connected.has("folder")}
            onClick={insertDateFolder}
          >
            + Date folder
          </button>
          <button
            type="button"
            className="nodrag rounded border border-zinc-700 px-1 py-0.5 text-[9px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
            disabled={connected.has("filename")}
            onClick={() => onChange("filename", "")}
          >
            × Clear
          </button>
          <button
            type="button"
            className="nodrag rounded border border-zinc-700 px-1 py-0.5 text-[9px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
            onClick={reset}
          >
            Reset
          </button>
        </div>
      </div>

      <div className="rounded border border-zinc-800 bg-zinc-950/60 px-2 py-1">
        <p className="text-[9px] font-semibold uppercase tracking-wider text-zinc-500">
          Will save as
        </p>
        <p
          className="mt-0.5 line-clamp-2 break-all font-mono text-[10px] text-zinc-300"
          title={preview.path}
        >
          {preview.path || "—"}
        </p>
        {preview.warnings.length > 0 && (
          <div className="mt-1 space-y-0.5">
            {preview.warnings.map((warning) => (
              <p
                key={warning}
                className="flex items-start gap-1 text-[9px] text-amber-400"
              >
                <WarningIcon className="mt-px h-3 w-3 shrink-0" />
                <span>{warning}</span>
              </p>
            ))}
          </div>
        )}
      </div>

      <div className="relative pl-3">
        {renderHandle("format", formatWidget, connected.has("format"))}
        <div
          className="flex flex-wrap items-center gap-1"
          role="group"
          aria-label="Image format"
        >
          {primaryFormats.map((candidate) => (
            <button
              key={candidate}
              type="button"
              aria-pressed={format === candidate}
              aria-label={`Use ${candidate.toUpperCase()} format`}
              className={`nodrag rounded border px-1.5 py-0.5 text-[10px] ${
                format === candidate
                  ? "border-cyan-700 bg-cyan-950/60 text-cyan-200"
                  : "border-zinc-700 text-zinc-400 hover:bg-zinc-800"
              }`}
              disabled={connected.has("format")}
              onClick={() => onChange("format", candidate)}
            >
              {candidate.toUpperCase()}
            </button>
          ))}
          {extraFormats.length > 0 && (
            <select
              aria-label="More image formats"
              className="nodrag rounded border border-zinc-700 bg-zinc-950 px-1 py-0.5 text-[10px] text-zinc-300 outline-none focus:border-cyan-500"
              value={extraFormats.includes(format) ? format : ""}
              disabled={connected.has("format")}
              onChange={(event) => {
                if (event.target.value) {
                  onChange("format", event.target.value);
                }
              }}
            >
              <option value="">More</option>
              {extraFormats.map((candidate) => (
                <option key={String(candidate)} value={String(candidate)}>
                  {String(candidate).toUpperCase()}
                </option>
              ))}
            </select>
          )}
          <button
            type="button"
            className="nodrag ml-auto rounded border border-zinc-700 px-1 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
            aria-expanded={advancedOpen}
            aria-controls="save-media-advanced"
            aria-label="Save Media settings"
            title="Save Media settings"
            onClick={() => setAdvancedOpen((current) => !current)}
          >
            <SettingsIcon className="h-3 w-3" />
          </button>
        </div>
        {facts.videoSuffix !== undefined && (
          <p className="mt-0.5 text-[9px] text-zinc-600">
            Videos keep their source extension and ignore image format settings.
          </p>
        )}
      </div>

      {advancedOpen && (
        <div
          id="save-media-advanced"
          className="space-y-1 rounded border border-zinc-800 bg-zinc-950/60 px-2 py-1"
        >
          <p className="text-[9px] text-zinc-600">
            Model label fills %model%; metadata tokens arrive later.
          </p>
          {[modelWidget, qualityWidget, compressWidget, optimizeWidget].map(
            (widget) => (
              <div key={widget.name} className="relative pl-3">
                {renderHandle(widget.name, widget, connected.has(widget.name))}
                <WidgetField
                  widget={widget}
                  value={values[widget.name]}
                  connected={connected.has(widget.name)}
                  onChange={(value) => onChange(widget.name, value)}
                />
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}
