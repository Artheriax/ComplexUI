export const SAVE_FILENAME_TOKENS = [
  { token: "date", label: "Date" },
  { token: "time", label: "Time" },
  { token: "counter", label: "Counter" },
  { token: "model", label: "Model" },
  { token: "width", label: "Width" },
  { token: "height", label: "Height" },
  { token: "batch", label: "Batch #" },
] as const;

export const SAVE_DISABLED_TOKENS = [
  { token: "seed", label: "Seed" },
  { token: "input_name", label: "Input name" },
  { token: "input_folder", label: "Input folder" },
] as const;

const TOKEN_PATTERN = /%([a-z_]+)%/gi;
const INVALID_SEGMENT_CHARS = new Set('<>:"/\\|?*');
const MEDIA_SUFFIXES = [
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".bmp",
  ".gif",
  ".tif",
  ".tiff",
  ".mp4",
  ".webm",
  ".mov",
  ".avi",
  ".mkv",
  ".m4v",
];
const FORMAT_SUFFIXES: Record<string, string> = {
  png: ".png",
  jpg: ".jpg",
  webp: ".webp",
  bmp: ".bmp",
  tiff: ".tiff",
  gif: ".gif",
};
const DISABLED: Set<string> = new Set(
  SAVE_DISABLED_TOKENS.map((item) => item.token),
);

export interface SavePreviewOptions {
  folder: string;
  filename: string;
  format: string;
  modelName: string;
  width?: number;
  height?: number;
  batchIndex?: number;
  batchCount?: number;
  counter?: number;
  videoSuffix?: string;
  now?: Date;
}

export interface SavePreview {
  path: string;
  warnings: string[];
}

export function sanitizeOutputSegment(value: string): string {
  let cleaned = "";
  for (const character of value) {
    const code = character.codePointAt(0) ?? 0;
    cleaned +=
      code < 32 || INVALID_SEGMENT_CHARS.has(character) ? "_" : character;
  }
  return cleaned
    .trim()
    .replace(/^[ .]+|[ .]+$/g, "")
    .slice(0, 120);
}

export function sanitizeModelName(value: string): string {
  return value
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^[._-]+|[._-]+$/g, "")
    .slice(0, 60);
}

function pad(value: number): string {
  return String(Math.max(0, Math.trunc(value))).padStart(3, "0");
}

function localDate(now: Date): string {
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function localTime(now: Date): string {
  const hours = String(now.getHours()).padStart(2, "0");
  const minutes = String(now.getMinutes()).padStart(2, "0");
  const seconds = String(now.getSeconds()).padStart(2, "0");
  return `${hours}-${minutes}-${seconds}`;
}

function isAbsoluteFolder(value: string): boolean {
  return /^[A-Za-z]:[\\/]/.test(value) || /^\\\\/.test(value) || /^\//.test(value);
}

function stripMediaSuffix(value: string): string {
  const lower = value.toLowerCase();
  const suffix = MEDIA_SUFFIXES.find((candidate) => lower.endsWith(candidate));
  return suffix ? value.slice(0, -suffix.length) : value;
}

function expandTokens(
  template: string,
  options: {
    model: string;
    now: Date;
    width?: number;
    height?: number;
    counter: number;
    batch: number;
    allowCounter: boolean;
    warnings: Set<string>;
  },
): string {
  return template.replace(TOKEN_PATTERN, (match, rawToken: string) => {
    const token = rawToken.toLowerCase();
    if (DISABLED.has(token)) {
      options.warnings.add(`%${token}% requires metadata support.`);
      return match;
    }
    if (token === "counter") {
      if (!options.allowCounter) {
        options.warnings.add("%counter% can only be used in the filename.");
        return match;
      }
      return pad(options.counter);
    }
    if (token === "batch" || token === "index") {
      if (!options.allowCounter) {
        options.warnings.add("%batch% can only be used in the filename.");
        return match;
      }
      return pad(options.batch);
    }
    if (token === "date") {
      return localDate(options.now);
    }
    if (token === "time") {
      return localTime(options.now);
    }
    if (token === "model") {
      return options.model;
    }
    if (token === "width" || token === "height") {
      const value = token === "width" ? options.width : options.height;
      if (value === undefined) {
        options.warnings.add(
          `%${token}% is filled in once the image is known.`,
        );
        return match;
      }
      return String(Math.trunc(value));
    }
    options.warnings.add(`Unknown token %${token}%.`);
    return match;
  });
}

export function previewSavePath(options: SavePreviewOptions): SavePreview {
  const now = options.now ?? new Date();
  const model = sanitizeModelName(options.modelName) || "image";
  const warnings = new Set<string>();
  const counter = options.counter ?? 1;
  const batch = options.batchIndex ?? 0;
  const folderTemplate = options.folder.trim();
  const expandedFolder = expandTokens(folderTemplate, {
    model,
    now,
    width: options.width,
    height: options.height,
    counter,
    batch,
    allowCounter: false,
    warnings,
  }).trim();
  const folder = expandedFolder || "outputs";
  const absolute = isAbsoluteFolder(folder);
  const base = absolute
    ? folder.replace(/[\\/]+$/, "")
    : expandedFolder
      ? `outputs/${expandedFolder.replace(/^[\\/]+|[\\/]+$/g, "")}`.replace(
          /\/+$/,
          "",
        )
      : "outputs";

  const filenameTemplate = options.filename.trim();
  if (!filenameTemplate) {
    warnings.add("Filename template is required.");
  }
  const hasCounter = /%counter%/i.test(filenameTemplate);
  if (!hasCounter) {
    warnings.add("Filename template must include %counter%.");
  }
  const hasBatch = /%(batch|index)%/i.test(filenameTemplate);
  const batchCount = options.batchCount ?? 1;
  let effectiveFilename = filenameTemplate;
  if (batchCount > 1 && !hasBatch && effectiveFilename) {
    const parts = effectiveFilename.split("/");
    const lastIndex = parts.length - 1;
    parts[lastIndex] = `${stripMediaSuffix(parts[lastIndex])}_%batch%`;
    effectiveFilename = parts.join("/");
  }

  const expanded = expandTokens(effectiveFilename, {
    model,
    now,
    width: options.width,
    height: options.height,
    counter,
    batch,
    allowCounter: true,
    warnings,
  });
  const segments: string[] = [];
  for (const rawSegment of expanded.split("/")) {
    if (rawSegment === "" || rawSegment === ".") {
      continue;
    }
    if (rawSegment === "..") {
      warnings.add("Filename segments may not be '..'.");
      continue;
    }
    const segment = sanitizeOutputSegment(rawSegment);
    if (segment) {
      segments.push(segment);
    }
  }
  if (segments.length === 0) {
    return { path: base, warnings: [...warnings] };
  }
  const extension =
    options.videoSuffix ??
    FORMAT_SUFFIXES[options.format.toLowerCase()] ??
    ".png";
  segments[segments.length - 1] = `${stripMediaSuffix(
    segments[segments.length - 1],
  )}${extension}`;
  const windows = /^[A-Za-z]:[\\/]|^\\\\/.test(base);
  const separator = windows ? "\\" : "/";
  const normalizedBase = windows ? base.replace(/\//g, "\\") : base;
  return {
    path: `${normalizedBase}${separator}${segments.join(separator)}`,
    warnings: [...warnings],
  };
}
