import DOMPurify from "dompurify";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Config, Data, Frame, Layout } from "plotly.js-dist-min";

import type {
  ImageRichOutput,
  PlotlyRichOutput,
  RichOutput,
  TableRichOutput,
  TextRichOutput,
} from "../types/graph";

interface RichOutputViewProps {
  output: RichOutput;
  interactive?: boolean;
  compact?: boolean;
}

function TruncationNote({ children }: { children: string }) {
  return <div className="mt-1 text-[10px] italic text-amber-400/80">{children}</div>;
}

function TextOutputView({
  output,
  compact,
}: {
  output: TextRichOutput;
  compact: boolean;
}) {
  return (
    <div>
      <pre
        className={`overflow-auto whitespace-pre-wrap break-words rounded bg-zinc-950/70 p-1.5 font-mono text-[10px] leading-relaxed text-zinc-200 ${
          compact ? "max-h-36" : "max-h-72"
        }`}
      >
        {output.text}
      </pre>
      {output.truncated && <TruncationNote>Text output truncated.</TruncationNote>}
    </div>
  );
}

function cellText(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

function TableOutputView({
  output,
  compact,
}: {
  output: TableRichOutput;
  compact: boolean;
}) {
  return (
    <div>
      <div className="mb-1 text-[10px] text-zinc-500">
        {output.total_rows} rows x {output.total_cols} cols — first {output.rows.length}
      </div>
      <div
        className={`overflow-auto rounded border border-zinc-800 ${
          compact ? "max-h-40" : "max-h-80"
        }`}
      >
        <table className="w-full border-collapse text-[10px]">
          <thead className="sticky top-0">
            <tr>
              {output.columns.map((column, index) => (
                <th
                  key={`${column}-${index}`}
                  className="border-b border-zinc-800 bg-zinc-900 px-1.5 py-0.5 text-left font-medium text-zinc-400"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {output.rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="odd:bg-zinc-900/40">
                {output.columns.map((_, columnIndex) => (
                  <td
                    key={columnIndex}
                    className="max-w-48 truncate px-1.5 py-0.5 text-zinc-300"
                    title={cellText(row[columnIndex])}
                  >
                    {cellText(row[columnIndex])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {output.truncated && <TruncationNote>Table preview truncated.</TruncationNote>}
    </div>
  );
}

function ImageOutputView({
  output,
  compact,
}: {
  output: ImageRichOutput;
  compact: boolean;
}) {
  return (
    <img
      src={`data:${output.mime};base64,${output.data}`}
      alt={output.alt}
      className={`mx-auto max-w-full rounded bg-white/95 object-contain ${
        compact ? "max-h-48" : "max-h-[28rem]"
      }`}
    />
  );
}

function HtmlOutputView({ html }: { html: string }) {
  const sanitized = useMemo(
    () => DOMPurify.sanitize(html, { USE_PROFILES: { html: true } }),
    [html],
  );
  return (
    <div
      className="rich-html max-h-80 overflow-auto rounded bg-zinc-950/60 p-2"
      dangerouslySetInnerHTML={{ __html: sanitized }}
    />
  );
}

function PlotlyOutputView({
  output,
  interactive,
  compact,
}: {
  output: PlotlyRichOutput;
  interactive: boolean;
  compact: boolean;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    let disposed = false;
    let plotly: typeof import("plotly.js-dist-min").default | null = null;
    const container = containerRef.current;

    void (async () => {
      try {
        const module = await import("plotly.js-dist-min");
        plotly = module.default;
        if (disposed || !container) {
          return;
        }
        const layout: Partial<Layout> = {
          autosize: true,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#d4d4d8", size: 10 },
          margin: { l: 44, r: 20, t: 36, b: 40 },
          ...(output.figure.layout as Partial<Layout> | undefined),
        };
        const config: Partial<Config> = {
          responsive: true,
          displayModeBar: interactive,
          staticPlot: !interactive,
        };
        await plotly.newPlot(
          container,
          (output.figure.data ?? []) as Data[],
          layout,
          config,
        );
        const frames = (output.figure.frames ?? []) as Frame[];
        if (frames.length > 0) {
          plotly.addFrames(container, frames);
        }
        if (!disposed) {
          setState("ready");
        }
      } catch {
        if (!disposed) {
          setState("error");
        }
      }
    })();

    return () => {
      disposed = true;
      if (plotly && container) {
        plotly.purge(container);
      }
    };
  }, [output.figure, interactive]);

  return (
    <div className="relative">
      <div
        ref={containerRef}
        role={interactive ? "figure" : "img"}
        style={{ width: "100%", height: compact ? 180 : 320 }}
        aria-label="Plotly figure"
      />
      {state === "loading" && (
        <div
          role="status"
          className="absolute inset-0 flex items-center justify-center text-[10px] text-zinc-500"
        >
          Loading figure…
        </div>
      )}
      {state === "error" && (
        <div
          role="alert"
          className="absolute inset-0 flex items-center justify-center text-[10px] text-red-300"
        >
          Unable to render Plotly figure.
        </div>
      )}
    </div>
  );
}

export function RichOutputView({
  output,
  interactive = false,
  compact = false,
}: RichOutputViewProps) {
  switch (output.kind) {
    case "text":
      return <TextOutputView output={output} compact={compact} />;
    case "table":
      return <TableOutputView output={output} compact={compact} />;
    case "image":
      return <ImageOutputView output={output} compact={compact} />;
    case "html":
      return (
        <div>
          <HtmlOutputView html={output.html} />
          {output.truncated && <TruncationNote>HTML output truncated.</TruncationNote>}
        </div>
      );
    case "plotly":
      return (
        <PlotlyOutputView
          output={output}
          interactive={interactive}
          compact={compact}
        />
      );
  }
}
