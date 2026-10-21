import { useState } from "react";

import { DatasetLibrary } from "./DatasetLibrary";
import { ModelsLibrary } from "./ModelsLibrary";

type AssetTab = "files" | "models";

export function AssetLibrary() {
  const [tab, setTab] = useState<AssetTab>("files");
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        role="tablist"
        aria-label="Asset library"
        className="flex border-b border-zinc-800 bg-zinc-950/40"
      >
        {(
          [
            ["files", "Files"],
            ["models", "Models"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
            className={`flex-1 border-b-2 px-2 py-1.5 text-[11px] font-medium ${
              tab === id
                ? "border-cyan-500 text-cyan-200"
                : "border-transparent text-zinc-500 hover:bg-zinc-900 hover:text-zinc-300"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="flex min-h-0 flex-1 flex-col">
        {tab === "files" ? <DatasetLibrary /> : <ModelsLibrary />}
      </div>
    </div>
  );
}
