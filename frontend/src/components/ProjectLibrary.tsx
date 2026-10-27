import { useMemo, useState } from "react";

import type { ProjectSummary } from "../lib/api";
import { useGraphStore } from "../store/useGraphStore";
import { ProjectHistoryDialog } from "./ProjectHistoryDialog";
import { SpinnerIcon, TrashIcon, WarningIcon } from "./icons";

function formatDate(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function ProjectCard({
  project,
  active,
  onHistory,
}: {
  project: ProjectSummary;
  active: boolean;
  onHistory: (project: ProjectSummary) => void;
}) {
  const openProjectInWindow = useGraphStore(
    (state) => state.openProjectInWindow,
  );
  const deleteProject = useGraphStore((state) => state.deleteProject);

  const open = () => {
    void openProjectInWindow(project.slug);
  };

  const remove = () => {
    if (
      !window.confirm(
        `Delete “${project.name}”? It and its version snapshots move to the workspace trash.`,
      )
    ) {
      return;
    }
    void deleteProject(project.slug);
  };

  return (
    <div
      className={`rounded-lg border p-2.5 transition-colors ${
        active
          ? "border-cyan-700 bg-cyan-950/20"
          : "border-zinc-800 bg-zinc-900 hover:border-zinc-700"
      }`}
    >
      <button
        type="button"
        onClick={open}
        className="block w-full text-left"
      >
        <div className="flex items-center justify-between gap-1">
          <span className="truncate text-xs font-semibold text-zinc-100">
            {project.name}
          </span>
          <span className="shrink-0 rounded bg-zinc-800/80 px-1 py-px text-[9px] text-zinc-500">
            {project.node_count} nodes
          </span>
        </div>
        <p className="mt-0.5 truncate text-[10px] text-zinc-500">
          Saved {formatDate(project.saved_at)}
        </p>
      </button>
      <div className="mt-2 flex items-center justify-between gap-2">
        <span className="text-[10px] text-zinc-600">
          {project.version_count} version
          {project.version_count === 1 ? "" : "s"}
        </span>
        <div className="flex items-center gap-1">
          {project.version_count > 0 && (
            <button
              type="button"
              onClick={() => onHistory(project)}
              className="rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              History
            </button>
          )}
          <button
            type="button"
            title={`Delete ${project.name}`}
            aria-label={`Delete ${project.name}`}
            onClick={remove}
            className="rounded border border-zinc-700 p-1 text-zinc-500 hover:border-red-800 hover:bg-red-950/40 hover:text-red-300"
          >
            <TrashIcon className="h-3 w-3" />
          </button>
        </div>
      </div>
    </div>
  );
}

export function ProjectLibrary() {
  const projects = useGraphStore((state) => state.projects);
  const projectsLoaded = useGraphStore((state) => state.projectsLoaded);
  const projectsError = useGraphStore((state) => state.projectsError);
  const currentSlug = useGraphStore((state) => state.projectSlug);
  const loadProjects = useGraphStore((state) => state.loadProjects);
  const [query, setQuery] = useState("");
  const [historyProject, setHistoryProject] =
    useState<ProjectSummary | null>(null);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return projects;
    }
    return projects.filter((project) =>
      `${project.name} ${project.slug}`.toLowerCase().includes(needle),
    );
  }, [projects, query]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="space-y-1.5 border-b border-zinc-800 p-2.5">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search workflows…"
          aria-label="Search workflows"
          className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-cyan-600"
        />
        <button
          type="button"
          onClick={() => {
            void loadProjects();
          }}
          className="w-full rounded-md border border-zinc-800 px-2 py-1 text-[10px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
        >
          Refresh workflow list
        </button>
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto p-2.5">
        {!projectsLoaded && !projectsError && (
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <SpinnerIcon className="h-3.5 w-3.5 animate-spin" /> Loading workflows…
          </div>
        )}
        {projectsError && (
          <div className="flex items-start gap-2 rounded border border-red-900/60 bg-red-950/40 p-2 text-[11px] text-red-300">
            <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{projectsError}</span>
          </div>
        )}
        {projectsLoaded && !projectsError && projects.length === 0 && (
          <p className="px-1 py-2 text-[11px] leading-relaxed text-zinc-600">
            No saved workflows yet. Build one, then press Ctrl+S or use File → Save.
          </p>
        )}
        {projectsLoaded && !projectsError && projects.length > 0 && visible.length === 0 && (
          <p className="px-1 py-2 text-[11px] text-zinc-600">
            No workflows match “{query.trim()}”.
          </p>
        )}
        {visible.map((project) => (
          <ProjectCard
            key={project.slug}
            project={project}
            active={project.slug === currentSlug}
            onHistory={setHistoryProject}
          />
        ))}
      </div>
      <div className="border-t border-zinc-800 p-2.5 text-[10px] leading-relaxed text-zinc-600">
        Saved workflows live in the local workspace. Overwriting keeps 30 version snapshots.
      </div>
      {historyProject && (
        <ProjectHistoryDialog
          project={historyProject}
          onClose={() => setHistoryProject(null)}
        />
      )}
    </div>
  );
}

