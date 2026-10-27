// @vitest-environment jsdom
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { DatasetPreview, DatasetSummary } from "../lib/api";
import { fetchDatasetPreview } from "../lib/api";
import { DatasetPreviewDialog } from "./DatasetPreviewDialog";

vi.mock("../lib/api", () => ({
  datasetRawUrl: (id: string) => `/api/uploads/${id}/raw`,
  fetchDatasetPreview: vi.fn(),
}));

const mockedFetch = vi.mocked(fetchDatasetPreview);

function dataset(id: string, name: string): DatasetSummary {
  return {
    id,
    name,
    path: `inputs/${name}`,
    format: "csv",
    kind: "table",
    size_bytes: 10,
    modified_at: "2026-01-01T00:00:00Z",
  };
}

function preview(target: DatasetSummary, column: string): DatasetPreview {
  return {
    dataset: target,
    columns: [column],
    rows: [["value"]],
    truncated: false,
    row_limit: 20,
  };
}

afterEach(() => {
  cleanup();
  mockedFetch.mockReset();
});

describe("DatasetPreviewDialog", () => {
  it("does not show the previous dataset while a new preview loads", async () => {
    const first = dataset("a", "first.csv");
    const second = dataset("b", "second.csv");
    let resolveFirst: ((value: DatasetPreview) => void) | undefined;
    let resolveSecond: ((value: DatasetPreview) => void) | undefined;
    mockedFetch
      .mockImplementationOnce(
        () =>
          new Promise<DatasetPreview>((resolve) => {
            resolveFirst = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise<DatasetPreview>((resolve) => {
            resolveSecond = resolve;
          }),
      );

    const { rerender } = render(
      <DatasetPreviewDialog dataset={first} onClose={() => {}} />,
    );

    await act(async () => {
      resolveFirst?.(preview(first, "old-column"));
    });
    expect(screen.getByText("old-column")).toBeTruthy();

    rerender(<DatasetPreviewDialog dataset={second} onClose={() => {}} />);
    expect(screen.queryByText("old-column")).toBeNull();
    expect(screen.getByText("Loading preview…")).toBeTruthy();

    await act(async () => {
      resolveSecond?.(preview(second, "new-column"));
    });
    expect(screen.getByText("new-column")).toBeTruthy();
  });
});
