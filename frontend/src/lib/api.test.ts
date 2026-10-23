import { afterEach, expect, test, vi } from "vitest";

import {
  deleteOutput,
  openOutput,
  openSavedOutput,
  outputRawUrl,
  pickFolder,
  revealOutput,
  revealSavedOutput,
  savedMediaRawUrl,
} from "./api";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

test("outputRawUrl preserves date folders", () => {
  expect(outputRawUrl("outputs/2026-10-23/my image.png")).toBe(
    "/api/outputs/2026-10-23/my%20image.png/raw",
  );
  expect(outputRawUrl("outputs/result.png")).toBe(
    "/api/outputs/result.png/raw",
  );
});

test("output actions encode nested ids", async () => {
  const fetchMock = vi.fn(async () => jsonResponse({ status: "opened" }));
  vi.stubGlobal("fetch", fetchMock);

  await openOutput("2026-10-23/result.png");
  await revealOutput("2026-10-23/result.png");

  expect(fetchMock).toHaveBeenNthCalledWith(
    1,
    "/api/outputs/2026-10-23/result.png/open",
    expect.objectContaining({ method: "POST" }),
  );
  expect(fetchMock).toHaveBeenNthCalledWith(
    2,
    "/api/outputs/2026-10-23/result.png/reveal",
    expect.objectContaining({ method: "POST" }),
  );
});

test("delete encodes nested ids", async () => {
  const fetchMock = vi.fn(async () =>
    jsonResponse({ status: "deleted", output: {} }),
  );
  vi.stubGlobal("fetch", fetchMock);

  await deleteOutput("2026-10-23/result.png");

  expect(fetchMock).toHaveBeenCalledWith(
    "/api/outputs/2026-10-23/result.png",
    expect.objectContaining({ method: "DELETE" }),
  );
});

test("builds artifact URLs for external media paths", () => {
  expect(savedMediaRawUrl("C:\\Users\\me\\out.png")).toBe(
    `/api/artifacts/raw?path=${encodeURIComponent("C:\\Users\\me\\out.png")}`,
  );
  expect(savedMediaRawUrl("outputs/2026-10-23/out.png")).toBe(
    "/api/outputs/2026-10-23/out.png/raw",
  );
});

test("routes external open and reveal to the artifact API", async () => {
  const fetchMock = vi.fn(async () =>
    jsonResponse({ status: "opened" }),
  );
  vi.stubGlobal("fetch", fetchMock);

  await openSavedOutput("C:\\Users\\me\\out.png");
  await revealSavedOutput("C:\\Users\\me\\out.png");

  expect(fetchMock).toHaveBeenNthCalledWith(
    1,
    "/api/artifacts/open",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ path: "C:\\Users\\me\\out.png" }),
    }),
  );
  expect(fetchMock).toHaveBeenNthCalledWith(
    2,
    "/api/artifacts/reveal",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ path: "C:\\Users\\me\\out.png" }),
    }),
  );
});

test("picks a folder through the native dialog endpoint", async () => {
  const fetchMock = vi.fn(async () =>
    jsonResponse({ path: "C:\\Users\\me\\images" }),
  );
  vi.stubGlobal("fetch", fetchMock);

  await expect(pickFolder("C:\\Users\\me")).resolves.toBe(
    "C:\\Users\\me\\images",
  );
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/fs/pick-folder",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ initial: "C:\\Users\\me" }),
    }),
  );
});
