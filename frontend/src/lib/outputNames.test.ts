import { expect, test } from "vitest";

import {
  previewSavePath,
  sanitizeModelName,
  sanitizeOutputSegment,
} from "./outputNames";

const NOW = new Date(2026, 9, 23, 16, 5, 9);

test("builds the default dated output path", () => {
  expect(
    previewSavePath({
      folder: "%date%",
      filename: "%model%_%counter%",
      format: "png",
      modelName: "dream shaper",
      now: NOW,
    }),
  ).toEqual({
    path: "outputs/2026-10-23/dream_shaper_001.png",
    warnings: [],
  });
});

test("supports absolute destinations and date folders", () => {
  expect(
    previewSavePath({
      folder: "C:/Users/Akali video/%date%",
      filename: "%counter%",
      format: "jpg",
      modelName: "sdxl",
      now: NOW,
    }).path,
  ).toBe("C:\\Users\\Akali video\\2026-10-23\\001.jpg");

  expect(
    previewSavePath({
      folder: "/home/user/images",
      filename: "%date%_%time%_%counter%",
      format: "webp",
      modelName: "sdxl",
      now: NOW,
    }).path,
  ).toBe("/home/user/images/2026-10-23_16-05-09_001.webp");
});

test("adds a batch index when the template omits it", () => {
  const preview = previewSavePath({
    folder: "",
    filename: "%model%_%counter%",
    format: "png",
    modelName: "batch",
    batchCount: 3,
    batchIndex: 2,
    now: NOW,
  });
  expect(preview.path).toBe("outputs/batch_001_002.png");
});

test("uses explicit batch tokens and nested folders", () => {
  const preview = previewSavePath({
    folder: "Akali/%date%",
    filename: "shots/%batch%/%counter%",
    format: "png",
    modelName: "model",
    batchCount: 4,
    batchIndex: 3,
    now: NOW,
  });
  expect(preview.path).toBe("outputs/Akali/2026-10-23/shots/003/001.png");
});

test("resolves width and height when known", () => {
  expect(
    previewSavePath({
      folder: "",
      filename: "%width%x%height%_%counter%",
      format: "png",
      modelName: "model",
      width: 768,
      height: 1024,
      now: NOW,
    }).path,
  ).toBe("outputs/768x1024_001.png");
});

test("warns about missing dimensions, disabled tokens and missing counter", () => {
  const preview = previewSavePath({
    folder: "%date%",
    filename: "%seed%_%counter%_%width%",
    format: "png",
    modelName: "model",
    now: NOW,
  });
  expect(preview.warnings).toContain("%seed% requires metadata support.");
  expect(preview.warnings).toContain(
    "%width% is filled in once the image is known.",
  );
  expect(preview.path).toContain("%seed%_001_%width%");

  const missingCounter = previewSavePath({
    folder: "",
    filename: "image",
    format: "png",
    modelName: "model",
    now: NOW,
  });
  expect(missingCounter.warnings).toContain(
    "Filename template must include %counter%.",
  );
});

test("uses the video suffix when provided", () => {
  expect(
    previewSavePath({
      folder: "",
      filename: "%model%_%counter%",
      format: "png",
      modelName: "clip",
      videoSuffix: ".mp4",
      now: NOW,
    }).path,
  ).toBe("outputs/clip_001.mp4");
});

test("sanitizes model names and path segments", () => {
  expect(sanitizeModelName("  dream/shaper 8 ")).toBe("dream_shaper_8");
  expect(sanitizeOutputSegment(" .hidden<file>. ")).toBe("hidden_file_");
  expect(sanitizeOutputSegment("..")).toBe("");
});
