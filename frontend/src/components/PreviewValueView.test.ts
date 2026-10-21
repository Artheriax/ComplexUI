// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, expect, test } from "vitest";

import type {
  ImagePreview,
  VideoPreview,
  VisionModelPreview,
} from "../types/graph";
import { PreviewValueView } from "./PreviewValueView";

afterEach(cleanup);

test("renders image thumbnails with dimensions", () => {
  const preview: ImagePreview = {
    kind: "image",
    width: 320,
    height: 240,
    mode: "RGB",
    thumbnail: "data:image/jpeg;base64,abc",
  };

  render(createElement(PreviewValueView, { value: preview }));

  expect(screen.getByAltText("Node output").getAttribute("src")).toBe(
    preview.thumbnail,
  );
  expect(screen.getByText("320x240")).toBeTruthy();
  expect(screen.getByText("RGB")).toBeTruthy();
});

test("shows local model provenance and fingerprint", () => {
  const preview: VisionModelPreview = {
    kind: "vision_model",
    backend: "ultralytics",
    task: "detection",
    model: "yolo11n",
    weights: "DEFAULT",
    device: "cpu",
    classes: [],
    source: "local",
    model_type: "yolo",
    path: "C:/repo/models/yolo/custom.pt",
    format: "pt",
    fingerprint: "1024:123",
  };

  render(createElement(PreviewValueView, { value: preview }));

  expect(screen.getByText("local")).toBeTruthy();
  expect(screen.getByText("C:/repo/models/yolo/custom.pt")).toBeTruthy();
  expect(screen.getByText(/fingerprint 1024:123/)).toBeTruthy();
});

test("renders playable video outputs with metadata", () => {
  const preview: VideoPreview = {
    kind: "video",
    path: "outputs/demo.mp4",
    url: "/api/outputs/demo.mp4/raw",
    fps: 12,
    frames: 24,
    width: 160,
    height: 120,
    duration: 2,
    poster: null,
  };

  render(createElement(PreviewValueView, { value: preview }));

  const video = screen.getByText(/Your browser does not support video/);
  expect(video.tagName).toBe("VIDEO");
  expect(video.getAttribute("src")).toBe("/api/outputs/demo.mp4/raw");
  expect(screen.getByText("160x120")).toBeTruthy();
  expect(screen.getByText("12 fps")).toBeTruthy();
  expect(screen.getByText("24 frames")).toBeTruthy();
});
