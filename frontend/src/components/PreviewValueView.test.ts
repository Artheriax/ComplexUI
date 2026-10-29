// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, expect, test } from "vitest";

import type {
  ConditioningPreview,
  DiffusionModelPreview,
  ImageBatchPreview,
  GuiderPreview,
  ImagePreview,
  LatentPreview,
  NoisePreview,
  SamplerPreview,
  SigmasPreview,
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
test("renders image batches and diffusion model metadata", () => {
  const batch: ImageBatchPreview = {
    kind: "image_batch",
    count: 2,
    width: 512,
    height: 512,
    mode: "RGB",
    thumbnail: "data:image/jpeg;base64,batch",
  };
  const model: DiffusionModelPreview = {
    kind: "diffusion_model",
    arch: "sdxl",
    name: "dreamshaperXL_lightning",
    class: "UNet2DConditionModel",
    parameters: 2567459428,
    device: "cuda",
    dtype: "torch.float16",
    storage_dtype: "fp16",
    quantization: {
      format: "int8_w8a8",
      label: "INT8 W8A8",
      runtime: "comfy_kitchen",
      source: "comfy_quant",
      mode: "pre_quantized",
      weight_bits: 8,
      valid: true,
    },
    memory: {
      weights_bytes: 2 * 1024 ** 3,
      quantization_overhead_bytes: 0,
      resident_estimate_bytes: 3 * 1024 ** 3,
      offload_ram_estimate_bytes: 2 * 1024 ** 3,
      confidence: "medium",
      assumptions: [],
      unknown: false,
    },
    source: "local",
    model_type: "diffusion",
    path: "models/diffusion/model.safetensors",
    format: "safetensors",
    fingerprint: "1024:123",
    preset: "sdxl_lightning",
    recommended: { steps: 6, cfg: 2 },
    attention: { backend: "sdpa" },
    offload: true,
    compiled: true,
  };
  const latent: LatentPreview = {
    kind: "latent",
    shape: [1, 4, 128, 128],
    dtype: "torch.float16",
    device: "cpu",
    width: 1024,
    height: 1024,
    batch_size: 1,
  };

  const { rerender } = render(createElement(PreviewValueView, { value: batch }));
  expect(screen.getByAltText("Image batch")).toBeTruthy();
  expect(screen.getByText("2 images")).toBeTruthy();
  expect(screen.getByText("512x512")).toBeTruthy();

  rerender(createElement(PreviewValueView, { value: model }));
  expect(screen.getByText("sdxl")).toBeTruthy();
  expect(screen.getByText("dreamshaperXL_lightning")).toBeTruthy();
  expect(screen.getByText("sdxl_lightning")).toBeTruthy();
  expect(screen.getByText(/params/)).toBeTruthy();
  expect(screen.getByText("INT8 W8A8")).toBeTruthy();
  expect(screen.getByText("~3.00 GB resident")).toBeTruthy();
  expect(screen.getByText("store fp16")).toBeTruthy();
  expect(screen.getByText("attn sdpa")).toBeTruthy();
  expect(screen.getByText("offload")).toBeTruthy();
  expect(screen.getByText("compiled")).toBeTruthy();

  rerender(createElement(PreviewValueView, { value: latent }));
  expect(screen.getByText("latent [1,4,128,128]")).toBeTruthy();
  expect(screen.getByText("1024x1024")).toBeTruthy();
});

test("renders the sigma schedule and advanced sampling contracts", () => {
  const sigmas: SigmasPreview = {
    kind: "sigmas",
    sigmas: [14.6, 8.2, 3.4, 1.1, 0],
    steps: 4,
    scheduler: "karras",
    denoise: 1,
    sigma_max: 14.6,
    sigma_min: 0.029,
    arch: "flux1",
  };
  const sampler: SamplerPreview = {
    kind: "sampler",
    name: "dpmpp_2m",
    custom: false,
  };
  const noise: NoisePreview = {
    kind: "noise",
    noise_type: "fractal",
    noise_seed: 42,
    alpha: 1,
    k: 1,
    scale: 0.1,
  };
  const guider: GuiderPreview = {
    kind: "guider",
    cfg: 7,
    guidance: 3.5,
    guidance_mode: "inherit",
    start_percent: 0.1,
    end_percent: 0.8,
    arch: "sd15",
    model: "dreamshaper_8",
    positive_text: "a cozy room",
    negative_text: "blurry",
  };

  const { rerender } = render(
    createElement(PreviewValueView, { value: sigmas }),
  );
  expect(screen.getByLabelText("Sigma schedule")).toBeTruthy();
  expect(screen.getByText("4 steps")).toBeTruthy();
  expect(screen.getByText("karras")).toBeTruthy();
  expect(screen.getByText("flux1")).toBeTruthy();

  rerender(createElement(PreviewValueView, { value: sampler }));
  expect(screen.getByText("dpmpp_2m")).toBeTruthy();

  rerender(createElement(PreviewValueView, { value: noise }));
  expect(screen.getByText("fractal")).toBeTruthy();
  expect(screen.getByText(/noise seed 42/)).toBeTruthy();

  rerender(createElement(PreviewValueView, { value: guider }));
  expect(screen.getByText("CFG 7")).toBeTruthy();
  expect(screen.getByText(/guidance 3\.5000/)).toBeTruthy();
  expect(screen.getByText(/CFG 0\.1000–0\.8000/)).toBeTruthy();
  expect(screen.getByText("dreamshaper_8")).toBeTruthy();
});

test("renders generic batch counts", () => {
  render(
    createElement(PreviewValueView, {
      value: { kind: "batch", count: 4 },
    }),
  );

  expect(screen.getByText("4 items")).toBeTruthy();
});

test("renders reference images and inpainting metadata on conditioning", () => {
  const conditioning: ConditioningPreview = {
    kind: "conditioning",
    arch: "flux2",
    shape: [1, 512, 4096],
    pooled_shape: null,
    attention_mask_shape: [1, 512],
    encoder_count: 1,
    ref_count: 2,
    inpaint: true,
    text: "put both products on the table",
  };

  render(createElement(PreviewValueView, { value: conditioning }));

  expect(screen.getByText("2 reference images")).toBeTruthy();
  expect(screen.getByText("inpainting mask attached")).toBeTruthy();
  expect(
    screen.getByText("put both products on the table"),
  ).toBeTruthy();
});
