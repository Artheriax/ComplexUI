import { describe, expect, test } from "vitest";

import type { ModelMemoryEstimate, ModelQuantizationInfo } from "./api";
import {
  formatModelSize,
  formatParameterCount,
  memoryDetail,
  memoryTag,
  quantizationDetail,
  quantizationTag,
} from "./modelMetadata";

const quantization: ModelQuantizationInfo = {
  format: "int8_w8a8",
  label: "INT8 W8A8",
  runtime: "comfy_kitchen",
  source: "comfy_quant",
  mode: "pre_quantized",
  variant: "rowwise",
  weight_bits: 8,
  activation: "dynamic",
  valid: true,
};

const memory: ModelMemoryEstimate = {
  weights_bytes: 2 * 1024 ** 3,
  quantization_overhead_bytes: 0,
  resident_estimate_bytes: 3 * 1024 ** 3,
  offload_ram_estimate_bytes: 2.5 * 1024 ** 3,
  confidence: "medium",
  assumptions: [],
  unknown: false,
};

describe("model metadata formatting", () => {
  test("quantization tags expose a factual label and details", () => {
    expect(quantizationTag(quantization)).toBe("INT8 W8A8");
    expect(quantizationDetail(quantization)).toBe(
      "rowwise · 8-bit · dynamic · comfy_kitchen",
    );
    expect(quantizationTag(null)).toBeNull();
  });

  test("memory tags show resident and offload estimates", () => {
    expect(memoryTag(memory)).toBe("~3.00 GB resident");
    expect(memoryDetail(memory)).toBe(
      "estimate: 2.00 GB weights · 2.50 GB offload RAM · medium confidence",
    );
    expect(
      memoryTag({
        ...memory,
        unknown: true,
      }),
    ).toBe("memory unknown");
  });

  test("parameter counts and sizes stay compact", () => {
    expect(formatParameterCount(2_567_459_428)).toBe("2.57B");
    expect(formatParameterCount(1_234_567)).toBe("1.2M");
    expect(formatParameterCount(undefined)).toBeNull();
    expect(formatModelSize(512)).toBe("512 B");
    expect(formatModelSize(2 * 1024 ** 3)).toBe("2.00 GB");
  });
});
