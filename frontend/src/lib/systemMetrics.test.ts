import { expect, test } from "vitest";

import {
  clampPercent,
  formatBytes,
  formatPercent,
  formatTemperature,
} from "./systemMetrics";

test("formats bytes using binary units", () => {
  expect(formatBytes(512)).toBe("512 B");
  expect(formatBytes(1024)).toBe("1 KB");
  expect(formatBytes(8 * 1024 ** 3)).toBe("8.0 GB");
  expect(formatBytes(null)).toBe("N/A");
});

test("formats percentages and temperatures safely", () => {
  expect(formatPercent(45.4)).toBe("45%");
  expect(formatPercent(null)).toBe("N/A");
  expect(formatTemperature(62.6)).toBe("63°C");
  expect(formatTemperature(null)).toBe("N/A");
  expect(clampPercent(120)).toBe(100);
  expect(clampPercent(-4)).toBe(0);
});
