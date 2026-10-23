// @vitest-environment jsdom
import { afterEach, expect, test, vi } from "vitest";

import { copyImageFromUrl } from "./clipboard";

class FakeClipboardItem {
  constructor(readonly items: Record<string, Promise<Blob>>) {}
}

function stubClipboard(): ReturnType<typeof vi.fn> {
  const write = vi.fn(async () => undefined);
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { write },
  });
  vi.stubGlobal("ClipboardItem", FakeClipboardItem);
  return write;
}

function imageResponse(blob: Blob): Response {
  return {
    ok: true,
    status: 200,
    blob: async () => blob,
  } as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: undefined,
  });
});

test("copies PNG images without re-encoding", async () => {
  const write = stubClipboard();
  const blob = new Blob([new Uint8Array([1, 2, 3])], { type: "image/png" });
  vi.stubGlobal("fetch", vi.fn(async () => imageResponse(blob)));

  await copyImageFromUrl("/api/outputs/result.png/raw");

  expect(write).toHaveBeenCalledTimes(1);
  const items = write.mock.calls[0][0] as FakeClipboardItem[];
  const copied = await items[0].items["image/png"];
  expect(copied.type).toBe("image/png");
  expect(new Uint8Array(await copied.arrayBuffer())).toEqual(
    new Uint8Array([1, 2, 3]),
  );
});

test("converts other image formats to PNG before copying", async () => {
  const write = stubClipboard();
  const jpeg = new Blob([new Uint8Array([4, 5, 6])], { type: "image/jpeg" });
  const png = new Blob([new Uint8Array([7, 8, 9])], { type: "image/png" });
  vi.stubGlobal("fetch", vi.fn(async () => imageResponse(jpeg)));
  vi.stubGlobal(
    "createImageBitmap",
    vi.fn(async () => ({ width: 2, height: 2, close: vi.fn() })),
  );
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    drawImage: vi.fn(),
  } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(
    (callback) => callback(png),
  );

  await copyImageFromUrl("/api/outputs/result.jpg/raw");

  const items = write.mock.calls[0][0] as FakeClipboardItem[];
  expect(await items[0].items["image/png"]).toBe(png);
});

test("reports unsupported clipboard APIs", async () => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: undefined,
  });
  vi.stubGlobal("ClipboardItem", undefined);

  await expect(
    copyImageFromUrl("/api/outputs/result.png/raw"),
  ).rejects.toThrow("not supported");
});
