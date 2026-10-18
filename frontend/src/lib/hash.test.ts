import { describe, expect, it } from "vitest";

import { normalizeLineEndings, sha256Hex } from "./hash";

describe("normalizeLineEndings", () => {
  it("converts CRLF and lone CR to LF", () => {
    expect(normalizeLineEndings("a\r\nb\rc\n")).toBe("a\nb\nc\n");
  });

  it("leaves LF-only text untouched", () => {
    expect(normalizeLineEndings("a\nb\n")).toBe("a\nb\n");
  });
});

describe("sha256Hex", () => {
  it("matches known SHA-256 vectors", async () => {
    expect(await sha256Hex("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    expect(await sha256Hex("")).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
  });

  it("hashes CRLF and LF variants identically (backend parity)", async () => {
    expect(await sha256Hex("x = 1\r\ny = 2\r\n")).toBe(
      await sha256Hex("x = 1\ny = 2\n"),
    );
  });
});
