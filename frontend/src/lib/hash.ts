/**
 * SHA-256 code hashing mirroring `backend/app/security.py`.
 *
 * The backend hashes the code after line-ending normalization, so the
 * frontend must do exactly the same to keep `code_hash` values comparable.
 */

export function normalizeLineEndings(code: string): string {
  return code.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

export async function sha256Hex(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(normalizeLineEndings(text));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}
