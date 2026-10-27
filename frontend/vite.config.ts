import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The launcher sets COMPLEXUI_BACKEND_URL so custom ports work without editing
// this file. Manual `npm run dev` falls back to the default backend port.
const backendUrl =
  process.env.COMPLEXUI_BACKEND_URL ?? "http://127.0.0.1:8000";
const backendWsUrl = backendUrl.replace(
  /^http(s?):\/\//,
  (_match, secure: string) => (secure ? "wss://" : "ws://"),
);

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": backendUrl,
      "/ws": {
        target: backendWsUrl,
        ws: true,
      },
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
