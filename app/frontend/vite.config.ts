import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend calls same-origin /api and /ws; Vite proxies them to the
// FastAPI backend during dev so there are no CORS surprises.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1", // bind IPv4 explicitly (default localhost may resolve to ::1 only)
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
