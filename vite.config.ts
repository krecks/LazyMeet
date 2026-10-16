import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

// The app runs in a Tauri window: `npm run dev` serves frontend/ with HMR on
// the port tauri.conf.json's devUrl expects; `npm run build` bundles it into
// frontend/dist for the installer (frontendDist). `--mode smoke` swaps the
// @tauri-apps/api modules for the stub the smoke test boots against
// (frontend/smoke/tauri-api.ts) and starts frontend/smoke/main.ts instead of
// the app.
function smokeEntry(): Plugin {
  const appEntry = '<script type="module" src="/ts/boot.ts"></script>';
  return {
    name: "lazymeet:smoke-entry",
    transformIndexHtml: {
      // Must run before Vite's own HTML transform, which rewrites the app
      // entry into the emitted bundle.
      order: "pre",
      handler(html) {
        if (!html.includes(appEntry)) {
          throw new Error(`smoke build: ${appEntry} not found in index.html`);
        }
        return html.replace(appEntry, '<script type="module" src="/smoke/main.ts"></script>');
      },
    },
  };
}

export default defineConfig(({ mode }) => {
  const smoke = mode === "smoke";
  const stub = fileURLToPath(new URL("./frontend/smoke/tauri-api.ts", import.meta.url));
  return {
    root: "frontend",
    // Relative asset URLs so the bundle also works when the webview loads it
    // from a tauri:// origin rather than from the dev server root.
    base: "./",
    clearScreen: false,
    plugins: smoke ? [smokeEntry()] : [],
    resolve: smoke
      ? {
          alias: [
            { find: "@tauri-apps/api/core", replacement: stub },
            { find: "@tauri-apps/api/event", replacement: stub },
            { find: "@tauri-apps/api/webview", replacement: stub },
          ],
        }
      : undefined,
    build: {
      // The smoke bundle must never be the one an installer ships: keep it in
      // its own output directory (frontendDist points at frontend/dist).
      outDir: smoke ? "dist-smoke" : "dist",
      emptyOutDir: true,
      target: "es2022",
    },
    server: {
      port: 1420,
      strictPort: true,
    },
  };
});
