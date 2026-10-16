// Ambient types for the frontend's global environment. The scripts are classic
// <script> files sharing one page scope, so the only things TypeScript cannot
// infer are the ones the browser/Tauri provide: the Tauri v2 bridge
// (withGlobalTauri in tauri.conf.json) and the error shape core.js throws.
//
// Command and event payloads stay `any` on purpose: the backend is the source
// of truth for their shape, and a new command shouldn't need a type here.

interface TauriEvent<T = any> {
  event: string;
  id: number;
  payload: T;
}

interface TauriGlobal {
  core: {
    invoke(cmd: string, args?: Record<string, unknown>): Promise<any>;
    convertFileSrc(path: string): string;
  };
  event: {
    listen(event: string, handler: (event: TauriEvent) => void): Promise<() => void>;
  };
  webview: {
    getCurrentWebview(): {
      onDragDropEvent(handler: (event: TauriEvent) => void): Promise<() => void>;
    };
  };
}

interface Window {
  __TAURI__: TauriGlobal;
}

// core.js re-throws backend failures as an Error carrying the backend's status
// code; frontend code branches on `.status` (404/409/422).
interface ApiError extends Error {
  status?: number;
}
