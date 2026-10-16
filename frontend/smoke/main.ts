// Smoke-test entry. In smoke builds Vite rewrites index.html to load this
// module instead of ts/boot.ts (see vite.config.ts): boot the real app against
// the aliased Tauri stub, then run the DOM assertions.
import "../ts/boot";
import { runSmokeChecks } from "./check";

runSmokeChecks();
