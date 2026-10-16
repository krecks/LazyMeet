// Assertions for the headless smoke test (see scripts/smoke_frontend.sh).
// `main.ts` runs this right after boot; the result is one <pre id="smoke-result">
// that the shell greps for.
import { openPrepare } from "../ts/prepare";

export async function runSmokeChecks(): Promise<void> {
  const problems: string[] = [];
  const finish = () => {
    if (document.getElementById("smoke-result")) return;
    const result = document.createElement("pre");
    result.id = "smoke-result";
    result.textContent = problems.length ? "SMOKE_FAIL: " + problems.join("; ") : "SMOKE_OK";
    document.body.appendChild(result);
  };

  window.addEventListener("error", (e) => problems.push("uncaught error: " + e.message));
  window.addEventListener("unhandledrejection", (e) => {
    problems.push("unhandled rejection: " + ((e.reason && e.reason.message) || e.reason));
  });

  // Boot is async (event listeners + the setup status round-trip). Wait for it
  // to settle before judging the DOM: #app visible means it finished, #fatal
  // means it failed. The budget is virtual time under --virtual-time-budget.
  const hidden = (id: string) => document.getElementById(id)!.classList.contains("hidden");
  const settled = () => !hidden("app") || !hidden("fatal");
  for (let i = 0; i < 100 && !settled(); i++) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  if (!hidden("boot")) problems.push("boot screen still visible");
  if (!hidden("setup")) problems.push("setup screen visible despite ready models");
  if (!hidden("fatal")) {
    problems.push("fatal screen: " + document.getElementById("fatal-msg")!.textContent!.trim());
  }
  if (hidden("app")) problems.push("main screen not visible");
  if (!document.getElementById("meeting-list")!.textContent!.includes("No meetings yet")) {
    problems.push("meeting-list empty state missing");
  }
  if (!document.getElementById("speaker-list")!.textContent!.includes("No saved voices")) {
    problems.push("speaker-list empty state missing");
  }

  try {
    await openPrepare("/tmp/smoke.wav");
    if (hidden("prepare")) problems.push("prepare screen did not open");
    if ((document.getElementById("prep-transcribe") as HTMLButtonElement).disabled) {
      problems.push("transcribe button stayed disabled after a successful probe");
    }
    if (!document.querySelector("#prep-range-list .range-row")) {
      problems.push("preselected range row missing");
    }
  } catch (e: any) {
    problems.push("openPrepare threw: " + e.message);
  }
  finish();
}
