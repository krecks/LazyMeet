#!/usr/bin/env bash
# Headless smoke test for the vanilla frontend.
#
# Boots frontend/index.html in a Chromium-based browser with a stubbed Tauri
# bridge (no Rust backend, no models) and fails if the app throws, never shows
# the main screen, or renders the wrong first-run state. Catches the class of
# regressions cargo tests cannot see: broken script order, missing globals,
# boot-flow changes, and a broken prepare screen.
#
# Usage: ./scripts/smoke_frontend.sh
#   BROWSER=/path/to/chrome    override browser detection
#   SMOKE_REQUIRE_BROWSER=1    fail instead of skipping when no browser exists
#                              (used by CI, where a silent skip would hide rot)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

find_browser() {
  if [[ -n "${BROWSER:-}" ]]; then
    command -v "$BROWSER"
    return
  fi
  local name path
  for name in google-chrome google-chrome-stable chromium chromium-browser microsoft-edge msedge; do
    if path="$(command -v "$name" 2>/dev/null)"; then
      echo "$path"
      return 0
    fi
  done
  for name in "Google Chrome" "Chromium" "Microsoft Edge" "Brave Browser"; do
    path="/Applications/$name.app/Contents/MacOS/$name"
    if [[ -x "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

if ! browser="$(find_browser)"; then
  if [[ -n "${SMOKE_REQUIRE_BROWSER:-}" ]]; then
    echo "smoke_frontend: FAIL — no Chromium-based browser found (SMOKE_REQUIRE_BROWSER is set)" >&2
    exit 1
  fi
  echo "smoke_frontend: SKIP — no Chromium-based browser found"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp -R "$root/frontend/." "$tmp/"

# Stub of the Tauri bridge. Covers everything boot and the prepare screen call;
# unknown commands resolve to null, so a new command at boot shows up as a
# smoke failure rather than a crash in the real app.
cat > "$tmp/__smoke_mock.js" <<'EOF'
window.__TAURI__ = {
  core: {
    invoke: async (cmd) => {
      if (cmd === "get_setup") return { ready: true, items: [] };
      if (cmd === "list_meetings") return [];
      if (cmd === "get_speakers") return [];
      if (cmd === "get_config") return { values: {} };
      if (cmd === "probe_media") {
        return { duration: 3, peaks: Array.from({ length: 64 }, (_, i) => Math.abs(Math.sin(i))) };
      }
      return null;
    },
    convertFileSrc: (path) => "asset://localhost/" + path,
  },
  event: { listen: async () => () => {} },
  webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => () => {} }) },
};
EOF

# Runs after the last app script, so all globals exist and boot's microtasks
# have drained. Writes one <pre> the shell greps for.
cat > "$tmp/__smoke_check.js" <<'EOF'
(function () {
  const problems = [];
  const result = document.createElement("pre");
  const finish = () => {
    if (document.getElementById("smoke-result")) return;
    result.id = "smoke-result";
    result.textContent = problems.length ? "SMOKE_FAIL: " + problems.join("; ") : "SMOKE_OK";
    document.body.appendChild(result);
  };

  window.addEventListener("error", (e) => problems.push("uncaught error: " + e.message));
  window.addEventListener("unhandledrejection", (e) => {
    problems.push("unhandled rejection: " + ((e.reason && e.reason.message) || e.reason));
  });

  setTimeout(async () => {
    // Boot state first — opening the prepare screen hides #app again.
    const hidden = (id) => document.getElementById(id).classList.contains("hidden");
    if (!hidden("boot")) problems.push("boot screen still visible");
    if (!hidden("setup")) problems.push("setup screen visible despite ready models");
    if (!hidden("fatal")) {
      problems.push("fatal screen: " + document.getElementById("fatal-msg").textContent.trim());
    }
    if (hidden("app")) problems.push("main screen not visible");
    if (!document.getElementById("meeting-list").textContent.includes("No meetings yet")) {
      problems.push("meeting-list empty state missing");
    }
    if (!document.getElementById("speaker-list").textContent.includes("No saved voices")) {
      problems.push("speaker-list empty state missing");
    }

    try {
      await openPrepare("/tmp/smoke.wav");
      if (hidden("prepare")) problems.push("prepare screen did not open");
      if (document.getElementById("prep-transcribe").disabled) {
        problems.push("transcribe button stayed disabled after a successful probe");
      }
      if (!document.querySelector("#prep-range-list .range-row")) {
        problems.push("preselected range row missing");
      }
    } catch (e) {
      problems.push("openPrepare threw: " + e.message);
    }
    finish();
  }, 0);
})();
EOF

# Inject the mock before the first app script and the checker after the last.
awk '
  /<script src="js\/core\.js"><\/script>/ && !mock { print "<script src=\"__smoke_mock.js\"></script>"; mock = 1 }
  { print }
  /<script src="js\/boot\.js"><\/script>/ { print "<script src=\"__smoke_check.js\"></script>"; check = 1 }
  END { exit !(mock && check) }
' "$tmp/index.html" > "$tmp/index.injected.html" ||
  { echo "smoke_frontend: FAIL — could not find the core.js/boot.js script tags in index.html" >&2; exit 1; }
mv "$tmp/index.injected.html" "$tmp/index.html"

flags=(
  --headless --disable-gpu --no-first-run --no-default-browser-check
  --disable-dev-shm-usage --virtual-time-budget=2000
  --user-data-dir="$tmp/profile" --dump-dom
)
if [[ "$(id -u)" -eq 0 ]]; then
  flags+=(--no-sandbox) # CI containers run as root; the sandbox needs a non-root user
fi

"$browser" "${flags[@]}" "file://$tmp/index.html" > "$tmp/dom.html" 2> "$tmp/browser.log" || true

if grep -q 'SMOKE_OK' "$tmp/dom.html"; then
  echo "smoke_frontend: PASS ($(basename "$browser"))"
  exit 0
fi

echo "smoke_frontend: FAIL" >&2
if grep -o 'SMOKE_FAIL[^<]*' "$tmp/dom.html" >&2; then
  :
else
  echo "  no verdict in the DOM — browser output:" >&2
  tail -n 10 "$tmp/browser.log" >&2
fi
exit 1
