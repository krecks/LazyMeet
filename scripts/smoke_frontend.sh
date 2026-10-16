#!/usr/bin/env bash
# Headless smoke test for the frontend bundle.
#
# Builds the real Vite bundle in smoke mode (the @tauri-apps/api modules are
# aliased to a stub — see vite.config.ts), serves it over HTTP (ES modules
# don't load from file://), and boots it in a headless Chromium-based browser.
# Fails if the app throws, never shows the main screen, or renders the wrong
# first-run state. Catches the class of regressions cargo tests cannot see: a
# broken bundle, boot-flow changes, and a broken prepare screen.
#
# Usage: ./scripts/smoke_frontend.sh
#   BROWSER=/path/to/chrome    override browser detection
#   SMOKE_REQUIRE_BROWSER=1    fail instead of skipping when no browser exists
#                              (used by CI, where a silent skip would hide rot)
#
# Needs the npm dependencies installed (npm ci) and python3 (throwaway server).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

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

npm run build:smoke

# Pick a free port and serve the smoke bundle. Module scripts are subject to
# CORS, so file:// cannot be used.
port="$(python3 - <<'PY'
import socket
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
)"
python3 -m http.server "$port" --bind 127.0.0.1 --directory frontend/dist-smoke >/dev/null 2>&1 &
server=$!
tmp="$(mktemp -d)"
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; rm -rf "$tmp"' EXIT

# Wait for the server to accept connections (python3 rather than curl, so the
# script has no dependencies beyond what it already uses).
python3 - "$port" <<'PY'
import socket, sys, time
port = int(sys.argv[1])
for _ in range(100):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
        sys.exit(0)
    except OSError:
        time.sleep(0.05)
sys.exit("smoke_frontend: server did not start")
PY

flags=(
  --headless --disable-gpu --no-first-run --no-default-browser-check
  --disable-dev-shm-usage --virtual-time-budget=2000
  --user-data-dir="$tmp/profile" --dump-dom
)
if [[ "$(id -u)" -eq 0 ]]; then
  flags+=(--no-sandbox) # CI containers run as root; the sandbox needs a non-root user
fi

"$browser" "${flags[@]}" "http://127.0.0.1:$port/" > "$tmp/dom.html" 2> "$tmp/browser.log" || true

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
