"""Build the Python backend into a Tauri sidecar binary.

Runs PyInstaller against ``backend/lazymeet-backend.spec`` and copies the result
to ``src-tauri/binaries/lazymeet-backend-<target-triple>(.exe)`` — the naming
Tauri's ``externalBin`` mechanism expects. The target triple is taken from
``rustc -vV`` so it matches the platform Tauri builds for.

Usage: python scripts/build_sidecar.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "backend" / "lazymeet-backend.spec"
DIST = ROOT / "dist"
BINARIES = ROOT / "src-tauri" / "binaries"


def target_triple() -> str:
    out = subprocess.run(["rustc", "-vV"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("host:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("could not determine target triple from `rustc -vV`")


def main() -> None:
    triple = target_triple()
    exe_suffix = ".exe" if sys.platform.startswith("win") else ""

    print(f"Building sidecar for {triple} ...")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(SPEC)],
        cwd=str(ROOT),
        check=True,
    )

    built = DIST / f"lazymeet-backend{exe_suffix}"
    if not built.exists():
        raise SystemExit(f"expected PyInstaller output at {built}, not found")

    BINARIES.mkdir(parents=True, exist_ok=True)
    dest = BINARIES / f"lazymeet-backend-{triple}{exe_suffix}"
    shutil.copy2(built, dest)
    if not exe_suffix:
        dest.chmod(0o755)
    print(f"Sidecar ready: {dest}")


if __name__ == "__main__":
    main()
