"""Build the Python backend into the app's bundled backend directory.

Runs PyInstaller against ``backend/lazymeet-backend.spec`` (onedir mode — it
starts in ~1 s, where a self-extracting onefile binary spent many seconds
unpacking on every launch) and copies the result to
``src-tauri/binaries/backend/``. tauri.conf.json ships that directory as a
bundle resource and the Rust shell spawns ``lazymeet-backend`` from it.

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


def main() -> None:
    exe_suffix = ".exe" if sys.platform.startswith("win") else ""

    built = DIST / "lazymeet-backend"
    if built.exists() and not built.is_dir():
        built.unlink()  # stale onefile output from an older build

    print("Building backend (PyInstaller onedir) ...")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(SPEC)],
        cwd=str(ROOT),
        check=True,
    )

    exe = built / f"lazymeet-backend{exe_suffix}"
    if not exe.exists():
        raise SystemExit(f"expected PyInstaller output at {exe}, not found")

    BINARIES.mkdir(parents=True, exist_ok=True)
    for stale in BINARIES.glob("lazymeet-backend-*"):  # old onefile sidecars
        stale.unlink()
    dest = BINARIES / "backend"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(built, dest)
    if not exe_suffix:
        (dest / "lazymeet-backend").chmod(0o755)
    print(f"Backend ready: {dest}")


if __name__ == "__main__":
    main()
