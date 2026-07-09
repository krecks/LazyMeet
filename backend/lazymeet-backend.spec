# PyInstaller spec for the LazyMeet backend.
# Freezes the FastAPI app + AI stack in onedir mode: a onefile binary would
# re-extract ~70 MB to a temp dir on every launch, adding many seconds before
# the app window shows anything. The directory is shipped as a Tauri bundle
# resource. Build via: python scripts/build_sidecar.py
#
# collect_all pulls in the native libraries and data files that sherpa-onnx,
# llama-cpp-python, imageio-ffmpeg and soundfile load at runtime.

import os

from PyInstaller.utils.hooks import collect_all

# SPECPATH is the directory containing this spec (backend/); the repo root is its
# parent. This keeps the build working regardless of the current directory.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))
ENTRY = os.path.join(ROOT, "scripts", "sidecar_entry.py")

datas, binaries, hiddenimports = [], [], []
for pkg in ("sherpa_onnx", "sherpa_onnx_core", "llama_cpp", "imageio_ffmpeg", "soundfile"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "fastapi",
    "anyio",
]

a = Analysis(
    [ENTRY],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lazymeet-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,        # stdout must be visible so Tauri can read "READY <port>"
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="lazymeet-backend",
)
