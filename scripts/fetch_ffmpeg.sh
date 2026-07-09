#!/usr/bin/env bash
# Fetch the per-platform static ffmpeg binary Tauri bundles as an externalBin
# (src-tauri/binaries/ffmpeg-<target-triple>[.exe]).
#
# The binaries come from pinned imageio-ffmpeg 0.6.0 wheels — the exact same
# ffmpeg the old Python sidecar shipped (a wheel is just a zip). Checksums are
# pinned; bump all four together when upgrading.
#
# Usage: scripts/fetch_ffmpeg.sh [target-triple]   (defaults to the host)
set -euo pipefail

cd "$(dirname "$0")/.."
DEST_DIR="src-tauri/binaries"
mkdir -p "$DEST_DIR"

TRIPLE="${1:-$(rustc -vV | sed -n 's/^host: //p')}"

case "$TRIPLE" in
  aarch64-apple-darwin)
    URL="https://files.pythonhosted.org/packages/40/5c/f3d8a657d362cc93b81aab8feda487317da5b5d31c0e1fdfd5e986e55d17/imageio_ffmpeg-0.6.0-py3-none-macosx_11_0_arm64.whl"
    SHA256="b1ae3173414b5fc5f538a726c4e48ea97edc0d2cdc11f103afee655c463fa742"
    INNER_GLOB="imageio_ffmpeg/binaries/ffmpeg-*"
    EXT=""
    ;;
  x86_64-apple-darwin)
    URL="https://files.pythonhosted.org/packages/da/58/87ef68ac83f4c7690961bce288fd8e382bc5f1513860fc7f90a9c1c1c6bf/imageio_ffmpeg-0.6.0-py3-none-macosx_10_9_intel.macosx_10_9_x86_64.whl"
    SHA256="9d2baaf867088508d4a3458e61eeb30e945c4ad8016025545f66c4b5aaef0a61"
    INNER_GLOB="imageio_ffmpeg/binaries/ffmpeg-*"
    EXT=""
    ;;
  x86_64-pc-windows-msvc)
    URL="https://files.pythonhosted.org/packages/2c/c6/fa760e12a2483469e2bf5058c5faff664acf66cadb4df2ad6205b016a73d/imageio_ffmpeg-0.6.0-py3-none-win_amd64.whl"
    SHA256="02fa47c83703c37df6bfe4896aab339013f62bf02c5ebf2dce6da56af04ffc0a"
    INNER_GLOB="imageio_ffmpeg/binaries/ffmpeg-*.exe"
    EXT=".exe"
    ;;
  x86_64-unknown-linux-gnu)
    URL="https://files.pythonhosted.org/packages/a0/2d/43c8522a2038e9d0e7dbdf3a61195ecc31ca576fb1527a528c877e87d973/imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl"
    SHA256="c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282"
    INNER_GLOB="imageio_ffmpeg/binaries/ffmpeg-*"
    EXT=""
    ;;
  *)
    echo "no pinned ffmpeg for target triple: $TRIPLE" >&2
    exit 1
    ;;
esac

DEST="$DEST_DIR/ffmpeg-$TRIPLE$EXT"
if [ -f "$DEST" ]; then
  echo "already present: $DEST"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "downloading $URL"
curl -fsSL "$URL" -o "$TMP/wheel.zip"

echo "$SHA256  $TMP/wheel.zip" | shasum -a 256 -c - >/dev/null 2>&1 \
  || echo "$SHA256  $TMP/wheel.zip" | sha256sum -c - >/dev/null

unzip -q "$TMP/wheel.zip" -d "$TMP/wheel" "$INNER_GLOB"
BIN="$(find "$TMP/wheel" -name 'ffmpeg-*' -type f | head -1)"
[ -n "$BIN" ] || { echo "no ffmpeg binary inside the wheel" >&2; exit 1; }

cp "$BIN" "$DEST"
chmod 0755 "$DEST"
echo "installed $DEST"
