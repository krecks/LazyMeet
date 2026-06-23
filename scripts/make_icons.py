"""Generate the LazyMeet app icon set from a single drawn source.

Produces the PNG/ICO/ICNS files Tauri references for bundling. Run from the repo
root: ``python scripts/make_icons.py``. (CI can instead use ``cargo tauri icon``.)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ICONS = Path(__file__).resolve().parent.parent / "src-tauri" / "icons"


def draw_source(size: int = 1024) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Rounded square background.
    pad = int(size * 0.06)
    d.rounded_rectangle([pad, pad, size - pad, size - pad], radius=int(size * 0.22),
                        fill=(24, 27, 34, 255))
    # Speech-bubble "M" mark in accent colour.
    accent = (110, 168, 254, 255)
    cx = size // 2
    # Three vertical bars suggesting speakers / waveform.
    bar_w = int(size * 0.09)
    gap = int(size * 0.06)
    heights = [0.30, 0.46, 0.22]
    start_x = cx - (bar_w * 3 + gap * 2) // 2
    base_y = int(size * 0.70)
    for i, h in enumerate(heights):
        x = start_x + i * (bar_w + gap)
        bh = int(size * h)
        d.rounded_rectangle([x, base_y - bh, x + bar_w, base_y], radius=bar_w // 2, fill=accent)
    # Little tail to read as a chat bubble.
    d.polygon([(cx - int(size * 0.05), base_y + int(size * 0.02)),
               (cx + int(size * 0.08), base_y + int(size * 0.02)),
               (cx + int(size * 0.02), base_y + int(size * 0.12))], fill=accent)
    return img


def main() -> None:
    ICONS.mkdir(parents=True, exist_ok=True)
    src = draw_source(1024)
    src.save(ICONS / "icon.png")
    src.resize((32, 32)).save(ICONS / "32x32.png")
    src.resize((128, 128)).save(ICONS / "128x128.png")
    src.resize((256, 256)).save(ICONS / "128x128@2x.png")
    src.resize((256, 256)).save(ICONS / "icon.ico", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
    src.save(ICONS / "icon.icns")
    # Windows Store style logos (optional but referenced by some templates).
    print("wrote icons to", ICONS)


if __name__ == "__main__":
    main()
