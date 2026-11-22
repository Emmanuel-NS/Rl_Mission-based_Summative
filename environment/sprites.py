"""Turn the generated art in assets/raw into game sprites in assets/sprites.

    uv run python -m environment.sprites

The raw pictures sit on a flat magenta background. This script keys it out,
trims the edges, and cuts the walking and car sheets into frames. The interior
backdrops are only resized. Pillow comes with matplotlib, so nothing extra is
installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "raw"
OUT = ROOT / "assets" / "sprites"

KEYED = ("pharmacy", "home", "tree", "office", "hotel", "bike_side", "bike_top")
SHEETS = {"patient_walk": ("patient", 4), "rider_walk": ("rider", 4), "cars_top": ("car", 4)}
BACKDROPS = ("backdrop", "pharmacy_interior", "lobby_interior")


def key_magenta(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    border = np.concatenate([rgb[:12].reshape(-1, 3), rgb[-12:].reshape(-1, 3)])
    background = np.median(border, axis=0)
    distance = np.sqrt(((rgb - background) ** 2).sum(axis=-1))
    alpha = np.clip((distance - 55.0) / 60.0, 0.0, 1.0)

    mask = Image.fromarray((alpha * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.7))
    alpha = np.asarray(mask).astype(np.float32) / 255.0

    edge = alpha < 0.98
    limit = green + 30.0
    red = np.where(edge, np.minimum(red, limit), red)
    blue = np.where(edge, np.minimum(blue, limit), blue)

    out = np.dstack([red, green, blue, alpha * 255.0]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def trim(image: Image.Image) -> Image.Image:
    box = image.getchannel("A").point(lambda v: 255 if v > 12 else 0).getbbox()
    return image.crop(box) if box else image


def cut_sheet(sheet: Image.Image, count: int) -> list[Image.Image]:
    cell = sheet.width // count
    frames = [trim(sheet.crop((i * cell, 0, (i + 1) * cell, sheet.height))) for i in range(count)]
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    padded = []
    for frame in frames:
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        canvas.paste(frame, ((width - frame.width) // 2, height - frame.height))
        padded.append(canvas)
    return padded


def build() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in KEYED:
        sprite = trim(key_magenta(Image.open(RAW / f"{name}.jpg")))
        path = OUT / f"{name}.png"
        sprite.save(path)
        written.append(path)

    for source, (prefix, count) in SHEETS.items():
        sheet = key_magenta(Image.open(RAW / f"{source}.jpg"))
        for number, frame in enumerate(cut_sheet(sheet, count)):
            path = OUT / f"{prefix}_{number}.png"
            frame.save(path)
            written.append(path)

    for name in BACKDROPS:
        path = OUT / f"{name}.jpg"
        Image.open(RAW / f"{name}.jpg").convert("RGB").save(path, quality=92)
        written.append(path)
    return written


if __name__ == "__main__":
    for path in build():
        print(path.relative_to(ROOT))
