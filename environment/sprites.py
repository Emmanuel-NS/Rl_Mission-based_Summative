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
POSES = {"rider_stand": ("rider_stand", 3), "patient_stand": ("patient_stand", 3)}
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


def cut_sheet(sheet: Image.Image, count: int, align_left: bool = False) -> list[Image.Image]:
    """Cut a row of poses. Standing poses are aligned on the back of the body so
    the person does not slide when an arm reaches out."""
    cell = sheet.width // count
    frames = [trim(sheet.crop((i * cell, 0, (i + 1) * cell, sheet.height))) for i in range(count)]
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    padded = []
    for frame in frames:
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        left = 0 if align_left else (width - frame.width) // 2
        canvas.paste(frame, (left, height - frame.height))
        padded.append(canvas)
    return padded


def drop_ground_box(frame: Image.Image) -> Image.Image:
    """The scanning pose shows the box on the floor. After the handover it is
    with the patient, so cut away the part of the legs row that lies beyond a gap."""
    data = np.asarray(frame).copy()
    height, width = data.shape[:2]
    top, bottom = int(height * 0.76), int(height * 0.93)
    band = (data[top:bottom, :, 3] > 120).any(axis=0)
    gap = 0
    box_left = width
    for x in range(int(width * 0.25), width):
        gap = 0 if band[x] else gap + 1
        if gap >= 10:
            box_left = next((k for k in range(x, width) if band[k]), width)
            break
    red = data[top:, :, 0].astype(int)
    green = data[top:, :, 1].astype(int)
    blue = data[top:, :, 2].astype(int)
    shoe = (green - red < 14) & (red + green + blue < 260)
    erase = np.zeros(shoe.shape, dtype=bool)
    erase[:, max(0, box_left - 2):] = True
    keep = shoe.copy()
    keep[:, box_left + 4:] = False
    keep[: bottom - top, :] = False
    erase &= ~keep
    alpha = data[top:, :, 3]
    alpha[erase] = 0
    data[top:, :, 3] = alpha
    return Image.fromarray(data, "RGBA")


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

    for source, (prefix, count) in POSES.items():
        sheet = key_magenta(Image.open(RAW / f"{source}.jpg"))
        frames = cut_sheet(sheet, count, align_left=True)
        if prefix == "rider_stand":
            frames.append(drop_ground_box(frames[2]))
        for number, frame in enumerate(frames):
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
