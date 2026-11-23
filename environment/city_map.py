"""Top-down Kigali city for the rider view, drawn once and scrolled like a map."""

from __future__ import annotations

import random

from environment.custom_env import (
    ARTERY_ROW,
    AVENUES,
    COLS,
    DESTINATIONS,
    PHARMACY_NODE,
    edge_nodes,
    ROWS,
    STREETS,
)

NX0, NY0 = 430, 340
DX, DY = 440, 340
WORLD_W = 2 * NX0 + (COLS - 1) * DX
WORLD_H = 2 * NY0 + (ROWS - 1) * DY
ROAD_W = 128
ARTERY_W = 136
BIKE_LANE = 26

PAVEMENT = (214, 210, 198)
ASPHALT = (60, 64, 69)
BIKE_ASPHALT = (54, 84, 76)
GRASS = (132, 178, 114)

LANDMARK_SIZE = {"pharmacy": (156, 104), "home": (112, 88), "office": (164, 112), "hotel": (184, 112)}


def node_px(node: int) -> tuple[int, int]:
    return NX0 + (node % COLS) * DX, NY0 + (node // COLS) * DY


def road_half(row: int) -> int:
    return (ARTERY_W if row == ARTERY_ROW else ROAD_W) // 2


def car_lane_offset(half: int) -> float:
    """Distance from the road centre to the middle of a car lane."""
    return (half - BIKE_LANE) / 2.0


def bike_lane_offset(half: int) -> float:
    """Distance from the road centre to the middle of the rider's lane at the kerb."""
    return half - BIKE_LANE / 2.0


def lane_half(node: int, horizontal: bool) -> int:
    """Half width of the road that a rider on this node travels along."""
    return road_half(node // COLS) if horizontal else ROAD_W // 2


def edge_geometry(edge: int):
    """Start, end, length, orientation, road half width and junction halves of a road."""
    a, b = edge_nodes(edge)
    ax, ay = node_px(a)
    bx, by = node_px(b)
    horizontal = ay == by
    length = abs(bx - ax) + abs(by - ay)
    if horizontal:
        half = road_half(a // COLS)
        junction_a = junction_b = ROAD_W // 2
    else:
        half = ROAD_W // 2
        junction_a, junction_b = road_half(a // COLS), road_half(b // COLS)
    return ax, ay, bx, by, length, horizontal, half, junction_a, junction_b


def _xs() -> list[int]:
    return [NX0 + c * DX for c in range(COLS)]


def _ys() -> list[int]:
    return [NY0 + r * DY for r in range(ROWS)]


def _lot(i: int, j: int):
    """Rectangle of block (i, j); -1 and COLS-1 / ROWS-1 are the outer ring."""
    xs, ys = _xs(), _ys()
    left = 0 if i < 0 else xs[i] + ROAD_W // 2
    right = WORLD_W if i + 1 >= COLS else xs[i + 1] - ROAD_W // 2
    top = 0 if j < 0 else ys[j] + road_half(j)
    bottom = WORLD_H if j + 1 >= ROWS else ys[j + 1] - road_half(j + 1)
    return left, top, right, bottom


def _anchor(node: int):
    """Block and corner next to a junction, so the landmark sits beside its road."""
    col, row = node % COLS, node // COLS
    for di, dj, right, bottom in ((0, 0, False, False), (-1, 0, True, False), (0, -1, False, True), (-1, -1, True, True)):
        i, j = col + di, row + dj
        if 0 <= i <= COLS - 2 and 0 <= j <= ROWS - 2:
            return i, j, right, bottom
    return 0, 0, False, False


def landmark_rect(kind: str, node: int):
    i, j, right, bottom = _anchor(node)
    left, top, rgt, bot = _lot(i, j)
    w, h = LANDMARK_SIZE[kind]
    x = rgt - 18 - w if right else left + 18
    y = bot - 18 - h if bottom else top + 18
    return x, y, w, h


def landmarks() -> dict[str, tuple[int, int, int, int]]:
    result = {"pharmacy": landmark_rect("pharmacy", PHARMACY_NODE)}
    for number, dest in enumerate(DESTINATIONS):
        result[f"dest{number}"] = landmark_rect(dest.kind, dest.node)
    return result


def build_city(pygame, fonts) -> object:
    """Return the static city as one big surface."""
    rng = random.Random(21)
    world = pygame.Surface((WORLD_W, WORLD_H))
    world.fill(PAVEMENT)
    shadows = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
    xs, ys = _xs(), _ys()

    buildings: list[tuple] = []
    for j in range(-1, ROWS):
        for i in range(-1, COLS):
            left, top, right, bottom = _lot(i, j)
            lot = pygame.Rect(left + 12, top + 12, right - left - 24, bottom - top - 24)
            if lot.width < 40 or lot.height < 40:
                continue
            outer = i < 0 or j < 0 or i >= COLS - 1 or j >= ROWS - 1
            if not outer and rng.random() < 0.2:
                holds_landmark = any(lot.colliderect(pygame.Rect(*rect)) for rect in landmarks().values())
                if not holds_landmark:
                    _park(pygame, world, lot, rng)
                    continue
            cols = 1 if lot.width < 230 else (3 if lot.width > 700 else 2)
            rows = 1 if lot.height < 170 else 2
            for a in range(cols):
                for b in range(rows):
                    cell = pygame.Rect(
                        lot.x + a * lot.width // cols,
                        lot.y + b * lot.height // rows,
                        lot.width // cols,
                        lot.height // rows,
                    ).inflate(-rng.randint(12, 26), -rng.randint(12, 26))
                    style = rng.choices(["concrete", "clay", "glass", "green", "slate"], [4, 3, 2, 1, 2])[0]
                    buildings.append((cell, style))

    for cell, _style in buildings:
        pygame.draw.rect(shadows, (20, 24, 30, 82), cell.move(14, 14), border_radius=6)
    world.blit(shadows, (0, 0))
    for cell, style in buildings:
        _roof(pygame, world, cell, style, rng)

    asphalt = ASPHALT
    for y_pos, row in zip(ys, range(ROWS)):
        half = road_half(row)
        pygame.draw.rect(world, asphalt, pygame.Rect(0, y_pos - half, WORLD_W, half * 2))
        pygame.draw.rect(world, BIKE_ASPHALT, pygame.Rect(0, y_pos - half, WORLD_W, BIKE_LANE))
        pygame.draw.rect(world, BIKE_ASPHALT, pygame.Rect(0, y_pos + half - BIKE_LANE, WORLD_W, BIKE_LANE))
        pygame.draw.line(world, (150, 148, 140), (0, y_pos - half), (WORLD_W, y_pos - half), 2)
        pygame.draw.line(world, (150, 148, 140), (0, y_pos + half), (WORLD_W, y_pos + half), 2)
    for x_pos in xs:
        half = ROAD_W // 2
        pygame.draw.rect(world, asphalt, pygame.Rect(x_pos - half, 0, half * 2, WORLD_H))
        pygame.draw.rect(world, BIKE_ASPHALT, pygame.Rect(x_pos - half, 0, BIKE_LANE, WORLD_H))
        pygame.draw.rect(world, BIKE_ASPHALT, pygame.Rect(x_pos + half - BIKE_LANE, 0, BIKE_LANE, WORLD_H))
        pygame.draw.line(world, (150, 148, 140), (x_pos - half, 0), (x_pos - half, WORLD_H), 2)
        pygame.draw.line(world, (150, 148, 140), (x_pos + half, 0), (x_pos + half, WORLD_H), 2)

    for y_pos, row in zip(ys, range(ROWS)):
        _hline_markings(pygame, world, y_pos, row == ARTERY_ROW)
    for x_pos in xs:
        _vline_markings(pygame, world, x_pos)
    for x_pos in xs:
        for y_pos, row in zip(ys, range(ROWS)):
            half_h = road_half(row)
            half_v = ROAD_W // 2
            pygame.draw.rect(world, asphalt, pygame.Rect(x_pos - half_v, y_pos - half_h, half_v * 2, half_h * 2))
            _crossings(pygame, world, x_pos, y_pos, half_v, half_h)

    for number, (y_pos, row) in enumerate(zip(ys, range(ROWS))):
        for k in range(COLS):
            _tree_row(pygame, world, k, y_pos, row, xs, rng)

    marks = landmarks()
    shadow2 = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
    for key, (x, y, w, h) in marks.items():
        pygame.draw.rect(shadow2, (20, 24, 30, 110), pygame.Rect(x + 16, y + 16, w, h), border_radius=8)
    world.blit(shadow2, (0, 0))
    _landmark(pygame, world, "pharmacy", marks["pharmacy"], fonts)
    for number, dest in enumerate(DESTINATIONS):
        _landmark(pygame, world, dest.kind, marks[f"dest{number}"], fonts)

    for row, y_pos in enumerate(ys):
        text = fonts(17, True).render(AVENUES[row], True, (200, 206, 210))
        world.blit(text, (xs[2] + DX // 2 - text.get_width() // 2, y_pos - road_half(row) + 8))
    for col, x_pos in enumerate(xs):
        text = fonts(17, True).render(STREETS[col], True, (200, 206, 210))
        text = pygame.transform.rotate(text, 90)
        world.blit(text, (x_pos - ROAD_W // 2 + 8, ys[1] + DY // 2 - text.get_height() // 2))
    return world


def _park(pygame, world, lot, rng) -> None:
    pygame.draw.rect(world, GRASS, lot, border_radius=10)
    pygame.draw.rect(world, (112, 160, 96), lot, width=3, border_radius=10)
    pygame.draw.line(world, (226, 220, 200), lot.topleft, lot.bottomright, 8)
    pygame.draw.line(world, (226, 220, 200), lot.topright, lot.bottomleft, 8)
    center = lot.center
    pygame.draw.circle(world, (226, 220, 200), center, 34)
    pygame.draw.circle(world, (146, 196, 222), center, 26)
    pygame.draw.circle(world, (200, 228, 242), center, 12)
    count = max(6, lot.width * lot.height // 4500)
    for _ in range(count):
        x = rng.randint(lot.left + 14, lot.right - 14)
        y = rng.randint(lot.top + 14, lot.bottom - 14)
        if abs(x - center[0]) < 44 and abs(y - center[1]) < 44:
            continue
        radius = rng.randint(12, 20)
        pygame.draw.circle(world, (60, 96, 56), (x + 6, y + 7), radius)
        pygame.draw.circle(world, (76, 128, 70), (x, y), radius)
        pygame.draw.circle(world, (104, 158, 92), (x - 4, y - 4), radius // 2)


def _roof(pygame, world, cell, style: str, rng) -> None:
    base = {
        "concrete": (226, 224, 218),
        "clay": (186, 100, 72),
        "glass": (96, 142, 178),
        "green": (110, 156, 112),
        "slate": (118, 124, 132),
    }[style]
    pygame.draw.rect(world, base, cell, border_radius=5)
    inner = cell.inflate(-14, -14)
    light = tuple(min(255, c + 22) for c in base)
    dark = tuple(max(0, c - 40) for c in base)
    pygame.draw.rect(world, light, inner, border_radius=4)
    pygame.draw.rect(world, dark, cell, width=2, border_radius=5)
    if style == "clay":
        pygame.draw.line(world, dark, (cell.centerx, cell.top + 3), (cell.centerx, cell.bottom - 3), 3)
        for step in range(cell.top + 14, cell.bottom - 10, 12):
            pygame.draw.line(world, tuple(max(0, c - 18) for c in base), (cell.left + 4, step), (cell.right - 4, step), 1)
    elif style == "glass":
        for x in range(inner.left + 10, inner.right, 22):
            pygame.draw.line(world, (170, 204, 228), (x, inner.top), (x, inner.bottom), 1)
        for y in range(inner.top + 10, inner.bottom, 22):
            pygame.draw.line(world, (170, 204, 228), (inner.left, y), (inner.right, y), 1)
    elif style == "green":
        for _ in range(5):
            pygame.draw.circle(world, (74, 128, 78), (rng.randint(inner.left + 8, inner.right - 8), rng.randint(inner.top + 8, inner.bottom - 8)), 7)
    if style in ("concrete", "glass", "slate"):
        for _ in range(rng.randint(1, 3)):
            unit = pygame.Rect(rng.randint(inner.left + 4, max(inner.left + 5, inner.right - 22)), rng.randint(inner.top + 4, max(inner.top + 5, inner.bottom - 18)), 16, 12)
            pygame.draw.rect(world, (74, 78, 84), unit.move(3, 3), border_radius=2)
            pygame.draw.rect(world, (210, 214, 218), unit, border_radius=2)
            pygame.draw.circle(world, (120, 124, 130), unit.center, 4)


def _bike_symbol(pygame, world, x: int, y: int, horizontal: bool) -> None:
    """A small white wheel pair painted in the rider's lane."""
    for k in (-9, 9):
        centre = (x + k, y) if horizontal else (x, y + k)
        pygame.draw.circle(world, (226, 236, 230), centre, 5, 2)
    a, b = ((x - 9, y), (x + 9, y)) if horizontal else ((x, y - 9), (x, y + 9))
    pygame.draw.line(world, (226, 236, 230), a, b, 2)


def _hline_markings(pygame, world, y: int, artery: bool) -> None:
    half = (ARTERY_W if artery else ROAD_W) // 2
    edge = half - BIKE_LANE
    for side in (-1, 1):
        pygame.draw.line(world, (240, 240, 236), (0, y + side * edge), (WORLD_W, y + side * edge), 2)
        for x in range(180, WORLD_W, 360):
            _bike_symbol(pygame, world, x, y + side * (half - BIKE_LANE // 2), True)
    if artery:
        pygame.draw.line(world, (236, 196, 60), (0, y - 3), (WORLD_W, y - 3), 3)
        pygame.draw.line(world, (236, 196, 60), (0, y + 3), (WORLD_W, y + 3), 3)
    else:
        for x in range(0, WORLD_W, 54):
            pygame.draw.line(world, (236, 196, 60), (x, y), (x + 28, y), 3)


def _vline_markings(pygame, world, x: int) -> None:
    half = ROAD_W // 2
    edge = half - BIKE_LANE
    for side in (-1, 1):
        pygame.draw.line(world, (240, 240, 236), (x + side * edge, 0), (x + side * edge, WORLD_H), 2)
        for y in range(120, WORLD_H, 300):
            _bike_symbol(pygame, world, x + side * (half - BIKE_LANE // 2), y, False)
    for y in range(0, WORLD_H, 54):
        pygame.draw.line(world, (236, 196, 60), (x, y), (x, y + 28), 3)


def _crossings(pygame, world, x: int, y: int, half_v: int, half_h: int) -> None:
    white = (244, 244, 240)
    for k in range(-half_v + 6, half_v - 4, 14):
        pygame.draw.rect(world, white, pygame.Rect(x + k, y - half_h - 30, 8, 24))
        pygame.draw.rect(world, white, pygame.Rect(x + k, y + half_h + 6, 8, 24))
    for k in range(-half_h + 6, half_h - 4, 14):
        pygame.draw.rect(world, white, pygame.Rect(x - half_v - 30, y + k, 24, 8))
        pygame.draw.rect(world, white, pygame.Rect(x + half_v + 6, y + k, 24, 8))


def _tree_row(pygame, world, col: int, y: int, row: int, xs: list[int], rng) -> None:
    if col >= COLS - 1:
        return
    x0, x1 = xs[col] + 80, xs[col + 1] - 80
    half = road_half(row)
    for x in range(x0, x1, 96):
        for side in (-1, 1):
            cy = y + side * (half + 6)
            pygame.draw.circle(world, (58, 92, 56), (x + 5, cy + 6), 11)
            pygame.draw.circle(world, (84, 138, 78), (x, cy), 11)
            pygame.draw.circle(world, (118, 170, 104), (x - 3, cy - 3), 5)


def _landmark(pygame, world, kind: str, rect, fonts) -> None:
    x, y, w, h = rect
    box = pygame.Rect(x, y, w, h)
    if kind == "pharmacy":
        pygame.draw.rect(world, (244, 247, 245), box, border_radius=8)
        pygame.draw.rect(world, (14, 120, 80), box, width=6, border_radius=8)
        cx, cy = box.center
        pygame.draw.rect(world, (22, 163, 74), pygame.Rect(cx - 8, cy - 28, 16, 56), border_radius=3)
        pygame.draw.rect(world, (22, 163, 74), pygame.Rect(cx - 28, cy - 8, 56, 16), border_radius=3)
    elif kind == "home":
        pygame.draw.rect(world, (196, 106, 76), box, border_radius=6)
        pygame.draw.rect(world, (170, 84, 60), pygame.Rect(box.centerx, box.top, w // 2, h), border_top_right_radius=6, border_bottom_right_radius=6)
        pygame.draw.line(world, (232, 196, 170), (box.centerx, box.top + 2), (box.centerx, box.bottom - 2), 3)
        pygame.draw.rect(world, (240, 236, 228), box, width=3, border_radius=6)
    elif kind == "office":
        pygame.draw.rect(world, (84, 132, 178), box, border_radius=6)
        for gx in range(box.left + 12, box.right - 6, 20):
            pygame.draw.line(world, (182, 214, 238), (gx, box.top + 4), (gx, box.bottom - 4), 2)
        for gy in range(box.top + 12, box.bottom - 6, 20):
            pygame.draw.line(world, (182, 214, 238), (box.left + 4, gy), (box.right - 4, gy), 2)
        pygame.draw.rect(world, (236, 242, 248), box, width=4, border_radius=6)
    else:
        pygame.draw.rect(world, (232, 218, 190), box, border_radius=8)
        pygame.draw.rect(world, (176, 140, 90), box, width=5, border_radius=8)
        pygame.draw.circle(world, (176, 140, 90), box.center, 26)
        text = fonts(30, True).render("H", True, (255, 250, 238))
        world.blit(text, text.get_rect(center=box.center))
