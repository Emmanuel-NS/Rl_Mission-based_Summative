"""Top-down Kigali search map for the FARUMASI patient."""

from __future__ import annotations

import os

import numpy as np

from environment.custom_env import (
    COLS,
    DISTRICT_NAMES,
    KIOSKS,
    MED_NAMES,
    PARTNERS,
    PUBLIC,
    ROWS,
    FarumasiShiftEnv,
)

WIDTH, HEIGHT = 1280, 760
MAP_LEFT, MAP_TOP = 28, 78
MAP_W, MAP_H = 860, 640

INK = (28, 38, 34)
PAPER = (244, 239, 228)
GREEN = (15, 118, 110)
GREEN_DARK = (12, 74, 68)
AMBER = (180, 83, 9)
RED = (153, 27, 27)
BLUE = (29, 78, 216)
CARD = (255, 252, 247)
LINE = (214, 206, 192)
PATIENT = (217, 119, 6)
DISTRICT_TINT = ((228, 236, 226), (226, 234, 244), (246, 236, 220))


class MapRenderer:
    def __init__(self, human: bool) -> None:
        self.human = human
        self._pygame = None
        self.screen = None
        self.font = None
        self.small = None
        self.bold = None
        self.clock = None

    def close(self) -> None:
        if self._pygame is not None and self.screen is not None:
            self._pygame.display.quit()
            self.screen = None

    def draw(self, env: FarumasiShiftEnv):
        pygame = self._ensure()
        self._scene(env)
        if self.human:
            pygame.display.flip()
            self.clock.tick(env.metadata["render_fps"])
            return None
        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    def _ensure(self):
        if not self.human:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        if self._pygame is None:
            import pygame

            pygame.init()
            pygame.font.init()
            self._pygame = pygame
            self.font = pygame.font.SysFont("segoeui", 18)
            self.small = pygame.font.SysFont("segoeui", 14)
            self.bold = pygame.font.SysFont("segoeui", 24, bold=True)
            self.clock = pygame.time.Clock()
            flags = 0 if self.human else pygame.HIDDEN
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
            pygame.display.set_caption("FARUMASI — find a genuine pack")
        return self._pygame

    def _scene(self, env: FarumasiShiftEnv) -> None:
        pygame = self._pygame
        self.screen.fill(PAPER)
        header = pygame.Rect(0, 0, WIDTH, 64)
        pygame.draw.rect(self.screen, GREEN_DARK, header)
        self._text("FARUMASI", 28, 10, (255, 255, 255), self.bold)
        self._text("Live pharmacy search  ·  Kigali", 28, 38, (214, 235, 228), self.small)
        self._map(env)
        self._panel(env)

    def _map(self, env: FarumasiShiftEnv) -> None:
        pygame = self._pygame
        tile_w = MAP_W / COLS
        tile_h = MAP_H / ROWS
        for col in range(COLS):
            district = 0 if col < 5 else 1 if col < 10 else 2
            rect = pygame.Rect(MAP_LEFT + col * tile_w, MAP_TOP, tile_w + 1, MAP_H)
            pygame.draw.rect(self.screen, DISTRICT_TINT[district], rect)
        for col in range(COLS + 1):
            x = MAP_LEFT + col * tile_w
            pygame.draw.line(self.screen, (232, 226, 214), (x, MAP_TOP), (x, MAP_TOP + MAP_H), 1)
        for row in range(ROWS + 1):
            y = MAP_TOP + row * tile_h
            pygame.draw.line(self.screen, (232, 226, 214), (MAP_LEFT, y), (MAP_LEFT + MAP_W, y), 1)
        for col in (7, 14):
            x = MAP_LEFT + col * tile_w
            pygame.draw.line(self.screen, LINE, (x, MAP_TOP), (x, MAP_TOP + MAP_H), 4)
        for index, name in enumerate(DISTRICT_NAMES):
            self._text(name, MAP_LEFT + int((0, 7, 14)[index] * tile_w) + 8, MAP_TOP + tile_h + 6, GREEN_DARK, self.font)
        home_x = MAP_LEFT + (env.home[0] + 0.5) * tile_w
        home_y = MAP_TOP + (env.home[1] + 0.5) * tile_h
        pygame.draw.circle(self.screen, (120, 113, 100), (int(home_x), int(home_y)), 7)
        self._text("home", int(home_x) - 16, int(home_y) + 10, INK, self.small)
        for site in (*PARTNERS, *KIOSKS, PUBLIC):
            self._building(env, site, tile_w, tile_h)
        self._patient(env, tile_w, tile_h)

    def _building(self, env: FarumasiShiftEnv, site, tile_w: float, tile_h: float) -> None:
        pygame = self._pygame
        cx = MAP_LEFT + (site.x + 0.5) * tile_w
        cy = MAP_TOP + (site.y + 0.5) * tile_h
        rect = pygame.Rect(0, 0, 46, 34)
        rect.center = (cx, cy - 6)
        if site.kind == "partner":
            known = bool(env.partner_known[site.site_id])
            stock = int(env.partner_stock[site.site_id]) if known else None
            colour = GREEN if stock else (120, 140, 136) if known else GREEN
            pygame.draw.rect(self.screen, colour, rect, border_radius=6)
            pygame.draw.line(self.screen, CARD, (cx, cy - 18), (cx, cy + 2), 3)
            pygame.draw.line(self.screen, CARD, (cx - 8, cy - 8), (cx + 8, cy - 8), 3)
            label = site.name.split()[0]
            if known:
                label = f"{label} {stock}"
        elif site.kind == "kiosk":
            pygame.draw.rect(self.screen, AMBER, rect, border_radius=4)
            if env.kiosk_known[site.site_id]:
                pygame.draw.rect(self.screen, RED, rect, width=3, border_radius=4)
                label = "unsafe"
            else:
                label = "cheap"
        else:
            pygame.draw.rect(self.screen, BLUE, rect, border_radius=6)
            label = "clinic"
        self._text(label, rect.x - 4, rect.bottom + 1, INK, self.small)

    def _patient(self, env: FarumasiShiftEnv, tile_w: float, tile_h: float) -> None:
        pygame = self._pygame
        cx = int(MAP_LEFT + (env.px + 0.5) * tile_w)
        cy = int(MAP_TOP + (env.py + 0.5) * tile_h)
        pygame.draw.circle(self.screen, (255, 244, 214), (cx, cy), 14)
        pygame.draw.circle(self.screen, PATIENT, (cx, cy), 14, 3)
        pygame.draw.circle(self.screen, INK, (cx, cy - 3), 3)

    def _panel(self, env: FarumasiShiftEnv) -> None:
        pygame = self._pygame
        panel = pygame.Rect(910, 78, 346, 640)
        pygame.draw.rect(self.screen, CARD, panel, border_radius=16)
        pygame.draw.rect(self.screen, LINE, panel, width=1, border_radius=16)
        self._text("Prescription", 932, 94, INK, self.bold)
        self._text(MED_NAMES[env.med], 932, 126, GREEN_DARK, self.font)
        self._bar(932, 160, 300, env.health, RED if env.health < 0.35 else GREEN)
        self._text(f"Health {env.health:.0%}", 932, 176, INK, self.small)
        phone = pygame.Rect(932, 214, 300, 280)
        pygame.draw.rect(self.screen, (22, 32, 30), phone, border_radius=18)
        if env.app_open:
            self._text("FARUMASI live", 948, 228, (167, 243, 208), self.font)
            y = 258
            for site in PARTNERS:
                stock = int(env.partner_stock[site.site_id])
                open_ = bool(env.partner_open[site.site_id])
                state = f"{stock} in stock" if open_ and stock else "unavailable"
                self._text(f"{site.name[:22]}  {state}", 948, y, (236, 253, 245), self.small)
                y += 28
            self._text("Stalls marked unlicensed", 948, 450, (254, 202, 202), self.small)
        else:
            self._text("FARUMASI", 948, 250, (167, 243, 208), self.bold)
            self._text("App is closed.", 948, 292, (214, 235, 228), self.font)
            self._text("Open it to see which licensed", 948, 324, (214, 235, 228), self.small)
            self._text("pharmacies have this medicine.", 948, 346, (214, 235, 228), self.small)
        self._text(env.last_event.replace("_", " "), 932, 520, INK, self.font)
        self._text(f"Step reward {env.last_reward:+.2f}    search {env.return_so_far:+.2f}", 932, 552, INK, self.small)
        self._text("Avoid a cheap stall. A licensed pack", 932, 600, GREEN_DARK, self.small)
        self._text("is the only safe way to finish.", 932, 622, GREEN_DARK, self.small)
        self._text(f"Step {env.steps}", 932, 660, INK, self.small)

    def _bar(self, x: int, y: int, width: int, fraction: float, colour: tuple[int, int, int]) -> None:
        pygame = self._pygame
        pygame.draw.rect(self.screen, LINE, pygame.Rect(x, y, width, 12), border_radius=6)
        inner = max(0, int(width * fraction))
        if inner:
            pygame.draw.rect(self.screen, colour, pygame.Rect(x, y, inner, 12), border_radius=6)

    def _text(self, text: str, x: int, y: int, colour: tuple[int, int, int], font) -> None:
        surface = font.render(text, True, colour)
        self.screen.blit(surface, (x, y))


def save_environment_png(path: str, seed: int = 0) -> str:
    """Write one frame of the search so the report has an environment visual."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = FarumasiShiftEnv(render_mode="rgb_array")
    env.reset(seed=seed)
    env.step(4)
    env.step(1)
    frame = env.render()
    env.close()
    import pygame

    pygame.image.save(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), path)
    return path
