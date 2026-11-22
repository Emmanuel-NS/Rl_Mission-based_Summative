"""Pygame view of one FARUMASI dispatch shift.

The window is the view used in play.py: three Kigali districts, six partner
pharmacies, the request at the front of the queue, and the running return.
"""

from __future__ import annotations

import os

import numpy as np

from environment.custom_env import (
    AMOXICILLIN,
    MED_NAMES,
    METFORMIN,
    N_PHARMACIES,
    ORS,
    PHARMACIES,
    FarumasiShiftEnv,
)

WIDTH = 1200
HEIGHT = 720
BG = (245, 242, 234)
INK = (28, 32, 28)
GREEN = (18, 84, 66)
CARD = (255, 252, 247)
MUTED = (90, 96, 90)
AMBER = (180, 83, 9)
TEAL = (14, 116, 144)
LEAF = (22, 101, 52)
ALERT = (153, 27, 27)
LINE = (220, 214, 204)


class DispatchRenderer:
    def __init__(self) -> None:
        self._pygame = None
        self.screen = None
        self.clock = None
        self.font = None
        self.small = None
        self.bold = None

    def render(self, env: FarumasiShiftEnv, human: bool):
        pygame = self._ensure(human)
        self._draw(env)
        if human:
            pygame.display.flip()
            self.clock.tick(env.metadata["render_fps"])
            return None
        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    def close(self) -> None:
        if self._pygame is not None and self.screen is not None:
            self._pygame.display.quit()
            self.screen = None

    def _ensure(self, human: bool):
        if not human:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        if self._pygame is None:
            import pygame

            pygame.init()
            pygame.font.init()
            self._pygame = pygame
            self.font = pygame.font.SysFont("segoeui", 18)
            self.small = pygame.font.SysFont("segoeui", 14)
            self.bold = pygame.font.SysFont("segoeui", 22, bold=True)
            self.clock = pygame.time.Clock()
        if self.screen is None:
            flags = 0 if human else self._pygame.HIDDEN
            self.screen = self._pygame.display.set_mode((WIDTH, HEIGHT), flags)
            self._pygame.display.set_caption("FARUMASI shift dispatcher")
        return self._pygame

    def _draw(self, env: FarumasiShiftEnv) -> None:
        pygame = self._pygame
        self.screen.fill(BG)
        header = pygame.Rect(0, 0, WIDTH, 64)
        pygame.draw.rect(self.screen, GREEN, header)
        self._text("FARUMASI", 24, 18, (255, 255, 255), self.bold)
        self._text("Kigali partner shift  ·  metformin is scarce", 160, 22, (214, 232, 224), self.font)

        districts = ("Nyarugenge", "Gasabo", "Kicukiro")
        for column, district in enumerate(districts):
            left = 24 + column * 250
            self._text(district, left, 80, GREEN, self.bold)
            pharmacies = [pharmacy for pharmacy in PHARMACIES if pharmacy.district_id == column]
            for row, pharmacy in enumerate(pharmacies):
                self._pharmacy_card(env, pharmacy, left, 112 + row * 210)

        panel = pygame.Rect(780, 80, 396, 616)
        pygame.draw.rect(self.screen, CARD, panel, border_radius=12)
        pygame.draw.rect(self.screen, LINE, panel, width=1, border_radius=12)
        self._text("Current request", 800, 96, GREEN, self.bold)
        order = env.queue[0] if env.queue else None
        if order is None:
            self._text("Shift complete", 800, 132, INK, self.font)
        else:
            kind = "urgent" if order.urgency >= 0.8 else "routine"
            district = ("Nyarugenge", "Gasabo", "Kicukiro")[order.district]
            self._text(f"#{order.order_id}  {MED_NAMES[order.med]}", 800, 132, INK, self.font)
            self._text(f"{kind}   ·   {district}", 800, 158, MUTED, self.small)
            self._text(f"deferrals {order.deferrals}    strikes {order.strikes}", 800, 180, MUTED, self.small)
        self._text(f"Last action   {env.last_event}", 800, 214, INK, self.small)
        self._text(f"Step reward   {env.last_reward:+.2f}", 800, 236, INK, self.small)
        self._text(f"Shift return  {env.return_so_far:+.2f}", 800, 258, INK, self.font)
        filled = f"{env.urgent_met_filled}/{env.urgent_met_total}"
        self._text(f"Urgent metformin filled   {filled}", 800, 286, AMBER, self.small)
        restock = "arrived" if env.restocked else "pending"
        self._text(f"Mid-shift metformin drop   {restock}", 800, 308, MUTED, self.small)
        if env.closed_pharmacy >= 0:
            self._text(f"Closed today   {PHARMACIES[env.closed_pharmacy].name}", 800, 330, ALERT, self.small)

        self._text("Queue", 800, 368, GREEN, self.bold)
        for row, line in enumerate(env.queue_preview(7)):
            self._text(line, 800, 398 + row * 22, INK, self.small)
        self._text("Objective: fill urgent metformin without emptying the shelf early.", 800, 640, MUTED, self.small)

    def _pharmacy_card(self, env: FarumasiShiftEnv, pharmacy, left: int, top: int) -> None:
        pygame = self._pygame
        rect = pygame.Rect(left, top, 230, 190)
        chosen = env.last_action == pharmacy.pharmacy_id and env.last_event == "dispensed"
        border = AMBER if chosen else LINE
        pygame.draw.rect(self.screen, CARD, rect, border_radius=10)
        pygame.draw.rect(self.screen, border, rect, width=2 if chosen else 1, border_radius=10)
        self._text(pharmacy.name, left + 12, top + 10, INK, self.small)
        status = "open" if env.open_mask[pharmacy.pharmacy_id] else "closed"
        status_color = LEAF if status == "open" else ALERT
        self._text(status, left + 12, top + 32, status_color, self.small)
        self._text(f"price ×{pharmacy.price_ratio:.2f}", left + 120, top + 32, MUTED, self.small)
        stocks = (
            (METFORMIN, "Metformin", AMBER, env.stock[pharmacy.pharmacy_id, METFORMIN], 6),
            (AMOXICILLIN, "Amoxicillin", TEAL, env.stock[pharmacy.pharmacy_id, AMOXICILLIN], 8),
            (ORS, "ORS", LEAF, env.stock[pharmacy.pharmacy_id, ORS], 8),
        )
        for row, (med, label, color, amount, cap) in enumerate(stocks):
            y = top + 64 + row * 36
            self._text(label, left + 12, y, MUTED, self.small)
            bar = pygame.Rect(left + 100, y + 4, 110, 10)
            pygame.draw.rect(self.screen, LINE, bar, border_radius=4)
            width = int(110 * min(1.0, amount / cap))
            if width > 0:
                pygame.draw.rect(self.screen, color, pygame.Rect(left + 100, y + 4, width, 10), border_radius=4)
            expiry = "  exp" if env.near[pharmacy.pharmacy_id, med] > 0 else ""
            self._text(str(int(amount)) + expiry, left + 12, y + 16, INK, self.small)

    def _text(self, text: str, x: int, y: int, color: tuple[int, int, int], font) -> None:
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))


def save_environment_png(path: str, seed: int = 0) -> str:
    """Write one frame of the shift so the report has an environment visual."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = FarumasiShiftEnv(render_mode="rgb_array")
    env.reset(seed=seed)
    env.step(2)
    frame = env.render()
    env.close()
    import pygame

    pygame.image.save(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), path)
    return path
