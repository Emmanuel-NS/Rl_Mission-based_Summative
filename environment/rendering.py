"""FARUMASI rider: a delivery game in Kigali.

The rider moves through three kinds of scene. The city is a top-down map where
she chooses roads through live traffic and rain. The pharmacy and the patient's
building are side-on scenes where she walks in, knocks, checks the prescription,
hands the box over, and scans the QR code. Art is in assets/sprites.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np

from environment import city_map
from environment.custom_env import (
    ARTERY_ROW,
    CARRY,
    COLLECT,
    COLS,
    COVER,
    DEADLINE,
    DESTINATIONS,
    ENTER,
    HAND,
    KNOCK,
    MED_NAMES,
    N_EDGES,
    P_BUILDING,
    P_ENTRANCE,
    P_INSIDE,
    P_OUTSIDE,
    P_PATIENT,
    P_RIDING,
    PARK,
    PHARMACY_NAME,
    PHARMACY_NODE,
    REMOUNT,
    RETURN_MEDS,
    ROWS,
    SCAN,
    VERIFY,
    WAIT,
    FarumasiRiderEnv,
    edge_id,
    edge_nodes,
    neighbour,
)

WIDTH, HEIGHT = 1280, 760
VIEW_H = 540
FPS = 30
SPRITES = Path(__file__).resolve().parent.parent / "assets" / "sprites"
SIDEWALK_TOP = 404
CURB_Y = 470
FEET_Y = 462
BUILDING_X = 860
BIKE_X = 250
START_X = 440
LOBBY_RECEPTION_X = 930
LOBBY_PATIENT_X = 600
INT_COUNTER_X = 650
INT_ENTER_X = 240
INT_FEET_Y = 508

INK = (24, 32, 30)
GREEN = (15, 118, 110)
GREEN_DARK = (10, 61, 56)
GREEN_SOFT = (22, 163, 74)
AMBER = (217, 119, 6)
RED = (185, 28, 28)
BLUE = (37, 99, 235)
CARD = (252, 250, 246)
MUTED = (110, 120, 116)
TRAFFIC_COLOURS = ((74, 190, 112), (250, 204, 21), (249, 115, 22), (220, 38, 38))

FAIL_EVENTS = {"rx_violation", "medicine_ruined", "order_cancelled", "delivery_abandoned"}

EVENT_TEXT = {
    "mission_started": "New order from Kigali City Pharmacy",
    "entered_pharmacy": "Walked into the pharmacy",
    "order_collected": "Order scanned out and boxed",
    "rode": "Riding through Kigali",
    "covered_box": "Rain cover fitted on the box",
    "waited": "Waiting",
    "parked": "Parked. Box carried in hand",
    "gate_opened": "Knocked. The gate is open",
    "checked_in": "Checked in at reception",
    "reached_patient": "Standing in front of the patient",
    "rx_ok": "Prescription matches the order",
    "rx_wrong": "Prescription does not match",
    "handed_over": "Box is in the patient's hands",
    "delivered": "QR scanned. Delivery confirmed",
    "returning": "Riding the medicine back",
    "returned_to_pharmacy": "Medicine returned to the pharmacist",
    "rx_violation": "Medicine handed over without a valid prescription",
    "medicine_ruined": "Rain ruined the medicine",
    "order_cancelled": "The patient cancelled: too late",
    "delivery_abandoned": "Delivery abandoned",
    "invalid": "That is not possible here",
}


def _ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


class DeliveryRenderer:
    def __init__(self, human: bool) -> None:
        self.human = human
        self._pygame = None
        self.screen = None
        self.clock = None
        self._fonts: dict[tuple[int, bool], object] = {}
        self.tick = 0
        self.phase_walk = 0
        self.fade = 0.0
        self.initialised = False
        self.scene = "ext"
        self.building = "pharmacy"
        self.rider_x = START_X
        self.rider_face = 1
        self.walking = False
        self.bike_x: float | None = BIKE_X
        self.patient_x: float | None = None
        self.gate_open = False
        self.city_pos = (0.0, 0.0)
        self.heading = -90.0
        self.rider_off = (0.0, 0.0)
        self.cars: list[dict] = []
        self._traffic_key = b""
        self._rain = []
        self._car_cache: dict[tuple[int, int], object] = {}

    # ------------------------------------------------------------- lifecycle

    def close(self) -> None:
        if self._pygame is not None and self.screen is not None:
            self._pygame.display.quit()
            self.screen = None

    def _ensure(self):
        if not self.human:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        if self._pygame is None:
            import pygame

            pygame.init()
            pygame.font.init()
            self._pygame = pygame
            self.clock = pygame.time.Clock()
            flags = 0 if self.human else pygame.HIDDEN
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
            pygame.display.set_caption("FARUMASI Rider: deliver safely, deliver fast")
            self._load_art()
        return self._pygame

    def _font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = self._pygame.font.SysFont("segoeui", size, bold=bold)
        return self._fonts[key]

    def _load_art(self) -> None:
        pygame = self._pygame

        def load(name: str, height: int | None = None, width: int | None = None):
            image = pygame.image.load(str(SPRITES / f"{name}.png")).convert_alpha()
            w, h = image.get_size()
            if height is not None:
                width = int(w * height / h)
            elif width is not None:
                height = int(h * width / w)
            return pygame.transform.smoothscale(image, (width, height))

        self.building_art = {
            "pharmacy": load("pharmacy", height=300),
            "home": load("home", width=560),
            "office": load("office", height=336),
            "hotel": load("hotel", height=336),
        }
        self.tree = load("tree", height=250)
        self.bike_side = load("bike_side", width=250)
        self.rider_ext = [load(f"rider_{i}", height=196) for i in range(4)]
        self.rider_int = [load(f"rider_{i}", height=262) for i in range(4)]
        self.patient_ext = [load(f"patient_{i}", height=186) for i in range(4)]
        self.patient_int = [load(f"patient_{i}", height=246) for i in range(4)]
        self.bike_top = load("bike_top", height=108)
        covered = self.bike_top.copy()
        w, h = covered.get_size()
        tarp = pygame.Rect(int(w * 0.1), int(h * 0.69), int(w * 0.8), int(h * 0.31))
        pygame.draw.rect(covered, (24, 58, 112), tarp, border_radius=5)
        pygame.draw.line(covered, (92, 140, 204), (tarp.left + 4, tarp.top + 5), (tarp.right - 4, tarp.top + 5), 2)
        pygame.draw.line(covered, (92, 140, 204), (tarp.left + 4, tarp.centery), (tarp.right - 4, tarp.centery), 1)
        self.bike_top_covered = covered
        self.car_art = [load(f"car_{i}", height=64) for i in range(4)]

        def cover_image(name: str, height: int):
            image = pygame.image.load(str(SPRITES / f"{name}.jpg")).convert()
            w, h = image.get_size()
            return pygame.transform.smoothscale(image, (int(w * height / h), height))

        raw = pygame.image.load(str(SPRITES / "backdrop.jpg")).convert()
        w, h = raw.get_size()
        top = raw.subsurface((0, 0, w, int(h * 0.88))).copy()
        scaled = pygame.transform.smoothscale(top, (int(top.get_width() * SIDEWALK_TOP / top.get_height()), SIDEWALK_TOP))
        sky = pygame.Surface((scaled.get_width() * 2, SIDEWALK_TOP))
        sky.blit(scaled, (0, 0))
        sky.blit(pygame.transform.flip(scaled, True, False), (scaled.get_width(), 0))
        self.sky = sky.subsurface((0, 0, WIDTH, SIDEWALK_TOP)).copy()
        self.interior = {
            "pharmacy": cover_image("pharmacy_interior", 720),
            "lobby": cover_image("lobby_interior", 720),
        }
        self.world = city_map.build_city(pygame, self._font).convert()
        self.landmarks = city_map.landmarks()
        self.overlay = pygame.Surface((WIDTH, VIEW_H), pygame.SRCALPHA)
        rng = random.Random(3)
        self._rain = [(rng.uniform(0, WIDTH), rng.uniform(0, VIEW_H), rng.uniform(14, 26)) for _ in range(150)]
        px, py = city_map.node_px(PHARMACY_NODE)
        self.city_pos = (float(px), float(py))
        self.rider_off = (0.0, 17.0)

    # --------------------------------------------------------------- drawing

    def draw(self, env: FarumasiRiderEnv):
        pygame = self._ensure()
        self._sync_traffic(env)
        if not self.human:
            self._layout(env)
            self._frame(env, popup=self._static_popup(env), t=0.7)
            frame = pygame.surfarray.array3d(self.screen)
            return np.transpose(frame, (1, 0, 2))

        if env.last_action < 0 or not self.initialised:
            self._layout(env)
            self.initialised = True
            self._play(env, [("hold", 96, "mission_start")])
            return None
        self._play(env, self._plan(env))
        return None

    def _present(self) -> None:
        pygame = self._pygame
        pygame.display.flip()
        self.tick += 1
        self.clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)

    # ------------------------------------------------------------- scene plan

    def _layout(self, env: FarumasiRiderEnv) -> None:
        kind = env.dest.kind
        phase = env.phase
        self.gate_open = False
        self.patient_x = None
        self.bike_x = BIKE_X
        self.rider_face = 1
        self.walking = False
        if phase == P_OUTSIDE:
            self.scene, self.building, self.rider_x = "ext", "pharmacy", START_X
        elif phase == P_INSIDE:
            self.scene, self.building, self.rider_x, self.bike_x = "int", "pharmacy", INT_COUNTER_X, None
        elif phase == P_RIDING:
            self.scene = "city"
            px, py = city_map.node_px(env.node)
            self.city_pos = (float(px), float(py))
        elif phase == P_ENTRANCE:
            self.scene, self.building, self.rider_x = "ext", kind, START_X
        elif phase == P_BUILDING:
            if kind == "home":
                self.scene, self.building, self.rider_x = "ext", kind, self._gate_x()
                self.gate_open, self.patient_x = True, self._door_x() + 40
            else:
                self.scene, self.building, self.rider_x, self.bike_x = "int", "lobby", LOBBY_RECEPTION_X, None
        else:
            if kind == "home":
                self.scene, self.building, self.rider_x = "ext", kind, self._door_x() - 130
                self.gate_open, self.patient_x = True, self._door_x() + 40
            else:
                self.scene, self.building, self.rider_x, self.bike_x = "int", "lobby", LOBBY_PATIENT_X - 130, None
                self.patient_x = LOBBY_PATIENT_X

    def _gate_x(self) -> float:
        return BUILDING_X + 0.21 * self.building_art["home"].get_width()

    def _door_x(self) -> float:
        return BUILDING_X + 0.02 * self.building_art["home"].get_width()

    def _plan(self, env: FarumasiRiderEnv) -> list[tuple]:
        action, event, kind = env.last_action, env.last_event, env.dest.kind
        plan: list[tuple] = []
        if event == "invalid":
            return [("hold", 12, "invalid")]
        if action <= 3:
            frames = int(min(100, 18 + 10 * env.last_minutes))
            plan.append(("ride", env.prev_node, env.node, frames))
            plan.append(("hold", 30 if env.last_wet else 12, "wet" if env.last_wet else None))
        elif action == COVER:
            plan.append(("hold", 42, "covered_box"))
        elif action == WAIT:
            plan.append(("hold", 22, "waited"))
        elif action == ENTER:
            plan += [
                ("walk", self.rider_x, BUILDING_X, 44),
                ("fade", {"scene": "int", "building": "pharmacy", "rider_x": INT_ENTER_X, "bike_x": None}),
                ("walk", INT_ENTER_X, INT_COUNTER_X, 32),
                ("hold", 30, "entered_pharmacy"),
            ]
        elif action == COLLECT:
            px, py = city_map.node_px(PHARMACY_NODE)
            plan += [
                ("hold", 60, "order_collected"),
                ("fade", {"scene": "city", "city_pos": (float(px), float(py)), "heading": -90.0, "rider_off": (0.0, 17.0)}),
                ("hold", 16, None),
            ]
        elif action == PARK:
            plan += [
                ("fade", {"scene": "ext", "building": kind, "rider_x": START_X, "bike_x": BIKE_X, "patient_x": None, "gate_open": False, "rider_face": 1}),
                ("hold", 36, "parked"),
            ]
        elif action == KNOCK:
            if kind == "home":
                plan += [
                    ("walk", self.rider_x, self._gate_x(), 42),
                    ("set", {"gate_open": True, "patient_x": self._door_x() + 40}),
                    ("hold", 58, "gate_opened"),
                ]
            else:
                plan += [
                    ("walk", self.rider_x, BUILDING_X, 40),
                    ("fade", {"scene": "int", "building": "lobby", "rider_x": 250, "bike_x": None}),
                    ("walk", 250, LOBBY_RECEPTION_X, 44),
                    ("hold", 56, "checked_in"),
                ]
        elif action == CARRY:
            if kind == "home":
                plan += [("walk", self.rider_x, self._door_x() - 130, 46), ("hold", 26, "reached_patient")]
            else:
                plan += [
                    ("set", {"patient_x": LOBBY_PATIENT_X}),
                    ("walk", self.rider_x, LOBBY_PATIENT_X - 130, 48),
                    ("hold", 26, "reached_patient"),
                ]
        elif action == VERIFY:
            plan.append(("hold", 66, event))
        elif action == HAND:
            plan.append(("hold", 50, "handed_over"))
        elif action == SCAN:
            plan += [("hold", 84, "scan"), ("hold", 100, "mission_passed")]
        elif action == REMOUNT:
            px, py = city_map.node_px(env.node)
            plan += [
                ("walk", self.rider_x, 120, 42, -1),
                ("fade", {"scene": "city", "city_pos": (float(px), float(py)), "heading": 90.0, "rider_off": (0.0, -17.0)}),
                ("hold", 22, "returning"),
            ]
        elif action == RETURN_MEDS:
            plan += [
                ("fade", {"scene": "int", "building": "pharmacy", "rider_x": INT_ENTER_X, "bike_x": None, "rider_face": 1}),
                ("walk", INT_ENTER_X, INT_COUNTER_X, 32),
                ("hold", 72, "returned_to_pharmacy"),
                ("hold", 100, "mission_returned"),
            ]
        if event in FAIL_EVENTS:
            plan.append(("hold", 110, "fail"))
        return plan

    def _play(self, env: FarumasiRiderEnv, plan: list[tuple]) -> None:
        for segment in plan:
            kind = segment[0]
            if kind == "set":
                self.__dict__.update(segment[1])
            elif kind == "hold":
                _, frames, popup = segment
                self.walking = False
                for frame in range(frames):
                    self._frame(env, popup=popup, t=frame / max(1, frames))
                    self._present()
            elif kind == "walk":
                x0, x1, frames = segment[1], segment[2], segment[3]
                face = segment[4] if len(segment) > 4 else (1 if x1 >= x0 else -1)
                self.rider_face = face
                self.walking = True
                for frame in range(frames):
                    self.rider_x = x0 + (x1 - x0) * _ease(frame / max(1, frames - 1))
                    self.phase_walk += 1
                    self._frame(env, popup=None, t=0.0)
                    self._present()
                self.rider_x = x1
                self.walking = False
            elif kind == "fade":
                self.walking = False
                for frame in range(10):
                    self.fade = (frame + 1) / 10
                    self._frame(env, popup=None, t=0.0)
                    self._present()
                self.__dict__.update(segment[1])
                for frame in range(10):
                    self.fade = 1.0 - (frame + 1) / 10
                    self._frame(env, popup=None, t=0.0)
                    self._present()
                self.fade = 0.0
            elif kind == "ride":
                self._ride(env, segment[1], segment[2], segment[3])

    def _ride(self, env: FarumasiRiderEnv, a: int, b: int, frames: int) -> None:
        ax, ay = city_map.node_px(a)
        bx, by = city_map.node_px(b)
        dx, dy = bx - ax, by - ay
        heading = {(0, -1): 0.0, (1, 0): -90.0, (0, 1): 180.0, (-1, 0): 90.0}[(int(np.sign(dx)), int(np.sign(dy)))]
        offset = {0.0: (17.0, 0.0), -90.0: (0.0, 17.0), 180.0: (-17.0, 0.0), 90.0: (0.0, -17.0)}[heading]
        old_heading, old_offset = self.heading, self.rider_off
        turn = ((heading - old_heading + 180.0) % 360.0) - 180.0
        self.walking = False
        for frame in range(frames):
            progress = frame / max(1, frames - 1)
            lead = _ease(min(1.0, progress * 5.0))
            self.heading = old_heading + turn * lead
            self.rider_off = (
                old_offset[0] + (offset[0] - old_offset[0]) * lead,
                old_offset[1] + (offset[1] - old_offset[1]) * lead,
            )
            move = _ease(progress)
            self.city_pos = (ax + dx * move, ay + dy * move)
            self._frame(env, popup=None, t=0.0)
            self._present()
        self.heading, self.rider_off = heading, offset
        self.city_pos = (float(bx), float(by))

    def _static_popup(self, env: FarumasiRiderEnv):
        event = env.last_event
        if env.last_action < 0:
            return "mission_start"
        if event in FAIL_EVENTS:
            return "fail"
        if event == "delivered":
            return "scan"
        if event == "returned_to_pharmacy":
            return "returned_to_pharmacy"
        return event

    # ---------------------------------------------------------------- frames

    def _frame(self, env: FarumasiRiderEnv, popup, t: float) -> None:
        if self.scene == "city":
            self._city(env)
        else:
            self._side_scene(env)
        if self.scene != "int":
            if env.raining:
                self._rain_layer(0.55)
            elif env.rain_soon:
                self._tint(36)
        self._top_bar(env)
        if popup is not None:
            self._popup(env, popup, t)
        self._hud(env)
        if self.fade > 0.0:
            veil = self._pygame.Surface((WIDTH, VIEW_H))
            veil.set_alpha(int(255 * self.fade))
            self.screen.blit(veil, (0, 0))

    def _tint(self, alpha: int) -> None:
        veil = self._pygame.Surface((WIDTH, VIEW_H), self._pygame.SRCALPHA)
        veil.fill((20, 30, 48, alpha))
        self.screen.blit(veil, (0, 0))

    def _rain_layer(self, strength: float) -> None:
        pygame = self._pygame
        self._tint(int(70 * strength))
        layer = self.overlay
        layer.fill((0, 0, 0, 0))
        for x, y, length in self._rain:
            yy = (y + self.tick * 20) % VIEW_H
            xx = (x - self.tick * 4) % WIDTH
            pygame.draw.line(layer, (206, 220, 240, 150), (xx, yy), (xx - 4, yy + length), 1)
        self.screen.blit(layer, (0, 0))

    # -------------------------------------------------------------- city view

    def _sync_traffic(self, env: FarumasiRiderEnv) -> None:
        key = env.traffic.tobytes()
        if key == self._traffic_key or self._pygame is None:
            return
        self._traffic_key = key
        rng = random.Random(77)
        cars = []
        for edge in range(N_EDGES):
            level = int(env.traffic[edge])
            a, b = edge_nodes(edge)
            ax, ay = city_map.node_px(a)
            bx, by = city_map.node_px(b)
            length = math.hypot(bx - ax, by - ay) - 150
            per_lane = (1, 2, 4, 4)[level]
            speed = (4.2, 2.4, 0.8, 0.0)[level]
            for lane in (1, -1):
                for k in range(per_lane):
                    cars.append(
                        {
                            "edge": edge,
                            "lane": lane,
                            "s": (k + rng.uniform(0.0, 0.5)) * length / per_lane,
                            "length": length,
                            "speed": speed * rng.uniform(0.9, 1.1),
                            "art": rng.randrange(4),
                        }
                    )
        self.cars = cars

    def _car_sprite(self, art: int, angle: int):
        key = (art, angle)
        if key not in self._car_cache:
            self._car_cache[key] = self._pygame.transform.rotate(self.car_art[art], angle)
        return self._car_cache[key]

    def _city(self, env: FarumasiRiderEnv) -> None:
        pygame = self._pygame
        cx, cy = self.city_pos
        cam_x = int(max(0, min(city_map.WORLD_W - WIDTH, cx - WIDTH * 0.5)))
        cam_y = int(max(0, min(city_map.WORLD_H - VIEW_H, cy - VIEW_H * 0.5)))
        self.screen.blit(self.world, (0, 0), pygame.Rect(cam_x, cam_y, WIDTH, VIEW_H))

        layer = self.overlay
        layer.fill((0, 0, 0, 0))
        for edge in range(N_EDGES):
            level = int(env.traffic[edge])
            a, b = edge_nodes(edge)
            ax, ay = city_map.node_px(a)
            bx, by = city_map.node_px(b)
            horizontal = ay == by
            half = (city_map.road_half(ay_row(ay)) if horizontal else city_map.ROAD_W // 2) - 3
            rect = pygame.Rect(min(ax, bx) - cam_x + 44, min(ay, by) - cam_y - half, abs(bx - ax) - 88, half * 2)
            if not horizontal:
                rect = pygame.Rect(ax - cam_x - half, min(ay, by) - cam_y + 56, half * 2, abs(by - ay) - 112)
            if rect.right < 0 or rect.left > WIDTH or rect.bottom < 0 or rect.top > VIEW_H:
                continue
            if level > 0:
                tint = TRAFFIC_COLOURS[level]
                pygame.draw.rect(layer, (*tint, 74), rect, border_radius=10)
        self.screen.blit(layer, (0, 0))

        for car in self.cars:
            self._draw_car(car, cam_x, cam_y)
        for edge in range(N_EDGES):
            if int(env.traffic[edge]) == 3:
                self._barrier(edge, cam_x, cam_y)

        self._markers(env, cam_x, cam_y)
        self._rider_top(env, cam_x, cam_y)

    def _draw_car(self, car: dict, cam_x: int, cam_y: int) -> None:
        a, b = edge_nodes(car["edge"])
        ax, ay = city_map.node_px(a)
        bx, by = city_map.node_px(b)
        span = max(1.0, car["length"])
        s = (car["s"] + car["speed"] * self.tick) % span
        if car["lane"] < 0:
            s = span - s
        fraction = (s + 75) / (span + 150)
        x = ax + (bx - ax) * fraction
        y = ay + (by - ay) * fraction
        horizontal = ay == by
        if horizontal:
            east = car["lane"] > 0
            y += 20 if east else -20
            angle = -90 if east else 90
        else:
            south = car["lane"] > 0
            x += -20 if south else 20
            angle = 180 if south else 0
        sx, sy = int(x - cam_x), int(y - cam_y)
        if -80 < sx < WIDTH + 80 and -80 < sy < VIEW_H + 80:
            sprite = self._car_sprite(car["art"], angle)
            shade = self._pygame.Surface(sprite.get_size(), self._pygame.SRCALPHA)
            self.screen.blit(sprite, sprite.get_rect(center=(sx, sy)))

    def _barrier(self, edge: int, cam_x: int, cam_y: int) -> None:
        pygame = self._pygame
        a, b = edge_nodes(edge)
        ax, ay = city_map.node_px(a)
        bx, by = city_map.node_px(b)
        x, y = (ax + bx) // 2 - cam_x, (ay + by) // 2 - cam_y
        if not (-100 < x < WIDTH + 100 and -100 < y < VIEW_H + 100):
            return
        horizontal = ay == by
        size = (16, 80) if horizontal else (80, 16)
        rect = pygame.Rect(0, 0, *size)
        rect.center = (x, y)
        pygame.draw.rect(self.screen, (240, 240, 236), rect, border_radius=3)
        steps = 6
        for k in range(0, steps, 2):
            if horizontal:
                part = pygame.Rect(rect.x, rect.y + k * rect.height // steps, rect.width, rect.height // steps)
            else:
                part = pygame.Rect(rect.x + k * rect.width // steps, rect.y, rect.width // steps, rect.height)
            pygame.draw.rect(self.screen, (220, 38, 38), part)
        pygame.draw.polygon(self.screen, (250, 204, 21), [(x, y - 62), (x - 18, y - 30), (x + 18, y - 30)])
        pygame.draw.polygon(self.screen, INK, [(x, y - 62), (x - 18, y - 30), (x + 18, y - 30)], 2)
        self._text("!", x, y - 44, INK, self._font(20, True), center=True)

    def _markers(self, env: FarumasiRiderEnv, cam_x: int, cam_y: int) -> None:
        pygame = self._pygame
        pulse = 4 * math.sin(self.tick * 0.16)
        if env.returning:
            key, colour, title = "pharmacy", GREEN_SOFT, "RETURN HERE"
            target = self.landmarks["pharmacy"]
        else:
            index = DESTINATIONS.index(env.dest)
            key, colour, title = f"dest{index}", (250, 204, 21), "DELIVER HERE"
            target = self.landmarks[key]
        x, y, w, h = target
        cx, cy = x + w // 2 - cam_x, y + h // 2 - cam_y
        on_screen = -60 < cx < WIDTH + 60 and -60 < cy < VIEW_H + 60
        if on_screen:
            ring = pygame.Surface((220, 220), pygame.SRCALPHA)
            pygame.draw.circle(ring, (*colour, 70), (110, 110), int(70 + pulse))
            pygame.draw.circle(ring, (*colour, 220), (110, 110), int(70 + pulse), 5)
            self.screen.blit(ring, (cx - 110, cy - 110))
            label = f"{title}: {env.dest.name if not env.returning else PHARMACY_NAME}"
            width = self._font(15, True).size(label)[0] + 26
            tag = pygame.Rect(0, 0, width, 30)
            tag.midbottom = (cx, cy - 76)
            pygame.draw.rect(self.screen, (20, 28, 26), tag, border_radius=15)
            pygame.draw.rect(self.screen, colour, tag, width=2, border_radius=15)
            self._text(label, tag.centerx, tag.centery, (255, 255, 255), self._font(15, True), center=True)
        else:
            rx, ry = self.city_pos[0] - cam_x, self.city_pos[1] - cam_y
            angle = math.atan2(cy - ry, cx - rx)
            ex = WIDTH / 2 + math.cos(angle) * (WIDTH / 2 - 56)
            ey = VIEW_H / 2 + math.sin(angle) * (VIEW_H / 2 - 56)
            tip = (ex + math.cos(angle) * 26, ey + math.sin(angle) * 26)
            left = (ex + math.cos(angle + 2.4) * 22, ey + math.sin(angle + 2.4) * 22)
            right = (ex + math.cos(angle - 2.4) * 22, ey + math.sin(angle - 2.4) * 22)
            pygame.draw.polygon(self.screen, colour, [tip, left, right])
            pygame.draw.polygon(self.screen, INK, [tip, left, right], 3)
            km = math.hypot(cx - rx, cy - ry) / 200.0
            self._text(f"{km:.1f} km", ex, ey + (-34 if ey > VIEW_H / 2 else 34), (255, 255, 255), self._font(16, True), center=True)

    def _rider_top(self, env: FarumasiRiderEnv, cam_x: int, cam_y: int) -> None:
        pygame = self._pygame
        x = self.city_pos[0] + self.rider_off[0] - cam_x
        y = self.city_pos[1] + self.rider_off[1] - cam_y
        art = self.bike_top_covered if env.covered else self.bike_top
        sprite = pygame.transform.rotate(art, self.heading)
        shadow = pygame.Surface((70, 70), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 90), pygame.Rect(18, 6, 36, 60))
        self.screen.blit(pygame.transform.rotate(shadow, self.heading), (x - 35 + 8, y - 35 + 8))
        if env.raining:
            beam = pygame.Surface((160, 160), pygame.SRCALPHA)
            pygame.draw.polygon(beam, (255, 244, 190, 70), [(80, 80), (30, 0), (130, 0)])
            self.screen.blit(pygame.transform.rotate(beam, self.heading), (x - 80, y - 80))
        self.screen.blit(sprite, sprite.get_rect(center=(int(x), int(y))))
        if env.last_wet and not self.walking:
            self._text("wet!", x + 26, y - 40, RED, self._font(18, True))

    # --------------------------------------------------------------- side scene

    def _side_scene(self, env: FarumasiRiderEnv) -> None:
        pygame = self._pygame
        if self.scene == "int":
            image = self.interior["pharmacy" if self.building == "pharmacy" else "lobby"]
            self.screen.blit(image, (-(image.get_width() - WIDTH) // 2, -110))
            self._walker(self.rider_int, self.rider_x, INT_FEET_Y)
            if self.patient_x is not None:
                self._patient(self.patient_int, self.patient_x, INT_FEET_Y)
            return
        self.screen.blit(self.sky, (0, 0))
        pygame.draw.rect(self.screen, (194, 188, 176), pygame.Rect(0, SIDEWALK_TOP, WIDTH, CURB_Y - SIDEWALK_TOP))
        pygame.draw.rect(self.screen, (158, 152, 142), pygame.Rect(0, SIDEWALK_TOP, WIDTH, 8))
        for x in range(0, WIDTH + 96, 96):
            pygame.draw.line(self.screen, (170, 164, 154), (x, SIDEWALK_TOP + 8), (x - 26, CURB_Y), 2)
        for x, scale in ((90, 1.0), (1200, 0.9)):
            tree = self.tree if scale == 1.0 else pygame.transform.smoothscale(self.tree, (int(self.tree.get_width() * scale), int(self.tree.get_height() * scale)))
            self.screen.blit(tree, (x - tree.get_width() // 2, SIDEWALK_TOP + 6 - tree.get_height()))
        art = self.building_art["pharmacy" if self.building == "pharmacy" else self.building]
        shadow = pygame.Surface((art.get_width() + 40, 26), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 70), shadow.get_rect())
        self.screen.blit(shadow, (BUILDING_X - shadow.get_width() // 2, SIDEWALK_TOP - 8))
        self.screen.blit(art, (BUILDING_X - art.get_width() // 2, SIDEWALK_TOP + 4 - art.get_height()))
        self._building_sign(env)
        pygame.draw.rect(self.screen, (226, 222, 214), pygame.Rect(0, CURB_Y - 6, WIDTH, 8))
        pygame.draw.rect(self.screen, (66, 70, 76), pygame.Rect(0, CURB_Y + 2, WIDTH, VIEW_H - CURB_Y))
        for x in range(10, WIDTH, 120):
            pygame.draw.rect(self.screen, (236, 206, 96), pygame.Rect(x, 508, 60, 5), border_radius=2)
        if self.bike_x is not None:
            self.screen.blit(self.bike_side, (self.bike_x - self.bike_side.get_width() // 2, FEET_Y + 6 - self.bike_side.get_height()))
        if self.patient_x is not None:
            self._patient(self.patient_ext, self.patient_x, FEET_Y)
        self._walker(self.rider_ext, self.rider_x, FEET_Y)

    def _building_sign(self, env: FarumasiRiderEnv) -> None:
        pygame = self._pygame
        if self.building == "pharmacy":
            title, detail, colour = PHARMACY_NAME, "licensed partner", GREEN_SOFT
        else:
            title, detail = env.dest.name, {"home": "patient's home", "office": "patient's office", "hotel": "patient's hotel"}[env.dest.kind]
            colour = (250, 204, 21)
        art = self.building_art["pharmacy" if self.building == "pharmacy" else self.building]
        top = SIDEWALK_TOP + 4 - art.get_height()
        title_font, detail_font = self._font(19, True), self._font(14)
        width = max(title_font.size(title)[0], detail_font.size(detail)[0]) + 40
        rect = pygame.Rect(0, 0, width, 52)
        rect.midbottom = (BUILDING_X, max(112, top - 8))
        card = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(card, (255, 255, 255, 240), card.get_rect(), border_radius=12)
        pygame.draw.rect(card, colour, pygame.Rect(0, 0, 9, rect.height), border_top_left_radius=12, border_bottom_left_radius=12)
        self.screen.blit(card, rect.topleft)
        self._text(title, rect.x + 22, rect.y + 5, INK, title_font)
        self._text(detail, rect.x + 22, rect.y + 29, MUTED, detail_font)

    def _walker(self, frames: list, x: float, feet_y: int) -> None:
        pygame = self._pygame
        index = (self.phase_walk // 4) % 4 if self.walking else 0
        sprite = frames[index]
        if self.rider_face < 0:
            sprite = pygame.transform.flip(sprite, True, False)
        bob = int(abs(math.sin(self.phase_walk * 0.42)) * 5) if self.walking else 0
        shadow = pygame.Surface((130, 22), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 90), shadow.get_rect())
        self.screen.blit(shadow, (x - 65, feet_y - 11))
        self.screen.blit(sprite, (x - sprite.get_width() // 2, feet_y - sprite.get_height() - bob))

    def _patient(self, frames: list, x: float, feet_y: int) -> None:
        pygame = self._pygame
        sprite = pygame.transform.flip(frames[0], True, False)
        shadow = pygame.Surface((120, 20), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 90), shadow.get_rect())
        self.screen.blit(shadow, (x - 60, feet_y - 10))
        self.screen.blit(sprite, (x - sprite.get_width() // 2, feet_y - sprite.get_height()))

    # -------------------------------------------------------------------- HUD

    def logo(self, surface, x: int, y: int, size: int) -> None:
        pygame = self._pygame
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(surface, (22, 163, 74), rect, border_radius=size // 4)
        pygame.draw.rect(surface, (255, 255, 255), rect, width=max(2, size // 14), border_radius=size // 4)
        bar = max(3, size // 6)
        pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(rect.centerx - bar // 2, rect.y + size // 5, bar, size * 3 // 5), border_radius=2)
        pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(rect.x + size // 5, rect.centery - bar // 2, size * 3 // 5, bar), border_radius=2)
        pygame.draw.arc(surface, (255, 255, 255), rect.inflate(size // 3, size // 3).move(0, size // 6), math.pi * 1.05, math.pi * 1.95, max(2, size // 16))

    def _top_bar(self, env: FarumasiRiderEnv) -> None:
        pygame = self._pygame
        bar = pygame.Surface((WIDTH, 56), pygame.SRCALPHA)
        bar.fill((8, 52, 48, 236))
        self.screen.blit(bar, (0, 0))
        self.logo(self.screen, 14, 8, 40)
        self._text("FARUMASI", 64, 6, (255, 255, 255), self._font(26, True))
        self._text("RIDER", 66, 34, (134, 239, 172), self._font(14, True))
        mission = f"{MED_NAMES[env.med]}  to  {env.dest.name}"
        if env.returning:
            mission = "Return the medicine to Kigali City Pharmacy"
        self._text(mission, 250, 8, (255, 255, 255), self._font(20, True))
        self._text(f"Order FR-{env.order_id}   Patient: {env.patient}", 250, 32, (178, 226, 214), self._font(14))
        total = int(env.clock_start + env.minutes)
        self._text(f"{total // 60:02d}:{total % 60:02d}", 880, 6, (255, 255, 255), self._font(28, True))
        weather = "RAIN" if env.raining else ("CLOUDS" if env.rain_soon else ("RUSH HOUR" if env.rush else "CLEAR"))
        wcolour = (147, 197, 253) if env.raining else ((253, 224, 71) if env.rain_soon or env.rush else (134, 239, 172))
        self._text(weather, 880, 36, wcolour, self._font(13, True))
        ratio = min(1.0, env.minutes / DEADLINE)
        self._text(f"Promise {int(DEADLINE)} min", 1010, 8, (178, 226, 214), self._font(14))
        self._text(f"{env.minutes:.0f} min used", 1010, 28, (255, 255, 255), self._font(16, True))
        colour = GREEN_SOFT if ratio < 0.7 else (AMBER if ratio < 1.0 else RED)
        self._bar(1150, 21, 108, ratio, colour, height=14)

    def _hud(self, env: FarumasiRiderEnv) -> None:
        pygame = self._pygame
        pygame.draw.rect(self.screen, (226, 220, 208), pygame.Rect(0, VIEW_H, WIDTH, HEIGHT - VIEW_H))
        pygame.draw.rect(self.screen, (10, 61, 56), pygame.Rect(0, VIEW_H, WIDTH, 4))
        self._action_card(env, pygame.Rect(16, 556, 408, 190))
        self._nav_card(env, pygame.Rect(440, 556, 396, 190))
        self._order_card(env, pygame.Rect(852, 556, 412, 190))

    def _card(self, rect) -> None:
        pygame = self._pygame
        pygame.draw.rect(self.screen, (196, 189, 175), rect.move(0, 3), border_radius=16)
        pygame.draw.rect(self.screen, CARD, rect, border_radius=16)

    def _action_card(self, env: FarumasiRiderEnv, rect) -> None:
        self._card(rect)
        label = env.action_label()
        colour = RED if env.last_event in FAIL_EVENTS else GREEN_DARK
        self._text("CURRENT ACTION", rect.x + 20, rect.y + 12, MUTED, self._font(14, True))
        size = 36
        while size > 18 and self._font(size, True).size(label)[0] > rect.width - 36:
            size -= 2
        self._text(label, rect.x + 20, rect.y + 34 + (36 - size) // 2, colour, self._font(size, True))
        self._text(env.current_place(), rect.x + 20, rect.y + 84, INK, self._font(19, True))
        self._text(EVENT_TEXT.get(env.last_event, env.last_event), rect.x + 20, rect.y + 112, MUTED, self._font(15))
        reward_colour = GREEN_SOFT if env.last_reward >= 0 else RED
        self._text("STEP REWARD", rect.x + 20, rect.y + 142, MUTED, self._font(13, True))
        self._text(f"{env.last_reward:+.2f}", rect.x + 20, rect.y + 156, reward_colour, self._font(28, True))
        self._text("TOTAL", rect.x + 214, rect.y + 142, MUTED, self._font(13, True))
        total_colour = GREEN_SOFT if env.return_so_far >= 0 else RED
        self._text(f"{env.return_so_far:+.2f}", rect.x + 214, rect.y + 156, total_colour, self._font(28, True))

    def _nav_card(self, env: FarumasiRiderEnv, rect) -> None:
        pygame = self._pygame
        self._card(rect)
        self._text("LIVE TRAFFIC", rect.x + 16, rect.y + 10, MUTED, self._font(14, True))
        for number, name in enumerate(("clear", "busy", "heavy", "blocked")):
            x = rect.x + 134 + number * 62
            pygame.draw.circle(self.screen, TRAFFIC_COLOURS[number], (x, rect.y + 19), 5)
            self._text(name, x + 9, rect.y + 11, MUTED, self._font(12))
        view = pygame.Rect(rect.x + 12, rect.y + 34, rect.width - 24, rect.height - 46)
        pygame.draw.rect(self.screen, (234, 238, 230), view, border_radius=10)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(view)

        def point(node: int) -> tuple[float, float]:
            col, row = node % COLS, node // COLS
            return view.x + 32 + col * (view.width - 64) / (COLS - 1), view.y + 24 + row * (view.height - 48) / (ROWS - 1)

        for edge in range(N_EDGES):
            a, b = edge_nodes(edge)
            pa, pb = point(a), point(b)
            pygame.draw.line(self.screen, (255, 255, 255), pa, pb, 11)
            pygame.draw.line(self.screen, TRAFFIC_COLOURS[int(env.traffic[edge])], pa, pb, 6)
        dist = env.time_to(env.target_node)
        node = env.node
        path = [node]
        for _ in range(12):
            if node == env.target_node:
                break
            best, cost = None, float("inf")
            for direction in range(4):
                other = neighbour(node, direction)
                if other is None:
                    continue
                total = env.edge_minutes(edge_id(node, other)) + dist[other]
                if total < cost:
                    best, cost = other, total
            node = best
            path.append(node)
        if len(path) > 1 and env.phase == P_RIDING:
            pygame.draw.lines(self.screen, (26, 115, 232), False, [point(n) for n in path], 3)
        px, py = point(PHARMACY_NODE)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(px), int(py)), 11)
        pygame.draw.circle(self.screen, GREEN_SOFT, (int(px), int(py)), 9)
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(int(px) - 1, int(py) - 6, 3, 12))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(int(px) - 6, int(py) - 1, 12, 3))
        dx, dy = point(env.dest.node)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(dx), int(dy)), 11)
        pygame.draw.circle(self.screen, (250, 204, 21), (int(dx), int(dy)), 9)
        letter = {"home": "H", "office": "O", "hotel": "Ho"}[env.dest.kind]
        self._text(letter, int(dx), int(dy), INK, self._font(11, True), center=True)
        wx = (self.city_pos[0] - city_map.NX0) / (city_map.DX * (COLS - 1))
        wy = (self.city_pos[1] - city_map.NY0) / (city_map.DY * (ROWS - 1))
        if self.scene != "city":
            rx, ry = point(env.node)
        else:
            rx, ry = view.x + 32 + wx * (view.width - 64), view.y + 24 + wy * (view.height - 48)
        halo = pygame.Surface((50, 50), pygame.SRCALPHA)
        pygame.draw.circle(halo, (66, 133, 244, 80), (25, 25), 10 + int(3 * abs(math.sin(self.tick * 0.12))))
        self.screen.blit(halo, (rx - 25, ry - 25))
        pygame.draw.circle(self.screen, (255, 255, 255), (int(rx), int(ry)), 8)
        pygame.draw.circle(self.screen, (26, 115, 232), (int(rx), int(ry)), 5)
        self.screen.set_clip(old_clip)
        eta = dist[env.node] if env.phase == P_RIDING else 0.0
        if env.phase == P_RIDING:
            self._text(f"Fastest ride now: {eta:.0f} min", rect.x + 18, rect.bottom - 22, GREEN_DARK, self._font(14, True))

    def _order_card(self, env: FarumasiRiderEnv, rect) -> None:
        pygame = self._pygame
        self._card(rect)
        self._text("ORDER", rect.x + 18, rect.y + 10, MUTED, self._font(14, True))
        self._text(MED_NAMES[env.med], rect.x + 18, rect.y + 28, INK, self._font(22, True))
        kind = {"home": "Home delivery", "office": "Office delivery", "hotel": "Hotel delivery"}[env.dest.kind]
        self._text(f"{kind}  -  {env.dest.area}", rect.x + 18, rect.y + 58, MUTED, self._font(15))
        if not env.rx_required:
            text, colour = "No prescription needed", MUTED
        elif not env.rx_checked:
            text, colour = "Prescription required: not checked yet", AMBER
        elif env.rx_valid:
            text, colour = "Prescription checked: valid", GREEN_SOFT
        else:
            text, colour = "Prescription WRONG: return the medicine", RED
        chip = pygame.Rect(rect.x + 16, rect.y + 84, rect.width - 32, 28)
        pygame.draw.rect(self.screen, tuple(int(c * 0.12 + 255 * 0.88) for c in colour), chip, border_radius=14)
        pygame.draw.rect(self.screen, colour, chip, width=2, border_radius=14)
        self._text(text, chip.centerx, chip.centery, colour, self._font(15, True), center=True)
        self._text("MEDICINE CONDITION", rect.x + 18, rect.y + 122, MUTED, self._font(13, True))
        bar_colour = RED if env.condition < 0.55 else GREEN_SOFT
        self._bar(rect.x + 18, rect.y + 142, 220, env.condition, bar_colour, height=14)
        self._text(f"{env.condition:.0%}", rect.x + 248, rect.y + 138, INK, self._font(17, True))
        cover = "Rain cover ON" if env.covered else "Rain cover off"
        self._text(cover, rect.x + 18, rect.y + 164, BLUE if env.covered else MUTED, self._font(14, True))
        weather = "Raining" if env.raining else ("Rain coming" if env.rain_soon else "Dry")
        self._text(weather, rect.x + 190, rect.y + 164, BLUE if env.raining else MUTED, self._font(14, True))

    # ------------------------------------------------------------------ popups

    def _popup(self, env: FarumasiRiderEnv, key: str, t: float) -> None:
        if key == "mission_start":
            self._mission_card(env, t)
            return
        if key == "scan":
            self._scan_phone(env, t)
            return
        if key == "mission_passed":
            self._big(f"MISSION PASSED", f"Delivered in {env.minutes:.0f} min   Reward {env.return_so_far:+.2f}", GREEN_SOFT, t)
            return
        if key == "mission_returned":
            self._big("MISSION COMPLETE", f"Wrong prescription returned safely   Reward {env.return_so_far:+.2f}", BLUE, t)
            return
        if key == "fail":
            self._big("MISSION FAILED", f"{EVENT_TEXT.get(env.last_event, '')}   Reward {env.return_so_far:+.2f}", RED, t)
            if env.last_event == "rx_violation":
                veil = self._pygame.Surface((WIDTH, VIEW_H), self._pygame.SRCALPHA)
                veil.fill((220, 38, 38, int(80 * (1.0 - t))))
                self.screen.blit(veil, (0, 0))
            return
        if key == "wet":
            self._toast("RAIN ON THE BOX. COVER IT!", RED, t)
            return
        banner = self._banner(env, key)
        if banner is None:
            return
        title, detail, colour = banner
        pygame = self._pygame
        slide = (1.0 - _ease(t * 4.0)) * -50
        panel = pygame.Surface((560, 100), pygame.SRCALPHA)
        pygame.draw.rect(panel, (12, 20, 18, 220), panel.get_rect(), border_radius=18)
        pygame.draw.rect(panel, colour, panel.get_rect(), width=4, border_radius=18)
        left = 24 if self.scene == "ext" else WIDTH // 2 - 280
        self.screen.blit(panel, (left, 72 + slide))
        self._text(title, left + 280, 100 + slide, (255, 255, 255), self._font(34, True), center=True)
        self._text(detail, left + 280, 142 + slide, (214, 235, 228), self._font(17), center=True)
        if key in ("rx_ok", "rx_wrong"):
            self._rx_paper(env, key == "rx_ok", t)
        if key == "gate_opened":
            self._speech("Knock knock!", 520, 240)
        reward = env.last_reward
        if key in ("handed_over", "order_collected") and reward:
            self._float_reward(reward, t)

    def _banner(self, env: FarumasiRiderEnv, key: str):
        med = MED_NAMES[env.med]
        table = {
            "entered_pharmacy": ("KIGALI CITY PHARMACY", "Walking in to collect the order", GREEN_SOFT),
            "order_collected": ("ORDER COLLECTED", f"{med} sealed in the FARUMASI box", GREEN_SOFT),
            "covered_box": ("BOX COVERED", "Rain cover on. The medicine stays dry", BLUE),
            "waited": ("WAITING", "Every minute counts against the promise", AMBER),
            "parked": ("PARKED", "Box in hand. The rider walks in on foot", GREEN_SOFT),
            "gate_opened": ("GATE OPENED", f"{env.patient} comes to the door", GREEN_SOFT),
            "checked_in": ("CHECKED IN", "Reception calls the patient down", GREEN_SOFT),
            "reached_patient": ("FACE TO FACE", f"With {env.patient}", GREEN_SOFT),
            "rx_ok": ("PRESCRIPTION VALID", "Name, medicine and date all match", GREEN_SOFT),
            "rx_wrong": ("PRESCRIPTION WRONG", "Do not hand it over. Take it back.", RED),
            "handed_over": ("HANDED OVER", f"Medicine placed in the hands of {env.patient}", GREEN_SOFT),
            "returning": ("RIDING BACK", "The medicine returns to the pharmacist", AMBER),
            "returned_to_pharmacy": ("RETURNED", "Pharmacist will correct the prescription", BLUE),
            "invalid": ("NOT ALLOWED", "That action is not possible here", RED),
        }
        return table.get(key)

    def _toast(self, text: str, colour, t: float) -> None:
        pygame = self._pygame
        panel = pygame.Rect(0, 0, 440, 44)
        panel.midtop = (WIDTH // 2, 70)
        pygame.draw.rect(self.screen, (20, 28, 26), panel, border_radius=22)
        pygame.draw.rect(self.screen, colour, panel, width=3, border_radius=22)
        self._text(text, panel.centerx, panel.centery, (255, 255, 255), self._font(18, True), center=True)

    def _big(self, title: str, detail: str, colour, t: float) -> None:
        pygame = self._pygame
        shade = pygame.Surface((WIDTH, VIEW_H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, int(120 * min(1.0, t * 5))))
        self.screen.blit(shade, (0, 0))
        scale = 0.6 + 0.4 * _ease(t * 4)
        font = self._font(int(76 * scale), True)
        for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3)):
            self.screen.blit(font.render(title, True, (10, 10, 10)), font.render(title, True, colour).get_rect(center=(WIDTH // 2 + dx, 250 + dy)))
        self.screen.blit(font.render(title, True, colour), font.render(title, True, colour).get_rect(center=(WIDTH // 2, 250)))
        self._text(detail, WIDTH // 2, 326, (255, 255, 255), self._font(24, True), center=True)

    def _mission_card(self, env: FarumasiRiderEnv, t: float) -> None:
        pygame = self._pygame
        shade = pygame.Surface((WIDTH, VIEW_H), pygame.SRCALPHA)
        shade.fill((6, 24, 22, 150))
        self.screen.blit(shade, (0, 0))
        card = pygame.Rect(0, 0, 700, 330)
        card.center = (WIDTH // 2, 300)
        pygame.draw.rect(self.screen, CARD, card, border_radius=24)
        pygame.draw.rect(self.screen, GREEN, card, width=5, border_radius=24)
        self.logo(self.screen, card.x + 30, card.y + 24, 64)
        self._text("FARUMASI RIDER", card.x + 112, card.y + 26, GREEN_DARK, self._font(34, True))
        self._text("NEW DELIVERY", card.x + 114, card.y + 66, MUTED, self._font(18, True))
        lines = [
            ("Order", f"FR-{env.order_id}   {MED_NAMES[env.med]}"),
            ("Patient", f"{env.patient}"),
            ("Deliver to", f"{env.dest.name} ({env.dest.kind})"),
            ("Prescription", "Required: check it before you hand over" if env.rx_required else "Not required"),
            ("Promise", f"{int(DEADLINE)} minutes, {'rush hour traffic' if env.rush else 'normal traffic'}"),
        ]
        y = card.y + 116
        for name, value in lines:
            self._text(name, card.x + 36, y, MUTED, self._font(17, True))
            self._text(value, card.x + 190, y, INK, self._font(19, True))
            y += 38

    def _speech(self, text: str, x: int, y: int) -> None:
        pygame = self._pygame
        bubble = pygame.Rect(0, 0, 200, 46)
        bubble.center = (x, y)
        pygame.draw.rect(self.screen, (255, 255, 255), bubble, border_radius=20)
        pygame.draw.rect(self.screen, INK, bubble, width=2, border_radius=20)
        self._text(text, bubble.centerx, bubble.centery, INK, self._font(20, True), center=True)

    def _float_reward(self, reward: float, t: float) -> None:
        colour = GREEN_SOFT if reward >= 0 else RED
        lift = 80 * _ease(t)
        alpha = int(255 * (1.0 - max(0.0, t - 0.55) / 0.45))
        font = self._font(52, True)
        text = font.render(f"{reward:+.2f}", True, colour)
        text.set_alpha(alpha)
        self.screen.blit(text, (WIDTH // 2 - 60, 270 - lift))

    def _rx_paper(self, env: FarumasiRiderEnv, ok: bool, t: float) -> None:
        pygame = self._pygame
        paper = pygame.Rect(0, 0, 250, 300)
        paper.center = (WIDTH // 2 + 280, 330 - int(14 * math.sin(t * 6)))
        pygame.draw.rect(self.screen, (250, 249, 244), paper, border_radius=8)
        pygame.draw.rect(self.screen, (180, 176, 164), paper, width=2, border_radius=8)
        self._text("PRESCRIPTION", paper.x + 24, paper.y + 14, GREEN_DARK, self._font(20, True))
        self._text("Rx", paper.x + 24, paper.y + 44, GREEN, self._font(34, True))
        self._text(MED_NAMES[env.med], paper.x + 24, paper.y + 96, INK, self._font(17, True))
        self._text(f"Patient: {env.patient}", paper.x + 24, paper.y + 124, MUTED, self._font(14))
        for k in range(4):
            pygame.draw.line(self.screen, (214, 210, 200), (paper.x + 24, paper.y + 164 + k * 16), (paper.right - 24, paper.y + 164 + k * 16), 2)
        colour = GREEN_SOFT if ok else RED
        pygame.draw.circle(self.screen, colour, (paper.right - 56, paper.bottom - 56), 36, 5)
        if ok:
            pygame.draw.lines(self.screen, colour, False, [(paper.right - 74, paper.bottom - 56), (paper.right - 58, paper.bottom - 40), (paper.right - 34, paper.bottom - 74)], 6)
        else:
            pygame.draw.line(self.screen, colour, (paper.right - 74, paper.bottom - 74), (paper.right - 38, paper.bottom - 38), 6)
            pygame.draw.line(self.screen, colour, (paper.right - 38, paper.bottom - 74), (paper.right - 74, paper.bottom - 38), 6)

    def _scan_phone(self, env: FarumasiRiderEnv, t: float) -> None:
        pygame = self._pygame
        body = pygame.Rect(0, 0, 300, 470)
        body.center = (WIDTH // 2, 290)
        pygame.draw.rect(self.screen, (14, 18, 20), body, border_radius=36)
        screen = body.inflate(-20, -24)
        pygame.draw.rect(self.screen, (246, 249, 246), screen, border_radius=26)
        head = pygame.Rect(screen.x, screen.y, screen.width, 74)
        pygame.draw.rect(self.screen, GREEN, head, border_top_left_radius=26, border_top_right_radius=26)
        self.logo(self.screen, head.x + 16, head.y + 15, 44)
        self._text("FARUMASI", head.x + 70, head.y + 10, (255, 255, 255), self._font(24, True))
        self._text("Confirm delivery", head.x + 72, head.y + 40, (204, 240, 226), self._font(14))
        qr = pygame.Rect(0, 0, 190, 190)
        qr.center = (screen.centerx, screen.y + 200)
        rng = random.Random(env.order_id)
        cell = qr.width // 21
        pygame.draw.rect(self.screen, (255, 255, 255), qr.inflate(16, 16), border_radius=8)
        for r in range(21):
            for c in range(21):
                finder = (r < 7 and c < 7) or (r < 7 and c > 13) or (r > 13 and c < 7)
                if finder:
                    on = r in (0, 6) or c in (0, 6) or c in (14, 20) or (r in (14, 20)) or (2 <= r % 14 <= 4 and 2 <= c % 14 <= 4)
                else:
                    on = rng.random() < 0.5
                if on:
                    pygame.draw.rect(self.screen, INK, pygame.Rect(qr.x + c * cell, qr.y + r * cell, cell, cell))
        if t < 0.65:
            y = qr.y + int((t / 0.65) * qr.height)
            pygame.draw.line(self.screen, (22, 163, 74), (qr.x - 10, y), (qr.right + 10, y), 4)
            self._text("Scanning the patient's QR code...", screen.centerx, screen.y + 330, MUTED, self._font(15, True), center=True)
        else:
            pygame.draw.circle(self.screen, GREEN_SOFT, (screen.centerx, screen.y + 330), 38)
            pygame.draw.lines(self.screen, (255, 255, 255), False, [(screen.centerx - 18, screen.y + 330), (screen.centerx - 4, screen.y + 346), (screen.centerx + 20, screen.y + 314)], 7)
            self._text("DELIVERY CONFIRMED", screen.centerx, screen.y + 392, GREEN_DARK, self._font(19, True), center=True)
            self._text(f"Order FR-{env.order_id}", screen.centerx, screen.y + 418, MUTED, self._font(14), center=True)

    # ---------------------------------------------------------------- helpers

    def _bar(self, x: int, y: int, width: int, fraction: float, colour, height: int = 12) -> None:
        pygame = self._pygame
        pygame.draw.rect(self.screen, (214, 206, 192), pygame.Rect(x, y, width, height), border_radius=height // 2)
        inner = max(0, int(width * max(0.0, min(1.0, fraction))))
        if inner:
            pygame.draw.rect(self.screen, colour, pygame.Rect(x, y, inner, height), border_radius=height // 2)

    def _text(self, text: str, x: float, y: float, colour, font, center: bool = False) -> None:
        surface = font.render(text, True, colour)
        rect = surface.get_rect()
        if center:
            rect.center = (int(x), int(y))
        else:
            rect.topleft = (int(x), int(y))
        self.screen.blit(surface, rect)


def ay_row(y: int) -> int:
    return int(round((y - city_map.NY0) / city_map.DY))


def save_logo(path: str) -> str:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    pygame.font.init()
    surface = pygame.Surface((520, 140), pygame.SRCALPHA)
    helper = DeliveryRenderer(False)
    helper._pygame = pygame
    helper.logo(surface, 4, 4, 130)
    font = pygame.font.SysFont("segoeui", 64, bold=True)
    surface.blit(font.render("FARUMASI", True, (10, 61, 56)), (152, 18))
    small = pygame.font.SysFont("segoeui", 26, bold=True)
    surface.blit(small.render("RIDER  |  Safe, fast delivery in Kigali", True, (22, 163, 74)), (156, 94))
    pygame.image.save(surface, path)
    return path


def save_environment_png(path: str, seed: int = 0, stage: str = "city") -> str:
    """Write one frame. stage is city, rain, pharmacy, lobby or scan."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from environment.baselines import heuristic_action

    def build(wanted_kind: str | None):
        for attempt in range(200):
            env = FarumasiRiderEnv(render_mode="rgb_array")
            env.reset(seed=seed + attempt)
            if wanted_kind is None or env.dest.kind == wanted_kind:
                return env
        raise RuntimeError("no matching seed")

    env = build("hotel" if stage in ("lobby", "scan") else None)
    if stage == "rain":
        env.rain_at = 0
    if stage == "pharmacy":
        env.step(ENTER)
    elif stage == "city" or stage == "rain":
        env.rx_required = False
        for _ in range(14):
            env.step(heuristic_action(env))
            if env.phase == P_RIDING and env.steps >= 6:
                break
    elif stage == "lobby":
        while env.phase != P_PATIENT:
            env.step(heuristic_action(env))
    elif stage == "scan":
        env.rx_required = False
        while env.outcome == "":
            env.step(heuristic_action(env))
    frame = env.render()
    env.close()
    import pygame

    pygame.image.save(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), path)
    return path
