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
    N_ACTIONS,
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

WIDTH, HEIGHT = 1280, 820
VIEW_H = 540
CTRL_Y = VIEW_H + 8
CTRL_H = 40
CARD_Y = VIEW_H + 56
FPS = 30
SHORT_ACTIONS = (
    "N",
    "E",
    "S",
    "W",
    "Cover",
    "Wait",
    "Enter",
    "Collect",
    "Park",
    "Knock",
    "Carry",
    "Verify",
    "Hand over",
    "Scan",
    "Walk back",
    "Return",
)
SPRITES = Path(__file__).resolve().parent.parent / "assets" / "sprites"
SIDEWALK_TOP = 404
CURB_Y = 470
FEET_Y = 462
BUILDING_X = 860
BIKE_X = 250
START_X = 440
SHELTER_BIKE_X = 650  # the bike is parked under the shelter ...
SHELTER_FROM_X = 770  # ... the rider steps off beside it ...
SHELTER_X = 880  # ... and waits a little further in
AWNING = (480, 1000)  # left and right edge of the shelter roof: it covers the bike and the rider
AWNING_TOP, AWNING_BOTTOM = 212, 246
LOBBY_RECEPTION_X = 330
LOBBY_ENTER_X = 110
MEET_INT_X = 600
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
CAR_LEN = 66
CAR_PITCH = 78
CARS_PER_LANE = (1, 2, 3, 4)
TRAFFIC_SPEED = (4.2, 2.4, 0.9, 0.3)

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
        self._geo: list[tuple] = []
        self._car_rng = random.Random(77)
        self.pose_r: int | None = None
        self.pose_p: int | None = None
        self.patient_walking = False
        # auto = agent drives; self = player clicks actions; paused = freeze
        self.control_mode = "auto"
        self._pending_action: int | None = None
        self._mode_rects: dict[str, object] = {}
        self._action_rects: list[tuple[object, int]] = []
        self._last_env: FarumasiRiderEnv | None = None
        self.patient_phase = 0
        self._rain = []
        self._car_cache: dict[tuple[int, int], object] = {}
        self._road_state: dict = {}
        self.moving = False  # True while the bike is rolling, so the next ride does not start from a standstill
        self.rain_on = False

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
        self.rider_pose = {
            "ext": [load(f"rider_stand_{i}", height=206) for i in range(4)],
            "int": [load(f"rider_stand_{i}", height=268) for i in range(4)],
        }
        self.patient_pose = {
            "ext": [pygame.transform.flip(load(f"patient_stand_{i}", height=190), True, False) for i in range(3)],
            "int": [pygame.transform.flip(load(f"patient_stand_{i}", height=250), True, False) for i in range(3)],
        }
        self.bike_top = load("bike_top", height=62)
        covered = self.bike_top.copy()
        w, h = covered.get_size()
        tarp = pygame.Rect(int(w * 0.1), int(h * 0.69), int(w * 0.8), int(h * 0.31))
        pygame.draw.rect(covered, (24, 58, 112), tarp, border_radius=5)
        pygame.draw.line(covered, (92, 140, 204), (tarp.left + 4, tarp.top + 5), (tarp.right - 4, tarp.top + 5), 2)
        pygame.draw.line(covered, (92, 140, 204), (tarp.left + 4, tarp.centery), (tarp.right - 4, tarp.centery), 1)
        self.bike_top_covered = covered
        self.car_art = [load(f"car_{i}", height=66) for i in range(4)]

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
            "lobby": pygame.transform.flip(cover_image("lobby_interior", 720), True, False),
        }
        self.world = city_map.build_city(pygame, self._font).convert()
        self.landmarks = city_map.landmarks()
        self.overlay = pygame.Surface((WIDTH, VIEW_H), pygame.SRCALPHA)
        rng = random.Random(3)
        self._rain = [(rng.uniform(0, WIDTH), rng.uniform(0, VIEW_H), rng.uniform(14, 26)) for _ in range(150)]
        px, py = city_map.node_px(PHARMACY_NODE)
        self.city_pos = (float(px), float(py))
        self.rider_off = self._lane_offset(-90.0, PHARMACY_NODE)
        self._geo = [city_map.edge_geometry(edge) for edge in range(N_EDGES)]

    @staticmethod
    def _lane_offset(heading: float, node: int) -> tuple[float, float]:
        """Where the rider sits across the road: in the kerb-side lane, on the right."""
        horizontal = abs(math.sin(math.radians(heading))) > 0.5
        reach = city_map.bike_lane_offset(city_map.lane_half(node, horizontal))
        ux, uy = -math.sin(math.radians(heading)), -math.cos(math.radians(heading))
        return (-uy * reach, ux * reach)

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
        self._poll_events()
        while self.control_mode == "paused":
            # Keep the control bar live so Self Play / Auto Play highlight updates.
            if self._last_env is not None and not self.walking and not self.moving:
                self._frame(self._last_env, popup=self._static_popup(self._last_env), t=0.7)
                pygame.display.flip()
            self.clock.tick(FPS)
            self._poll_events()

    def _poll_events(self) -> None:
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.control_mode = "auto" if self.control_mode == "paused" else "paused"
                    self._pending_action = None
                elif event.key == pygame.K_a:
                    self.control_mode = "auto"
                    self._pending_action = None
                elif event.key == pygame.K_s:
                    self.control_mode = "self"
                    self._pending_action = None
                elif event.key == pygame.K_p:
                    self.control_mode = "paused"
                    self._pending_action = None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)

    def _handle_click(self, pos) -> None:
        for mode, rect in self._mode_rects.items():
            if rect.collidepoint(pos):
                self.control_mode = mode
                self._pending_action = None
                return
        if self.control_mode != "self":
            return
        for rect, action in self._action_rects:
            if rect.collidepoint(pos):
                self._pending_action = action
                return

    def hold_frame(self, env: FarumasiRiderEnv) -> None:
        """Redraw the current scene once (used while waiting for Self Play / Pause)."""
        self._ensure()
        self._last_env = env
        self._frame(env, popup=self._static_popup(env), t=0.7)
        self._present()

    def wait_for_action(self, env: FarumasiRiderEnv) -> int | None:
        """Block until the player picks a legal action, or the mode leaves Self Play."""
        self._pending_action = None
        while self.control_mode == "self" and self._pending_action is None:
            self.hold_frame(env)
        action = self._pending_action
        self._pending_action = None
        return action

    # ------------------------------------------------------------- scene plan

    def _layout(self, env: FarumasiRiderEnv) -> None:
        kind = env.dest.kind
        phase = env.phase
        self.gate_open = False
        self.patient_x = None
        self.bike_x = BIKE_X
        self.rider_face = 1
        self.walking = False
        self.patient_walking = False
        self.pose_r = None
        self.pose_p = None
        if phase == P_OUTSIDE:
            self.scene, self.building, self.rider_x = "ext", "pharmacy", START_X
        elif phase == P_INSIDE:
            self.scene, self.building, self.rider_x, self.bike_x = "int", "pharmacy", INT_COUNTER_X, None
        elif phase == P_RIDING:
            self.scene = "city"
            px, py = city_map.node_px(env.node)
            self.city_pos = (float(px), float(py))
            prev = getattr(env, "prev_node", env.node)
            if prev != env.node:
                qx, qy = city_map.node_px(prev)
                self.heading = {(0, -1): 0.0, (1, 0): -90.0, (0, 1): 180.0, (-1, 0): 90.0}[(int(np.sign(px - qx)), int(np.sign(py - qy)))]
                self.rider_off = self._lane_offset(self.heading, prev)
        elif phase == P_ENTRANCE:
            self.scene, self.building, self.rider_x = "ext", kind, START_X
        elif phase == P_BUILDING:
            if kind == "home":
                self.scene, self.building, self.rider_x = "ext", kind, self._meet("ext")[0] - 60
                self.gate_open, self.patient_x = True, self._meet("ext")[1]
            else:
                self.scene, self.building, self.rider_x, self.bike_x = "int", "lobby", LOBBY_RECEPTION_X, None
        else:
            home = kind == "home"
            self.scene, self.building = ("ext", kind) if home else ("int", "lobby")
            if not home:
                self.bike_x = None
            self.rider_x, self.patient_x = self._meet(self.scene)
            self.gate_open = home
            self.pose_r = 3 if env.handed else 0
            self.pose_p = 2 if env.handed else 0

    def _gate_x(self) -> float:
        return BUILDING_X + 0.21 * self.building_art["home"].get_width()

    def _meet(self, scene: str) -> tuple[float, float]:
        """Where the rider and the patient stand, body centre, when they meet."""
        if scene == "int":
            return float(MEET_INT_X), float(MEET_INT_X + 185)
        patient = self._gate_x() - 10
        return patient - 140, patient

    def _plan(self, env: FarumasiRiderEnv) -> list[tuple]:
        action, event, kind = env.last_action, env.last_event, env.dest.kind
        plan: list[tuple] = []
        if event == "invalid":
            return [("hold", 12, "invalid")]
        if self.scene == "shelter" and action not in (COVER, WAIT):
            plan += self._leave_shelter()
        if action <= 3:
            frames = int(min(110, 24 + 11 * env.last_minutes))
            # the bike keeps rolling through junctions; a warning appears while it moves
            plan.append(("ride", env.prev_node, env.node, frames, "wet" if env.last_wet else None))
        elif action in (COVER, WAIT) and self.scene in ("city", "shelter"):
            plan += self._shelter_plan(env, action)
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
                ("fade", {"scene": "city", "city_pos": (float(px), float(py)), "heading": -90.0, "rider_off": self._lane_offset(-90.0, PHARMACY_NODE)}),
                ("hold", 16, None),
            ]
        elif action == PARK:
            plan += [
                ("fade", {"scene": "ext", "building": kind, "rider_x": START_X, "bike_x": BIKE_X, "patient_x": None, "gate_open": False, "rider_face": 1}),
                ("hold", 36, "parked"),
            ]
        elif action == KNOCK:
            if kind == "home":
                rider_at, patient_at = self._meet("ext")
                plan += [
                    ("walk", self.rider_x, rider_at - 60, 42),
                    ("hold", 24, None),
                    ("set", {"gate_open": True, "patient_x": patient_at, "pose_p": 0}),
                    ("hold", 58, "gate_opened"),
                ]
            else:
                plan += [
                    ("walk", self.rider_x, BUILDING_X, 40),
                    ("fade", {"scene": "int", "building": "lobby", "rider_x": LOBBY_ENTER_X, "bike_x": None}),
                    ("walk", LOBBY_ENTER_X, LOBBY_RECEPTION_X, 44),
                    ("hold", 56, "checked_in"),
                ]
        elif action == CARRY:
            if kind == "home":
                plan += [
                    ("walk", self.rider_x, self._meet("ext")[0], 30),
                    ("set", {"pose_r": 0, "pose_p": 0}),
                    ("hold", 36, "reached_patient"),
                ]
            else:
                rider_at, patient_at = self._meet("int")
                plan += [
                    ("meet", rider_at, WIDTH + 80, patient_at, 64),
                    ("set", {"pose_r": 0, "pose_p": 0}),
                    ("hold", 36, "reached_patient"),
                ]
        elif action == VERIFY:
            plan.append(("hold", 66, event))
        elif action == HAND:
            plan += [
                ("set", {"pose_r": 1, "pose_p": 1}),
                ("hold", 26, None),
                ("set", {"pose_r": 3, "pose_p": 2}),
                ("hold", 50, "handed_over"),
            ]
        elif action == SCAN:
            plan += [("hold", 84, "scan"), ("hold", 100, "mission_passed")]
        elif action == REMOUNT:
            px, py = city_map.node_px(env.node)
            plan += [
                ("set", {"pose_r": None}),
                ("walk", self.rider_x, 120, 42, -1),
                ("fade", {"scene": "city", "city_pos": (float(px), float(py)), "heading": 90.0, "rider_off": self._lane_offset(90.0, env.node), "patient_x": None, "pose_p": None}),
                ("hold", 22, "returning"),
            ]
        elif action == RETURN_MEDS:
            plan += [
                ("fade", {"scene": "int", "building": "pharmacy", "rider_x": INT_ENTER_X, "bike_x": None, "rider_face": 1, "patient_x": None, "pose_r": None, "pose_p": None}),
                ("walk", INT_ENTER_X, INT_COUNTER_X, 32),
                ("hold", 72, "returned_to_pharmacy"),
                ("hold", 100, "mission_returned"),
            ]
        if event in FAIL_EVENTS:
            plan.append(("hold", 110, "fail"))
        return plan

    def _shelter_plan(self, env: FarumasiRiderEnv, action: int) -> list[tuple]:
        """Waiting and covering the box happen off the road, under one roof that covers bike and rider.

        The rider goes in once and stays there while the agent keeps waiting or covering. The
        way back to the road is only played when the agent rides on (see _leave_shelter).
        """
        if action == COVER:
            popup, stay = "covered_box", 40
        else:
            popup, stay = ("sheltering" if env.raining else "waited"), 36
        if self.scene == "shelter":
            return [("set", {"pose_r": 0}), ("hold", stay, popup)]
        self._road_state = {
            "scene": "city",
            "city_pos": self.city_pos,
            "heading": self.heading,
            "rider_off": self.rider_off,
            "bike_x": BIKE_X,
            "pose_r": None,
            "patient_x": None,
            "pose_p": None,
            "rider_face": 1,
        }
        return [
            ("fade", {"scene": "shelter", "building": "home", "rider_x": float(SHELTER_FROM_X), "bike_x": SHELTER_BIKE_X, "patient_x": None, "pose_r": None, "pose_p": None, "rider_face": 1, "gate_open": False}),
            ("walk", float(SHELTER_FROM_X), float(SHELTER_X), 20),
            ("set", {"pose_r": 0}),
            ("hold", stay, popup),
        ]

    def _leave_shelter(self) -> list[tuple]:
        """Back to the bike and out onto the road, exactly where the rider left it."""
        return [
            ("set", {"pose_r": None}),
            ("walk", float(SHELTER_X), float(SHELTER_FROM_X), 20),
            ("fade", dict(self._road_state)),
        ]

    def _play(self, env: FarumasiRiderEnv, plan: list[tuple]) -> None:
        for segment in plan:
            kind = segment[0]
            if kind != "ride":
                self.moving = False
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
            elif kind == "meet":
                # the rider steps up while the patient walks in from the side
                _, rider_to, patient_from, patient_to, frames = segment
                rider_from = self.rider_x
                self.rider_face = 1
                self.walking = True
                self.patient_walking = True
                self.patient_x = float(patient_from)
                for frame in range(frames):
                    progress = _ease(frame / max(1, frames - 1))
                    self.rider_x = rider_from + (rider_to - rider_from) * progress
                    self.patient_x = patient_from + (patient_to - patient_from) * progress
                    self.phase_walk += 1
                    self.patient_phase += 1
                    self._frame(env, popup=None, t=0.0)
                    self._present()
                self.rider_x, self.patient_x = float(rider_to), float(patient_to)
                self.walking = False
                self.patient_walking = False
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
                self._ride(env, segment[1], segment[2], segment[3], segment[4] if len(segment) > 4 else None)

    @staticmethod
    def _travel(progress: float, from_rest: bool, to_rest: bool) -> float:
        """Distance covered, 0 to 1. The bike only speeds up or slows down when it starts or ends a trip."""
        a = 0.25 if from_rest else 0.0
        d = 0.25 if to_rest else 0.0
        v = 1.0 / (1.0 - a / 2.0 - d / 2.0)
        if a > 0.0 and progress < a:
            return v * progress * progress / (2.0 * a)
        if d > 0.0 and progress > 1.0 - d:
            return 1.0 - v * (1.0 - progress) ** 2 / (2.0 * d)
        return v * (progress - a / 2.0)

    def _ride(self, env: FarumasiRiderEnv, a: int, b: int, frames: int, popup: str | None = None) -> None:
        ax, ay = city_map.node_px(a)
        bx, by = city_map.node_px(b)
        dx, dy = bx - ax, by - ay
        heading = {(0, -1): 0.0, (1, 0): -90.0, (0, 1): 180.0, (-1, 0): 90.0}[(int(np.sign(dx)), int(np.sign(dy)))]
        offset = self._lane_offset(heading, a)
        old_heading, old_offset = self.heading, self.rider_off
        turn = ((heading - old_heading + 180.0) % 360.0) - 180.0
        self.walking = False
        horizontal = dx != 0
        box = (city_map.ROAD_W // 2) if horizontal else city_map.road_half(b // COLS)
        length = abs(dx) + abs(dy)
        waited = 0
        previous = 0.0
        from_rest = not self.moving
        to_rest = b == env.target_node  # only slow down where the trip really ends
        if from_rest:
            while waited < 60 and self._junction_busy(a, horizontal):  # let crossing cars clear before pulling out
                waited += 1
                self._frame(env, popup=None, t=0.0)
                self._present()
        for frame in range(frames):
            progress = (frame + 1) / frames
            # stop short of a junction while cross traffic is still clearing it
            before = length * (1.0 - self._travel(previous, from_rest, to_rest))
            after = length * (1.0 - self._travel(progress, from_rest, to_rest))
            while before > box + 30 >= after and waited < 60 and self._junction_busy(b, horizontal):
                waited += 1
                self._frame(env, popup=None, t=0.0)
                self._present()
            previous = progress
            lead = _ease(min(1.0, progress * 5.0))
            self.heading = old_heading + turn * lead
            self.rider_off = (
                old_offset[0] + (offset[0] - old_offset[0]) * lead,
                old_offset[1] + (offset[1] - old_offset[1]) * lead,
            )
            move = self._travel(progress, from_rest, to_rest)
            self.city_pos = (ax + dx * move, ay + dy * move)
            show = popup if popup is not None and progress > 0.5 else None
            self._frame(env, popup=show, t=min(1.0, (progress - 0.5) * 2.0))
            self._present()
        self.heading, self.rider_off = heading, offset
        self.city_pos = (float(bx), float(by))
        self.moving = not to_rest

    def _junction_busy(self, node: int, rider_horizontal: bool) -> bool:
        """True while a car on the crossing road is inside this junction."""
        nx, ny = city_map.node_px(node)
        reach_x = city_map.ROAD_W // 2 + 14
        reach_y = city_map.road_half(node // COLS) + 14
        for car in self.cars:
            if car["alpha"] < 0.5:
                continue
            ax, ay, bx, by, length, horizontal, half, _ja, _jb = self._geo[car["edge"]]
            if horizontal == rider_horizontal:
                continue
            if car["dir"] > 0:
                sx, sy, ux, uy = ax, ay, (bx - ax) / length, (by - ay) / length
            else:
                sx, sy, ux, uy = bx, by, (ax - bx) / length, (ay - by) / length
            shift = city_map.car_lane_offset(half)
            x = sx + ux * car["pos"] - uy * shift
            y = sy + uy * car["pos"] + ux * shift
            if abs(x - nx) < reach_x + CAR_LEN / 2 * abs(ux) and abs(y - ny) < reach_y + CAR_LEN / 2 * abs(uy):
                return True
        return False

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
        self.rain_on = env.raining
        if self.scene == "city":
            self._city(env)
        else:
            self._side_scene(env)
        if self.scene == "shelter":
            self._porch(env)
        elif self.scene != "int":
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
        """Keep every road stocked with about as many cars as its traffic level asks for."""
        if self._pygame is None:
            return
        rng = self._car_rng
        first = not self.cars
        for edge in range(N_EDGES):
            _ax, _ay, _bx, _by, length, _h, _half, junction_a, junction_b = self._geo[edge]
            level = int(env.traffic[edge])
            for direction in (1, -1):
                start = junction_a if direction > 0 else junction_b
                end = junction_b if direction > 0 else junction_a
                room = length - start - end - 56  # stay short of the stop line
                if level == 3:
                    room = length / 2 - 60 - start
                wanted = max(1, min(CARS_PER_LANE[level], int(room // CAR_PITCH)))
                lane = [car for car in self.cars if car["edge"] == edge and car["dir"] == direction]
                if first:
                    for k in range(wanted):
                        slot = room / wanted
                        self.cars.append(self._new_car(edge, direction, start + 20 + slot * (k + rng.uniform(0.0, 0.1)), 1.0))
                elif len(lane) < wanted and self.tick % 24 == 0:
                    if all(car["pos"] > start + CAR_PITCH for car in lane):
                        self.cars.append(self._new_car(edge, direction, float(start), 0.0))

    def _new_car(self, edge: int, direction: int, pos: float, alpha: float) -> dict:
        rng = self._car_rng
        return {"edge": edge, "dir": direction, "pos": pos, "v": 0.0, "art": rng.randrange(4), "pace": rng.uniform(0.92, 1.08), "alpha": alpha}

    @staticmethod
    def _signal_green(horizontal: bool, tick: int) -> bool:
        phase = tick % 390
        return phase < 120 if horizontal else 195 <= phase < 315

    def _next_lane(self, edge: int, direction: int) -> tuple[int, int, bool]:
        """The road a car drives onto after the junction: straight on, or round to the far side of the map."""
        a, b = edge_nodes(edge)
        far, near = (b, a) if direction > 0 else (a, b)
        step = far - near
        ahead = far + step
        horizontal = abs(step) == 1
        inside = 0 <= ahead < COLS * ROWS and (not horizontal or ahead // COLS == far // COLS)
        if inside:
            return edge_id(far, ahead), (1 if ahead > far else -1), False
        if horizontal:
            row = far // COLS
            first, last = row * COLS, row * COLS + COLS - 1
        else:
            col = far % COLS
            first, last = col, (ROWS - 1) * COLS + col
        if step > 0:
            return edge_id(first, first + step), 1, True
        return edge_id(last + step, last), -1, True

    def _step_cars(self, env: FarumasiRiderEnv) -> None:
        """Move the traffic one frame. Cars keep a gap, stop at red, and give way to the rider."""
        lanes: dict[tuple[int, int], list[dict]] = {}
        for car in self.cars:
            lanes.setdefault((car["edge"], car["dir"]), []).append(car)
        rider_x = self.city_pos[0] + self.rider_off[0]
        rider_y = self.city_pos[1] + self.rider_off[1]
        heading = math.radians(self.heading)
        rider_ux, rider_uy = -math.sin(heading), -math.cos(heading)
        moved: list[tuple[dict, int, int]] = []
        for (edge, direction), group in lanes.items():
            ax, ay, bx, by, length, horizontal, half, junction_a, junction_b = self._geo[edge]
            far_junction = junction_b if direction > 0 else junction_a
            stop_line = length - far_junction - CAR_LEN / 2 - 8
            if direction > 0:
                sx, sy, ux, uy = ax, ay, (bx - ax) / length, (by - ay) / length
            else:
                sx, sy, ux, uy = bx, by, (ax - bx) / length, (ay - by) / length
            speed = TRAFFIC_SPEED[int(env.traffic[edge])] * (0.8 if env.raining else 1.0)
            red = not self._signal_green(horizontal, self.tick)
            blocked = int(env.traffic[edge]) == 3
            group.sort(key=lambda car: -car["pos"])
            leader = None
            next_edge, next_dir, _wrapped = self._next_lane(edge, direction)
            beyond = [other["pos"] for other in lanes.get((next_edge, next_dir), ())]
            start_box = junction_a if direction > 0 else junction_b
            exit_blocked = bool(beyond) and min(beyond) < far_junction + CAR_LEN / 2 + CAR_PITCH + 8  # never block the junction
            far_x, far_y = city_map.node_px(edge_nodes(edge)[1 if direction > 0 else 0])
            far_row = edge_nodes(edge)[1 if direction > 0 else 0] // COLS
            rider_in_box = abs(rider_x - far_x) < city_map.ROAD_W // 2 + 30 and abs(rider_y - far_y) < city_map.road_half(far_row) + 30
            crossing = abs(ux * rider_ux + uy * rider_uy) < 0.5 or min(abs(rider_ux), abs(rider_uy)) > 0.12
            exit_blocked = exit_blocked or (rider_in_box and crossing)  # wait for the rider to clear the junction
            for car in group:
                if leader is None and beyond:
                    # the last car of the road ahead counts as the leader across the junction
                    gap = (length - car["pos"]) + min(beyond) - CAR_PITCH
                    limit_ahead = max(0.0, gap)
                else:
                    limit_ahead = None
                limit = speed * car["pace"]
                if car["pos"] > stop_line or car["pos"] < start_box + CAR_LEN / 2:
                    limit = max(limit, 3.0)  # nobody dawdles inside the junction
                if leader is not None:
                    limit = min(limit, max(0.0, leader["pos"] - car["pos"] - CAR_PITCH))
                elif limit_ahead is not None:
                    limit = min(limit, limit_ahead)
                if (red or exit_blocked) and car["pos"] <= stop_line + 0.5:
                    limit = min(limit, max(0.0, stop_line - car["pos"]))
                if blocked and car["pos"] <= length / 2 - CAR_LEN / 2 - 14:
                    limit = min(limit, max(0.0, length / 2 - CAR_LEN / 2 - 14 - car["pos"]))
                car["v"] = min(limit, car["v"] + 0.18)
                car["pos"] += car["v"]
                car["alpha"] = min(1.0, car["alpha"] + 0.05)
                leader = car
                if car["pos"] >= length - 6:
                    moved.append((car, edge, direction))
        for car, edge, direction in moved:
            next_edge, next_dir, wrapped = self._next_lane(edge, direction)
            ahead = [other["pos"] for other in self.cars if other["edge"] == next_edge and other["dir"] == next_dir and other is not car]
            if ahead and min(ahead) < CAR_PITCH + 6:
                car["pos"] = self._geo[edge][4] - 7
                car["v"] = 0.0
                continue
            car["edge"], car["dir"] = next_edge, next_dir
            car["pos"] = 0.0
            if wrapped:
                car["alpha"] = 0.0

    def _car_sprite(self, art: int, angle: int):
        key = (art, angle)
        if key not in self._car_cache:  # rotated once, then reused every frame
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
            if level == 0:
                continue
            ax, ay, bx, by, length, horizontal, half, junction_a, junction_b = self._geo[edge]
            reach = half - city_map.BIKE_LANE - 2
            if horizontal:
                rect = pygame.Rect(ax + junction_a - cam_x, ay - cam_y - reach, length - junction_a - junction_b, reach * 2)
            else:
                rect = pygame.Rect(ax - cam_x - reach, ay + junction_a - cam_y, reach * 2, length - junction_a - junction_b)
            if rect.right < 0 or rect.left > WIDTH or rect.bottom < 0 or rect.top > VIEW_H:
                continue
            pygame.draw.rect(layer, (*TRAFFIC_COLOURS[level], 74), rect, border_radius=10)
        self.screen.blit(layer, (0, 0))

        self._sync_traffic(env)
        self._step_cars(env)
        for car in self.cars:
            self._draw_car(car, cam_x, cam_y)
        for edge in range(N_EDGES):
            if int(env.traffic[edge]) == 3:
                self._barrier(edge, cam_x, cam_y)

        self._markers(env, cam_x, cam_y)
        self._rider_top(env, cam_x, cam_y)

    def _draw_car(self, car: dict, cam_x: int, cam_y: int) -> None:
        ax, ay, bx, by, length, _horizontal, half, _ja, _jb = self._geo[car["edge"]]
        if car["dir"] > 0:
            sx, sy, ux, uy = ax, ay, (bx - ax) / length, (by - ay) / length
        else:
            sx, sy, ux, uy = bx, by, (ax - bx) / length, (ay - by) / length
        shift = city_map.car_lane_offset(half)
        x = sx + ux * car["pos"] - uy * shift
        y = sy + uy * car["pos"] + ux * shift
        px, py = int(x - cam_x), int(y - cam_y)
        if -90 < px < WIDTH + 90 and -90 < py < VIEW_H + 90:
            angle = int(round(math.degrees(math.atan2(-ux, -uy))))
            sprite = self._car_sprite(car["art"], angle)
            if car["alpha"] < 1.0:
                sprite = sprite.copy()
                sprite.set_alpha(int(255 * car["alpha"]))
            self.screen.blit(sprite, sprite.get_rect(center=(px, py)))

    def _barrier(self, edge: int, cam_x: int, cam_y: int) -> None:
        pygame = self._pygame
        ax, ay, bx, by, _length, horizontal, half, _ja, _jb = self._geo[edge]
        x, y = (ax + bx) // 2 - cam_x, (ay + by) // 2 - cam_y
        if not (-100 < x < WIDTH + 100 and -100 < y < VIEW_H + 100):
            return
        across = 2 * (half - city_map.BIKE_LANE) - 4
        size = (16, across) if horizontal else (across, 16)
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
        if env.raining:
            self._umbrella_top(x - 4, y - 6)
        if env.last_wet and not self.walking:
            self._text("wet!", x + 26, y - 40, RED, self._font(18, True))

    def _umbrella_top(self, x: float, y: float) -> None:
        """Seen from above: a green and white umbrella over the rider, on a pole."""
        pygame = self._pygame
        radius, slices = 27, 8
        canopy = pygame.Surface((radius * 2 + 8, radius * 2 + 8), pygame.SRCALPHA)
        centre = (radius + 4, radius + 4)
        for k in range(slices):
            a0 = 2 * math.pi * k / slices + 0.2
            a1 = 2 * math.pi * (k + 1) / slices + 0.2
            points = [centre] + [(centre[0] + radius * math.cos(a0 + (a1 - a0) * s / 4), centre[1] + radius * math.sin(a0 + (a1 - a0) * s / 4)) for s in range(5)]
            pygame.draw.polygon(canopy, (22, 163, 74, 215) if k % 2 == 0 else (250, 250, 246, 215), points)
        pygame.draw.circle(canopy, (*INK, 255), centre, radius, 2)
        for k in range(slices):
            a0 = 2 * math.pi * k / slices + 0.2
            pygame.draw.line(canopy, (*INK, 255), centre, (centre[0] + radius * math.cos(a0), centre[1] + radius * math.sin(a0)), 1)
        pygame.draw.circle(canopy, (*INK, 255), centre, 3)
        shadow = pygame.Surface(canopy.get_size(), pygame.SRCALPHA)
        pygame.draw.circle(shadow, (0, 0, 0, 60), centre, radius)
        self.screen.blit(shadow, (x - centre[0] + 9, y - centre[1] + 9))
        self.screen.blit(canopy, (x - centre[0], y - centre[1]))

    def _umbrella_side(self, x: float, head_y: float) -> None:
        """Seen from the side: the rider walks in the rain under an umbrella."""
        pygame = self._pygame
        radius, rise = 84, 46
        cy = head_y - 6
        slices = 5
        for k in range(slices):
            a0 = math.pi * k / slices
            a1 = math.pi * (k + 1) / slices
            points = [(x + radius * math.cos(a0), cy - rise * math.sin(a0))]
            points += [(x + radius * math.cos(a0 + (a1 - a0) * s / 4), cy - rise * math.sin(a0 + (a1 - a0) * s / 4)) for s in range(1, 5)]
            points += [(x + radius * math.cos(a1), cy)]
            points += [(x + radius * math.cos(a0), cy)]
            pygame.draw.polygon(self.screen, GREEN_SOFT if k % 2 == 0 else (250, 250, 246), points)
        pygame.draw.line(self.screen, INK, (x - radius, cy), (x + radius, cy), 2)
        pygame.draw.arc(self.screen, INK, pygame.Rect(x - radius, cy - rise, radius * 2, rise * 2), 0, math.pi, 2)
        pygame.draw.line(self.screen, INK, (x, cy), (x, cy + 108), 3)
        pygame.draw.arc(self.screen, INK, pygame.Rect(x - 12, cy + 100, 12, 16), math.pi, 2 * math.pi, 3)

    # --------------------------------------------------------------- side scene

    def _side_scene(self, env: FarumasiRiderEnv) -> None:
        pygame = self._pygame
        if self.scene == "int":
            image = self.interior["pharmacy" if self.building == "pharmacy" else "lobby"]
            self.screen.blit(image, (-(image.get_width() - WIDTH) // 2, -110))
            self._actors("int", INT_FEET_Y)
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
        if self.scene != "shelter":
            self._building_sign(env)
        pygame.draw.rect(self.screen, (226, 222, 214), pygame.Rect(0, CURB_Y - 6, WIDTH, 8))
        pygame.draw.rect(self.screen, (66, 70, 76), pygame.Rect(0, CURB_Y + 2, WIDTH, VIEW_H - CURB_Y))
        for x in range(10, WIDTH, 120):
            pygame.draw.rect(self.screen, (236, 206, 96), pygame.Rect(x, 508, 60, 5), border_radius=2)
        if self.bike_x is not None:
            left = self.bike_x - self.bike_side.get_width() // 2
            top = FEET_Y + 6 - self.bike_side.get_height()
            self.screen.blit(self.bike_side, (left, top))
            if env.covered:
                self._tarp(left, top)
        if self.scene != "shelter":  # at the porch the rider is drawn after the rain
            self._actors("ext", FEET_Y)

    def _tarp(self, left: int, top: int) -> None:
        """The blue rain cover over the delivery box."""
        pygame = self._pygame
        w, h = self.bike_side.get_size()
        rect = pygame.Rect(left + int(w * 0.0), top + int(h * 0.08), int(w * 0.325), int(h * 0.40))
        pygame.draw.rect(self.screen, (24, 58, 112), rect, border_radius=8)
        pygame.draw.rect(self.screen, (92, 140, 204), rect, width=2, border_radius=8)
        for fraction in (0.33, 0.66):
            x = rect.x + int(rect.width * fraction)
            pygame.draw.line(self.screen, (92, 140, 204), (x, rect.y + 3), (x, rect.bottom - 3), 1)

    def _porch(self, env: FarumasiRiderEnv) -> None:
        """Rain falls on the road and the bike, but not under the awning where the rider waits."""
        pygame = self._pygame
        left, right = AWNING
        dry = pygame.Rect(left, AWNING_BOTTOM, right - left, FEET_Y + 12 - AWNING_BOTTOM)
        saved = self.screen.subsurface(dry).copy()
        if env.raining:
            self._rain_layer(0.55)
        elif env.rain_soon:
            self._tint(36)
        self.screen.blit(saved, dry.topleft)
        self._actors("ext", FEET_Y)
        # the awning itself, with posts down to the pavement
        for x in (left + 6, right - 10):
            pygame.draw.rect(self.screen, (92, 96, 102), pygame.Rect(x, AWNING_BOTTOM, 5, FEET_Y - AWNING_BOTTOM + 4))
        band = pygame.Rect(left, AWNING_TOP, right - left, AWNING_BOTTOM - AWNING_TOP)
        stripe = 26
        for k, x in enumerate(range(left, right, stripe)):
            colour = GREEN_SOFT if k % 2 == 0 else (250, 250, 246)
            pygame.draw.rect(self.screen, colour, pygame.Rect(x, band.y, min(stripe, right - x), band.height))
            if x + stripe <= right:
                pygame.draw.circle(self.screen, colour, (x + stripe // 2, band.bottom), stripe // 2 - 1)
        pygame.draw.rect(self.screen, INK, band, width=2)

    def _actors(self, key: str, feet_y: int) -> None:
        """The patient is drawn first so the rider stands in front of them."""
        pygame = self._pygame
        if self.patient_x is not None:
            if self.patient_walking:
                frames = self.patient_int if key == "int" else self.patient_ext
                sprite = pygame.transform.flip(frames[(self.patient_phase // 4) % 4], True, False)
                bob = int(abs(math.sin(self.patient_phase * 0.42)) * 5)
                left = self.patient_x - sprite.get_width() / 2
            else:
                sprite = self.patient_pose[key][self.pose_p or 0]
                bob = 0
                left = self.patient_x - 0.75 * sprite.get_width()
            shadow = pygame.Surface((120, 20), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow, (0, 0, 0, 90), shadow.get_rect())
            self.screen.blit(shadow, (self.patient_x - 60, feet_y - 10))
            self.screen.blit(sprite, (left, feet_y - sprite.get_height() - bob))
        if self.pose_r is not None and not self.walking:
            sprite = self.rider_pose[key][self.pose_r]
            shadow = pygame.Surface((130, 22), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow, (0, 0, 0, 90), shadow.get_rect())
            self.screen.blit(shadow, (self.rider_x - 65, feet_y - 11))
            self.screen.blit(sprite, (self.rider_x - 0.19 * sprite.get_width(), feet_y - sprite.get_height()))
            head = feet_y - sprite.get_height()
        else:
            frames = self.rider_int if key == "int" else self.rider_ext
            self._walker(frames, self.rider_x, feet_y)
            head = feet_y - frames[0].get_height()
        if self.rain_on and key == "ext" and self.scene != "shelter":
            self._umbrella_side(self.rider_x, head)

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
        self._last_env = env
        pygame.draw.rect(self.screen, (226, 220, 208), pygame.Rect(0, VIEW_H, WIDTH, HEIGHT - VIEW_H))
        pygame.draw.rect(self.screen, (10, 61, 56), pygame.Rect(0, VIEW_H, WIDTH, 4))
        self._control_bar(env)
        self._action_card(env, pygame.Rect(16, CARD_Y, 408, 210))
        self._nav_card(env, pygame.Rect(440, CARD_Y, 396, 210))
        self._order_card(env, pygame.Rect(852, CARD_Y, 412, 210))

    def _control_bar(self, env: FarumasiRiderEnv) -> None:
        pygame = self._pygame
        bar = pygame.Rect(16, CTRL_Y, WIDTH - 32, CTRL_H)
        pygame.draw.rect(self.screen, (244, 240, 232), bar, border_radius=10)
        pygame.draw.rect(self.screen, (196, 189, 175), bar, width=1, border_radius=10)
        self._mode_rects = {}
        labels = (("self", "Self Play"), ("auto", "Auto Play"), ("paused", "Pause"))
        x = bar.x + 10
        for mode, label in labels:
            active = self.control_mode == mode
            rect = pygame.Rect(x, bar.y + 6, 120, CTRL_H - 12)
            fill = GREEN if active else (255, 255, 255)
            text = (255, 255, 255) if active else GREEN_DARK
            pygame.draw.rect(self.screen, fill, rect, border_radius=8)
            pygame.draw.rect(self.screen, GREEN_DARK, rect, width=2, border_radius=8)
            self._text(label, rect.centerx, rect.centery, text, self._font(15, True), center=True)
            self._mode_rects[mode] = rect
            x += 128
        hint = {
            "auto": "Agent chooses each step",
            "self": "Click a legal action below",
            "paused": "Frozen - press Auto Play or Self Play to continue",
        }[self.control_mode]
        self._text(hint, x + 8, bar.y + 12, MUTED, self._font(14))

    def _card(self, rect) -> None:
        pygame = self._pygame
        pygame.draw.rect(self.screen, (196, 189, 175), rect.move(0, 3), border_radius=16)
        pygame.draw.rect(self.screen, CARD, rect, border_radius=16)

    def _action_card(self, env: FarumasiRiderEnv, rect) -> None:
        self._card(rect)
        self._action_rects = []
        if self.control_mode == "self":
            self._self_play_actions(env, rect)
            return
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

    def _self_play_actions(self, env: FarumasiRiderEnv, rect) -> None:
        pygame = self._pygame
        self._text("YOUR MOVE", rect.x + 16, rect.y + 10, MUTED, self._font(14, True))
        self._text(env.current_place(), rect.x + 120, rect.y + 10, INK, self._font(14, True))
        mask = env.action_mask()
        cols = 4
        pad = 8
        top = rect.y + 36
        cell_w = (rect.width - 24 - pad * (cols - 1)) / cols
        cell_h = 36
        for action in range(N_ACTIONS):
            row, col = divmod(action, cols)
            # two rows of 4 for ride/cover/wait/enter/collect..., then next rows
            x = rect.x + 12 + col * (cell_w + pad)
            y = top + row * (cell_h + 6)
            if y + cell_h > rect.bottom - 8:
                break
            box = pygame.Rect(int(x), int(y), int(cell_w), cell_h)
            legal = mask[action] > 0.5
            if legal:
                fill = (220, 252, 231)
                border = GREEN
                text = GREEN_DARK
                self._action_rects.append((box, action))
            else:
                fill = (238, 236, 230)
                border = (200, 196, 186)
                text = (160, 160, 154)
            pygame.draw.rect(self.screen, fill, box, border_radius=8)
            pygame.draw.rect(self.screen, border, box, width=2, border_radius=8)
            self._text(SHORT_ACTIONS[action], box.centerx, box.centery, text, self._font(13, True), center=True)

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
        left = 24 if self.scene in ("ext", "shelter") else WIDTH // 2 - 280
        self.screen.blit(panel, (left, 72 + slide))
        self._text(title, left + 280, 100 + slide, (255, 255, 255), self._font(34, True), center=True)
        self._text(detail, left + 280, 142 + slide, (214, 235, 228), self._font(17), center=True)
        if key in ("rx_ok", "rx_wrong"):
            self._rx_paper(env, key == "rx_ok", t)
        if key == "gate_opened":
            self._speech("Knock knock!", int(self.rider_x) + 100, 210)
        reward = env.last_reward
        if key in ("handed_over", "order_collected") and reward:
            self._float_reward(reward, t)

    def _banner(self, env: FarumasiRiderEnv, key: str):
        med = MED_NAMES[env.med]
        table = {
            "entered_pharmacy": ("KIGALI CITY PHARMACY", "Walking in to collect the order", GREEN_SOFT),
            "order_collected": ("ORDER COLLECTED", f"{med} sealed in the FARUMASI box", GREEN_SOFT),
            "covered_box": ("BOX COVERED", "Rain cover on. The medicine stays dry", BLUE),
            "waited": ("WAITING", "Pulled over at a house. The clock keeps running", AMBER),
            "sheltering": ("TAKING SHELTER", "Under a porch from the rain. The clock keeps running", BLUE),
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
        paper.center = (WIDTH - 170, 300 - int(10 * math.sin(t * 6)))
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
        real_screen = self.screen
        sheet = pygame.Surface((300, 470), pygame.SRCALPHA)
        self.screen = sheet
        body = pygame.Rect(0, 0, 300, 470)
        body.center = (150, 235)
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
        self.screen = real_screen
        scale = 0.62 * (0.75 + 0.25 * _ease(t * 5.0))
        inset = pygame.transform.smoothscale(sheet, (int(300 * scale), int(470 * scale)))
        self.screen.blit(inset, inset.get_rect(center=(1090, 300)))

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

    env = build("hotel" if stage in ("lobby", "scan") else ("home" if stage in ("home", "handover") else None))
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
    elif stage in ("lobby", "home"):
        env.rx_required = False
        while env.phase != P_PATIENT:
            env.step(heuristic_action(env))
    elif stage == "handover":
        env.rx_required = False
        while env.phase != P_PATIENT:
            env.step(heuristic_action(env))
        env.step(HAND)
    elif stage == "scan":
        env.rx_required = False
        while env.outcome == "":
            env.step(heuristic_action(env))
    frame = env.render()
    env.close()
    import pygame

    pygame.image.save(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), path)
    return path
