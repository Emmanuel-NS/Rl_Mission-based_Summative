"""FARUMASI rider: one medicine delivery across Kigali.

The rider starts outside Kigali City Pharmacy. She walks in, collects the
order, rides to a home, an office or a hotel, finds the way through live
traffic, covers the box when it rains, parks, walks in on foot, checks the
prescription when one is required, hands the box over, and scans the QR code.
If the prescription is wrong she carries the medicine back to the pharmacy.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gymnasium
import numpy as np
from gymnasium import spaces

ENV_VERSION = "rider-delivery-3"
MAX_STEPS = 80
DEADLINE = 30.0
CANCEL_AFTER = DEADLINE + 25.0
TIME_COST = 0.03
SHAPING = 0.04

COLS, ROWS = 6, 4
N_NODES = COLS * ROWS
N_EDGES = ROWS * (COLS - 1) + (ROWS - 1) * COLS
PHARMACY_NODE = 2 * COLS + 0
PHARMACY_NAME = "Kigali City Pharmacy"
ARTERY_ROW = 2
EXTRA_MINUTES = (0.0, 1.0, 3.0, 6.0)
TRAFFIC_NAMES = ("clear", "busy", "heavy", "blocked")

MEDICINES = (
    ("Metformin 500mg", True),
    ("Amoxicillin 250mg", True),
    ("ORS sachets", False),
    ("Paracetamol 500mg", False),
)
MED_NAMES = tuple(name for name, _ in MEDICINES)

RIDE_N, RIDE_E, RIDE_S, RIDE_W = 0, 1, 2, 3
COVER, WAIT, ENTER, COLLECT, PARK, KNOCK, CARRY = 4, 5, 6, 7, 8, 9, 10
VERIFY, HAND, SCAN, REMOUNT, RETURN_MEDS = 11, 12, 13, 14, 15
N_ACTIONS = 16
ACTION_NAMES = (
    "ride_north",
    "ride_east",
    "ride_south",
    "ride_west",
    "cover_box",
    "wait",
    "enter_pharmacy",
    "collect_order",
    "park_and_walk_in",
    "knock_or_check_in",
    "carry_to_patient",
    "verify_prescription",
    "hand_over",
    "scan_qr",
    "walk_back_to_bike",
    "return_medicine",
)
ACTION_LABELS = (
    "RIDING NORTH",
    "RIDING EAST",
    "RIDING SOUTH",
    "RIDING WEST",
    "COVERING THE BOX",
    "WAITING",
    "ENTERING THE PHARMACY",
    "COLLECTING THE ORDER",
    "PARKING AND WALKING IN",
    "KNOCKING AT THE GATE",
    "CARRYING TO THE PATIENT",
    "CHECKING THE PRESCRIPTION",
    "HANDING OVER",
    "SCANNING THE QR CODE",
    "WALKING BACK TO THE BIKE",
    "RETURNING THE MEDICINE",
)
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))

P_OUTSIDE, P_INSIDE, P_RIDING, P_ENTRANCE, P_BUILDING, P_PATIENT = range(6)
PHASE_NAMES = ("outside pharmacy", "inside pharmacy", "riding", "at the entrance", "inside the building", "with the patient")

AVENUES = ("KG 7 Ave", "KN 4 Ave", "KN 3 Rd", "KG 11 Ave")
STREETS = ("KN 1 Rd", "KG 2 St", "KG 9 Ave", "KG 5 Ave", "KG 13 Ave", "KN 8 Rd")


@dataclass(frozen=True)
class Destination:
    name: str
    kind: str
    col: int
    row: int
    area: str

    @property
    def node(self) -> int:
        return self.row * COLS + self.col


DESTINATIONS = (
    Destination("Home, Kimihurura", "home", 3, 0, "Kimihurura"),
    Destination("Home, Remera", "home", 5, 2, "Remera"),
    Destination("Kigali Heights, Kacyiru", "office", 4, 0, "Kacyiru"),
    Destination("Offices, Nyarutarama", "office", 3, 3, "Nyarutarama"),
    Destination("Radisson Blu Hotel", "hotel", 5, 3, "Kimihurura"),
    Destination("Marriott Hotel", "hotel", 4, 1, "Nyarugenge"),
)
KINDS = ("home", "office", "hotel")
PATIENT_NAMES = ("Uwase Aline", "Ingabire Claire", "Mukamana Diane", "Uwimana Grace", "Mutesi Sandra", "Umutoni Esther")


def node_xy(node: int) -> tuple[int, int]:
    return node % COLS, node // COLS


def edge_id(a: int, b: int) -> int:
    ca, ra = node_xy(a)
    cb, rb = node_xy(b)
    if ra == rb:
        return ra * (COLS - 1) + min(ca, cb)
    return ROWS * (COLS - 1) + min(ra, rb) * COLS + ca


def edge_nodes(edge: int) -> tuple[int, int]:
    horizontal = ROWS * (COLS - 1)
    if edge < horizontal:
        row, col = divmod(edge, COLS - 1)
        return row * COLS + col, row * COLS + col + 1
    row, col = divmod(edge - horizontal, COLS)
    return row * COLS + col, (row + 1) * COLS + col


def neighbour(node: int, direction: int) -> int | None:
    col, row = node_xy(node)
    dc, dr = DIRECTIONS[direction]
    nc, nr = col + dc, row + dr
    if 0 <= nc < COLS and 0 <= nr < ROWS:
        return nr * COLS + nc
    return None


def place_name(node: int) -> str:
    col, row = node_xy(node)
    return f"{AVENUES[row]} and {STREETS[col]}"


GLOBAL_FEATURES = 6 + 2 + 2 + 2 + 1 + 4 + 4 + 3 + 3 + 1 + len(MEDICINES) + 4
OBS_DIM = GLOBAL_FEATURES + N_EDGES + N_ACTIONS


def observation_catalogue() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    def add(name: str, range_: str, source: str, cadence: str = "every step") -> None:
        rows.append(
            {
                "index": len(rows),
                "name": name,
                "range": range_,
                "source": source,
                "cadence": cadence,
                "agent_path": "read by the policy before every action",
            }
        )

    for phase in PHASE_NAMES:
        add(f"phase_{phase.replace(' ', '_')}", "0 or 1", "where the rider is in the delivery")
    add("x", "0 to 1", "rider column on the Kigali grid")
    add("y", "0 to 1", "rider row on the Kigali grid")
    add("to_patient_dx", "0 to 1", "patient column minus rider column, rescaled")
    add("to_patient_dy", "0 to 1", "patient row minus rider row, rescaled")
    add("to_pharmacy_dx", "0 to 1", "pharmacy column minus rider column, rescaled")
    add("to_pharmacy_dy", "0 to 1", "pharmacy row minus rider row, rescaled")
    add("clock", "0 to 1", f"minutes used divided by {int(CANCEL_AFTER)}")
    add("raining", "0 or 1", "rain has started")
    add("rain_soon", "0 or 1", "dark clouds: rain starts within two steps")
    add("box_covered", "0 or 1", "the rain cover is on the delivery box")
    add("box_condition", "0 to 1", "falls when an uncovered box is rained on")
    add("rx_required", "0 or 1", "the order needs a valid prescription")
    add("rx_checked", "0 or 1", "the rider has checked the prescription")
    add("rx_ok_known", "0 or 1", "checked and valid")
    add("rx_bad_known", "0 or 1", "checked and wrong")
    for kind in KINDS:
        add(f"patient_at_{kind}", "0 or 1", "kind of address, shown on the order")
    add("returning", "0 or 1", "carrying the medicine back to the pharmacy")
    add("handed_over", "0 or 1", "the box is in the patient's hands")
    add("entrance_cleared", "0 or 1", "gate opened or reception has checked the rider in")
    add("steps", "0 to 1", f"decisions taken divided by {MAX_STEPS}")
    for number, name in enumerate(MED_NAMES):
        add(f"medicine_{number}", "0 or 1", f"order contains {name}")
    for name in ("north", "east", "south", "west"):
        add(f"traffic_{name}", "0 to 1", "traffic on the road leaving this junction (0 clear, 1 blocked)")
    for edge in range(N_EDGES):
        a, b = edge_nodes(edge)
        add(f"traffic_edge_{edge}", "0 to 1", f"live traffic between {place_name(a)} and {place_name(b)}")
    for number, name in enumerate(ACTION_NAMES):
        add(f"legal_{name}", "0 or 1", "legal-action mask", "every step")
    assert len(rows) == OBS_DIM, (len(rows), OBS_DIM)
    return rows


class FarumasiRiderEnv(gymnasium.Env):
    """One delivery from Kigali City Pharmacy to a patient."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._renderer = None
        self.np_random = np.random.default_rng()
        self.traffic = np.zeros(N_EDGES, dtype=np.int32)
        self._init_state()

    def _init_state(self) -> None:
        self.dest = DESTINATIONS[0]
        self.med = 0
        self.phase = P_OUTSIDE
        self.node = PHARMACY_NODE
        self.prev_node = PHARMACY_NODE
        self.steps = 0
        self.minutes = 0.0
        self.rain_at = 10**6
        self.incident_at = 10**6
        self.incident_edge = -1
        self.rush = False
        self.covered = False
        self.condition = 1.0
        self.rx_required = False
        self.rx_valid = True
        self.rx_checked = False
        self.returning = False
        self.handed = False
        self.cleared = False
        self.outcome = ""
        self.return_so_far = 0.0
        self.last_event = "mission_started"
        self.last_reward = 0.0
        self.last_action = -1
        self.last_minutes = 0.0
        self.last_wet = False
        self.clock_start = 17 * 60 + 5
        self.order_id = 2000
        self.patient = PATIENT_NAMES[0]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        self._init_state()
        rng = self.np_random
        self.dest = DESTINATIONS[int(rng.integers(0, len(DESTINATIONS)))]
        self.med = int(rng.integers(0, len(MEDICINES)))
        self.order_id = 2000 + int(rng.integers(0, 900))
        self.patient = PATIENT_NAMES[int(rng.integers(0, len(PATIENT_NAMES)))]
        self.rx_required = MEDICINES[self.med][1]
        self.rx_valid = not (self.rx_required and rng.random() < 0.3)
        self.rush = bool(rng.random() < 0.6)
        self.clock_start = (17 * 60 + 5) if self.rush else (11 * 60 + 20)
        self.traffic = rng.choice([0, 1, 2], size=N_EDGES, p=[0.55, 0.33, 0.12]).astype(np.int32)
        if self.rush:
            for col in range(COLS - 1):
                self.traffic[edge_id(ARTERY_ROW * COLS + col, ARTERY_ROW * COLS + col + 1)] = 2 if rng.random() < 0.65 else 3
        self.rain_at = int(rng.integers(3, 11)) if rng.random() < 0.65 else 10**6
        self.incident_at = int(rng.integers(5, 10))
        return self._observe(), self._info()

    # ------------------------------------------------------------------ step

    @property
    def raining(self) -> bool:
        return self.steps >= self.rain_at

    @property
    def rain_soon(self) -> bool:
        return (not self.raining) and (self.rain_at - self.steps) <= 2

    @property
    def target_node(self) -> int:
        return PHARMACY_NODE if self.returning else self.dest.node

    def edge_minutes(self, edge: int) -> float:
        return 1.0 + EXTRA_MINUTES[int(self.traffic[edge])] + (1.0 if self.raining else 0.0)

    def time_to(self, target: int) -> np.ndarray:
        """Traffic-aware minutes from every junction to target (Dijkstra)."""
        dist = np.full(N_NODES, np.inf)
        dist[target] = 0.0
        done = np.zeros(N_NODES, dtype=bool)
        for _ in range(N_NODES):
            candidates = np.where(done, np.inf, dist)
            node = int(np.argmin(candidates))
            if not np.isfinite(candidates[node]):
                break
            done[node] = True
            for direction in range(4):
                other = neighbour(node, direction)
                if other is None:
                    continue
                cost = dist[node] + self.edge_minutes(edge_id(node, other))
                if cost < dist[other]:
                    dist[other] = cost
        return dist

    def _tick_world(self) -> None:
        if self.steps == self.incident_at:
            self.incident_edge = int(self.np_random.integers(0, N_EDGES))
            self.traffic[self.incident_edge] = 3
        elif self.steps % 6 == 0:
            for edge in self.np_random.integers(0, N_EDGES, size=3):
                shift = 1 if self.np_random.random() < 0.5 else -1
                self.traffic[int(edge)] = int(np.clip(self.traffic[int(edge)] + shift, 0, 2))

    def step(self, action: int):
        action = int(action)
        self.last_action = action
        self.prev_node = self.node
        self.steps += 1
        self._tick_world()
        legal = self.action_mask()
        was_outdoors = self.phase in (P_RIDING, P_ENTRANCE) and self.phase_has_box
        terminated = False
        reward = 0.0
        if legal[action] < 0.5:
            reward, minutes, event = -0.2, 1.0, "invalid"
        else:
            reward, minutes, event, terminated = self._act(action)

        self.last_wet = False
        if was_outdoors and self.raining and not self.covered and not terminated:
            damage = 0.12 if action <= RIDE_W else 0.07
            self.condition = max(0.0, self.condition - damage)
            reward -= damage
            self.last_wet = True
        if self.condition <= 0.3 and not terminated:
            reward -= 4.0
            event, terminated = "medicine_ruined", True
            self.outcome = "ruined"

        self.minutes += minutes
        self.last_minutes = minutes
        reward -= TIME_COST * minutes
        if self.minutes >= CANCEL_AFTER and not terminated:
            reward -= 3.0
            event, terminated = "order_cancelled", True
            self.outcome = "cancelled"
        truncated = self.steps >= MAX_STEPS and not terminated
        if truncated:
            reward -= 2.0
            event = "delivery_abandoned"
            self.outcome = "abandoned"
        self.return_so_far += reward
        self.last_event = event
        self.last_reward = reward
        return self._observe(), float(reward), terminated, truncated, self._info()

    @property
    def phase_has_box(self) -> bool:
        return self.phase >= P_RIDING

    def _act(self, action: int):
        minutes, reward, terminated = 1.0, 0.0, False
        event = "waited"
        if action <= RIDE_W:
            target = neighbour(self.node, action)
            before = self.time_to(self.target_node)
            edge = edge_id(self.node, target)
            minutes = self.edge_minutes(edge)
            self.node = target
            reward += SHAPING * (before[self.prev_node] - before[self.node])
            event = "rode"
        elif action == COVER:
            self.covered = True
            event = "covered_box"
        elif action == WAIT:
            event = "waited"
        elif action == ENTER:
            self.phase = P_INSIDE
            event = "entered_pharmacy"
        elif action == COLLECT:
            self.phase = P_RIDING
            reward += 0.2
            event = "order_collected"
        elif action == PARK:
            self.phase = P_ENTRANCE
            event = "parked"
        elif action == KNOCK:
            self.phase = P_BUILDING
            self.cleared = True
            minutes = 2.0
            event = "gate_opened" if self.dest.kind == "home" else "checked_in"
        elif action == CARRY:
            self.phase = P_PATIENT
            minutes = 2.0
            event = "reached_patient"
        elif action == VERIFY:
            self.rx_checked = True
            reward += 0.3
            event = "rx_ok" if self.rx_valid else "rx_wrong"
        elif action == HAND:
            if self.rx_required and (not self.rx_checked or not self.rx_valid):
                reward -= 6.0
                event, terminated = "rx_violation", True
                self.outcome = "violation"
            else:
                self.handed = True
                self.covered = False
                reward += 0.5
                event = "handed_over"
        elif action == SCAN:
            quality = 0.6 + 0.4 * self.condition
            after = self.minutes + minutes
            bonus = 2.0 * float(np.clip((DEADLINE - after) / DEADLINE, 0.0, 1.0))
            late = 0.12 * max(0.0, after - DEADLINE)
            reward += (5.0 + bonus) * quality - late
            event, terminated = "delivered", True
            self.outcome = "delivered"
        elif action == REMOUNT:
            self.phase = P_RIDING
            self.returning = True
            self.covered = False
            minutes = 2.0
            event = "returning"
        elif action == RETURN_MEDS:
            reward += 2.5
            event, terminated = "returned_to_pharmacy", True
            self.outcome = "returned"
        return reward, minutes, event, terminated

    # ------------------------------------------------------------------ mask

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        phase = self.phase
        if phase == P_OUTSIDE:
            mask[[ENTER, WAIT]] = 1.0
        elif phase == P_INSIDE:
            mask[[COLLECT, WAIT]] = 1.0
        elif phase == P_RIDING:
            for direction in range(4):
                if neighbour(self.node, direction) is not None:
                    mask[direction] = 1.0
            mask[WAIT] = 1.0
            if not self.covered:
                mask[COVER] = 1.0
            if not self.returning and self.node == self.dest.node:
                mask[PARK] = 1.0
            if self.returning and self.node == PHARMACY_NODE:
                mask[RETURN_MEDS] = 1.0
        elif phase == P_ENTRANCE:
            mask[[KNOCK, WAIT]] = 1.0
            if not self.covered:
                mask[COVER] = 1.0
        elif phase == P_BUILDING:
            mask[CARRY] = 1.0
        elif phase == P_PATIENT:
            if self.handed:
                mask[SCAN] = 1.0
            else:
                mask[HAND] = 1.0
                if self.rx_required and not self.rx_checked:
                    mask[VERIFY] = 1.0
                if self.rx_checked and not self.rx_valid:
                    mask[REMOUNT] = 1.0
        return mask

    def action_label(self, action: int | None = None) -> str:
        action = self.last_action if action is None else action
        if action < 0:
            return "MISSION START"
        if action == KNOCK and self.dest.kind != "home":
            return "CHECKING IN AT RECEPTION"
        return ACTION_LABELS[action]

    # ----------------------------------------------------------- observation

    def _observe(self) -> np.ndarray:
        values: list[float] = []
        for phase in range(6):
            values.append(1.0 if self.phase == phase else 0.0)
        col, row = node_xy(self.node)
        values += [col / (COLS - 1), row / (ROWS - 1)]
        dcol, drow = node_xy(self.dest.node)
        values += [(dcol - col + COLS - 1) / (2 * (COLS - 1)), (drow - row + ROWS - 1) / (2 * (ROWS - 1))]
        pcol, prow = node_xy(PHARMACY_NODE)
        values += [(pcol - col + COLS - 1) / (2 * (COLS - 1)), (prow - row + ROWS - 1) / (2 * (ROWS - 1))]
        values.append(min(1.0, self.minutes / CANCEL_AFTER))
        values += [float(self.raining), float(self.rain_soon), float(self.covered), self.condition]
        values += [
            float(self.rx_required),
            float(self.rx_checked),
            float(self.rx_checked and self.rx_valid),
            float(self.rx_checked and not self.rx_valid),
        ]
        for kind in KINDS:
            values.append(1.0 if self.dest.kind == kind else 0.0)
        values += [float(self.returning), float(self.handed), float(self.cleared)]
        values.append(self.steps / MAX_STEPS)
        for number in range(len(MEDICINES)):
            values.append(1.0 if self.med == number else 0.0)
        for direction in range(4):
            other = neighbour(self.node, direction)
            values.append(0.0 if other is None else self.traffic[edge_id(self.node, other)] / 3.0)
        values += list(self.traffic.astype(np.float32) / 3.0)
        values += list(self.action_mask())
        observation = np.asarray(values, dtype=np.float32)
        assert observation.shape == (OBS_DIM,), observation.shape
        return np.clip(observation, 0.0, 1.0)

    def current_place(self) -> str:
        if self.phase in (P_OUTSIDE, P_INSIDE):
            return PHARMACY_NAME
        if self.phase in (P_ENTRANCE, P_BUILDING, P_PATIENT):
            return self.dest.name
        if self.node == self.dest.node and not self.returning:
            return self.dest.name
        if self.node == PHARMACY_NODE:
            return PHARMACY_NAME
        return place_name(self.node)

    def _info(self) -> dict:
        label = self.action_label()
        return {
            "event": self.last_event,
            "action_mask": self.action_mask(),
            "condition": round(self.condition, 3),
            "minutes": round(self.minutes, 1),
            "shift_return": self.return_so_far,
            "outcome": self.outcome,
            "decision": {
                "action": ACTION_NAMES[self.last_action] if self.last_action >= 0 else None,
                "action_label": label,
                "event": self.last_event,
                "reward": self.last_reward,
                "medicine": MED_NAMES[self.med],
                "pharmacy_name": self.current_place(),
                "shift_return": self.return_so_far,
            },
        }

    # ------------------------------------------------------------- rendering

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import DeliveryRenderer

            self._renderer = DeliveryRenderer(self.render_mode == "human")
        return self._renderer.draw(self)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---------------------------------------------- metrics the trainers read

    @property
    def delivered(self) -> int:
        return int(self.outcome == "delivered")

    @property
    def returned_ok(self) -> int:
        return int(self.outcome == "returned")

    @property
    def urgent_met_filled(self) -> int:
        return self.delivered + self.returned_ok

    @property
    def urgent_met_total(self) -> int:
        return 1

    @property
    def routine_met_filled(self) -> int:
        return 0

    @property
    def dispensed(self) -> int:
        return self.delivered

    @property
    def referred(self) -> int:
        return self.returned_ok
