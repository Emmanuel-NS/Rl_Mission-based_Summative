"""A patient searches Kigali for one genuine pack.

One episode is one person with a prescription. The city is a small map.
Licensed pharmacies may be out of stock, closed, or expensive. Unlicensed
stalls advertise the same medicine cheaply, and buying there can be harmful.
FARUMASI is an action: open it and every partner pharmacy shows live stock
and price, and every stall is marked unlicensed. Walking, waiting, and
guessing all spend the patient's health.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gymnasium
import numpy as np
from gymnasium import spaces

METFORMIN = 0
AMOXICILLIN = 1
ORS = 2
MED_NAMES = ("Metformin 500mg", "Amoxicillin 250mg", "ORS sachets")
DISTRICT_NAMES = ("Nyarugenge", "Gasabo", "Kicukiro")

NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
APP, INSPECT, BUY = 4, 5, 6
N_ACTIONS = 7
ACTION_NAMES = ("north", "east", "south", "west", "open_farumasi", "inspect", "buy")
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

COLS, ROWS = 21, 11
MAX_STEPS = 64
RESTOCK_AT = 14
ENV_VERSION = "patient-map-1"
DECAY = (0.028, 0.032, 0.020)

N_PARTNERS = 6
N_KIOSKS = 3
GLOBAL_FEATURES = 12
PARTNER_FEATURES = 6
KIOSK_FEATURES = 5
PUBLIC_FEATURES = 4
LOCAL_FEATURES = 4
OBS_DIM = (
    GLOBAL_FEATURES
    + N_PARTNERS * PARTNER_FEATURES
    + N_KIOSKS * KIOSK_FEATURES
    + PUBLIC_FEATURES
    + LOCAL_FEATURES
    + N_ACTIONS
)


@dataclass(frozen=True)
class Site:
    site_id: int
    kind: str
    name: str
    x: int
    y: int
    district: str
    price: float


PARTNERS = (
    Site(0, "partner", "Nyamirambo Community", 2, 8, "Nyarugenge", 1.35),
    Site(1, "partner", "Kigali City Pharmacy", 5, 3, "Nyarugenge", 1.90),
    Site(2, "partner", "Remera Care", 8, 8, "Gasabo", 1.20),
    Site(3, "partner", "Kimironko Health Mart", 12, 3, "Gasabo", 1.75),
    Site(4, "partner", "Kicukiro Centre", 16, 8, "Kicukiro", 1.40),
    Site(5, "partner", "Gikondo Pharmacy", 19, 3, "Kicukiro", 1.95),
)
KIOSKS = (
    Site(0, "kiosk", "Nyamirambo roadside stall", 6, 5, "Nyarugenge", 0.70),
    Site(1, "kiosk", "Remera market table", 10, 6, "Gasabo", 0.65),
    Site(2, "kiosk", "Gikondo backstreet", 14, 5, "Kicukiro", 0.60),
)
PUBLIC = Site(0, "public", "District health centre", 10, 1, "Gasabo", 1.00)
HOMES = ((0, 0), (0, 10), (20, 0), (20, 10))


def _district(x: int) -> int:
    if x < 7:
        return 0
    if x < 14:
        return 1
    return 2


def _unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _rel(delta: int, span: int) -> float:
    return _unit((delta / span + 1.0) / 2.0)


def observation_catalogue() -> list[dict[str, str | int]]:
    """Meaning, range, source, and update cadence of every observation index."""
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

    add("patient_x", "0 to 1 (column / 14)", "patient position on the Kigali map")
    add("patient_y", "0 to 1 (row / 8)", "patient position on the Kigali map")
    add("health", "0 to 1", "falls every step; faster for an acute medicine")
    add("step", "0 to 1 (step / 64)", "decisions taken this search")
    add("medicine_metformin", "0 or 1", "the prescription drawn at the start of the episode")
    add("medicine_amoxicillin", "0 or 1", "the prescription drawn at the start of the episode")
    add("medicine_ors", "0 or 1", "the prescription drawn at the start of the episode")
    add("urgency", "0.4 or 1", "1 for metformin and amoxicillin, 0.4 for ORS")
    add("farumasi_open", "0 or 1", "1 after the patient opens the FARUMASI app")
    add("standing_on_a_site", "0 or 1", "1 if this tile is a pharmacy, stall, or clinic")
    add("site_known", "0 or 1", "1 if the site under the patient has been revealed")
    add("restock_happened", "0 or 1", "1 after the mid-search partner delivery", "once, at step 14")
    for site in PARTNERS:
        add(f"p{site.site_id}_rel_x", "0 to 1", f"map offset from the patient to {site.name}")
        add(f"p{site.site_id}_rel_y", "0 to 1", f"map offset from the patient to {site.name}")
        add(
            f"p{site.site_id}_stock",
            "0 to 1 (packs / 2), else 0",
            "partner shelf for the prescribed medicine; hidden until FARUMASI or a visit",
        )
        add(
            f"p{site.site_id}_price",
            "0 to 1 ((price - 1) / 1), else 0",
            "licensed list price; hidden until FARUMASI or a visit",
        )
        add(
            f"p{site.site_id}_available",
            "0 or 1",
            "1 only when the patient knows this partner is open and in stock",
        )
        add(f"p{site.site_id}_known", "0 or 1", "1 after FARUMASI or after inspecting this pharmacy")
    for site in KIOSKS:
        add(f"k{site.site_id}_rel_x", "0 to 1", f"map offset from the patient to {site.name}")
        add(f"k{site.site_id}_rel_y", "0 to 1", f"map offset from the patient to {site.name}")
        add(f"k{site.site_id}_advertised_price", "0.25", "chalkboard price, visible without entering")
        add(f"k{site.site_id}_known_unlicensed", "0 or 1", "1 after FARUMASI or after inspecting the stall")
        add(f"k{site.site_id}_here", "0 or 1", "1 if the patient is standing on this stall")
    add("public_rel_x", "0 to 1", "map offset to the district health centre")
    add("public_rel_y", "0 to 1", "map offset to the district health centre")
    add("public_signed", "1", "the clinic is a signed public building")
    add("public_distance", "0 to 1", "Manhattan distance to the clinic, divided by 30")
    add("here_partner", "0 or 1", "tile under the patient")
    add("here_kiosk", "0 or 1", "tile under the patient")
    add("here_public", "0 or 1", "tile under the patient")
    add("here_home", "0 or 1", "tile under the patient")
    for name in ACTION_NAMES:
        add(f"mask_{name}", "0 or 1", "1 if this action is legal now", "every step, also enforced by the policy")
    if len(rows) != OBS_DIM:
        raise RuntimeError(f"catalogue has {len(rows)} rows, OBS_DIM is {OBS_DIM}")
    return rows


class FarumasiShiftEnv(gymnasium.Env):
    """Patient search. The class name stays so the trainers keep their import."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._renderer = None
        self.np_random = np.random.default_rng()
        self.sites = { (site.x, site.y): site for site in (*PARTNERS, *KIOSKS, PUBLIC) }
        self.px = self.py = 0
        self.home = (0, 7)
        self.med = METFORMIN
        self.health = 1.0
        self.steps = 0
        self.app_open = False
        self.restocked = False
        self.partner_stock = np.zeros(N_PARTNERS, dtype=np.int32)
        self.partner_open = np.ones(N_PARTNERS, dtype=bool)
        self.partner_known = np.zeros(N_PARTNERS, dtype=bool)
        self.kiosk_known = np.zeros(N_KIOSKS, dtype=bool)
        self.restock_id = 0
        self.obtained = False
        self.poisoned = False
        self.bought_public = False
        self.bought_partner = False
        self.return_so_far = 0.0
        self.last_event = "start"
        self.last_reward = 0.0
        self.last_action = -1

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        self.med = int(self.np_random.integers(0, 3))
        self.home = HOMES[int(self.np_random.integers(0, len(HOMES)))]
        self.px, self.py = self.home
        self.health = 1.0
        self.steps = 0
        self.app_open = False
        self.restocked = False
        self.obtained = False
        self.poisoned = False
        self.bought_public = False
        self.bought_partner = False
        self.return_so_far = 0.0
        self.last_event = "search_started"
        self.last_reward = 0.0
        self.last_action = -1
        self.partner_stock[:] = 0
        self.partner_open[:] = True
        self.partner_known[:] = False
        self.kiosk_known[:] = False
        home_district = _district(self.home[0])
        far_district = (home_district + 2) % 3
        grouped = {0: [], 1: [], 2: []}
        for site in PARTNERS:
            grouped[_district(site.x)].append(site.site_id)
        local = grouped[home_district]
        far = grouped[far_district]
        self.partner_open[local[0]] = False
        self.restock_id = int(local[1])
        self.partner_stock[far[0]] = 2
        self.partner_stock[far[1]] = 1
        observation = self._observe()
        return observation, self._info()

    def step(self, action: int):
        action = int(action)
        self.last_action = action
        self.steps += 1
        self._apply_restock()
        reward = 0.0
        terminated = False
        event = "walked"
        if action in (NORTH, EAST, SOUTH, WEST):
            reward, event = self._move(action)
        elif action == APP:
            reward, event = self._open_app()
        elif action == INSPECT:
            reward, event = self._inspect()
        elif action == BUY:
            reward, event, terminated = self._buy()
        else:
            reward, event = -0.2, "invalid"
        self.health = max(0.0, self.health - DECAY[self.med])
        if self.health <= 0.0 and not terminated:
            reward -= 2.5
            event = "health_failed"
            terminated = True
        truncated = self.steps >= MAX_STEPS and not terminated
        if truncated:
            reward -= 1.5
            event = "search_abandoned"
        self.return_so_far += reward
        self.last_event = event
        self.last_reward = reward
        return self._observe(), float(reward), terminated, truncated, self._info()

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for action, (dx, dy) in enumerate(DELTAS):
            nx, ny = self.px + dx, self.py + dy
            if 0 <= nx < COLS and 0 <= ny < ROWS:
                mask[action] = 1.0
        mask[APP] = 1.0
        site = self.sites.get((self.px, self.py))
        if site is not None and not self._known(site):
            mask[INSPECT] = 1.0
        if site is not None and self._can_buy(site):
            mask[BUY] = 1.0
        return mask

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import MapRenderer

            self._renderer = MapRenderer(self.render_mode == "human")
        return self._renderer.draw(self)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @property
    def urgent_met_filled(self) -> int:
        return int(self.obtained and not self.poisoned)

    @property
    def urgent_met_total(self) -> int:
        return 1

    @property
    def routine_met_filled(self) -> int:
        return 0

    @property
    def dispensed(self) -> int:
        return int(self.bought_partner)

    @property
    def referred(self) -> int:
        return int(self.bought_public)

    def _known(self, site: Site) -> bool:
        if site.kind == "partner":
            return bool(self.partner_known[site.site_id])
        if site.kind == "kiosk":
            return bool(self.kiosk_known[site.site_id])
        return True

    def _can_buy(self, site: Site) -> bool:
        if site.kind == "kiosk":
            return True
        if site.kind == "public":
            return True
        return bool(
            self.partner_known[site.site_id]
            and self.partner_open[site.site_id]
            and self.partner_stock[site.site_id] > 0
        )

    def _move(self, action: int) -> tuple[float, str]:
        dx, dy = DELTAS[action]
        nx, ny = self.px + dx, self.py + dy
        if not (0 <= nx < COLS and 0 <= ny < ROWS):
            return -0.2, "blocked"
        self.px, self.py = nx, ny
        return -0.01, "walked"

    def _open_app(self) -> tuple[float, str]:
        if self.app_open:
            return -0.12, "app_already_open"
        self.app_open = True
        self.partner_known[:] = True
        self.kiosk_known[:] = True
        return -0.04, "farumasi_opened"

    def _inspect(self) -> tuple[float, str]:
        site = self.sites.get((self.px, self.py))
        if site is None:
            return -0.15, "nothing_to_inspect"
        if site.kind == "partner":
            self.partner_known[site.site_id] = True
        elif site.kind == "kiosk":
            self.kiosk_known[site.site_id] = True
        return 0.02, "inspected"

    def _buy(self) -> tuple[float, str, bool]:
        site = self.sites.get((self.px, self.py))
        if site is None:
            return -0.2, "nothing_to_buy", False
        if site.kind == "kiosk":
            self.poisoned = True
            self.health = 0.0
            return -4.0, "harmful_product", True
        if site.kind == "public":
            self.health = max(0.0, self.health - 0.08)
            self.obtained = True
            self.bought_public = True
            return 1.15 + 0.35 * self.health, "public_dispensed", True
        if not self.partner_known[site.site_id]:
            return -0.25, "unknown_shelf", False
        if not self.partner_open[site.site_id] or self.partner_stock[site.site_id] <= 0:
            return -0.4, "out_of_stock", False
        self.partner_stock[site.site_id] -= 1
        self.obtained = True
        self.bought_partner = True
        travelled = abs(self.px - self.home[0]) + abs(self.py - self.home[1])
        reward = 2.15 - 0.55 * (site.price - 1.0) - 0.03 * travelled + 0.45 * self.health
        return reward, "genuine_dispensed", True

    def _apply_restock(self) -> None:
        if self.steps == RESTOCK_AT and not self.restocked:
            self.partner_stock[self.restock_id] += 1
            self.partner_open[self.restock_id] = True
            self.restocked = True

    def _observe(self) -> np.ndarray:
        observation = np.zeros(OBS_DIM, dtype=np.float32)
        cursor = 0

        def put(value: float) -> None:
            nonlocal cursor
            observation[cursor] = _unit(value)
            cursor += 1

        put(self.px / (COLS - 1))
        put(self.py / (ROWS - 1))
        put(self.health)
        put(self.steps / MAX_STEPS)
        for med in range(3):
            put(1.0 if self.med == med else 0.0)
        put(1.0 if self.med != ORS else 0.4)
        put(1.0 if self.app_open else 0.0)
        here = self.sites.get((self.px, self.py))
        put(1.0 if here is not None else 0.0)
        put(1.0 if here is not None and self._known(here) else 0.0)
        put(1.0 if self.restocked else 0.0)
        for site in PARTNERS:
            known = bool(self.partner_known[site.site_id])
            put(_rel(site.x - self.px, COLS - 1))
            put(_rel(site.y - self.py, ROWS - 1))
            put((self.partner_stock[site.site_id] / 2.0) if known else 0.0)
            put((site.price - 1.0) if known else 0.0)
            available = known and self.partner_open[site.site_id] and self.partner_stock[site.site_id] > 0
            put(1.0 if available else 0.0)
            put(1.0 if known else 0.0)
        for site in KIOSKS:
            put(_rel(site.x - self.px, COLS - 1))
            put(_rel(site.y - self.py, ROWS - 1))
            put(0.25)
            put(1.0 if self.kiosk_known[site.site_id] else 0.0)
            put(1.0 if (self.px, self.py) == (site.x, site.y) else 0.0)
        put(_rel(PUBLIC.x - self.px, COLS - 1))
        put(_rel(PUBLIC.y - self.py, ROWS - 1))
        put(1.0)
        distance = abs(PUBLIC.x - self.px) + abs(PUBLIC.y - self.py)
        put(distance / 30.0)
        put(1.0 if here is not None and here.kind == "partner" else 0.0)
        put(1.0 if here is not None and here.kind == "kiosk" else 0.0)
        put(1.0 if here is not None and here.kind == "public" else 0.0)
        put(1.0 if (self.px, self.py) == self.home else 0.0)
        mask = self.action_mask()
        observation[cursor : cursor + N_ACTIONS] = mask
        return observation

    def _info(self) -> dict:
        site = self.sites.get((self.px, self.py))
        decision = {
            "action": "none" if self.last_action < 0 else ACTION_NAMES[self.last_action],
            "medicine": MED_NAMES[self.med],
            "pharmacy_name": None if site is None else site.name,
            "place_kind": None if site is None else site.kind,
            "health": round(self.health, 3),
            "farumasi_open": self.app_open,
            "event": self.last_event,
        }
        return {
            "event": self.last_event,
            "shift_return": self.return_so_far,
            "decision": decision,
            "action_mask": self.action_mask(),
            "health": self.health,
            "medicine": MED_NAMES[self.med],
        }
