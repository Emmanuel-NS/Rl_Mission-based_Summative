"""Sequential FARUMASI pharmacy-shift dispatcher.

One episode is one Kigali morning shift. Patients have already submitted
medicine requests in the FARUMASI app. The agent stands in for the dispatcher:

* send the request at the front of the queue to one of six partner pharmacies,
* defer it so a more urgent request is served first, or
* refer the patient to a public facility.

Metformin stock is scarce. A pack dispensed to a routine refill cannot be
dispensed to a later patient who is already rationing a diabetes dose. That
cost shows up when the urgent request fails, and again in a bonus paid only
at the end of the shift. Nearest-pharmacy ranking cannot express that trade.
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
# Minutes by motorbike. Same district, neighbouring district, opposite ends of Kigali.
TRAVEL_MINUTES = (
    (15, 30, 45),
    (30, 15, 30),
    (45, 30, 15),
)

N_PHARMACIES = 6
DEFER = 6
REFER = 7
N_ACTIONS = 8
RESTOCK_AT = 8
MAX_STEPS = 72
PHARM_FEATURES = 8
N_GLOBAL = 18
OBS_DIM = N_PHARMACIES * PHARM_FEATURES + N_GLOBAL + N_ACTIONS

URGENT_MET_ORDERS = 4
ROUTINE_MET_ORDERS = 5
AMOX_ORDERS = 6
ORS_ORDERS = 9
METFORMIN_PACKS = 4


@dataclass(frozen=True)
class Pharmacy:
    pharmacy_id: int
    name: str
    district_id: int
    district: str
    price_ratio: float


PHARMACIES = (
    Pharmacy(0, "Nyamirambo Community", 0, "Nyarugenge", 1.35),
    Pharmacy(1, "Kigali City Pharmacy", 0, "Nyarugenge", 1.90),
    Pharmacy(2, "Remera Care", 1, "Gasabo", 1.20),
    Pharmacy(3, "Kimironko Health Mart", 1, "Gasabo", 1.75),
    Pharmacy(4, "Kicukiro Centre", 2, "Kicukiro", 1.40),
    Pharmacy(5, "Gikondo Pharmacy", 2, "Kicukiro", 1.95),
)

ACTION_NAMES = tuple(
    [f"dispense:{pharmacy.name}" for pharmacy in PHARMACIES] + ["defer", "refer_public"]
)


@dataclass
class Order:
    order_id: int
    med: int
    urgency: float
    district: int
    strikes: int = 0
    deferrals: int = 0

    @property
    def urgent_metformin(self) -> bool:
        return self.med == METFORMIN and self.urgency >= 0.9

    @property
    def routine_metformin(self) -> bool:
        return self.med == METFORMIN and self.urgency < 0.5


def observation_catalogue() -> list[dict[str, str | int]]:
    """Meaning, range, source, and update cadence of every observation index."""
    rows: list[dict[str, str | int]] = []
    index = 0
    for pharmacy in PHARMACIES:
        fields = (
            (
                "stock_metformin",
                "0 to 1 (packs / 6)",
                "partner inventory, decremented when this pharmacy dispenses",
            ),
            (
                "stock_amoxicillin",
                "0 to 1 (packs / 8)",
                "partner inventory, decremented when this pharmacy dispenses",
            ),
            (
                "stock_ors",
                "0 to 1 (packs / 8)",
                "partner inventory, decremented when this pharmacy dispenses",
            ),
            (
                "near_expiry_current_medicine",
                "0 or 1",
                "1 if this pharmacy holds a near-expiry pack of the current medicine",
            ),
            (
                "is_open",
                "0 or 1",
                "shift roster; a closed pharmacy keeps stock but cannot dispense",
            ),
            (
                "travel_minutes",
                "0 to 1 (minutes / 45)",
                "district travel matrix from the current patient to this pharmacy",
            ),
            (
                "price_ratio",
                "0 to 1 ((ratio - 1) / 1)",
                "pharmacy list price divided by the international reference price",
            ),
            (
                "can_dispense",
                "0 or 1",
                "open and stock of the current medicine is at least 1",
            ),
        )
        for name, span, source in fields:
            rows.append(
                {
                    "index": index,
                    "name": f"p{pharmacy.pharmacy_id}_{name}",
                    "range": span,
                    "source": f"{pharmacy.name}. {source}",
                    "cadence": "every decision",
                    "agent_path": "flat float vector into the policy network",
                }
            )
            index += 1

    global_fields = (
        ("medicine_metformin", "0 or 1", "one-hot of the request at the front of the queue"),
        ("medicine_amoxicillin", "0 or 1", "one-hot of the request at the front of the queue"),
        ("medicine_ors", "0 or 1", "one-hot of the request at the front of the queue"),
        ("urgency", "0 to 1", "1.0 urgent metformin, 0.85 acute antibiotic, 0.25 routine"),
        ("district_nyarugenge", "0 or 1", "patient district on the request"),
        ("district_gasabo", "0 or 1", "patient district on the request"),
        ("district_kicukiro", "0 or 1", "patient district on the request"),
        ("city_stock_current", "0 to 1 (packs / 8)", "sum of this medicine across open pharmacies"),
        ("remaining_urgent_metformin", "0 to 1 (count / 6)", "urgent metformin requests still in the queue, including the current one"),
        ("remaining_routine_metformin", "0 to 1 (count / 8)", "routine metformin requests still in the queue"),
        ("urgent_metformin_behind", "0 to 1 (count / 6)", "urgent metformin requests waiting behind the current one"),
        ("queue_length", "0 to 1 (count / 24)", "requests not yet closed"),
        ("shift_progress", "0 to 1 (step / 72)", "decisions already taken this shift"),
        ("strikes", "0 to 1 (strikes / 2)", "failed dispense attempts on the current request"),
        ("deferrals", "0 to 1 (deferrals / 2)", "times this request has been sent to the back of the queue"),
        ("any_pharmacy_can_dispense", "0 or 1", "at least one open pharmacy has the medicine"),
        ("restock_pending", "0 or 1", "1 until the mid-shift metformin drop has arrived"),
        ("near_expiry_available", "0 or 1", "an open pharmacy has a near-expiry pack of the current medicine"),
    )
    for name, span, source in global_fields:
        rows.append(
            {
                "index": index,
                "name": name,
                "range": span,
                "source": source,
                "cadence": "every decision",
                "agent_path": "flat float vector into the policy network",
            }
        )
        index += 1

    for action_id, action_name in enumerate(ACTION_NAMES):
        rows.append(
            {
                "index": index,
                "name": f"mask_{action_name}",
                "range": "0 or 1",
                "source": "dispatch screen: pharmacies that are closed or out of stock are greyed out; defer is off after two defers",
                "cadence": "every decision",
                "agent_path": "last 8 observation values; also used to mask illegal actions",
            }
        )
        index += 1
    if index != OBS_DIM:
        raise RuntimeError(f"catalogue has {index} rows, OBS_DIM is {OBS_DIM}")
    return rows


class FarumasiShiftEnv(gymnasium.Env):
    """Morning dispatch shift for FARUMASI partner pharmacies in Kigali."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._renderer = None

        self.stock = np.zeros((N_PHARMACIES, 3), dtype=np.int32)
        self.near = np.zeros((N_PHARMACIES, 3), dtype=np.int32)
        self.open_mask = np.ones(N_PHARMACIES, dtype=bool)
        self.queue: list[Order] = []
        self.steps = 0
        self.restocked = False
        self.restock_pharmacy = -1
        self.return_so_far = 0.0
        self.last_action = -1
        self.last_event = "shift_start"
        self.last_reward = 0.0
        self.last_decision: dict = {}
        self.urgent_met_total = URGENT_MET_ORDERS
        self.urgent_met_filled = 0
        self.urgent_met_failed = 0
        self.routine_met_filled = 0
        self.dispensed = 0
        self.referred = 0
        self.closed_pharmacy = -1

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.stock = np.zeros((N_PHARMACIES, 3), dtype=np.int32)
        self.near = np.zeros((N_PHARMACIES, 3), dtype=np.int32)
        self.open_mask = np.ones(N_PHARMACIES, dtype=bool)
        self.steps = 0
        self.restocked = False
        self.restock_pharmacy = -1
        self.return_so_far = 0.0
        self.last_action = -1
        self.last_event = "shift_start"
        self.last_reward = 0.0
        self.last_decision = {}
        self.urgent_met_filled = 0
        self.urgent_met_failed = 0
        self.routine_met_filled = 0
        self.dispensed = 0
        self.referred = 0
        self._sample_shift()
        observation = self._observe()
        return observation, self._info()

    def step(self, action: int):
        if not self.queue:
            observation = self._observe()
            return observation, 0.0, True, False, self._info()

        if self.steps == RESTOCK_AT and not self.restocked:
            self._apply_restock()

        action = int(action)
        order = self.queue[0]
        reward = 0.0
        event = "invalid"

        if action < N_PHARMACIES:
            reward, event = self._dispense(action, order)
        elif action == DEFER:
            reward, event = self._defer(order)
        else:
            reward, event = self._refer(order)

        self.steps += 1
        self.last_action = action
        self.last_event = event
        self.last_reward = float(reward)
        terminated = len(self.queue) == 0
        truncated = (not terminated) and self.steps >= MAX_STEPS
        if terminated or truncated:
            end_bonus = self._terminal_bonus()
            reward += end_bonus
            self.last_reward = float(reward)
        self.return_so_far += float(reward)
        self.last_decision = self._decision(action, order, event, float(reward))
        observation = self._observe()
        return observation, float(reward), terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import DispatchRenderer

            self._renderer = DispatchRenderer()
        return self._renderer.render(self, human=self.render_mode == "human")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        if not self.queue:
            mask[REFER] = 1.0
            return mask
        order = self.queue[0]
        for pharmacy_id in range(N_PHARMACIES):
            if self.open_mask[pharmacy_id] and self.stock[pharmacy_id, order.med] > 0:
                mask[pharmacy_id] = 1.0
        if order.deferrals < 2:
            mask[DEFER] = 1.0
        mask[REFER] = 1.0
        return mask

    def city_available(self, med: int) -> int:
        total = 0
        for pharmacy_id in range(N_PHARMACIES):
            if self.open_mask[pharmacy_id]:
                total += int(self.stock[pharmacy_id, med])
        return total

    def count_metformin(self, *, urgent: bool, exclude_current: bool) -> int:
        orders = self.queue[1:] if exclude_current else self.queue
        total = 0
        for order in orders:
            if urgent and order.urgent_metformin:
                total += 1
            if not urgent and order.routine_metformin:
                total += 1
        return total

    def queue_preview(self, limit: int = 6) -> list[str]:
        lines = []
        for order in self.queue[:limit]:
            kind = "urgent" if order.urgency >= 0.8 else "routine"
            lines.append(
                f"#{order.order_id} {MED_NAMES[order.med]} · {kind} · {DISTRICT_NAMES[order.district]}"
            )
        return lines

    def _sample_shift(self) -> None:
        rng = self.np_random
        for _ in range(METFORMIN_PACKS):
            self.stock[int(rng.integers(0, N_PHARMACIES)), METFORMIN] += 1
        for pharmacy_id in range(N_PHARMACIES):
            self.stock[pharmacy_id, AMOXICILLIN] = int(rng.integers(2, 5))
            self.stock[pharmacy_id, ORS] = int(rng.integers(3, 7))

        self.closed_pharmacy = int(rng.integers(0, N_PHARMACIES))
        self.open_mask[self.closed_pharmacy] = False
        hidden = int(self.stock[self.closed_pharmacy, METFORMIN])
        self.stock[self.closed_pharmacy, METFORMIN] = 0
        receivers = [i for i in range(N_PHARMACIES) if i != self.closed_pharmacy]
        for _ in range(hidden):
            self.stock[int(rng.choice(receivers)), METFORMIN] += 1

        holders = [i for i in range(N_PHARMACIES) if self.open_mask[i] and self.stock[i, METFORMIN] > 0]
        if holders:
            chosen = int(rng.choice(holders))
            self.near[chosen, METFORMIN] = 1

        orders: list[Order] = []
        next_id = 1
        for _ in range(URGENT_MET_ORDERS):
            orders.append(Order(next_id, METFORMIN, 1.0, int(rng.integers(0, 3))))
            next_id += 1
        for _ in range(ROUTINE_MET_ORDERS):
            orders.append(Order(next_id, METFORMIN, 0.25, int(rng.integers(0, 3))))
            next_id += 1
        for _ in range(AMOX_ORDERS):
            orders.append(Order(next_id, AMOXICILLIN, 0.85, int(rng.integers(0, 3))))
            next_id += 1
        for _ in range(ORS_ORDERS):
            orders.append(Order(next_id, ORS, 0.30, int(rng.integers(0, 3))))
            next_id += 1
        permutation = rng.permutation(len(orders))
        self.queue = [orders[int(i)] for i in permutation]
        self.urgent_met_total = URGENT_MET_ORDERS

    def _apply_restock(self) -> None:
        self.restocked = True
        open_ids = [i for i in range(N_PHARMACIES) if self.open_mask[i]]
        lowest = min(int(self.stock[i, METFORMIN]) for i in open_ids)
        choices = [i for i in open_ids if int(self.stock[i, METFORMIN]) == lowest]
        pick = int(self.np_random.choice(choices))
        self.stock[pick, METFORMIN] += 1
        self.restock_pharmacy = pick

    def _dispense(self, pharmacy_id: int, order: Order) -> tuple[float, str]:
        can = bool(self.open_mask[pharmacy_id] and self.stock[pharmacy_id, order.med] > 0)
        if not can:
            order.strikes += 1
            if order.strikes >= 2:
                reward = -1.0 + self._fail_penalty(order)
                self._drop_front("fail")
                return reward, "patient_left"
            return -1.0, "rejected_dispatch"

        self.stock[pharmacy_id, order.med] -= 1
        used_expiry = False
        if self.near[pharmacy_id, order.med] > 0:
            self.near[pharmacy_id, order.med] -= 1
            used_expiry = True
        travel = TRAVEL_MINUTES[order.district][PHARMACIES[pharmacy_id].district_id]
        price = PHARMACIES[pharmacy_id].price_ratio
        reward = 1.0
        reward -= 0.35 * (travel / 45.0)
        reward += 0.25 * (2.0 - price)
        if used_expiry:
            reward += 0.45
        if order.med == AMOXICILLIN and travel >= 45:
            reward -= 0.9
        if order.urgent_metformin:
            reward += 0.3
        if order.routine_metformin:
            reward += 0.15
        self._drop_front("dispense")
        return reward, "dispensed"

    def _defer(self, order: Order) -> tuple[float, str]:
        if order.deferrals >= 2:
            reward = self._fail_penalty(order)
            self._drop_front("fail")
            return reward, "deferred_out"
        order.deferrals += 1
        self.queue.append(self.queue.pop(0))
        wait_cost = 0.22 if order.urgency >= 0.8 else 0.05
        return -wait_cost, "deferred"

    def _refer(self, order: Order) -> tuple[float, str]:
        city = self.city_available(order.med)
        urgent_others = self.count_metformin(urgent=True, exclude_current=True)
        if city <= 0:
            reward = 0.55
        elif order.routine_metformin and city <= urgent_others:
            reward = -0.2 if not self.restocked else 0.45
        elif order.urgent_metformin:
            reward = -2.0
            self.urgent_met_failed += 1
        else:
            reward = -1.1
        self._drop_front("refer")
        return reward, "referred"

    def _fail_penalty(self, order: Order) -> float:
        if order.urgent_metformin:
            self.urgent_met_failed += 1
            return -2.5
        if order.med == AMOXICILLIN:
            return -1.5
        return -0.6

    def _drop_front(self, outcome: str) -> None:
        order = self.queue.pop(0)
        if outcome == "dispense":
            self.dispensed += 1
            if order.urgent_metformin:
                self.urgent_met_filled += 1
            elif order.routine_metformin:
                self.routine_met_filled += 1
        elif outcome == "refer":
            self.referred += 1

    def _terminal_bonus(self) -> float:
        filled_ratio = self.urgent_met_filled / max(1, self.urgent_met_total)
        leftover_expiry = int(self.near[:, METFORMIN].sum())
        unmet = self.urgent_met_total - self.urgent_met_filled
        return 8.0 * filled_ratio - 0.8 * leftover_expiry - 0.5 * unmet

    def _observe(self) -> np.ndarray:
        observation = np.zeros(OBS_DIM, dtype=np.float32)
        order = self.queue[0] if self.queue else None
        cursor = 0
        for pharmacy_id, pharmacy in enumerate(PHARMACIES):
            observation[cursor] = min(1.0, self.stock[pharmacy_id, METFORMIN] / 6.0)
            observation[cursor + 1] = min(1.0, self.stock[pharmacy_id, AMOXICILLIN] / 8.0)
            observation[cursor + 2] = min(1.0, self.stock[pharmacy_id, ORS] / 8.0)
            if order is not None:
                observation[cursor + 3] = 1.0 if self.near[pharmacy_id, order.med] > 0 else 0.0
                travel = TRAVEL_MINUTES[order.district][pharmacy.district_id]
                observation[cursor + 5] = travel / 45.0
                can = self.open_mask[pharmacy_id] and self.stock[pharmacy_id, order.med] > 0
                observation[cursor + 7] = 1.0 if can else 0.0
            observation[cursor + 4] = 1.0 if self.open_mask[pharmacy_id] else 0.0
            observation[cursor + 6] = (pharmacy.price_ratio - 1.0) / 1.0
            cursor += PHARM_FEATURES

        if order is not None:
            observation[cursor + order.med] = 1.0
            observation[cursor + 3] = float(order.urgency)
            observation[cursor + 4 + order.district] = 1.0
            observation[cursor + 7] = min(1.0, self.city_available(order.med) / 8.0)
            observation[cursor + 13] = min(1.0, order.strikes / 2.0)
            observation[cursor + 14] = min(1.0, order.deferrals / 2.0)
            observation[cursor + 17] = 1.0 if any(
                self.open_mask[i] and self.near[i, order.med] > 0 for i in range(N_PHARMACIES)
            ) else 0.0
        observation[cursor + 8] = min(1.0, self.count_metformin(urgent=True, exclude_current=False) / 6.0)
        observation[cursor + 9] = min(1.0, self.count_metformin(urgent=False, exclude_current=False) / 8.0)
        observation[cursor + 10] = min(1.0, self.count_metformin(urgent=True, exclude_current=True) / 6.0)
        observation[cursor + 11] = min(1.0, len(self.queue) / 24.0)
        observation[cursor + 12] = min(1.0, self.steps / MAX_STEPS)
        observation[cursor + 15] = 1.0 if order is not None and self.action_mask()[:N_PHARMACIES].sum() > 0 else 0.0
        observation[cursor + 16] = 0.0 if self.restocked else 1.0
        mask_start = N_PHARMACIES * PHARM_FEATURES + N_GLOBAL
        observation[mask_start:] = self.action_mask()
        return observation

    def _info(self) -> dict:
        order = self.queue[0] if self.queue else None
        return {
            "action_mask": self.action_mask().copy(),
            "event": self.last_event,
            "decision": dict(self.last_decision),
            "stock_metformin": self.stock[:, METFORMIN].astype(int).tolist(),
            "urgent_met_filled": self.urgent_met_filled,
            "urgent_met_total": self.urgent_met_total,
            "routine_met_filled": self.routine_met_filled,
            "dispensed": self.dispensed,
            "referred": self.referred,
            "shift_return": self.return_so_far,
            "queue_length": len(self.queue),
            "current_medicine": None if order is None else MED_NAMES[order.med],
            "current_district": None if order is None else DISTRICT_NAMES[order.district],
            "restocked": self.restocked,
            "closed_pharmacy": PHARMACIES[self.closed_pharmacy].name if self.closed_pharmacy >= 0 else None,
        }

    def _decision(self, action: int, order: Order, event: str, reward: float) -> dict:
        pharmacy_name = PHARMACIES[action].name if action < N_PHARMACIES else None
        return {
            "action": ACTION_NAMES[action],
            "action_id": action,
            "event": event,
            "reward": round(reward, 4),
            "order_id": order.order_id,
            "medicine": MED_NAMES[order.med],
            "urgency": "high" if order.urgency >= 0.8 else "routine",
            "patient_district": DISTRICT_NAMES[order.district],
            "pharmacy_id": action if action < N_PHARMACIES else None,
            "pharmacy_name": pharmacy_name,
        }
