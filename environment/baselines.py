"""Greedy dispatcher used as a reference line, not as a learned agent.

It keeps scarce metformin for urgent requests, sends a routine refill to the
back of the queue while a mid-shift drop is still useful, and otherwise picks
the open pharmacy that combines a short ride, a lower list price, and a
near-expiry pack.
"""

from __future__ import annotations

import numpy as np

from environment.custom_env import (
    AMOXICILLIN,
    DEFER,
    METFORMIN,
    N_PHARMACIES,
    PHARMACIES,
    REFER,
    TRAVEL_MINUTES,
    FarumasiShiftEnv,
)


def legal_random_action(env: FarumasiShiftEnv, rng: np.random.Generator) -> int:
    mask = env.action_mask()
    choices = np.flatnonzero(mask > 0.5)
    return int(rng.choice(choices))


def heuristic_action(env: FarumasiShiftEnv) -> int:
    order = env.queue[0]
    mask = env.action_mask()
    if order.routine_metformin:
        urgent_others = env.count_metformin(urgent=True, exclude_current=True)
        city = env.city_available(METFORMIN)
        if city <= urgent_others:
            urgent_behind = urgent_others
            if urgent_behind > 0 and mask[DEFER] > 0.5:
                return DEFER
            return REFER
    best_id = None
    best_score = 1e9
    for pharmacy_id in range(N_PHARMACIES):
        if mask[pharmacy_id] < 0.5:
            continue
        pharmacy = PHARMACIES[pharmacy_id]
        travel = TRAVEL_MINUTES[order.district][pharmacy.district_id]
        score = float(travel) + 18.0 * pharmacy.price_ratio
        if env.near[pharmacy_id, order.med] > 0:
            score -= 22.0
        if order.med == AMOXICILLIN and travel >= 45:
            score += 25.0
        if score < best_score:
            best_score = score
            best_id = pharmacy_id
    if best_id is None:
        return REFER
    return best_id


def episode_return(policy: str, seed: int) -> float:
    env = FarumasiShiftEnv()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    done = False
    while not done:
        if policy == "heuristic":
            action = heuristic_action(env)
        elif policy == "random":
            action = legal_random_action(env, rng)
        else:
            raise ValueError(policy)
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    total = float(env.return_so_far)
    env.close()
    return total


def compare_baselines(n_seeds: int = 30, start_seed: int = 0) -> dict[str, float]:
    random_scores = [episode_return("random", start_seed + i) for i in range(n_seeds)]
    heuristic_scores = [episode_return("heuristic", start_seed + i) for i in range(n_seeds)]
    return {
        "seeds": float(n_seeds),
        "random_mean": float(np.mean(random_scores)),
        "random_std": float(np.std(random_scores)),
        "heuristic_mean": float(np.mean(heuristic_scores)),
        "heuristic_std": float(np.std(heuristic_scores)),
        "gap": float(np.mean(heuristic_scores) - np.mean(random_scores)),
    }
