"""Reference riders: one that follows the delivery rules, and a legal-random one."""

from __future__ import annotations

import numpy as np

from environment.custom_env import (
    CARRY,
    COLLECT,
    COVER,
    ENTER,
    HAND,
    KNOCK,
    P_BUILDING,
    P_ENTRANCE,
    P_INSIDE,
    P_OUTSIDE,
    P_PATIENT,
    P_RIDING,
    PARK,
    REMOUNT,
    RETURN_MEDS,
    SCAN,
    VERIFY,
    FarumasiRiderEnv,
    edge_id,
    neighbour,
)


def legal_random_action(env: FarumasiRiderEnv, rng: np.random.Generator) -> int:
    choices = np.flatnonzero(env.action_mask() > 0.5)
    return int(rng.choice(choices))


def best_ride(env: FarumasiRiderEnv) -> int:
    """Direction with the lowest traffic-aware time to the target."""
    remaining = env.time_to(env.target_node)
    best, best_cost = 0, float("inf")
    for direction in range(4):
        other = neighbour(env.node, direction)
        if other is None:
            continue
        cost = env.edge_minutes(edge_id(env.node, other)) + remaining[other]
        if cost < best_cost:
            best, best_cost = direction, cost
    return best


def heuristic_action(env: FarumasiRiderEnv) -> int:
    """Follow the rules: cover the box, avoid traffic, check the prescription."""
    mask = env.action_mask()
    phase = env.phase
    if phase == P_OUTSIDE:
        return ENTER
    if phase == P_INSIDE:
        return COLLECT
    if (env.raining or env.rain_soon) and mask[COVER] > 0.5:
        return COVER
    if phase == P_RIDING:
        if env.returning and env.node == env.target_node:
            return RETURN_MEDS
        if not env.returning and env.node == env.dest.node:
            return PARK
        return best_ride(env)
    if phase == P_ENTRANCE:
        return KNOCK
    if phase == P_BUILDING:
        return CARRY
    if phase == P_PATIENT:
        if env.rx_required and not env.rx_checked:
            return VERIFY
        if env.rx_checked and not env.rx_valid:
            return REMOUNT
        if not env.handed:
            return HAND
        return SCAN
    return int(np.flatnonzero(mask > 0.5)[0])


def episode_return(policy: str, seed: int) -> float:
    env = FarumasiRiderEnv()
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
