"""A patient who opens FARUMASI before crossing the city.

The rule never buys from an unlicensed stall. It reveals live stock, walks to
the closest affordable partner that actually has the prescription, and uses
the public clinic only when no licensed shelf can help.
"""

from __future__ import annotations

import numpy as np

from environment.custom_env import (
    APP,
    BUY,
    EAST,
    KIOSKS,
    NORTH,
    PARTNERS,
    PUBLIC,
    SOUTH,
    WEST,
    FarumasiShiftEnv,
)

DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
KIOSK_TILES = {(site.x, site.y) for site in KIOSKS}


def legal_random_action(env: FarumasiShiftEnv, rng: np.random.Generator) -> int:
    choices = np.flatnonzero(env.action_mask() > 0.5)
    return int(rng.choice(choices))


def _steps_onto_kiosk(env: FarumasiShiftEnv, action: int) -> bool:
    dx, dy = DELTAS[action]
    return (env.px + dx, env.py + dy) in KIOSK_TILES


def _toward(env: FarumasiShiftEnv, tx: int, ty: int) -> int:
    mask = env.action_mask()
    horizontal = []
    vertical = []
    if tx > env.px:
        horizontal.append(EAST)
    elif tx < env.px:
        horizontal.append(WEST)
    if ty > env.py:
        vertical.append(SOUTH)
    elif ty < env.py:
        vertical.append(NORTH)
    order = horizontal + vertical if abs(tx - env.px) >= abs(ty - env.py) else vertical + horizontal
    order += [NORTH, EAST, SOUTH, WEST]
    for action in order:
        if mask[action] > 0.5 and not _steps_onto_kiosk(env, action):
            return action
    return int(np.flatnonzero(mask > 0.5)[0])


def heuristic_action(env: FarumasiShiftEnv) -> int:
    mask = env.action_mask()
    if not env.app_open and mask[APP] > 0.5:
        return APP
    here = env.sites.get((env.px, env.py))
    target = _target(env)
    if here is not None and (here.x, here.y) == (target.x, target.y) and mask[BUY] > 0.5:
        return BUY
    return _toward(env, target.x, target.y)


def _target(env: FarumasiShiftEnv):
    best = None
    best_score = 1e9
    for site in PARTNERS:
        if not env.partner_open[site.site_id] or env.partner_stock[site.site_id] <= 0:
            continue
        distance = abs(site.x - env.px) + abs(site.y - env.py)
        score = distance + 6.0 * (site.price - 1.0)
        if score < best_score:
            best = site
            best_score = score
    return PUBLIC if best is None else best


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
        "kiosks": float(len(KIOSKS)),
    }
