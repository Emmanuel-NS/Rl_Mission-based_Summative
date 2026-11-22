"""Behaviour checks for the FARUMASI patient search. No network training."""

import numpy as np

from environment.baselines import compare_baselines, heuristic_action
from environment.custom_env import (
    APP,
    BUY,
    N_ACTIONS,
    OBS_DIM,
    PARTNERS,
    FarumasiShiftEnv,
    observation_catalogue,
)


def test_catalogue_covers_every_index():
    rows = observation_catalogue()
    assert len(rows) == OBS_DIM
    assert [row["index"] for row in rows] == list(range(OBS_DIM))


def test_reset_observation_is_inside_the_box():
    env = FarumasiShiftEnv()
    observation, info = env.reset(seed=3)
    assert observation.shape == (OBS_DIM,)
    assert observation.dtype == np.float32
    assert float(observation.min()) >= 0.0
    assert float(observation.max()) <= 1.0
    assert info["action_mask"].shape == (N_ACTIONS,)
    assert env.app_open is False
    assert env.health == 1.0


def test_farumasi_reveals_partner_stock():
    env = FarumasiShiftEnv()
    env.reset(seed=1)
    assert not env.partner_known.any()
    _, reward, terminated, _, info = env.step(APP)
    assert info["event"] == "farumasi_opened"
    assert reward < 0.0
    assert not terminated
    assert env.partner_known.all()
    assert env.kiosk_known.all()
    assert "stock" in observation_catalogue()[14]["name"]


def test_unlicensed_purchase_ends_the_search():
    env = FarumasiShiftEnv()
    env.reset(seed=2)
    kiosk = next(site for site in env.sites.values() if site.kind == "kiosk")
    assert kiosk.kind == "kiosk"
    env.px, env.py = kiosk.x, kiosk.y
    _, reward, terminated, _, info = env.step(BUY)
    assert terminated
    assert reward < -3.0
    assert info["event"] == "harmful_product"
    assert env.poisoned


def test_licensed_purchase_takes_one_pack():
    env = FarumasiShiftEnv()
    env.reset(seed=4)
    site = PARTNERS[2]
    env.partner_known[site.site_id] = True
    env.partner_open[site.site_id] = True
    env.partner_stock[site.site_id] = 2
    env.px, env.py = site.x, site.y
    _, reward, terminated, _, info = env.step(BUY)
    assert terminated
    assert reward > 1.0
    assert info["event"] == "genuine_dispensed"
    assert int(env.partner_stock[site.site_id]) == 1


def test_search_reaches_a_terminal_state():
    env = FarumasiShiftEnv()
    env.reset(seed=5)
    done = False
    steps = 0
    while not done and steps < 80:
        action = int(np.flatnonzero(env.action_mask() > 0.5)[0])
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    assert done


def test_heuristic_beats_legal_random_on_the_same_searches():
    scores = compare_baselines(n_seeds=20, start_seed=0)
    assert scores["gap"] > 1.0, scores


def test_heuristic_opens_farumasi_before_buying():
    env = FarumasiShiftEnv()
    env.reset(seed=7)
    assert heuristic_action(env) == APP
