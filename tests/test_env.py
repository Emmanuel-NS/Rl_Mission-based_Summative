"""Behaviour checks for the FARUMASI rider delivery. No network training."""

import numpy as np

from environment.baselines import compare_baselines, heuristic_action
from environment.custom_env import (
    CARRY,
    COLLECT,
    COVER,
    ENTER,
    HAND,
    KNOCK,
    N_ACTIONS,
    OBS_DIM,
    P_ENTRANCE,
    P_PATIENT,
    P_RIDING,
    PARK,
    PHARMACY_NODE,
    REMOUNT,
    RETURN_MEDS,
    SCAN,
    VERIFY,
    FarumasiRiderEnv,
    observation_catalogue,
)


def ride_to_patient(env: FarumasiRiderEnv) -> None:
    while env.phase != P_PATIENT:
        env.step(heuristic_action(env))
        assert env.outcome == ""


def test_catalogue_covers_every_index():
    rows = observation_catalogue()
    assert len(rows) == OBS_DIM
    assert [row["index"] for row in rows] == list(range(OBS_DIM))


def test_reset_observation_is_inside_the_box():
    env = FarumasiRiderEnv()
    observation, info = env.reset(seed=3)
    assert observation.shape == (OBS_DIM,)
    assert float(observation.min()) >= 0.0
    assert float(observation.max()) <= 1.0
    assert info["action_mask"].shape == (N_ACTIONS,)
    assert env.node == PHARMACY_NODE
    assert [int(i) for i in np.flatnonzero(info["action_mask"])] == [5, ENTER]


def test_rider_cannot_ride_before_collecting_the_order():
    env = FarumasiRiderEnv()
    env.reset(seed=1)
    _, reward, _, _, info = env.step(1)
    assert info["event"] == "invalid"
    assert reward < 0.0
    env.step(ENTER)
    env.step(COLLECT)
    assert env.phase == P_RIDING


def test_uncovered_box_is_damaged_by_rain():
    env = FarumasiRiderEnv()
    env.reset(seed=1)
    env.rain_at = 0
    env.step(ENTER)
    env.step(COLLECT)
    env.step(heuristic_action(env) if heuristic_action(env) <= 3 else 1)
    wet = env.condition
    assert wet < 1.0
    env2 = FarumasiRiderEnv()
    env2.reset(seed=1)
    env2.rain_at = 0
    env2.step(ENTER)
    env2.step(COLLECT)
    env2.step(COVER)
    env2.step(heuristic_action(env2))
    assert env2.condition == 1.0


def test_wrong_prescription_must_be_returned():
    env = FarumasiRiderEnv()
    env.reset(seed=2)
    env.rx_required, env.rx_valid = True, False
    ride_to_patient(env)
    mask = env.action_mask()
    assert mask[VERIFY] > 0.5 and mask[REMOUNT] < 0.5
    env.step(VERIFY)
    assert env.action_mask()[REMOUNT] > 0.5
    env.step(REMOUNT)
    assert env.returning
    while env.outcome == "":
        env.step(heuristic_action(env))
    assert env.outcome == "returned"
    assert env.return_so_far > 0.0


def test_handing_over_without_checking_is_a_violation():
    env = FarumasiRiderEnv()
    env.reset(seed=4)
    env.rx_required, env.rx_valid = True, True
    ride_to_patient(env)
    _, reward, terminated, _, info = env.step(HAND)
    assert terminated
    assert reward < -5.0
    assert info["event"] == "rx_violation"


def test_full_delivery_ends_with_a_qr_scan():
    env = FarumasiRiderEnv()
    env.reset(seed=5)
    env.rx_required, env.rx_valid = False, True
    ride_to_patient(env)
    env.step(HAND)
    _, reward, terminated, _, info = env.step(SCAN)
    assert terminated
    assert info["event"] == "delivered"
    assert reward > 3.0
    assert env.urgent_met_filled == 1


def test_random_rider_reaches_an_end_state():
    env = FarumasiRiderEnv()
    env.reset(seed=6)
    rng = np.random.default_rng(0)
    done = False
    steps = 0
    while not done and steps < 100:
        action = int(rng.choice(np.flatnonzero(env.action_mask() > 0.5)))
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    assert done


def test_rule_following_rider_beats_random_by_a_wide_gap():
    scores = compare_baselines(n_seeds=20, start_seed=0)
    assert scores["gap"] > 5.0, scores


def test_entrance_needs_a_knock_before_carrying():
    env = FarumasiRiderEnv()
    env.reset(seed=7)
    while env.phase != P_ENTRANCE:
        env.step(heuristic_action(env))
    mask = env.action_mask()
    assert mask[KNOCK] > 0.5 and mask[CARRY] < 0.5
    env.step(KNOCK)
    assert env.action_mask()[CARRY] > 0.5
    assert PARK >= 0 and RETURN_MEDS >= 0
