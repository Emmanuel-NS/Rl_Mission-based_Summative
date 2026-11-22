"""Behaviour checks for the FARUMASI shift. No network training."""

import numpy as np

from environment.baselines import compare_baselines, heuristic_action
from environment.custom_env import (
    METFORMIN,
    OBS_DIM,
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
    assert info["action_mask"].shape == (8,)
    assert info["queue_length"] == 24


def test_successful_dispense_reduces_stock():
    env = FarumasiShiftEnv()
    env.reset(seed=1)
    order = env.queue[0]
    pharmacy_id = next(
        i
        for i in range(6)
        if env.open_mask[i] and env.stock[i, order.med] > 0
    )
    before = int(env.stock[pharmacy_id, order.med])
    _, reward, _, _, info = env.step(pharmacy_id)
    assert int(env.stock[pharmacy_id, order.med]) == before - 1
    assert info["event"] == "dispensed"
    assert reward > 0.0


def test_empty_pharmacy_does_not_consume_the_request():
    env = FarumasiShiftEnv()
    env.reset(seed=2)
    order_id = env.queue[0].order_id
    medicine = env.queue[0].med
    empty = next(i for i in range(6) if env.stock[i, medicine] == 0 or not env.open_mask[i])
    before = env.stock.copy()
    _, reward, terminated, _, info = env.step(empty)
    assert reward < 0.0
    assert np.array_equal(env.stock, before)
    assert not terminated
    assert env.queue[0].order_id == order_id
    assert info["event"] == "rejected_dispatch"


def test_defer_sends_the_request_to_the_back():
    env = FarumasiShiftEnv()
    env.reset(seed=4)
    front = env.queue[0].order_id
    second = env.queue[1].order_id
    _, _, _, _, info = env.step(6)
    assert info["event"] == "deferred"
    assert env.queue[0].order_id == second
    assert env.queue[-1].order_id == front


def test_shift_reaches_a_terminal_state():
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


def test_heuristic_beats_legal_random_on_the_same_shifts():
    scores = compare_baselines(n_seeds=20, start_seed=0)
    assert scores["gap"] > 5.0, scores


def test_deferring_a_routine_metformin_can_keep_a_pack():
    """A routine refill at the front must not be the only way to use metformin."""
    env = FarumasiShiftEnv()
    env.reset(seed=7)
    # Force a known scarce state: one pack, routine request in front, urgent behind.
    env.stock[:, METFORMIN] = 0
    env.stock[2, METFORMIN] = 1
    env.near[:, METFORMIN] = 0
    env.open_mask[:] = True
    routine = env.queue[0]
    routine.med = METFORMIN
    routine.urgency = 0.25
    urgent = env.queue[1]
    urgent.med = METFORMIN
    urgent.urgency = 1.0
    assert heuristic_action(env) == 6
