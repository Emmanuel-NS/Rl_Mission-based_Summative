import numpy as np

from training.evaluation import TEST_SEEDS, by_condition, rollout, summarise


def test_test_seeds_do_not_overlap_the_validation_seeds():
    from training.common import EVAL_SEEDS

    assert not set(TEST_SEEDS) & set(EVAL_SEEDS)
    assert len(TEST_SEEDS) == 100


def test_rule_following_rider_succeeds_and_random_rider_does_not():
    heuristic = [rollout("heuristic", seed) for seed in range(5000, 5012)]
    random = [rollout("random", seed) for seed in range(5000, 5012)]
    assert summarise(heuristic)["success"] > 0.9
    assert summarise(random)["success"] < 0.5
    assert summarise(heuristic)["return_mean"] > summarise(random)["return_mean"] + 5.0


def test_summary_and_conditions_are_consistent():
    records = [rollout("heuristic", seed) for seed in range(5000, 5010)]
    summary = summarise(records)
    assert sum(summary["outcomes"].values()) == summary["episodes"] == 10
    assert abs(sum(summary["action_share"].values()) - 1.0) < 1e-6
    groups = by_condition(records)
    assert groups["dry"]["n"] + groups["rain"]["n"] == 10
    assert all(np.isfinite(g["return_mean"]) for g in groups.values())
