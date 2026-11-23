"""Held-out evaluation: how well do the trained riders do on deliveries they never saw?

The sweep selects each checkpoint on the validation seeds 1000-1019. Those
seeds are used again and again while a run is training, so the number they give
is a little optimistic. This module scores every checkpoint on 100 fresh test
deliveries (seeds 5000-5099) that no run ever touched, and breaks the result
down by the conditions that change the best strategy: rain, rush hour, the
prescription, and the kind of address.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from environment.baselines import heuristic_action, legal_random_action
from environment.custom_env import ACTION_NAMES, ENV_VERSION, FarumasiRiderEnv
from training.common import greedy_action, load_checkpoint

TEST_SEEDS = tuple(range(5000, 5100))
OUTCOMES = ("delivered", "returned", "violation", "ruined", "cancelled", "abandoned")


def facets(env: FarumasiRiderEnv) -> dict:
    """The conditions of one delivery, read right after reset."""
    return {
        "rain": env.rain_at < 10**5,
        "rush": bool(env.rush),
        "rx_required": bool(env.rx_required),
        "rx_invalid": bool(env.rx_required and not env.rx_valid),
        "kind": env.dest.kind,
    }


def rollout(policy, seed: int) -> dict:
    """Play one delivery. policy is 'random', 'heuristic' or a function observation -> action."""
    env = FarumasiRiderEnv()
    observation, _ = env.reset(seed=seed)
    record = facets(env)
    rng = np.random.default_rng(seed + 10_000)
    actions: list[int] = []
    done = False
    while not done:
        if policy == "random":
            action = legal_random_action(env, rng)
        elif policy == "heuristic":
            action = heuristic_action(env)
        else:
            action = policy(observation)
        observation, _reward, terminated, truncated, _info = env.step(action)
        actions.append(int(action))
        done = terminated or truncated
    record.update(
        {
            "seed": seed,
            "return": float(env.return_so_far),
            "outcome": env.outcome or "abandoned",
            "minutes": float(env.minutes),
            "steps": len(actions),
            "condition": float(env.condition),
            "actions": actions,
        }
    )
    env.close()
    return record


def net_policy(net: torch.nn.Module):
    device = torch.device("cpu")
    return lambda observation: greedy_action(net, observation, device)


def summarise(records: list[dict]) -> dict:
    returns = np.array([r["return"] for r in records])
    outcomes = Counter(r["outcome"] for r in records)
    usage = Counter(a for r in records for a in r["actions"])
    total = max(1, sum(usage.values()))
    return {
        "episodes": len(records),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_sem": float(returns.std(ddof=1) / np.sqrt(len(returns))) if len(returns) > 1 else 0.0,
        "success": float(np.mean([r["outcome"] in ("delivered", "returned") for r in records])),
        "minutes_mean": float(np.mean([r["minutes"] for r in records])),
        "outcomes": {name: outcomes.get(name, 0) for name in OUTCOMES},
        "action_share": {ACTION_NAMES[a]: usage.get(a, 0) / total for a in range(len(ACTION_NAMES))},
        "wait_share": usage.get(5, 0) / total,
    }


def by_condition(records: list[dict]) -> dict:
    """Success rate and mean return for each condition that changes the best strategy."""
    groups = {
        "dry": lambda r: not r["rain"],
        "rain": lambda r: r["rain"],
        "normal traffic": lambda r: not r["rush"],
        "rush hour": lambda r: r["rush"],
        "no prescription": lambda r: not r["rx_required"],
        "valid prescription": lambda r: r["rx_required"] and not r["rx_invalid"],
        "invalid prescription": lambda r: r["rx_invalid"],
        "home": lambda r: r["kind"] == "home",
        "office": lambda r: r["kind"] == "office",
        "hotel": lambda r: r["kind"] == "hotel",
    }
    out = {}
    for name, test in groups.items():
        chosen = [r for r in records if test(r)]
        if not chosen:
            continue
        out[name] = {
            "n": len(chosen),
            "success": float(np.mean([r["outcome"] in ("delivered", "returned") for r in chosen])),
            "return_mean": float(np.mean([r["return"] for r in chosen])),
        }
    return out


def evaluate_path(path: str | Path, seeds=TEST_SEEDS) -> tuple[dict, list[dict]]:
    net, payload = load_checkpoint(path, torch.device("cpu"))
    policy = net_policy(net)
    records = [rollout(policy, seed) for seed in seeds]
    summary = summarise(records)
    summary["algo"] = payload.get("algo")
    summary["config_name"] = payload.get("config", {}).get("name")
    return summary, records


def evaluate_everything(
    log_root: str | Path = "logs",
    model_root: str | Path = "models",
    out_path: str | Path = "logs/test_evaluation.json",
) -> dict:
    """Score the baselines, all 40 sweep checkpoints, and the final models on the test seeds."""
    model_root = Path(model_root)
    result: dict = {"env_version": ENV_VERSION, "test_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]], "baselines": {}, "sweep": {}, "final": {}, "conditions": {}}
    for name in ("random", "heuristic"):
        records = [rollout(name, seed) for seed in TEST_SEEDS]
        result["baselines"][name] = summarise(records)
        result["conditions"][name] = by_condition(records)
        print(f"{name:<10} return {result['baselines'][name]['return_mean']:+.2f}", flush=True)
    for path in sorted(model_root.glob("*/*_exp*.pt")):
        if path.parent.name == "final":
            continue
        summary, records = evaluate_path(path)
        key = path.stem
        result["sweep"][key] = summary
        result["conditions"][key] = by_condition(records)
        print(f"{key:<14} return {summary['return_mean']:+.2f}  success {summary['success']:.0%}", flush=True)
    for path in sorted((model_root / "final").glob("*.pt")):
        summary, records = evaluate_path(path)
        result["final"][path.stem] = summary
        result["conditions"][path.stem] = by_condition(records)
        print(f"{path.stem:<18} return {summary['return_mean']:+.2f}  success {summary['success']:.0%}", flush=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate_everything()
