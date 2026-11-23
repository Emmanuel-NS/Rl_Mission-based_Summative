"""Train the winning setting of each algorithm for longer, with several seeds.

The sweep gives every setting the same short budget (30,000 frames), so it can
rank settings fairly but not show where each algorithm ends up. This step takes
the best setting of each algorithm from the sweep and trains it for a longer
budget on three training seeds. The checkpoint of each run is chosen on the
validation seeds (1000-1019); the report number comes from the untouched test
seeds (5000-5099), see training/evaluation.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from environment.custom_env import ENV_VERSION
from training.common import resolve_device

ALGOS = ("dqn", "reinforce", "ppo", "a2c")


def best_config(algo: str, log_root: Path = Path("logs")) -> dict:
    """The sweep setting with the highest validation return for this algorithm."""
    best = None
    for path in log_root.glob(f"{algo}/experiment_*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("env_version") != ENV_VERSION:
            continue
        if best is None or summary["best_eval_return"] > best["best_eval_return"]:
            best = summary
    if best is None:
        raise RuntimeError(f"no finished sweep for {algo}; run: uv run python main.py --sweep {algo}")
    return best["config"]


def run_final(trainers: dict, algos, seeds, frames: int, eval_every: int, device_name: str, force: bool = False) -> list[dict]:
    device = resolve_device(device_name)
    done = []
    for algo in algos:
        config = dict(best_config(algo))
        for seed in seeds:
            log_dir = Path("logs") / "final" / algo / f"seed_{seed}"
            model_path = Path("models") / "final" / f"{algo}_seed{seed}.pt"
            summary_path = log_dir / "summary.json"
            if summary_path.exists() and model_path.exists() and not force:
                previous = json.loads(summary_path.read_text(encoding="utf-8"))
                if previous.get("env_version") == ENV_VERSION and previous.get("frames", 0) >= frames:
                    print(f"skip {algo} seed {seed}: already finished", flush=True)
                    done.append(previous)
                    continue
            print(f"\n=== final {algo} (setting '{config['name']}') seed {seed}, {frames} frames ===", flush=True)
            done.append(
                trainers[algo](
                    config,
                    total_frames=frames,
                    device=device,
                    seed=seed,
                    log_dir=log_dir,
                    model_path=model_path,
                    eval_every=eval_every,
                )
            )
    return done


def promote_final(log_root: Path = Path("logs"), model_root: Path = Path("models")) -> dict | None:
    """Make the final run with the best validation return the agent that play.py shows."""
    import torch

    best = None
    for path in (log_root / "final").glob("*/seed_*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("env_version") != ENV_VERSION:
            continue
        if best is None or summary["best_eval_return"] > best["best_eval_return"]:
            best = dict(summary)
            best["seed"] = int(path.parent.name.split("_")[1])
    if best is None:
        return None
    source = model_root / "final" / f"{best['algo']}_seed{best['seed']}.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    torch.save(payload, model_root / "best_agent.pt")
    (model_root / "best_agent.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best
