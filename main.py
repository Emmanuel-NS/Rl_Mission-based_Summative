"""Train, compare, and inspect the FARUMASI rider agents.

uv run python main.py
    short PPO run so a fresh clone executes without extra setup

uv run python main.py --sweep all --device cpu --frames 30000
    10 settings for DQN, REINFORCE, PPO, and A2C (add --algos dqn reinforce to run only some)

uv run python main.py --final --frames 120000 --seeds 2
    the best setting of each algorithm, longer, on two training seeds

uv run python main.py --evaluate
    score everything on 100 unseen deliveries and draw the report figures
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from environment.baselines import compare_baselines
from environment.custom_env import ENV_VERSION, observation_catalogue
from environment.rendering import save_environment_png
from training.common import evaluate, heuristic_reference, load_checkpoint, promote_best, resolve_device
from training.dqn_training import train_dqn
from training.experiments import EXPERIMENTS
from training.evaluation import evaluate_everything
from training.final_training import ALGOS as FINAL_ALGOS, promote_final, run_final
from training.report_figures import build_report_figures
from training.results_doc import write_results
from training.pg_training import train_a2c, train_ppo, train_reinforce
from training.plots import build_plots, write_markdown_tables

TRAINERS = {
    "dqn": train_dqn,
    "reinforce": train_reinforce,
    "ppo": train_ppo,
    "a2c": train_a2c,
}


def _model_path(algo: str, experiment: int) -> Path:
    folder = "dqn" if algo == "dqn" else "pg"
    return Path("models") / folder / f"{algo}_exp{experiment:02d}.pt"


def _log_dir(algo: str, experiment: int) -> Path:
    return Path("logs") / algo / f"experiment_{experiment:02d}"


def sync_backup() -> None:
    raw = os.environ.get("FARUMASI_BACKUP", "").strip()
    if not raw:
        return
    target = Path(raw)
    try:
        target.mkdir(parents=True, exist_ok=True)
        for name in ("logs", "models"):
            source = Path(name)
            if source.exists():
                shutil.copytree(source, target / name, dirs_exist_ok=True)
        print(f"saved progress to {target}", flush=True)
    except OSError as error:  # another worker may be writing the same files; the next save will catch up
        print(f"could not copy progress to {target} this time: {error}", flush=True)


def run_one(algo: str, config: dict, args) -> dict:
    log_dir = _log_dir(algo, int(config["id"]))
    model_path = _model_path(algo, int(config["id"]))
    summary_path = log_dir / "summary.json"
    if summary_path.exists() and model_path.exists() and not args.force:
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("env_version") == ENV_VERSION:
            print(f"skip {algo} experiment {config['id']}: already finished", flush=True)
            return previous
    print(f"\n=== {algo} experiment {config['id']}: {config['name']} ===", flush=True)
    summary = TRAINERS[algo](
        config,
        total_frames=args.frames,
        device=resolve_device(args.device),
        seed=args.seed,
        log_dir=log_dir,
        model_path=model_path,
        eval_every=args.eval_every,
    )
    sync_backup()
    return summary


def run_sweep(args) -> None:
    algos = list(args.algos) if args.sweep == "all" else [args.sweep]
    for algo in algos:
        configs = EXPERIMENTS[algo]
        if args.experiment is not None:
            configs = [config for config in configs if config["id"] == args.experiment]
        for config in configs:
            run_one(algo, config, args)
    if set(algos) != set(TRAINERS):
        print("Sweep finished for", ", ".join(algos), "(tables and figures are made when every algorithm is done).", flush=True)
        return
    best = promote_best()
    figures = build_plots()
    table = write_markdown_tables()
    reference = heuristic_reference()
    Path("logs/heuristic_reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print("\nBest agent:", json.dumps({key: best[key] for key in ("algo", "experiment", "best_eval_return") if best}, indent=2))
    print("Rule-following rider on the same held-out deliveries:", reference)
    print("Figures:", figures)
    print("Tables:", table)
    sync_backup()


def describe() -> None:
    print(f"{'idx':>4}  {'name':<42}  range")
    for row in observation_catalogue():
        print(f"{row['index']:>4}  {row['name']:<42}  {row['range']}")
        print(f"      {row['source']}")


def demo(args) -> None:
    best_path = Path("models/best_agent.pt")
    if best_path.exists() and not args.force:
        device = resolve_device("cpu" if args.device == "auto" else args.device)
        net, payload = load_checkpoint(best_path, device)
        scores = evaluate(net, device)
        print(json.dumps({"checkpoint": payload.get("algo"), "experiment": payload.get("config", {}).get("id"), **scores}, indent=2))
        print("Full sweep: uv run python main.py --sweep all --device cpu --frames 30000 --eval-every 5000")
        return
    config = next(item for item in EXPERIMENTS["ppo"] if item["id"] == 1)
    args.frames = min(args.frames, 4096)
    args.eval_every = min(args.eval_every, args.frames)
    summary = run_one("ppo", config, args)
    promote_best()
    print(json.dumps(summary, indent=2))
    print("This was a short check. The assignment sweep is:")
    print("uv run python main.py --sweep all --device cpu --frames 30000 --eval-every 5000")


def main() -> None:
    parser = argparse.ArgumentParser(description="FARUMASI rider delivery reinforcement learning")
    parser.add_argument("--sweep", choices=["dqn", "reinforce", "ppo", "a2c", "all"])
    parser.add_argument("--experiment", type=int)
    parser.add_argument("--frames", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--final", action="store_true", help="train the best setting of each algorithm for longer")
    parser.add_argument("--seeds", type=int, default=2, help="training seeds for --final")
    parser.add_argument("--algos", nargs="+", default=list(FINAL_ALGOS), choices=list(FINAL_ALGOS))
    parser.add_argument("--evaluate", action="store_true", help="score every checkpoint on 100 unseen test deliveries")
    parser.add_argument("--screenshot", action="store_true")
    args = parser.parse_args()

    if args.describe:
        describe()
        return
    if args.baselines:
        print(json.dumps(compare_baselines(), indent=2))
        return
    if args.screenshot:
        path = save_environment_png("assets/environment.png")
        print(path)
        return
    if args.final:
        run_final(TRAINERS, args.algos, range(args.seeds), args.frames, args.eval_every, args.device, args.force)
        best = promote_final()
        print("Best long run:", json.dumps({key: best[key] for key in ("algo", "seed", "best_eval_return")} if best else None))
        sync_backup()
        return
    if args.evaluate:
        Path("logs/heuristic_reference.json").write_text(json.dumps(heuristic_reference(), indent=2), encoding="utf-8")
        evaluate_everything()
        print(build_plots())
        print(build_report_figures())
        print(write_markdown_tables())
        print(write_results())
        sync_backup()
        return
    if args.plots:
        print(build_plots())
        print(build_report_figures())
        print(write_markdown_tables())
        return
    if args.sweep:
        run_sweep(args)
        return
    demo(args)


if __name__ == "__main__":
    main()
