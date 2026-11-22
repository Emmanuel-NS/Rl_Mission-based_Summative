"""Play the trained FARUMASI dispatcher in the shift window.

    uv run python play.py
    uv run python play.py --episodes 3 --seed 1000
    uv run python play.py --heuristic
"""

from __future__ import annotations

import argparse
import json

from environment.baselines import heuristic_action
from environment.custom_env import FarumasiShiftEnv
from training.common import load_checkpoint, resolve_device, run_episode


def play_heuristic(episodes: int, seed: int) -> None:
    for episode in range(episodes):
        env = FarumasiShiftEnv(render_mode="human")
        env.reset(seed=seed + episode)
        done = False
        step = 0
        print(f"\n=== heuristic episode {episode + 1}  seed {seed + episode} ===")
        while not done:
            action = heuristic_action(env)
            _observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            decision = info["decision"]
            print(
                f"{step:02d}  {info['event']:<18} reward {reward:+7.2f}  "
                f"return {info['shift_return']:+7.2f}  {json.dumps(decision)}"
            )
            done = terminated or truncated
            step += 1
        print(
            f"genuine pack {env.urgent_met_filled}/{env.urgent_met_total}  "
            f"return {env.return_so_far:+.2f}"
        )
        env.close()


def play_agent(path: str, episodes: int, seed: int, device_name: str) -> None:
    device = resolve_device(device_name)
    net, payload = load_checkpoint(path, device)
    print(f"Loaded {payload.get('algo')} experiment {payload.get('config', {}).get('id')} from {path}")
    for episode in range(episodes):
        print(f"\n=== learned episode {episode + 1}  seed {seed + episode} ===")
        summary = run_episode(net, seed + episode, device, render_mode="human")
        for step, row in enumerate(summary["trace"]):
            print(
                f"{step:02d}  {row['event']:<18} reward {row['reward']:+7.2f}  "
                f"return {row['shift_return']:+7.2f}  {row['medicine']} -> {row['pharmacy']}"
            )
        print(
            f"genuine pack {summary['urgent_met_filled']}/{summary['urgent_met_total']}  "
            f"licensed pharmacy {summary['dispensed']}  public clinic {summary['referred']}  "
            f"return {summary['return']:+.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a FARUMASI dispatcher play one shift")
    parser.add_argument("--model", default="models/best_agent.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--heuristic", action="store_true")
    args = parser.parse_args()
    if args.heuristic:
        play_heuristic(args.episodes, args.seed)
        return
    play_agent(args.model, args.episodes, args.seed, args.device)


if __name__ == "__main__":
    main()
