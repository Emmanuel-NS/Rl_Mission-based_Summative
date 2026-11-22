"""Watch a FARUMASI rider deliver medicine across Kigali.

    uv run python play.py --heuristic --episodes 3 --seed 0
    uv run python play.py --episodes 3 --seed 1000
"""

from __future__ import annotations

import argparse

from environment.baselines import heuristic_action
from environment.custom_env import FarumasiRiderEnv
from training.common import load_checkpoint, resolve_device, run_episode


def print_step(step: int, reward: float, info: dict) -> None:
    decision = info["decision"]
    print(
        f"{step:02d}  {decision['action_label']:<26} {info['event']:<22} "
        f"reward {reward:+7.2f}  total {info['shift_return']:+7.2f}  "
        f"{info['minutes']:>4.0f} min  box {info['condition']:.0%}  {decision['pharmacy_name']}",
        flush=True,
    )


def print_result(outcome: str, minutes: float, total: float) -> None:
    words = {
        "delivered": "DELIVERED. QR scanned, medicine in the patient's hands",
        "returned": "RETURNED. Wrong prescription went back to the pharmacy",
        "violation": "FAILED. Medicine handed over without a valid prescription",
        "ruined": "FAILED. The medicine was ruined by rain",
        "cancelled": "FAILED. The patient cancelled: too late",
        "abandoned": "FAILED. Delivery abandoned",
    }
    print(f"{words.get(outcome, outcome)}   {minutes:.0f} min   return {total:+.2f}", flush=True)


def play_heuristic(episodes: int, seed: int) -> None:
    for episode in range(episodes):
        env = FarumasiRiderEnv(render_mode="human")
        env.reset(seed=seed + episode)
        done = False
        step = 0
        print(f"\n=== rider episode {episode + 1}  seed {seed + episode}  {env.dest.name} ===", flush=True)
        while not done:
            action = heuristic_action(env)
            _observation, reward, terminated, truncated, info = env.step(action)
            print_step(step, reward, info)
            env.render()
            done = terminated or truncated
            step += 1
        print_result(env.outcome, env.minutes, env.return_so_far)
        env.close()


def play_agent(path: str, episodes: int, seed: int, device_name: str) -> None:
    device = resolve_device(device_name)
    net, payload = load_checkpoint(path, device)
    print(f"Loaded {payload.get('algo')} experiment {payload.get('config', {}).get('id')} from {path}")
    for episode in range(episodes):
        print(f"\n=== learned episode {episode + 1}  seed {seed + episode} ===", flush=True)
        summary = run_episode(net, seed + episode, device, render_mode="human", on_step=print_step)
        outcome = "delivered" if summary["dispensed"] else ("returned" if summary["referred"] else "failed")
        print_result(outcome, 0.0, summary["return"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a FARUMASI rider deliver one order")
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
