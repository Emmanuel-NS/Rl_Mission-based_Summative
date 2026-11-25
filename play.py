"""Watch or drive a FARUMASI rider delivery in Kigali.

    uv run python play.py --episodes 3 --seed 1000
    uv run python play.py --heuristic --episodes 3 --seed 0

Window controls: Self Play, Auto Play, Pause (also keys S / A / Space).
"""

from __future__ import annotations

import argparse

from environment.baselines import heuristic_action
from environment.custom_env import MED_NAMES, FarumasiRiderEnv
from training.common import greedy_action, load_checkpoint, resolve_device


def print_step(step: int, reward: float, info: dict) -> None:
    decision = info["decision"]
    print(
        f"{step:02d}  {decision['action_label']:<26} {info['event']:<22} "
        f"reward {reward:+7.2f}  total {info['episode_return']:+7.2f}  "
        f"{info['minutes']:>4.0f} min  box {info['condition']:.0%}  {decision['pharmacy_name']}",
        flush=True,
    )


def print_order(env: FarumasiRiderEnv) -> None:
    traffic = "rush hour" if env.rush else "normal traffic"
    rain = "rain on the way" if env.rain_at < 10**5 else "dry"
    if env.rx_required:
        prescription = "prescription required, " + ("valid" if env.rx_valid else "INVALID")
    else:
        prescription = "no prescription needed"
    print(f"order: {MED_NAMES[env.med]} to {env.dest.name} ({env.dest.kind}); {traffic}; {rain}; {prescription}", flush=True)


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


def play(episodes: int, seed: int, model_path: str, device_name: str, heuristic: bool) -> None:
    device = resolve_device(device_name)
    net = None
    if not heuristic:
        net, payload = load_checkpoint(model_path, device)
        print(f"Loaded {payload.get('algo')} experiment {payload.get('config', {}).get('id')} from {model_path}")
    else:
        print("Playing with the rule-following rider")

    env = FarumasiRiderEnv(render_mode="human")
    results = []
    try:
        for episode in range(episodes):
            label = "rider" if heuristic else "learned"
            print(f"\n=== {label} episode {episode + 1}  seed {seed + episode} ===", flush=True)
            observation, _info = env.reset(seed=seed + episode)
            print_order(env)
            # First render creates the window and shows the mission card.
            env.render()
            renderer = env._renderer
            assert renderer is not None
            done = False
            step = 0
            while not done:
                while renderer.control_mode == "paused":
                    renderer.hold_frame(env)
                if renderer.control_mode == "self":
                    action = renderer.wait_for_action(env)
                    if action is None:
                        continue
                elif heuristic:
                    action = heuristic_action(env)
                else:
                    action = greedy_action(net, observation, device)
                observation, reward, terminated, truncated, info = env.step(action)
                print_step(step, reward, info)
                env.render()
                done = terminated or truncated
                step += 1
            outcome = env.outcome or "failed"
            print_result(outcome, env.minutes, env.return_so_far)
            results.append({"outcome": outcome, "return": float(env.return_so_far), "minutes": float(env.minutes)})
    finally:
        env.close()

    if not results:
        return
    done = sum(1 for item in results if item["outcome"] in ("delivered", "returned"))
    mean_return = sum(item["return"] for item in results) / len(results)
    mean_minutes = sum(item["minutes"] for item in results) / len(results)
    print(
        f"\nSUMMARY  {done}/{len(results)} deliveries finished correctly   "
        f"mean return {mean_return:+.2f}   mean time {mean_minutes:.0f} min",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch or drive a FARUMASI rider delivery")
    parser.add_argument("--model", default="models/best_agent.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--heuristic", action="store_true")
    args = parser.parse_args()
    play(args.episodes, args.seed, args.model, args.device, args.heuristic)


if __name__ == "__main__":
    main()
