"""Shared checkpoint, logging, and greedy evaluation helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torchrl.modules import MLP

from environment.baselines import heuristic_action
from environment.custom_env import ENV_VERSION, N_ACTIONS, OBS_DIM, FarumasiShiftEnv

EVAL_SEEDS = tuple(range(1000, 1020))


def align_frames(total_frames: int, frames_per_batch: int) -> int:
    if frames_per_batch <= 0:
        return total_frames
    remainder = total_frames % frames_per_batch
    if remainder == 0:
        return total_frames
    return total_frames + (frames_per_batch - remainder)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(name)


def make_policy_net(hidden: int, out_features: int = N_ACTIONS) -> MLP:
    return MLP(
        in_features=OBS_DIM,
        out_features=out_features,
        num_cells=[hidden, hidden],
        activation_class=torch.nn.ReLU,
    )


def make_value_net(hidden: int) -> MLP:
    return MLP(
        in_features=OBS_DIM,
        out_features=1,
        num_cells=[hidden, hidden],
        activation_class=torch.nn.ReLU,
    )


def masked_logits(net: torch.nn.Module, observation: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    logits = net(tensor).squeeze(0)
    mask = torch.as_tensor(observation[-N_ACTIONS:] > 0.5, device=device)
    return logits.masked_fill(~mask, -1e9)


@torch.no_grad()
def greedy_action(net: torch.nn.Module, observation: np.ndarray, device: torch.device) -> int:
    return int(torch.argmax(masked_logits(net, observation, device)).item())


def run_episode(net: torch.nn.Module, seed: int, device: torch.device, render_mode: str | None = None):
    env = FarumasiShiftEnv(render_mode=render_mode)
    observation, info = env.reset(seed=seed)
    done = False
    trace = []
    while not done:
        action = greedy_action(net, observation, device)
        observation, reward, terminated, truncated, info = env.step(action)
        trace.append(
            {
                "action": info["decision"].get("action"),
                "event": info["event"],
                "reward": reward,
                "medicine": info["decision"].get("medicine"),
                "pharmacy": info["decision"].get("pharmacy_name"),
                "shift_return": info["shift_return"],
            }
        )
        done = terminated or truncated
        if render_mode == "human":
            env.render()
    summary = {
        "return": float(env.return_so_far),
        "urgent_met_filled": int(env.urgent_met_filled),
        "urgent_met_total": int(env.urgent_met_total),
        "routine_met_filled": int(env.routine_met_filled),
        "dispensed": int(env.dispensed),
        "referred": int(env.referred),
        "trace": trace,
    }
    if render_mode == "human":
        env.render()
    env.close()
    return summary


def evaluate(net: torch.nn.Module, device: torch.device, seeds: tuple[int, ...] = EVAL_SEEDS) -> dict[str, float]:
    returns = []
    fills = []
    for seed in seeds:
        summary = run_episode(net, seed, device)
        returns.append(summary["return"])
        fills.append(summary["urgent_met_filled"] / max(1, summary["urgent_met_total"]))
    return {
        "eval_return": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_urgent_fill": float(np.mean(fills)),
    }


def heuristic_reference(seeds: tuple[int, ...] = EVAL_SEEDS) -> dict[str, float]:
    returns = []
    fills = []
    for seed in seeds:
        env = FarumasiShiftEnv()
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(heuristic_action(env))
            done = terminated or truncated
        returns.append(float(env.return_so_far))
        fills.append(env.urgent_met_filled / max(1, env.urgent_met_total))
        env.close()
    return {
        "heuristic_return": float(np.mean(returns)),
        "heuristic_return_std": float(np.std(returns)),
        "heuristic_urgent_fill": float(np.mean(fills)),
    }


class MetricLog:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "training_log.csv"
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = None

    def write(self, row: dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    matrices = [value for value in payload["state_dict"].values() if getattr(value, "ndim", 0) == 2]
    if not any(int(value.shape[1]) == OBS_DIM for value in matrices):
        raise RuntimeError(
            "This checkpoint belongs to the old dispatcher. Train the patient-map environment again."
        )
    net = make_policy_net(int(payload["hidden"]))
    net.load_state_dict(payload["state_dict"])
    net.to(device)
    net.eval()
    return net, payload


def write_summary(directory: Path, summary: dict) -> None:
    summary = dict(summary)
    summary["env_version"] = ENV_VERSION
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps(summary["config"], indent=2), encoding="utf-8")


def promote_best(log_root: Path = Path("logs"), model_root: Path = Path("models")) -> dict | None:
    best = None
    for path in log_root.glob("*/*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("env_version") != ENV_VERSION:
            continue
        if best is None or summary.get("best_eval_return", -1e9) > best.get("best_eval_return", -1e9):
            best = summary
    if best is None:
        return None
    algo = best["algo"]
    source = model_root / ("dqn" if algo == "dqn" else "pg") / f"{algo}_exp{best['experiment']:02d}.pt"
    if not source.exists():
        return best
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    destination = model_root / "best_agent.pt"
    torch.save(payload, destination)
    (model_root / "best_agent.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best
