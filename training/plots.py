"""Figures the report uses to compare the four algorithms."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from environment.custom_env import ENV_VERSION

ALGOS = ("dqn", "reinforce", "ppo", "a2c")
COLOURS = {
    "dqn": "#0f766e",
    "reinforce": "#b45309",
    "ppo": "#1d4ed8",
    "a2c": "#7c3aed",
}


def _read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items() if value != ""})
        return rows


def _summaries(log_root: Path) -> list[dict]:
    found = []
    for path in sorted(log_root.glob("*/*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("env_version") == ENV_VERSION:
            found.append(item)
    return found


def _best_summary(items: list[dict], algo: str) -> dict | None:
    group = [item for item in items if item["algo"] == algo]
    if not group:
        return None
    return max(group, key=lambda item: item.get("best_eval_return", float("-inf")))


def build_plots(log_root: str | Path = "logs", out_dir: str | Path = "logs/figures") -> list[str]:
    log_root = Path(log_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = _summaries(log_root)
    written: list[str] = []

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for algo in ALGOS:
        best = _best_summary(summaries, algo)
        if best is None:
            continue
        rows = _read_csv(log_root / algo / f"experiment_{best['experiment']:02d}" / "training_log.csv")
        frames = [row["frame"] for row in rows if "eval_return" in row]
        scores = [row["eval_return"] for row in rows if "eval_return" in row]
        if frames:
            axis.plot(frames, scores, label=f"{algo} exp {best['experiment']}", color=COLOURS[algo], linewidth=2)
    axis.set_xlabel("Environment frames")
    axis.set_ylabel("Greedy return on held-out shifts")
    axis.set_title("Best run of each algorithm")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "comparative_rewards.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    for axis, algo in zip(axes.ravel(), ALGOS):
        group = [item for item in summaries if item["algo"] == algo]
        for item in group:
            csv_path = log_root / algo / f"experiment_{item['experiment']:02d}" / "training_log.csv"
            if not csv_path.exists():
                continue
            rows = _read_csv(csv_path)
            frames = [row["frame"] for row in rows if "eval_return" in row]
            scores = [row["eval_return"] for row in rows if "eval_return" in row]
            if frames:
                axis.plot(frames, scores, alpha=0.85, linewidth=1.2)
        axis.set_title(algo)
        axis.set_xlabel("Frames")
        axis.set_ylabel("Eval return")
        axis.grid(True, alpha=0.3)
    fig.suptitle("All hyperparameter runs")
    fig.tight_layout()
    path = out_dir / "all_runs.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    dqn_best = _best_summary(summaries, "dqn")
    if dqn_best is not None:
        rows = _read_csv(log_root / "dqn" / f"experiment_{dqn_best['experiment']:02d}" / "training_log.csv")
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.plot([row["frame"] for row in rows], [row["loss"] for row in rows], color=COLOURS["dqn"])
        axis.set_xlabel("Environment frames")
        axis.set_ylabel("DQN loss")
        axis.set_title(f"DQN loss, experiment {dqn_best['experiment']}")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / "dqn_loss.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for algo in ("reinforce", "ppo", "a2c"):
        best = _best_summary(summaries, algo)
        if best is None:
            continue
        rows = _read_csv(log_root / algo / f"experiment_{best['experiment']:02d}" / "training_log.csv")
        axis.plot(
            [row["frame"] for row in rows],
            [row.get("entropy", 0.0) for row in rows],
            label=algo,
            color=COLOURS[algo],
            linewidth=2,
        )
    axis.set_xlabel("Environment frames")
    axis.set_ylabel("Policy entropy")
    axis.set_title("Entropy of the best policy-gradient run")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "policy_entropy.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, algo in zip(axes.ravel(), ALGOS):
        group = sorted((item for item in summaries if item["algo"] == algo), key=lambda item: item["experiment"])
        labels = [str(item["experiment"]) for item in group]
        values = [item.get("best_eval_return", 0.0) for item in group]
        urgent = [item.get("best_urgent_fill", 0.0) for item in group]
        axis.bar(labels, values, color=COLOURS[algo])
        axis.set_title(f"{algo} best eval return")
        axis.set_xlabel("Experiment")
        axis.set_ylabel("Return")
        for label, fill in zip(labels, urgent):
            axis.annotate(f"{fill:.0%}", (label, 0), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    fig.tight_layout()
    path = out_dir / "experiment_bars.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))
    return written


def write_markdown_tables(log_root: str | Path = "logs", path: str | Path = "logs/experiment_tables.md") -> str:
    log_root = Path(log_root)
    summaries = _summaries(log_root)
    lines = ["# Hyperparameter results", ""]
    for algo in ALGOS:
        group = sorted((item for item in summaries if item["algo"] == algo), key=lambda item: item["experiment"])
        lines.append(f"## {algo}")
        lines.append("")
        if not group:
            lines.append("No runs yet.")
            lines.append("")
            continue
        keys = [key for key in group[0]["config"].keys() if key not in ("id", "name")]
        header = ["exp", "name", *keys, "best_return", "urgent_fill"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for item in group:
            config = item["config"]
            cells = [str(item["experiment"]), str(config.get("name", ""))]
            for key in keys:
                value = config.get(key, "")
                cells.append(f"{value:.4g}" if isinstance(value, float) else str(value))
            cells.append(f"{item.get('best_eval_return', float('nan')):.2f}")
            cells.append(f"{item.get('best_urgent_fill', float('nan')):.2f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return str(destination)
