"""Evaluation figures for the report: convergence, test returns, generalisation, behaviour.

Reads the training logs and logs/test_evaluation.json (training/evaluation.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from training.plots import ALGOS, COLOURS, _read_csv, _summaries

OUTCOME_COLOURS = {
    "delivered": "#16a34a",
    "returned": "#0ea5e9",
    "violation": "#dc2626",
    "ruined": "#92400e",
    "cancelled": "#f59e0b",
    "abandoned": "#6b7280",
}


def _smooth(values: list[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    out = np.empty_like(array)
    for i in range(len(array)):
        out[i] = array[max(0, i - window + 1) : i + 1].mean()
    return out


def _load_eval(log_root: Path) -> dict | None:
    path = log_root / "test_evaluation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _best_sweep_keys(summaries: list[dict]) -> dict[str, str]:
    keys = {}
    for algo in ALGOS:
        group = [s for s in summaries if s["algo"] == algo]
        if group:
            best = max(group, key=lambda s: s["best_eval_return"])
            keys[algo] = f"{algo}_exp{best['experiment']:02d}"
    return keys


def _agents(result: dict, summaries: list[dict]) -> list[tuple[str, str, dict]]:
    """(label, colour, summary) for the baselines and the best agent of each algorithm."""
    rows = [
        ("random", "#9ca3af", result["baselines"]["random"]),
        ("rule-following", "#111827", result["baselines"]["heuristic"]),
    ]
    sweep = _best_sweep_keys(summaries)
    for algo in ALGOS:
        finals = [v for k, v in result.get("final", {}).items() if k.startswith(f"{algo}_seed")]
        if finals:
            rows.append((f"{algo} (long run)", COLOURS[algo], {**finals[0], "all": finals}))
        elif algo in sweep:
            rows.append((f"{algo} ({sweep[algo]})", COLOURS[algo], result["sweep"][sweep[algo]]))
    return rows


def convergence(log_root: Path, out_dir: Path, summaries: list[dict], result: dict | None) -> str:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for axis, algo in zip(axes, ALGOS):
        curves = []
        for item in (s for s in summaries if s["algo"] == algo):
            rows = _read_csv(log_root / algo / f"experiment_{item['experiment']:02d}" / "training_log.csv")
            rows = [r for r in rows if "train_return" in r]
            if len(rows) < 4:
                continue
            frames = np.array([r["frame"] for r in rows])
            curves.append((frames, _smooth([r["train_return"] for r in rows], 8)))
        if not curves:
            continue
        grid = np.linspace(2000, min(c[0][-1] for c in curves), 60)
        stack = np.array([np.interp(grid, f, y) for f, y in curves])
        axis.fill_between(grid, stack.min(0), stack.max(0), color=COLOURS[algo], alpha=0.18, label="range of 10 runs")
        axis.plot(grid, stack.mean(0), color=COLOURS[algo], linewidth=2.2, label="mean of 10 runs")
        if result:
            axis.axhline(result["baselines"]["heuristic"]["return_mean"], color="#111827", linestyle="--", linewidth=1, label="rule-following rider")
            axis.axhline(result["baselines"]["random"]["return_mean"], color="#9ca3af", linestyle=":", linewidth=1.2, label="random legal actions")
        axis.set_title(algo.upper())
        axis.set_xlabel("Environment frames")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Training return (smoothed)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Convergence: training return of every sweep run, by algorithm")
    fig.tight_layout()
    path = out_dir / "convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def test_returns(result: dict, summaries: list[dict], out_dir: Path) -> str:
    rows = _agents(result, summaries)
    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    labels, means, errors, colours = [], [], [], []
    for label, colour, summary in rows:
        labels.append(label)
        colours.append(colour)
        if "all" in summary:
            values = [s["return_mean"] for s in summary["all"]]
            means.append(float(np.mean(values)))
            errors.append(float(np.std(values)) if len(values) > 1 else summary["return_sem"])
        else:
            means.append(summary["return_mean"])
            errors.append(summary["return_sem"])
    y = np.arange(len(labels))
    axis.barh(y, means, xerr=errors, color=colours, capsize=3)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean return on 100 unseen test deliveries (error bar: standard error, or std over seeds)")
    axis.set_title("Held-out test return")
    for i, m in enumerate(means):
        axis.text(m + (0.1 if m >= 0 else -0.1), i, f"{m:+.2f}", va="center", ha="left" if m >= 0 else "right", fontsize=8)
    axis.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "test_returns.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def generalisation(result: dict, summaries: list[dict], out_dir: Path) -> str:
    sweep = _best_sweep_keys(summaries)
    chosen = [("rule-following", "heuristic", "#111827")]
    for algo in ALGOS:
        finals = sorted(k for k in result.get("final", {}) if k.startswith(f"{algo}_seed"))
        if finals:
            chosen.append((algo, finals[0], COLOURS[algo]))
        elif algo in sweep:
            chosen.append((algo, sweep[algo], COLOURS[algo]))
    names = ["dry", "rain", "normal traffic", "rush hour", "no prescription", "valid prescription", "invalid prescription", "home", "office", "hotel"]
    fig, axis = plt.subplots(figsize=(11, 4.2))
    width = 0.8 / len(chosen)
    for i, (label, key, colour) in enumerate(chosen):
        data = result["conditions"].get(key, {})
        values = [data.get(name, {}).get("success", np.nan) for name in names]
        axis.bar(np.arange(len(names)) + i * width, values, width, label=label, color=colour)
    axis.set_xticks(np.arange(len(names)) + 0.4 - width / 2, names, rotation=25, ha="right")
    axis.set_ylabel("Success rate")
    axis.set_ylim(0, 1.05)
    axis.text(0.0, -0.62, "Success = delivered, or a wrong prescription taken back to the pharmacy.", transform=axis.transAxes, fontsize=8)
    axis.set_title("Generalisation: success rate by condition on the test deliveries")
    axis.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.38))
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "generalization.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def outcomes(result: dict, summaries: list[dict], out_dir: Path) -> str:
    rows = _agents(result, summaries)
    fig, axis = plt.subplots(figsize=(9, 4))
    bottoms = np.zeros(len(rows))
    for outcome, colour in OUTCOME_COLOURS.items():
        values = np.array([s["outcomes"][outcome] / s["episodes"] for _l, _c, s in rows])
        axis.bar([label for label, _c, _s in rows], values, bottom=bottoms, color=colour, label=outcome)
        bottoms += values
    axis.set_ylabel("Share of test deliveries")
    axis.set_title("How the deliveries ended")
    axis.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    plt.setp(axis.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    path = out_dir / "outcomes.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def behaviour(result: dict, summaries: list[dict], out_dir: Path) -> str:
    rows = _agents(result, summaries)
    actions = list(rows[0][2]["action_share"].keys())
    matrix = np.array([[s["action_share"][a] for a in actions] for _l, _c, s in rows])
    fig, axis = plt.subplots(figsize=(10, 3.8))
    image = axis.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=0.35)
    axis.set_xticks(range(len(actions)), actions, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(len(rows)), [label for label, _c, _s in rows], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= 0.02:
                axis.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=7)
    axis.set_title("Share of decisions spent on each action (test deliveries)")
    fig.colorbar(image, ax=axis, fraction=0.025)
    fig.tight_layout()
    path = out_dir / "behaviour.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def validation_vs_test(result: dict, summaries: list[dict], out_dir: Path) -> str:
    fig, axis = plt.subplots(figsize=(5.4, 4.6))
    for algo in ALGOS:
        xs, ys = [], []
        for s in (s for s in summaries if s["algo"] == algo):
            key = f"{algo}_exp{s['experiment']:02d}"
            if key in result["sweep"]:
                xs.append(s["best_eval_return"])
                ys.append(result["sweep"][key]["return_mean"])
        axis.scatter(xs, ys, color=COLOURS[algo], label=algo, s=28)
    lims = [-5.2, 2]
    axis.plot(lims, lims, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Validation return used to pick the checkpoint (20 episodes)")
    axis.set_ylabel("Test return (100 unseen episodes)")
    axis.set_title("Validation versus test, 40 sweep runs")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "validation_vs_test.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def final_curves(log_root: Path, out_dir: Path) -> str | None:
    fig, axis = plt.subplots(figsize=(8.5, 4.4))
    drawn = False
    for algo in ALGOS:
        runs = []
        for path in sorted((log_root / "final" / algo).glob("seed_*/training_log.csv")):
            rows = [r for r in _read_csv(path) if "eval_return" in r]
            if rows:
                runs.append(([r["frame"] for r in rows], [r["eval_return"] for r in rows]))
        if not runs:
            continue
        grid = np.linspace(max(r[0][0] for r in runs), min(r[0][-1] for r in runs), 40)
        stack = np.array([np.interp(grid, f, y) for f, y in runs])
        axis.plot(grid, stack.mean(0), color=COLOURS[algo], linewidth=2, label=f"{algo} (mean of {len(runs)} seeds)")
        axis.fill_between(grid, stack.min(0), stack.max(0), color=COLOURS[algo], alpha=0.15)
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    axis.set_xlabel("Environment frames")
    axis.set_ylabel("Return on validation deliveries")
    axis.set_title("Long runs of the best setting of each algorithm")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "final_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def build_report_figures(log_root: str | Path = "logs", out_dir: str | Path = "logs/figures") -> list[str]:
    log_root, out_dir = Path(log_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = _summaries(log_root)
    result = _load_eval(log_root)
    written = [convergence(log_root, out_dir, summaries, result)]
    final = final_curves(log_root, out_dir)
    if final:
        written.append(final)
    if result:
        written += [
            test_returns(result, summaries, out_dir),
            generalisation(result, summaries, out_dir),
            outcomes(result, summaries, out_dir),
            behaviour(result, summaries, out_dir),
            validation_vs_test(result, summaries, out_dir),
        ]
    return written
