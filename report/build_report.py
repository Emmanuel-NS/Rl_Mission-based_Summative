"""Build the report as a .docx from the logs, the figures and report/narrative.py.

    uv run --no-project --with python-docx --with matplotlib --with pillow python report/build_report.py

Numbers in the tables come straight from logs/*/summary.json and logs/test_evaluation.json,
so the document cannot disagree with the files in the repository.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

from narrative import TEXT  # noqa: E402
import narrative_results  # noqa: E402,F401  (adds the results text to TEXT)

ENV_VERSION = "rider-delivery-3"
ALGOS = ("dqn", "reinforce", "ppo", "a2c")
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
FIG = ROOT / "logs" / "figures"
OUT = ROOT / "report" / "FARUMASI_RL_Report.docx"


def load_sweep() -> dict[str, list[dict]]:
    out = {a: [] for a in ALGOS}
    for path in sorted((ROOT / "logs").glob("*/experiment_*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("env_version") == ENV_VERSION:
            out[item["algo"]].append(item)
    for algo in out:
        out[algo].sort(key=lambda s: s["experiment"])
    return out


def load_final() -> dict[str, list[dict]]:
    out = {a: [] for a in ALGOS}
    for path in sorted((ROOT / "logs" / "final").glob("*/seed_*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["seed"] = int(path.parent.name.split("_")[1])
        out[item["algo"]].append(item)
    return out


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def changes(config: dict, base: dict) -> str:
    parts = []
    for key, value in config.items():
        if key in ("id", "name"):
            continue
        if key not in base or base[key] != value:
            parts.append(f"{key} {fmt(value)}")
    return ", ".join(parts) if parts else "baseline"


# ---------------------------------------------------------------- docx helpers


def shade(cell, colour: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:val"), "clear")
    element.set(qn("w:color"), "auto")
    element.set(qn("w:fill"), colour)
    props.append(element)


def set_cell(cell, text: str, size: float = 8, bold: bool = False, align=None, colour: str | None = None) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.space_before = Pt(0)
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    if colour:
        run.font.color.rgb = RGBColor.from_string(colour)


def table(doc, header: list[str], rows: list[list[str]], widths: list[float], size: float = 8, bold_rows: set[int] | None = None) -> None:
    grid = doc.add_table(rows=1, cols=len(header))
    grid.style = "Table Grid"
    grid.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(header):
        set_cell(grid.rows[0].cells[i], text, size, True, colour="FFFFFF")
        shade(grid.rows[0].cells[i], "0F766E")
    for r, row in enumerate(rows):
        cells = grid.add_row().cells
        for i, text in enumerate(row):
            set_cell(cells[i], text, size, bold=bool(bold_rows and r in bold_rows))
            if bold_rows and r in bold_rows:
                shade(cells[i], "DCFCE7")
    for row in grid.rows:
        for i, width in enumerate(widths):
            row.cells[i].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def caption(doc, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(8.5)


def para(doc, text: str, size: float = 10, bold: bool = False, after: float = 4) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.space_before = Pt(0)
    run = paragraph.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold


def bullets(doc, items: list[str], size: float = 10) -> None:
    for text in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(1)
        run = paragraph.add_run(text)
        run.font.size = Pt(size)


def figure(doc, path: Path, width: float, text: str) -> None:
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_after = Pt(0)
    caption(doc, text)


def figure_pair(doc, left: tuple[Path, str], right: tuple[Path, str], width: float = 3.2) -> None:
    grid = doc.add_table(rows=2, cols=2)
    grid.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (path, text) in enumerate((left, right)):
        cell = grid.rows[0].cells[i]
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].add_run().add_picture(str(path), width=Inches(width))
        set_cell(grid.rows[1].cells[i], text, 8, align=WD_ALIGN_PARAGRAPH.LEFT)
        grid.rows[1].cells[i].paragraphs[0].runs[0].italic = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def heading(doc, text: str, level: int = 1) -> None:
    paragraph = doc.add_heading(text, level=level)
    paragraph.paragraph_format.space_before = Pt(8 if level == 1 else 4)
    paragraph.paragraph_format.space_after = Pt(3)
    for run in paragraph.runs:
        run.font.color.rgb = RGBColor(0x0A, 0x3D, 0x38)
        run.font.size = Pt(14 if level == 1 else 11.5)


def montage() -> Path:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    names = [("environment.png", "Riding: lanes, lights, bike lane, minimap"), ("environment_rain.png", "Rain: umbrella and live traffic"), ("environment_pharmacy.png", "Collecting at the pharmacy"), ("environment_handover.png", "Face-to-face handover")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6))
    for axis, (name, title) in zip(axes.ravel(), names):
        axis.imshow(mpimg.imread(ROOT / "assets" / name))
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.tight_layout(pad=0.4)
    path = FIG / "environment_montage.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- content


def build() -> None:
    sweep = load_sweep()
    final = load_final()
    ev = json.loads((ROOT / "logs" / "test_evaluation.json").read_text(encoding="utf-8"))
    base = ev["baselines"]
    doc = Document()
    section = doc.sections[0]
    section.left_margin = section.right_margin = Inches(0.8)
    section.top_margin = section.bottom_margin = Inches(0.7)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("FARUMASI Rider: Reinforcement Learning for Safe, Fast Medicine Delivery in Kigali")
    run.bold = True
    run.font.size = Pt(17)
    run.font.color.rgb = RGBColor(0x0A, 0x3D, 0x38)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("Emmanuel Nsabagasani   |   Mission-Based Reinforcement Learning Summative   |   github.com/Emmanuel-NS/Rl_Mission-based_Summative").font.size = Pt(9)

    heading(doc, "1. Introduction")
    para(doc, TEXT["intro"])

    heading(doc, "2. Environment")
    para(doc, TEXT["env_overview"])
    figure(doc, montage(), 6.4, "Figure 1. The environment (pygame, 30 fps). Top-down Kigali with lanes, traffic lights, a bike lane and live traffic; side-on pharmacy and handover scenes; rain and umbrella.")
    heading(doc, "2.1 Why reinforcement learning", 2)
    para(doc, TEXT["why_rl"])
    heading(doc, "2.2 Actions", 2)
    table(
        doc,
        ["#", "Action", "Legal when", "Real-world step"],
        [list(row) for row in TEXT["actions"]],
        [0.4, 1.45, 2.35, 2.7],
        7.5,
    )
    heading(doc, "2.3 Observation (90 values, all scaled to 0-1)", 2)
    para(doc, TEXT["obs_intro"], 9.5)
    table(
        doc,
        ["Index", "Name", "Encoding", "Simulation rule", "Real-world source"],
        [list(row) for row in TEXT["observations"]],
        [0.5, 1.2, 1.4, 2.0, 1.8],
        7,
    )
    heading(doc, "2.4 Rewards, start state and terminal conditions", 2)
    table(doc, ["Event", "Reward"], [list(row) for row in TEXT["rewards"]], [4.4, 2.5], 7.5)
    para(doc, TEXT["start_terminal"], 9.5)

    heading(doc, "3. Methods")
    para(doc, TEXT["methods"])
    table(doc, ["Algorithm", "TorchRL pieces", "Specific controls tuned"], [list(row) for row in TEXT["method_rows"]], [0.9, 3.0, 3.0], 7.5)
    para(doc, TEXT["protocol"])

    heading(doc, "4. Hyperparameter experiments")
    para(doc, TEXT["hp_intro"])
    for algo in ALGOS:
        items = sweep[algo]
        base_config = items[0]["config"]
        best_validation = max(items, key=lambda s: s["best_eval_return"])["experiment"]
        rows = []
        bold = set()
        for i, item in enumerate(items):
            key = f"{algo}_exp{item['experiment']:02d}"
            test = ev["sweep"].get(key)
            rows.append(
                [
                    str(item["experiment"]),
                    item["config"]["name"].replace("_", " "),
                    changes(item["config"], base_config),
                    f"{item['best_eval_return']:+.2f}",
                    f"{test['return_mean']:+.2f}" if test else "",
                    f"{test['success']:.0%}" if test else "",
                    f"{test['wait_share']:.0%}" if test else "",
                ]
            )
            if item["experiment"] == best_validation:
                bold.add(i)
        heading(doc, f"4.{ALGOS.index(algo) + 1} {NAMES[algo]}", 2)
        table(
            doc,
            ["Exp", "Setting", "Change from baseline (exp 1)", "Val. return", "Test return", "Test success", "Share of WAIT"],
            rows,
            [0.35, 1.3, 2.35, 0.7, 0.7, 0.7, 0.7],
            7.5,
            bold,
        )
        para(doc, TEXT[f"hp_{algo}"], 9.5)
    caption(doc, "Tables 4.1-4.4. Ten settings per algorithm, 30,000 frames each, training seed 0. Val. return is the best of four validation checks (seeds 1000-1019) used to pick the checkpoint; Test return is the mean over 100 unseen deliveries (seeds 5000-5099). The highlighted row has the best validation return.")

    heading(doc, "5. Results and discussion")
    para(doc, TEXT["results_intro"])
    figure_pair(
        doc,
        (FIG / "comparative_rewards.png", "Figure 2. Validation return of the best run of each algorithm in the sweep."),
        (FIG / "convergence.png", "Figure 3. Training return of all ten runs per algorithm (mean and range)."),
    )
    para(doc, TEXT["convergence"], 9.5)
    figure_pair(
        doc,
        (FIG / "dqn_loss.png", "Figure 4. DQN training loss (best DQN run). The loss stays small because the target moves as the policy improves."),
        (FIG / "policy_entropy.png", "Figure 5. Policy entropy of the best REINFORCE, PPO and A2C runs."),
    )
    para(doc, TEXT["loss_entropy"], 9.5)
    heading(doc, "5.1 Long runs and the held-out test", 2)
    para(doc, TEXT["final_intro"])
    final_rows = []
    for algo in ALGOS:
        runs = final[algo]
        if not runs:
            continue
        tests = [ev["final"][f"{algo}_seed{r['seed']}"] for r in runs if f"{algo}_seed{r['seed']}" in ev["final"]]
        means = [t["return_mean"] for t in tests]
        final_rows.append(
            [
                NAMES[algo],
                runs[0]["config"]["name"].replace("_", " "),
                ", ".join(f"{m:+.2f}" for m in means),
                f"{sum(means) / len(means):+.2f}",
                f"{sum(t['success'] for t in tests) / len(tests):.0%}",
                f"{sum(t['wait_share'] for t in tests) / len(tests):.0%}",
            ]
        )
    final_rows.append(["Random legal actions", "", "", f"{base['random']['return_mean']:+.2f}", f"{base['random']['success']:.0%}", f"{base['random']['wait_share']:.0%}"])
    final_rows.append(["Rule-following rider", "", "", f"{base['heuristic']['return_mean']:+.2f}", f"{base['heuristic']['success']:.0%}", f"{base['heuristic']['wait_share']:.0%}"])
    table(doc, ["Agent", "Setting", "Test return per seed", "Mean", "Success", "WAIT share"], final_rows, [1.4, 1.5, 1.7, 0.7, 0.8, 0.8], 8)
    caption(doc, "Table 5.1. Best setting of each algorithm trained for 120,000 frames (two seeds) and scored on 100 unseen test deliveries.")
    figure_pair(
        doc,
        (FIG / "final_curves.png", "Figure 6. Validation return during the long runs (mean and range over seeds)."),
        (FIG / "test_returns.png", "Figure 7. Test return on 100 unseen deliveries, with the two baselines."),
    )
    para(doc, TEXT["final_discussion"], 9.5)
    heading(doc, "5.2 Generalisation and behaviour", 2)
    figure_pair(
        doc,
        (FIG / "generalization.png", "Figure 8. Success rate by delivery condition (test deliveries)."),
        (FIG / "behaviour.png", "Figure 9. Share of decisions spent on each action."),
    )
    para(doc, TEXT["behaviour"], 9.5)
    figure_pair(
        doc,
        (FIG / "outcomes.png", "Figure 10. How the test deliveries ended."),
        (FIG / "validation_vs_test.png", "Figure 11. Validation return against test return for the 40 sweep runs."),
    )
    para(doc, TEXT["outcomes"], 9.5)
    heading(doc, "5.3 Limitations", 2)
    bullets(doc, TEXT["limitations"], 9.5)

    heading(doc, "6. System implementation")
    para(doc, TEXT["system"], 9.5)

    heading(doc, "7. Conclusion")
    para(doc, TEXT["conclusion"])
    heading(doc, "Reproducing the results", 2)
    para(doc, TEXT["reproduce"], 9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print("saved", OUT)


if __name__ == "__main__":
    build()
