# FARUMASI Rider Delivery

Mission-based reinforcement learning for safe, on-time medicine delivery in Kigali.

A FARUMASI motorcycle rider collects an order at Kigali City Pharmacy and must reach the patient within a 30-minute promise with the medicine dry. Traffic, rain, and prescription checks change what a good policy looks like. The same custom environment is used to train and compare **DQN**, **REINFORCE**, **PPO**, and **A2C** (TorchRL).

**Author:** Emmanuel Nsabagasani  
**Repository:** [Emmanuel-NS/Rl_Mission-based_Summative](https://github.com/Emmanuel-NS/Rl_Mission-based_Summative)  
**Demo video:** [Google Drive](https://drive.google.com/file/d/1CYprfJ1ZD92d95xxVT6NEJ7aEZ99C7a5/view?usp=sharing)

---

## Results (held-out test, 100 deliveries)

| Agent | Mean return | Success |
| --- | ---: | ---: |
| Random legal actions | −4.14 | 2% |
| Rule-following rider | +6.54 | 100% |
| DQN (best, γ = 0.9) | +6.24 | 99–100% |
| PPO | +6.21 | 98–99% |
| A2C | +6.00 | 93–99% |
| REINFORCE | +2.79 | 42% / 81% |

Best checkpoint: `models/best_agent.pt` (Double DQN, experiment 4, 120k frames). Full tables and figures: [`docs/RESULTS.md`](docs/RESULTS.md), [`logs/figures/`](logs/figures/). Environment specification: [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run python main.py --baselines
uv run python play.py --episodes 3 --seed 1000
```

`play.py` loads `models/best_agent.pt` by default and shows the pygame window plus a step-by-step terminal log. Use `--heuristic` for the rule-following reference rider. Window controls: **Self Play**, **Auto Play**, **Pause**.

On a fresh clone without a checkpoint, `uv run python main.py` trains a short PPO baseline so the entry point still runs after `uv sync`.

---

## Training and evaluation

Sweep (10 settings × 4 algorithms, 30,000 frames each, validation seeds 1000–1019):

```powershell
uv run python main.py --sweep all --device cpu --frames 30000 --eval-every 5000
```

Long runs and held-out evaluation (best setting per algorithm, 120,000 frames, 2 seeds; test seeds 5000–5099):

```powershell
uv run python main.py --final --frames 120000 --seeds 2 --device cpu
uv run python main.py --evaluate
```

Completed sweep runs are skipped on restart unless `--force` is passed. Logs go to `logs/`, checkpoints to `models/dqn/`, `models/pg/`, and `models/final/`. The same pipeline is in `colab_train.ipynb` (CPU runtime; the env step is Python-bound, so a free T4 does not speed up the sweep usefully).

Report draft (from logs and figures):

```powershell
uv run --no-project --with python-docx --with matplotlib --with pillow python report/build_report.py
```

---

## Serving

```powershell
uv run python serve.py --port 8000
uv run python -m service.export
```

`POST /act` returns the recommended action, legal mask, and policy scores for a posted delivery state. `GET /schema` lists fields. Export writes TorchScript and a JSON weight dump. Encoding matches training (`tests/test_service.py`).

---

## Environment (summary)

- **Map:** 6×4 Kigali junctions; 16 discrete actions with a legal-action mask.
- **Observation:** 94 values (phase, position, rain, box condition, prescription flags, traffic, compass bearing, mask).
- **Reward:** time penalty; bonuses for collect / verify / hand over / QR scan (or return of an invalid Rx); large penalties for unsafe handover or a ruined box.
- **Terminals:** delivered, returned, violation, ruined, cancelled, abandoned.

`uv run python main.py --describe` prints the observation layout. Details in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

---

## Project layout

```text
main.py                 # CLI: baselines, sweep, final, evaluate
play.py                 # Interactive / auto play of best agent
serve.py                # JSON policy API
colab_train.ipynb       # Full training notebook
environment/            # Gymnasium env, city map, pygame renderer
training/               # DQN + PG (REINFORCE, PPO, A2C), eval, plots
models/                 # dqn/, pg/, final/, best_agent.pt
logs/                   # Run logs, summaries, figures
service/                # Serving + export
report/                 # Report builder (assignment template)
docs/                   # ENVIRONMENT.md, RESULTS.md
assets/                 # Sprites and screenshots
tests/
```

Sprites under `assets/raw/` were produced with an image model and cut into `assets/sprites/` via `python -m environment.sprites`. Environment logic, rendering, and training code are original to this project.

---

## License / course work

Submitted for the Machine Learning Techniques II mission-based RL summative. Code and results are those of Emmanuel Nsabagasani.
