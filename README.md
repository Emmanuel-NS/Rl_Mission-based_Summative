# FARUMASI rider delivery

Reinforcement learning for a FARUMASI rider in Kigali. The rider collects a medicine order at Kigali City Pharmacy and has to bring it to the patient inside a 30-minute promise, with the box unharmed. The roads are live: rush hour jams the main road, an incident can block a street, and rain can start at any time. At the destination the rider parks, walks to the gate, carries the box to the patient, checks the prescription when one is required, hands over, and scans the patient's QR code. A wrong prescription must go back to the pharmacy that dispatched it. The same city is used for DQN, REINFORCE, PPO, and A2C.

## Setup

[uv](https://docs.astral.sh/uv/) installs the dependencies and creates the virtual environment.

```powershell
uv sync
uv run python main.py --baselines
uv run python main.py --screenshot
```

`uv run python main.py` with no arguments evaluates `models/best_agent.pt` when that file exists. On a fresh clone it trains the baseline PPO setting for 4096 frames so the project runs after `uv sync`.

## Full training

The comparison used in the report is 10 settings for each algorithm, 20,000 environment frames each, on the same deliveries and the same held-out seeds (1000–1019). Open `colab_train.ipynb` on a normal CPU runtime. The city step is Python and the network is small, so a T4 does not make the sweep faster and the free GPU quota runs out before the files can be downloaded.

```powershell
uv run python main.py --sweep all --device cpu --frames 20000 --eval-every 5000
```

One algorithm, one setting:

```powershell
uv run python main.py --sweep ppo --experiment 1 --device cpu --frames 20000
```

After the sweep, `models/best_agent.pt` is the greedy policy with the highest mean return on the held-out deliveries. Curves and tables are written to `logs/figures/` and `logs/experiment_tables.md`. A finished experiment is skipped if you start the sweep again, so a Colab disconnect does not throw away completed runs. Pass `--force` to train that setting again. Results from older environments (dispatcher, patient map, patient street) are ignored.

## Watch a delivery

```powershell
uv run python play.py --episodes 3 --seed 1000
```

The window is a top-down Kigali with live traffic, a minimap showing the fastest route, and a mission card, in the style of a delivery game. Pharmacy, gate, and lobby scenes are shown side-on. Every step is also printed in the terminal. `--heuristic` plays the hand-written rider, which is the reference the agents are compared against.

## What the agent observes and does

The map is a 6 by 4 grid of Kigali junctions (KN 3 Rd is the main road). Orders go to one of six places: two homes, an office, an office tower, and two hotels.

Actions (16): ride north, east, south, west; cover the box with the rain tarp; wait; enter the pharmacy; collect the order; park; knock or check in at reception; carry the box; verify the prescription; hand over; scan the QR code; walk back to the bike; return the medicine to the pharmacy.

`uv run python main.py --describe` prints every observation index (90 values: phase, position, destination and pharmacy offsets, clock, rain, box condition, prescription knowledge, the traffic level on all 38 roads, and a legal-action mask in the last 16 values).

The rider is paid for a delivery that is quick and keeps the medicine dry, and pays for every minute and for rain damage. It is penalised for handing over a prescription medicine without verification or with an invalid prescription, and for a ruined box. Returning an invalid prescription to the pharmacy is a correct ending with a smaller reward. Heavy traffic costs extra minutes, so riding the main road at rush hour is a mistake that a side street avoids. The episode ends when the delivery is scanned, the medicine is returned, the box is ruined, the order is cancelled at 55 minutes, or the rider reaches 80 steps.

## Art

The buildings, the people, the cars, the bike, the trees and the interior backdrops in `assets/raw/` were generated with an AI image tool for this project. `uv run python -m environment.sprites` cuts them into the transparent sprites in `assets/sprites/` that the window uses. The FARUMASI logo is drawn in code. The environment, the rendering, and the training are written for this project.

## Layout

```text
main.py
play.py
environment/custom_env.py
environment/city_map.py
environment/rendering.py
environment/sprites.py
environment/baselines.py
training/dqn_training.py
training/pg_training.py
training/experiments.py
models/dqn/
models/pg/
logs/
assets/
tests/
```

## Video

From this folder, with the camera and the whole screen shared: say that FARUMASI riders have to deliver fast and safely, that rush hour and rain change the best strategy, and that a wrong prescription goes back to the pharmacy. Then run `uv run python play.py --episodes 6 --seed 1000` and leave the window and the terminal visible. Checkpoints from earlier environments do not fit this one, so train again before recording the learned agent. `uv run python play.py --heuristic --episodes 6 --seed 1000` shows the hand-written rider on the same city.
