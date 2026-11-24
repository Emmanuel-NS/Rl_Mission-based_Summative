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

The comparison used in the report is 10 settings for each algorithm, 30,000 environment frames each, on the same deliveries and the same held-out seeds (1000–1019). Open `colab_train.ipynb` on a normal CPU runtime. The city step is Python and the network is small, so a T4 does not make the sweep faster and the free GPU quota runs out before the files can be downloaded.

```powershell
uv run python main.py --sweep all --device cpu --frames 30000 --eval-every 5000
```

One algorithm, one setting:

```powershell
uv run python main.py --sweep ppo --experiment 1 --device cpu --frames 30000
```

After the sweep, `models/best_agent.pt` is the greedy policy with the highest mean return on the validation deliveries. Curves and tables are written to `logs/figures/` and `logs/experiment_tables.md`. A finished experiment is skipped if you start the sweep again, so a Colab disconnect does not throw away completed runs. Pass `--force` to train that setting again. Results from older environments (dispatcher, patient map, patient street) are ignored.

## Long runs and held-out evaluation

The sweep gives every setting 30,000 frames so that settings can be compared fairly. Two more steps show where each algorithm ends up and whether the result holds on deliveries it never saw.

```powershell
uv run python main.py --final --frames 120000 --seeds 2 --device cpu
uv run python main.py --evaluate
```

`--final` trains the best setting of each algorithm for 120,000 frames on two training seeds (logs in `logs/final/`, checkpoints in `models/final/`) and makes the run with the best validation return the `models/best_agent.pt` that `play.py` shows. `--evaluate` scores the random rider, the rule-following rider, all 40 sweep checkpoints and the long runs on 100 test deliveries (seeds 5000-5099) that no run used, writes `logs/test_evaluation.json`, and draws the report figures in `logs/figures/`: convergence, test returns, generalisation by condition, outcomes, behaviour, and validation against test. The same two commands are the last cells of `colab_train.ipynb`.

Headline result on the 100 test deliveries: random legal actions -4.14 (2% success), the rule-following rider +6.54 (100%), DQN +6.24 (both seeds, 99-100%), PPO +6.21 (98-99%), A2C +6.00 (93-99%) and REINFORCE +2.79 (42% and 81%) after 120,000 frames. No long run handed over without a valid prescription. The full tables (all 40 sweep runs, the long runs, success by condition) are in `docs/RESULTS.md`, written by `--evaluate`; the report draft is built from the same files by `report/build_report.py`.

## Use the trained agent from another system

```powershell
uv run python serve.py --port 8000
uv run python -m service.export
```

`serve.py` is a small JSON API (standard library only). The rider app would post the delivery state to `POST /act` and get the recommended action, the legal actions and the policy's scores; `GET /schema` lists the state fields. `service.export` writes a TorchScript file and a plain-JSON copy of the weights (three dense layers) for an on-device implementation. The state is encoded with the environment's own encoder, so serving cannot drift from training. `tests/test_service.py` covers the encoder, the API, and the export. The full environment description (actions, 94 observation values with their sources, rewards, start and terminal conditions) is in `docs/ENVIRONMENT.md`.

## Watch a delivery

```powershell
uv run python play.py --episodes 3 --seed 1000
```

The window is a top-down Kigali with live traffic, a minimap showing the fastest route, and a mission card, in the style of a delivery game. Cars stay in their lanes, stop at the traffic lights, queue behind a blocked road, and wait for the rider; the rider uses the green bike lane at the kerb and waits when cross traffic is still in a junction, and rolls through junctions without stopping in them. In the rain the rider rides under an umbrella; when the agent chooses WAIT or COVER, the rider pulls over and stands under the porch of a nearby house instead of stopping in the road. Pharmacy, gate, and lobby scenes are shown side-on. At the patient the two stand face to face: the rider holds out the box, the patient takes it, and the rider scans the QR code. Every step is also printed in the terminal. `--heuristic` plays the hand-written rider, which is the reference the agents are compared against.

## What the agent observes and does

The map is a 6 by 4 grid of Kigali junctions (KN 3 Rd is the main road). Orders go to one of six places: two homes, an office, an office tower, and two hotels.

Actions (16): ride north, east, south, west; cover the box with the rain tarp; wait; enter the pharmacy; collect the order; park; knock or check in at reception; carry the box; verify the prescription; hand over; scan the QR code; walk back to the bike; return the medicine to the pharmacy.

`uv run python main.py --describe` prints every observation index (94 values: phase, position, destination and pharmacy offsets, clock, rain, box condition, prescription knowledge, the traffic level on all 38 roads, a compass bearing to the destination, and a legal-action mask in the last 16 values).

The rider is paid for a delivery that is quick and keeps the medicine dry, and pays for every minute and for rain damage. It is penalised for handing over a prescription medicine without verification or with an invalid prescription, and for a ruined box. Returning an invalid prescription to the pharmacy is a correct ending with a smaller reward. Heavy traffic costs extra minutes, so riding the main road at rush hour is a mistake that a side street avoids. The episode ends when the delivery is scanned, the medicine is returned, the box is ruined, the order is cancelled at 55 minutes, or the rider reaches 80 steps.

## Art

The buildings, the people (walking and standing poses), the cars, the bike, the trees and the interior backdrops in `assets/raw/` were generated with an AI image tool for this project. `uv run python -m environment.sprites` cuts them into the transparent sprites in `assets/sprites/` that the window uses. The FARUMASI logo is drawn in code. The environment, the rendering, and the training are written for this project.

## Layout

```text
main.py
play.py
serve.py
docs/ENVIRONMENT.md
service/agent.py
service/server.py
service/export.py
environment/custom_env.py
environment/city_map.py
environment/rendering.py
environment/sprites.py
environment/baselines.py
training/dqn_training.py
training/pg_training.py
training/experiments.py
training/final_training.py
training/evaluation.py
training/report_figures.py
models/dqn/
models/pg/
logs/
assets/
tests/
```

## Video

From this folder, with the camera and the whole screen shared: say that FARUMASI riders have to deliver fast and safely, that rush hour and rain change the best strategy, and that a wrong prescription goes back to the pharmacy. Then run `uv run python play.py --episodes 6 --seed 1000` and leave the window and the terminal visible. `uv run python play.py --heuristic --episodes 6 --seed 1000` shows the hand-written rider on the same city.
