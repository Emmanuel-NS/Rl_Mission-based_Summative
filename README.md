# FARUMASI shift dispatcher

Reinforcement learning for one morning shift on FARUMASI, the Kigali pharmacy platform. Patients have already sent medicine requests. The agent decides, one request at a time, which partner pharmacy dispenses it, whether to defer it, or whether to refer the patient to a public facility.

Metformin is scarce: four packs, four urgent requests, and five routine refills. A pack used on a routine refill is not there when a later patient is already rationing a dose. That cost is paid when the urgent request fails, and again as a bonus at the end of the shift. A nearest-pharmacy rule does not carry that consequence. The same environment is used for DQN, REINFORCE, PPO, and A2C.

## Setup

[uv](https://docs.astral.sh/uv/) installs the dependencies and creates the virtual environment.

```powershell
uv sync
uv run python main.py --baselines
uv run python main.py --screenshot
```

`uv run python main.py` with no arguments evaluates `models/best_agent.pt` when that file exists. On a fresh clone it trains the baseline PPO setting for 4096 frames so the project runs after `uv sync`.

## Full training

The comparison used in the report is 10 settings for each algorithm, 20,000 environment frames each, on the same shifts and the same held-out seeds (1000–1019). Open `colab_train.ipynb` on a normal CPU runtime. The pharmacy step is Python and the network is small, so a T4 does not make the sweep faster and the free GPU quota runs out before the files can be downloaded.

```powershell
uv run python main.py --sweep all --device cpu --frames 20000 --eval-every 5000
```

One algorithm, one setting:

```powershell
uv run python main.py --sweep ppo --experiment 1 --device cpu --frames 20000
```

After the sweep, `models/best_agent.pt` is the greedy policy with the highest mean return on the held-out shifts. Curves and tables are written to `logs/figures/` and `logs/experiment_tables.md`. A finished experiment is skipped if you start the sweep again, so a Colab disconnect does not throw away completed runs. Pass `--force` to train that setting again.

## Watch a shift

```powershell
uv run python play.py --episodes 3 --seed 1000
```

This opens the Kigali dispatch window and prints every decision in the terminal: action, reward, running return, medicine, and pharmacy. `--heuristic` plays the hand-written rule on the same window.

## What the agent observes and does

Actions, in order: dispense at Nyamirambo Community, Kigali City Pharmacy, Remera Care, Kimironko Health Mart, Kicukiro Centre, Gikondo Pharmacy, defer the request, refer to a public facility.

`uv run python main.py --describe` prints every observation index, its range, and where the number comes from. Stock, opening hours, travel time, list price, and the legal-action mask are updated on every decision. The mask is the last eight values. Closed pharmacies and empty shelves are not legal dispense actions.

A shift starts with a shuffled morning queue, one pharmacy closed, four metformin packs on the open pharmacies, and one near-expiry metformin pack. A one-pack drop arrives after eight decisions. The shift ends when the queue is empty, or after 72 decisions.

The reward pays for a completed dispense, subtracts travel and a high list price, and adds a bonus for using a near-expiry pack. Deferring an urgent request costs more than deferring a routine one. Referring a patient who could have been served is penalised. The large bonus for urgent metformin filled, and the penalty for a near-expiry pack still on the shelf, are added only when the shift ends.

Each learned decision is also a JSON record (`pharmacy_id`, medicine, urgency, district, reward). That is the payload a FARUMASI partner screen can store or send on.

## Layout

```text
main.py
play.py
environment/custom_env.py
environment/rendering.py
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

From this folder, with the camera and the whole screen shared: say what FARUMASI is dispatching, what the agent is trying to do, and how the reward treats a scarce metformin pack. Then run `uv run python play.py --episodes 3 --seed 1000` and leave the window and the terminal visible for most of the recording. Point at one defer or referral where the agent keeps a pack, and read the urgent-metformin line at the end of the shift.
