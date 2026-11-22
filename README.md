# FARUMASI shift dispatcher

Reinforcement learning for one patient searching Kigali for a genuine pack. Licensed pharmacies can be closed, expensive, or out of stock. Unlicensed stalls advertise the same medicine cheaply, and buying there is harmful. FARUMASI is an action the patient can take: open the app and every partner shelf updates live, while every stall is marked unlicensed. The same map is used for DQN, REINFORCE, PPO, and A2C.

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

This opens the Kigali map and prints every step in the terminal. The window advances once a second so the walk can be watched. `--heuristic` plays the hand-written patient, who opens FARUMASI and then crosses toward a licensed shelf.

## What the agent observes and does

Actions, in order: north, east, south, west, open FARUMASI, inspect the building you are standing on, buy.

`uv run python main.py --describe` prints every observation index. Partner stock and price stay hidden until the app is opened or that pharmacy is inspected. Stall positions and their cheap advertised price are visible immediately. The licence is not. The legal-action mask is the last seven values.

A search starts at a corner of the city. The pharmacies in that district are closed or empty. The genuine packs are across Kigali, and on step 14 a nearby partner receives one pack, which FARUMASI shows if the app is open. Three stalls will sell something harmful. Health falls on every step. The search ends when the patient buys a genuine pack, buys from a stall, runs out of health, or reaches 64 steps.

The reward pays most for a nearby affordable licensed pack, less for the public clinic, and a large penalty for a harmful product. Opening FARUMASI costs a little time and reveals the safe shelves.

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

From this folder, with the camera and the whole screen shared: say that the patient has to find a genuine pack, that some stalls are unlicensed, and that FARUMASI shows live stock before the walk. Then run `uv run python play.py --episodes 3 --seed 1000` and leave the map and the terminal visible for most of the recording. The old dispatcher checkpoints do not match this map, so train again before recording the learned agent. `uv run python play.py --heuristic --episodes 3 --seed 1000` shows the hand-written patient on the same map.
