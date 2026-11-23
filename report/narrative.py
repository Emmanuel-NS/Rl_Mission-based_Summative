"""Text of the report. Numbers that come from the logs are inserted by build_report.py, not typed here."""

TEXT: dict = {}

TEXT["intro"] = (
    "FARUMASI is a digital pharmacy platform in Kigali that brings medicine to patients' homes, offices and hotels. "
    "A delivery is a chain of decisions: which road to ride in live traffic, whether to cover the box when rain starts, "
    "how to get from the street to the patient, and when it is safe to hand over a prescription medicine. "
    "This project turns one delivery into a reinforcement learning problem and compares a value-based method (DQN) with three "
    "policy-gradient methods (REINFORCE, PPO, A2C), all implemented with TorchRL and trained on the same environment. "
    "The goal of the agent is to complete each delivery inside the 30-minute promise with the medicine dry, and never to hand over "
    "a prescription medicine without a valid prescription."
)

TEXT["env_overview"] = (
    "One episode is one delivery. The rider starts outside Kigali City Pharmacy, collects the order, rides across a 6 by 4 grid of "
    "Kigali junctions (KN 3 Rd is the busy main road), parks, walks in, carries the box to the patient, checks the prescription when one is "
    "required, hands the box over and has the patient scan a QR code. If the prescription is wrong the rider takes the medicine back "
    "to the pharmacy. The environment is a Gymnasium class (environment/custom_env.py) drawn with pygame (environment/rendering.py)."
)

TEXT["why_rl"] = (
    "The problem is sequential and the choices cannot be scored one at a time. Actions change a state that persists (position, box "
    "condition, whether the prescription has been checked). The trade-offs pay off late: covering the box costs a minute now and avoids "
    "0.12 of box condition on every later ride step; checking the prescription costs a minute and avoids a -6 failure; a side street is "
    "longer but avoids a jammed main road worth +3 to +6 minutes. The world is stochastic (rush hour, an incident that blocks a random road "
    "at step 5-9, traffic that drifts every 6 steps, rain in 65% of orders, lasting 6-14 steps), and the rider only partly observes it: a prescription is valid "
    "or not (30% invalid) only after it is checked, and rain is hinted at only two steps ahead. Shortest-path routing alone cannot solve "
    "this. A hand-written rider that combines routing, covering and checking scores well, and the agents have to discover that "
    "behaviour from reward alone."
)

TEXT["actions"] = [
    ("0-3", "ride N, E, S, W", "riding, and a road exists", "drive to the next junction (1 min + 1 busy, 3 heavy, 6 blocked, +1 in rain)"),
    ("4", "cover the box", "riding or at the entrance, uncovered", "fit the rain cover"),
    ("5", "wait (take shelter)", "riding, only while it rains", "pull over under a porch until the rain stops; box stays dry"),
    ("6", "enter pharmacy", "outside the pharmacy", "walk in"),
    ("7", "collect order", "inside the pharmacy", "pharmacist seals the box"),
    ("8", "park and walk in", "riding, at the destination", "park the bike"),
    ("9", "knock / check in", "at the entrance", "gate or reception (2 min)"),
    ("10", "carry to patient", "inside the building", "walk to the patient (2 min)"),
    ("11", "verify prescription", "with patient, required, unchecked", "check name, medicine, date"),
    ("12", "hand over", "with patient, not yet handed over", "give the box"),
    ("13", "scan QR", "after handover", "patient confirms in the FARUMASI app"),
    ("14", "walk back to bike", "wrong prescription found", "start the return trip (2 min)"),
    ("15", "return medicine", "back at the pharmacy", "give it to the pharmacist"),
]

TEXT["obs_intro"] = (
    "Every value is produced by the simulator after each decision and enters the agent as one 94-value vector through the first "
    "layer of the policy network. The last column gives the live source in the FARUMASI app."
)

TEXT["observations"] = [
    ("0-5", "phase", "one-hot, 6 phases", "set by enter, collect, park, knock, carry", "order status in the rider app"),
    ("6-7", "x, y", "column/5, row/3", "changes with each ride action", "GPS snapped to a junction"),
    ("8-11", "offset to patient, to pharmacy", "(target - rider) rescaled", "destination drawn at reset", "geocoded address, pharmacy location"),
    ("12", "clock", "minutes used / 55", "each action adds its minutes", "order-accepted timestamp"),
    ("13-14", "raining, rain soon", "0 or 1", "rain start <= steps < rain end; start within 2 steps", "weather service and nowcast"),
    ("15-16", "box covered, condition", "0 or 1; 0 to 1", "cover action; falls 0.12 per ride step when uncovered in rain (not while sheltering)", "rider toggle; humidity sensor"),
    ("17-20", "prescription: required, checked, known ok, known wrong", "0 or 1 each", "drawn from the medicine; set by verify; hidden until verified", "order system; rider's check"),
    ("21-23", "address kind", "one-hot home, office, hotel", "drawn at reset", "order"),
    ("24-27", "returning, handed over, entrance cleared, steps/80", "0 or 1; 0 to 1", "set by the matching actions", "app state and counter"),
    ("28-31", "medicine", "one-hot, 4 medicines", "drawn at reset", "order"),
    ("32-35", "traffic around the junction", "level/3 on four roads", "traffic table", "live traffic service"),
    ("36-39", "compass bearing", "0 or 1 for N, E, S, W", "riding that way gets closer to the destination", "phone compass and address"),
    ("40-77", "traffic on every road", "level/3 on 38 roads", "random start, drift, incident, rush-hour jam on KN 3 Rd", "live traffic, 4 bands"),
    ("78-93", "legal-action mask", "0 or 1 for 16 actions", "derived from phase and flags", "app business rules"),
]

TEXT["rewards"] = [
    ("every minute used", "-0.03"),
    ("riding closer to the target by traffic-aware time", "+0.12 per minute saved (negative when riding away)"),
    ("order collected / prescription checked / handed over", "+0.2 / +0.3 / +0.5"),
    ("QR scan, delivered", "(5 + up to 2 for being early) x (0.6 + 0.4 x box condition) - 0.12 per minute past 30"),
    ("wrong prescription returned to the pharmacy", "+2.5"),
    ("rain on an uncovered box", "condition falls 0.12 (riding) or 0.07 and the same is subtracted"),
    ("hand over prescription medicine without a valid check", "-6, episode ends"),
    ("box condition at or below 0.3", "-4, episode ends"),
    ("illegal action", "-0.2 and a minute passes"),
    ("order cancelled at 55 min / 80 steps reached", "-3 / -2, episode ends"),
]

TEXT["start_terminal"] = (
    "Start state: the rider is outside the pharmacy (junction 12), box condition 1.0, clock 0. Each reset draws the destination (6 addresses), "
    "the medicine (4, two need a prescription, 30% of those prescriptions are invalid), rush hour (60%), the traffic on all 38 roads, "
    "rain (65% of orders, starting at step 3-10) and an incident at step 5-9. Terminal conditions: delivered (QR scan) and returned "
    "(wrong prescription taken back) are successes; violation (hand-over without a valid check), ruined (box condition at or below 0.3), "
    "cancelled (55 minutes) and abandoned (80 decisions) are failures."
)

TEXT["methods"] = (
    "All four agents interact with the same environment through the same TorchRL wrapper. Illegal actions are masked: the mask is the "
    "last 16 observation values, fed to the e-greedy module and Q-value actor for DQN and to MaskedCategorical for the policy-gradient "
    "methods. Policies are MLPs with two hidden layers (ReLU); the policy-gradient methods use a separate value network. Optimiser: Adam."
)

TEXT["method_rows"] = [
    ("DQN", "Collector, replay buffer, QValueActor, EGreedyModule (masked), DQNLoss, HardUpdate", "learning rate, gamma, epsilon anneal length, replay size, batch, target-update interval, double DQN, loss, network width"),
    ("REINFORCE", "Collector, ProbabilisticActor, MaskedCategorical, ReinforceLoss with a value baseline", "learning rate, gamma, entropy bonus, with or without baseline, rollout length, network width, gradient clip"),
    ("PPO", "ClipPPOLoss with GAE, several epochs of minibatches per rollout", "learning rate, gamma, clip range, entropy, GAE lambda, epochs, minibatch, rollout length, width"),
    ("A2C", "A2CLoss with GAE, one update per rollout", "learning rate, gamma, entropy, GAE lambda, rollout length, gradient clip, width"),
]

TEXT["protocol"] = (
    "Evaluation protocol. Every sweep run has 30,000 environment frames and training seed 0, so settings are compared on equal budgets. "
    "During a run the greedy policy is scored every 5,000 frames on 20 validation deliveries (seeds 1000-1019); the best snapshot is kept. "
    "Because those seeds are used to choose, the number reported for each checkpoint is its score on 100 test deliveries (seeds 5000-5099) "
    "that no run ever touched. Return is the sum of rewards; success means delivered or a wrong prescription taken back."
)

TEXT["hp_intro"] = (
    "Each algorithm has ten different settings, changing the controls that are specific to it (DQN: exploration, replay, target updates; "
    "policy gradients: rollout length, entropy, advantage estimation, clipping). The first setting of each table is the baseline; the "
    "third column shows what changed. Test return and success come from the 100 unseen deliveries, the last column shows how often the "
    "greedy policy chose WAIT."
)

TEXT["system"] = (
    "The project runs after `uv sync` with `uv run python main.py` (main.py), watches an agent with `uv run python play.py` (a 30 fps pygame "
    "window plus a step-by-step terminal log), and serves the trained policy with `uv run python serve.py`: a small JSON API where the "
    "rider app posts the delivery state to POST /act and receives the recommended action, the legal actions and the policy's scores. "
    "The state is encoded by the environment's own encoder, so serving cannot drift from training. `python -m service.export` writes a "
    "TorchScript file and a plain JSON copy of the weights for an on-device version. Tests cover the environment rules, the evaluation, the encoder, the API and the export."
)

TEXT["reproduce"] = (
    "git clone https://github.com/Emmanuel-NS/Rl_Mission-based_Summative.git; uv sync; uv run python main.py --sweep all --device cpu --frames 30000; "
    "uv run python main.py --final --frames 120000 --seeds 2; uv run python main.py --evaluate; uv run python play.py --episodes 6 --seed 1000. "
    "Logs are in logs/, figures in logs/figures/, checkpoints in models/, and the full tables in docs/RESULTS.md."
)
