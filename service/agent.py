"""Use a trained rider policy from outside the training code.

The FARUMASI rider app (or a dispatcher dashboard) would describe the delivery
as a small JSON "state". This module turns that state into the same 94-value
observation the agent was trained on, masks the actions that are not legal,
and returns the action the policy recommends with its scores.

    state = {"phase": "riding", "node": 14, "destination": 4, "medicine": 0, ...}
    RiderAgent.load("models/best_agent.pt").recommend(state)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from environment.custom_env import (
    ACTION_LABELS,
    ACTION_NAMES,
    DESTINATIONS,
    MAX_STEPS,
    N_EDGES,
    OBS_DIM,
    PHASE_NAMES,
    FarumasiRiderEnv,
)
from training.common import load_checkpoint

STATE_FIELDS = {
    "phase": f"one of {list(PHASE_NAMES)} or its index 0-5",
    "node": "junction index 0-23 (row * 6 + column); the pharmacy is 12",
    "destination": f"0-{len(DESTINATIONS) - 1}: " + ", ".join(f"{i}={d.name}" for i, d in enumerate(DESTINATIONS)),
    "medicine": "0 Metformin, 1 Amoxicillin, 2 ORS, 3 Paracetamol",
    "minutes": "minutes used since the order was accepted",
    "steps": "decisions taken so far",
    "raining": "true when rain has started and not yet stopped",
    "rain_soon": "true when dark clouds say rain is within two decisions",
    "box_covered": "true when the rain cover is on the box",
    "box_condition": "0 to 1, 1 is a dry box",
    "rx_required": "true when the medicine needs a prescription",
    "rx_checked": "true after the rider has checked the prescription",
    "rx_valid": "the result of the check (only used once rx_checked is true)",
    "returning": "true when carrying a wrong prescription back",
    "handed_over": "true when the box is in the patient's hands",
    "entrance_cleared": "true when the gate is open or reception has checked the rider in",
    "traffic": f"list of {N_EDGES} road levels, 0 clear, 1 busy, 2 heavy, 3 blocked",
}


def state_to_observation(state: dict) -> np.ndarray:
    """Encode a JSON state with the environment's own encoder, so training and serving cannot drift apart."""
    env = FarumasiRiderEnv()
    env.reset(seed=0)
    phase = state.get("phase", 0)
    env.phase = PHASE_NAMES.index(phase) if isinstance(phase, str) else int(phase)
    env.node = env.prev_node = int(state.get("node", env.node))
    env.dest = DESTINATIONS[int(state.get("destination", 0))]
    env.med = int(state.get("medicine", 0))
    env.minutes = float(state.get("minutes", 0.0))
    env.steps = int(state.get("steps", 0))
    if state.get("raining"):
        env.rain_at = 0
        env.rain_end = 10**6
    elif state.get("rain_soon"):
        env.rain_at = env.steps + 1
        env.rain_end = 10**6
    else:
        env.rain_at = env.rain_end = 10**6
    env.covered = bool(state.get("box_covered", False))
    env.condition = float(state.get("box_condition", 1.0))
    env.rx_required = bool(state.get("rx_required", False))
    env.rx_checked = bool(state.get("rx_checked", False))
    env.rx_valid = bool(state.get("rx_valid", True))
    env.returning = bool(state.get("returning", False))
    env.handed = bool(state.get("handed_over", False))
    env.cleared = bool(state.get("entrance_cleared", False))
    traffic = state.get("traffic")
    if traffic is not None:
        if len(traffic) != N_EDGES:
            raise ValueError(f"traffic needs {N_EDGES} values, got {len(traffic)}")
        env.traffic = np.asarray(traffic, dtype=np.int32)
    env.steps = min(env.steps, MAX_STEPS)
    observation = env._observe()
    env.close()
    return observation


class RiderAgent:
    """A trained policy that answers: what should the rider do now?"""

    def __init__(self, net: torch.nn.Module, meta: dict) -> None:
        self.net = net
        self.meta = meta

    @classmethod
    def load(cls, path: str | Path = "models/best_agent.pt") -> "RiderAgent":
        net, payload = load_checkpoint(path, torch.device("cpu"))
        meta = {"algo": payload.get("algo"), "experiment": payload.get("config", {}).get("id"), "config": payload.get("config", {})}
        return cls(net, meta)

    @torch.no_grad()
    def recommend(self, state: dict | None = None, observation=None) -> dict:
        if observation is None:
            observation = state_to_observation(state or {})
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (OBS_DIM,):
            raise ValueError(f"observation needs {OBS_DIM} values, got {observation.shape}")
        logits = self.net(torch.as_tensor(observation).unsqueeze(0)).squeeze(0)
        legal = observation[-len(ACTION_NAMES):] > 0.5
        masked = logits.masked_fill(~torch.as_tensor(legal), -1e9)
        probabilities = torch.softmax(masked, dim=0)
        action = int(torch.argmax(masked))
        return {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "label": ACTION_LABELS[action].title(),
            "legal_actions": [ACTION_NAMES[i] for i in np.flatnonzero(legal)],
            "scores": {ACTION_NAMES[i]: round(float(probabilities[i]), 4) for i in np.flatnonzero(legal)},
            "policy": self.meta,
        }
