"""Export the trained policy so it can run without this training code.

    uv run python -m service.export

Writes two files next to the checkpoint:
    best_agent.torchscript.pt   TorchScript module for a server or an Android/iOS PyTorch runtime
    best_agent_policy.json      the weights as plain JSON (three dense layers with ReLU),
                                small enough for an in-app JavaScript or Kotlin implementation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from environment.custom_env import OBS_DIM
from training.common import load_checkpoint


def export(model: str = "models/best_agent.pt") -> tuple[Path, Path]:
    net, payload = load_checkpoint(model, torch.device("cpu"))
    folder = Path(model).parent
    scripted = torch.jit.trace(net, torch.zeros(1, OBS_DIM))
    script_path = folder / "best_agent.torchscript.pt"
    scripted.save(str(script_path))
    layers = []
    linear = [module for module in net.modules() if isinstance(module, torch.nn.Linear)]
    for index, layer in enumerate(linear):
        layers.append(
            {
                "weight": layer.weight.detach().tolist(),
                "bias": layer.bias.detach().tolist(),
                "activation": "relu" if index < len(linear) - 1 else "none",
            }
        )
    json_path = folder / "best_agent_policy.json"
    json_path.write_text(
        json.dumps(
            {
                "algo": payload.get("algo"),
                "observation_size": OBS_DIM,
                "output": "one score per action; set the scores of illegal actions to -inf, then take the arg max",
                "mask": "the last 16 observation values are the legal-action mask",
                "layers": layers,
            }
        ),
        encoding="utf-8",
    )
    return script_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the trained rider policy")
    parser.add_argument("--model", default="models/best_agent.pt")
    args = parser.parse_args()
    for path in export(args.model):
        print(path)


if __name__ == "__main__":
    main()
