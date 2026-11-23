"""Deep Q-Network training on the shared FARUMASI delivery."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate

from environment.custom_env import MAX_STEPS
from training.common import (
    MetricLog,
    align_frames,
    evaluate,
    make_policy_net,
    save_checkpoint,
    write_summary,
)
from training.torchrl_env import make_env


def _finished_return(batch) -> float | None:
    done = batch.get(("next", "done"), default=None)
    episode_reward = batch.get(("next", "episode_reward"), default=None)
    if done is None or episode_reward is None:
        return None
    finished = episode_reward[done.view(episode_reward.shape[0], -1).any(dim=-1)]
    if finished.numel() == 0:
        return None
    return float(finished.float().mean())


def _snapshot(loss: DQNLoss, actor: QValueActor, net: torch.nn.Module) -> None:
    with loss.value_network_params.to_module(actor):
        weights = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    net.load_state_dict(weights)


def train_dqn(
    config: dict,
    *,
    total_frames: int,
    device: torch.device,
    seed: int,
    log_dir: Path,
    model_path: Path,
    eval_every: int = 5000,
) -> dict:
    torch.manual_seed(seed)
    env = make_env(seed)
    hidden = int(config["hidden"])
    net = make_policy_net(hidden).to(device)
    actor = QValueActor(
        net,
        in_keys=["observation"],
        spec=env.action_spec,
        action_mask_key="action_mask",
    ).to(device)
    explorer = EGreedyModule(
        spec=env.action_spec,
        eps_init=float(config["eps_start"]),
        eps_end=float(config["eps_end"]),
        annealing_num_steps=int(config["eps_anneal"]),
        action_mask_key="action_mask",
        device=device,
    )
    policy = TensorDictSequential(actor, explorer)
    loss = DQNLoss(
        actor,
        loss_function="smooth_l1" if config["loss"] == "smooth_l1" else "l2",
        delay_value=True,
        double_dqn=bool(config["double_dqn"]),
        action_space=env.action_spec,
    )
    loss.make_value_estimator(gamma=float(config["gamma"]))
    target = HardUpdate(loss, value_network_update_interval=int(config["target_every"]))
    optimizer = torch.optim.Adam(loss.parameters(), lr=float(config["lr"]))
    replay = TensorDictReplayBuffer(
        storage=LazyTensorStorage(int(config["buffer"])),
        batch_size=int(config["batch"]),
    )
    frames_per_batch = int(config["frames_per_batch"])
    total_frames = align_frames(total_frames, frames_per_batch)
    collector = Collector(
        env,
        policy,
        backend="direct",
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        env_device="cpu",
        policy_device=device,
        max_frames_per_traj=MAX_STEPS,
        set_truncated=True,
        auto_register_policy_transforms=False,
    )
    log = MetricLog(log_dir)
    best_return = float("-inf")
    best_fill = 0.0
    frame = 0
    next_eval = min(eval_every, total_frames)
    started = time.perf_counter()
    try:
        for batch in collector:
            frame += int(batch.numel())
            replay.extend(batch.reshape(-1))
            explorer.step(int(batch.numel()))
            loss_value = 0.0
            grad_norm = 0.0
            updates = max(1, int(batch.numel()) // 4)
            if len(replay) >= int(config["batch"]):
                for _ in range(updates):
                    sample = replay.sample()
                    if sample.device != device:
                        sample = sample.to(device)
                    optimizer.zero_grad()
                    loss_td = loss(sample)
                    loss_td["loss"].backward()
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(loss.parameters(), float(config["grad_clip"]))
                    )
                    optimizer.step()
                    target.step()
                    loss_value = float(loss_td["loss"].detach())
                collector.update_policy_weights_()
            train_return = _finished_return(batch)
            row = {
                "frame": frame,
                "train_return": "" if train_return is None else f"{train_return:.6f}",
                "loss": f"{loss_value:.6f}",
                "entropy": "",
                "epsilon": f"{float(explorer.eps):.6f}",
                "eval_return": "",
                "eval_success": "",
                "grad_norm": f"{grad_norm:.6f}",
            }
            if frame >= next_eval:
                _snapshot(loss, actor, net)
                scores = evaluate(net, device)
                row["eval_return"] = f"{scores['eval_return']:.6f}"
                row["eval_success"] = f"{scores['eval_success']:.6f}"
                if scores["eval_return"] > best_return:
                    best_return = scores["eval_return"]
                    best_fill = scores["eval_success"]
                    save_checkpoint(
                        model_path,
                        {
                            "algo": "dqn",
                            "hidden": hidden,
                            "state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                            "config": config,
                            "eval_return": best_return,
                        },
                    )
                next_eval += eval_every
            log.write(row)
            print(
                f"dqn exp {config['id']:02d}  frame {frame:6d}  "
                f"eps {float(explorer.eps):.3f}  loss {loss_value:.4f}  "
                f"train {train_return if train_return is not None else float('nan'):.2f}"
                + (
                    f"  eval {row['eval_return']}  success {row['eval_success']}"
                    if row["eval_return"]
                    else ""
                ),
                flush=True,
            )
    finally:
        log.close()
        collector.shutdown()
        env.close()
    if best_return == float("-inf"):
        _snapshot(loss, actor, net)
        scores = evaluate(net, device)
        best_return = scores["eval_return"]
        best_fill = scores["eval_success"]
        save_checkpoint(
            model_path,
            {
                "algo": "dqn",
                "hidden": hidden,
                "state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "config": config,
                "eval_return": best_return,
            },
        )
    summary = {
        "algo": "dqn",
        "experiment": int(config["id"]),
        "config": config,
        "frames": frame,
        "best_eval_return": best_return,
        "best_success": best_fill,
        "seconds": round(time.perf_counter() - started, 2),
        "device": str(device),
    }
    write_summary(log_dir, summary)
    return summary
