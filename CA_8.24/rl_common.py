from __future__ import annotations

from collections import deque
from pathlib import Path
import csv
import random
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_env import RobotEnvConfig, RobotEnvironment, STAY_ACTION


SCENARIOS = {
    "P0": [(11, 11), (13, 33)],
    "P1": [(9, 8), (23, 8)],
    "P2": [(9, 22), (23, 22)],
    "P3": [(13, 29), (20, 29)],
    "P4": [(4, 10), (28, 10)],
    "Best-Static": [(15, 32), (18, 32)],
}


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(args: Any, seed: int, algorithm: str) -> RobotEnvConfig:
    return RobotEnvConfig(
        map_path=getattr(args, "map_path", "map.json"),
        target_area=tuple(getattr(args, "target_area", (3, 32, 2, 7))),
        num_persons=int(args.num_persons),
        robot_start_positions=SCENARIOS[args.scenario],
        seed=int(seed),
        train_seed=getattr(args, "train_seed", None),
        eval_seed=int(seed),
        algorithm=algorithm,
        scenario_id=args.scenario,
        target_ratio=float(args.target_ratio),
        termination_ratio=float(args.termination_ratio),
        exit_service_steps=int(args.exit_service_steps),
        friction_mu=float(args.friction_mu),
        no_progress_limit=int(args.no_progress_limit),
        max_steps_guard=int(args.max_steps_guard),
        robot_repulsion_cutoff=float(args.robot_repulsion_cutoff),
        robot_repulsion_amplitude=float(args.robot_repulsion_amplitude),
        robot_friction_beta=float(args.robot_friction_beta),
        record_step_metrics=False,
    )


def obs_dict_to_list(env: RobotEnvironment, obs: dict[str, np.ndarray]) -> list[np.ndarray]:
    return [obs[agent] for agent in env.agents]


def concat_state(obs_list: list[np.ndarray]) -> np.ndarray:
    if not obs_list:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(obs_list).astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: list[float], window: int = 10) -> list[float]:
    if not values:
        return []
    out = []
    q: deque[float] = deque()
    total = 0.0
    for value in values:
        q.append(float(value))
        total += float(value)
        if len(q) > window:
            total -= q.popleft()
        out.append(total / len(q))
    return out


def plot_training_curve(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        plt.rcParams.update({
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No RL fine-tuning episodes", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title)
        fig.savefig(path)
        plt.close(fig)
        return
    episodes = [int(row["episode"]) for row in rows]
    t80 = [float(row["t80"]) if row.get("t80") not in (None, "") else np.nan for row in rows]
    invalid = [float(row.get("invalid_action_count", 0)) for row in rows]

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(episodes, t80, color="#4E79A7", alpha=0.45, label="T80")
    axes[0].plot(episodes, moving_average(t80, 10), color="#4E79A7", linewidth=2, label="T80 MA10")
    axes[0].set_ylabel("T80 steps")
    axes[0].legend()
    axes[0].set_title(title)
    axes[1].plot(episodes, invalid, color="#E15759", alpha=0.65)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Invalid actions")
    fig.savefig(path)
    plt.close(fig)


def summarize_convergence(rows: list[dict[str, Any]], window: int = 20) -> dict[str, Any]:
    successful = [row for row in rows if row.get("success_80")]
    if not successful:
        return {
            "episodes": len(rows),
            "success_80_rate": 0.0,
            "first_window_mean_t80": None,
            "last_window_mean_t80": None,
            "delta_t80_last_minus_first": None,
        }
    t80_values = [float(row["t80"]) for row in rows if row.get("t80") not in (None, "")]
    first = t80_values[:window]
    last = t80_values[-window:]
    first_mean = float(np.mean(first)) if first else None
    last_mean = float(np.mean(last)) if last else None
    return {
        "episodes": len(rows),
        "success_80_rate": len(successful) / max(1, len(rows)),
        "first_window_mean_t80": first_mean,
        "last_window_mean_t80": last_mean,
        "delta_t80_last_minus_first": None if first_mean is None or last_mean is None else last_mean - first_mean,
    }


def static_policy_summary(args: Any, seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        env = RobotEnvironment(make_config(args, seed, "Static-2R"))
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            _, _, dones, _ = env.step([STAY_ACTION for _ in env.agents])
            done = dones["__all__"]
        row = env.episode_summary()
        row["episode"] = len(rows) + 1
        rows.append(row)
    return rows
