from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn_train import QNetwork
from qmix_train import AgentQNetwork
from rl_common import SCENARIOS, obs_dict_to_list
from robot_env import RobotEnvConfig, RobotEnvironment, STAY_ACTION


def make_env(args: argparse.Namespace, scenario: str, method: str, seed: int) -> RobotEnvironment:
    starts = [] if method == "No-Robot" else SCENARIOS[scenario]
    return RobotEnvironment(
        RobotEnvConfig(
            num_persons=args.num_persons,
            robot_start_positions=starts,
            seed=seed,
            eval_seed=seed,
            scenario_id=scenario,
            algorithm=method,
            target_ratio=args.target_ratio,
            termination_ratio=args.termination_ratio,
            exit_service_steps=args.exit_service_steps,
            friction_mu=args.friction_mu,
            robot_repulsion_cutoff=args.robot_repulsion_cutoff,
            robot_repulsion_amplitude=args.robot_repulsion_amplitude,
            robot_friction_beta=args.robot_friction_beta,
            record_step_metrics=False,
        )
    )


def load_idqn_models(model_root: Path, scenario: str, env: RobotEnvironment) -> list[QNetwork]:
    models = []
    for idx in range(env.num_agents):
        model = QNetwork(env.observation_space.shape[0], env.action_space.n)
        state = torch.load(
            model_root / scenario / "idqn" / f"idqn_robot_{idx}.pth",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def load_qmix_model(model_root: Path, scenario: str, env: RobotEnvironment) -> AgentQNetwork:
    model = AgentQNetwork(env.observation_space.shape[0], env.action_space.n)
    checkpoint = torch.load(
        model_root / scenario / "qmix" / "qmix_model.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["agent_net"])
    model.eval()
    return model


def argmax_action(model: torch.nn.Module, obs: np.ndarray) -> int:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(model(obs_t), dim=1).item())


def run_episode(
    args: argparse.Namespace,
    scenario: str,
    method: str,
) -> tuple[RobotEnvironment, np.ndarray, list[list[tuple[int, int]]]]:
    env = make_env(args, scenario, method, args.seed)
    observations, _ = env.reset(seed=args.seed)
    heatmap = np.zeros((env.rows, env.cols), dtype=np.float64)
    trajectories: list[list[tuple[int, int]]] = [
        [tuple(pos)] for pos in env.robot_positions
    ]

    idqn_models = None
    qmix_model = None
    model_root = Path(args.model_dir)
    if method == "IDQN":
        idqn_models = load_idqn_models(model_root, scenario, env)
    elif method == "QMIX":
        qmix_model = load_qmix_model(model_root, scenario, env)

    done = False
    while not done:
        for person in env.persons:
            if not person.escaped:
                heatmap[person.position] += 1.0

        if method == "No-Robot":
            actions: list[int] = []
        elif method == "Static":
            actions = [STAY_ACTION for _ in env.agents]
        elif method == "IDQN":
            assert idqn_models is not None
            actions = [
                argmax_action(idqn_models[idx], observations[agent_id])
                for idx, agent_id in enumerate(env.agents)
            ]
        elif method == "QMIX":
            assert qmix_model is not None
            actions = [
                argmax_action(qmix_model, obs)
                for obs in obs_dict_to_list(env, observations)
            ]
        else:
            raise ValueError(f"unknown method: {method}")

        observations, _, dones, _ = env.step(actions)
        done = dones["__all__"]
        for idx, pos in enumerate(env.robot_positions):
            trajectories[idx].append(tuple(pos))

    return env, heatmap, trajectories


def draw_map_background(ax: plt.Axes, env: RobotEnvironment) -> None:
    base = np.ones((env.rows, env.cols, 3), dtype=np.float64)
    for r, c in env.wall_cells:
        base[r, c] = (0.08, 0.08, 0.08)
    for r, c in env.fire_cells:
        base[r, c] = (0.95, 0.25, 0.18)
    for r, c in env.exit_cells:
        base[r, c] = (0.10, 0.55, 0.20)
    ax.imshow(base, origin="upper", interpolation="nearest")
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def save_heatmap(
    path: Path,
    env: RobotEnvironment,
    heatmap: np.ndarray,
    scenario: str,
    method: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_map_background(ax, env)
    masked = np.ma.masked_where(heatmap <= 0, heatmap)
    im = ax.imshow(masked, origin="upper", cmap="magma", alpha=0.72, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pedestrian cell-visits")
    ax.set_title(f"{scenario} {method} pedestrian heatmap, seed {env.seed_value}")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_trajectory(
    path: Path,
    env: RobotEnvironment,
    trajectories: list[list[tuple[int, int]]],
    scenario: str,
    method: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_map_background(ax, env)
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2"]
    for idx, trajectory in enumerate(trajectories):
        if not trajectory:
            continue
        rows = [pos[0] for pos in trajectory]
        cols = [pos[1] for pos in trajectory]
        color = colors[idx % len(colors)]
        ax.plot(cols, rows, color=color, linewidth=2.2, marker="o", markersize=2.5, label=f"robot {idx + 1}")
        ax.scatter(cols[0], rows[0], color=color, s=80, marker="s", edgecolor="white", linewidth=0.8)
        ax.scatter(cols[-1], rows[-1], color=color, s=90, marker="*", edgecolor="white", linewidth=0.8)
    if trajectories:
        ax.legend(loc="upper left", frameon=True)
    ax.set_title(f"{scenario} {method} robot trajectories, seed {env.seed_value}")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--methods", nargs="+", default=["No-Robot", "Static", "IDQN", "QMIX"])
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--num-persons", type=int, default=70)
    parser.add_argument("--target-ratio", type=float, default=0.8)
    parser.add_argument("--termination-ratio", type=float, default=0.8)
    parser.add_argument("--exit-service-steps", type=int, default=2)
    parser.add_argument("--friction-mu", type=float, default=0.6)
    parser.add_argument("--robot-repulsion-cutoff", type=float, default=6.0)
    parser.add_argument("--robot-repulsion-amplitude", type=float, default=0.35)
    parser.add_argument("--robot-friction-beta", type=float, default=1.2)
    parser.add_argument("--model-dir", default="result/models/final_target_ep1500")
    parser.add_argument("--output-dir", default="result/visual/final_target_ep1500/visuals")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for scenario in args.scenarios:
        for method in args.methods:
            env, heatmap, trajectories = run_episode(args, scenario, method)
            prefix = output_dir / f"{scenario}_{method.lower().replace('-', '_')}_seed{args.seed}"
            save_heatmap(prefix.with_name(prefix.name + "_heatmap.png"), env, heatmap, scenario, method)
            save_trajectory(prefix.with_name(prefix.name + "_trajectory.png"), env, trajectories, scenario, method)
            print(
                f"{scenario} {method}: t80={env.t80} escaped={env.escaped_persons} "
                f"heatmap/trajectory saved",
                flush=True,
            )


if __name__ == "__main__":
    main()
