from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from robot_env import RobotEnvironment, STAY_ACTION
from rl_common import (
    make_config,
    plot_training_curve,
    set_global_seeds,
    static_policy_summary,
    summarize_convergence,
    write_csv,
)


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        *,
        device: torch.device,
        hidden_size: int = 128,
        lr: float = 1e-3,
        gamma: float = 1.0,
        batch_size: int = 64,
        buffer_size: int = 20000,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.memory: deque = deque(maxlen=buffer_size)
        self.model = QNetwork(state_size, action_size, hidden_size).to(device)
        self.target_model = QNetwork(state_size, action_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target_network()

    def update_target_network(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.model(state_t), dim=1).item())

    def replay(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_states_t).max(dim=1).values
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q
        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())


def evaluate_idqn(agents: list[DQNAgent], args: argparse.Namespace, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        env = RobotEnvironment(make_config(args, seed, "IDQN-2R"))
        observations, _ = env.reset(seed=seed)
        done = False
        total_actions = 0
        stay_actions = 0
        per_agent_total = [0 for _ in env.agents]
        per_agent_stay = [0 for _ in env.agents]
        while not done:
            actions = {}
            for idx, agent_id in enumerate(env.agents):
                actions[agent_id] = agents[idx].act(observations[agent_id], epsilon=0.0)
                total_actions += 1
                per_agent_total[idx] += 1
                if actions[agent_id] == STAY_ACTION:
                    stay_actions += 1
                    per_agent_stay[idx] += 1
            observations, _, dones, _ = env.step(actions)
            done = dones["__all__"]
        row = env.episode_summary()
        row["total_action_count"] = total_actions
        row["stay_action_count"] = stay_actions
        row["stay_action_rate"] = stay_actions / max(1, total_actions)
        row["move_action_rate"] = 1.0 - row["stay_action_rate"]
        for idx, pos in enumerate(env.robot_positions):
            row[f"robot{idx + 1}_final"] = pos
            row[f"robot{idx + 1}_stay_action_rate"] = (
                per_agent_stay[idx] / max(1, per_agent_total[idx])
            )
        row["static_like_policy"] = (
            row["robot_path_length"] == 0 and row["stay_action_rate"] >= 0.99
        )
        row["episode"] = len(rows) + 1
        rows.append(row)
    return rows


def train_idqn(args: argparse.Namespace) -> dict:
    set_global_seeds(args.train_seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    probe_env = RobotEnvironment(make_config(args, args.train_seed, "IDQN-2R"))
    agents = [
        DQNAgent(
            probe_env.observation_space.shape[0],
            probe_env.action_space.n,
            device=device,
            hidden_size=args.hidden_size,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
        )
        for _ in probe_env.agents
    ]

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    epsilon = args.epsilon_start
    train_rows = []
    for episode in range(1, args.episodes + 1):
        seed = args.train_seed + episode
        env = RobotEnvironment(make_config(args, seed, "IDQN-2R"))
        observations, _ = env.reset(seed=seed)
        done = False
        losses = []
        while not done:
            actions = {}
            for idx, agent_id in enumerate(env.agents):
                actions[agent_id] = agents[idx].act(observations[agent_id], epsilon)
            next_observations, rewards, dones, _ = env.step(actions)
            done = dones["__all__"]
            for idx, agent_id in enumerate(env.agents):
                agents[idx].remember(
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_observations[agent_id],
                    float(done),
                )
                loss = agents[idx].replay()
                if loss is not None:
                    losses.append(loss)
            observations = next_observations

        if episode % args.target_update == 0:
            for agent in agents:
                agent.update_target_network()
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)

        row = env.episode_summary()
        row.update({
            "episode": episode,
            "epsilon": epsilon,
            "loss": float(np.mean(losses)) if losses else None,
            "device": str(device),
        })
        train_rows.append(row)
        if episode == 1 or episode % args.log_interval == 0:
            print(
                f"IDQN episode {episode}/{args.episodes}: "
                f"t80={row['t80']} return={row['mean_episode_return']} "
                f"invalid={row['invalid_action_count']} eps={epsilon:.3f}"
            )

    for idx, agent in enumerate(agents):
        torch.save(agent.model.state_dict(), model_dir / f"idqn_robot_{idx}.pth")

    eval_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds))
    eval_rows = evaluate_idqn(agents, args, eval_seeds)
    static_rows = static_policy_summary(args, eval_seeds)

    stem = f"idqn_{args.scenario}_n{args.num_persons}_ep{args.episodes}"
    train_path = output_dir / f"{stem}_train.csv"
    eval_path = output_dir / f"{stem}_eval.csv"
    static_path = output_dir / f"{stem}_static_eval.csv"
    write_csv(train_path, train_rows)
    write_csv(eval_path, eval_rows)
    write_csv(static_path, static_rows)
    plot_training_curve(output_dir / f"{stem}_curve.png", train_rows, f"IDQN-2R {args.scenario}")

    train_summary = summarize_convergence(train_rows, window=max(5, min(20, args.episodes // 3)))
    eval_summary = summarize_convergence(eval_rows, window=max(1, len(eval_rows)))
    static_summary = summarize_convergence(static_rows, window=max(1, len(static_rows)))
    report_path = output_dir / f"{stem}_report.md"
    report_path.write_text(
        "\n".join([
            "# IDQN-2R Training Report",
            "",
            f"- Scenario: `{args.scenario}`",
            f"- Episodes: `{args.episodes}`",
            f"- Device: `{device}`",
            f"- Train summary: `{train_summary}`",
            f"- Greedy eval summary: `{eval_summary}`",
            f"- Static same-start eval summary: `{static_summary}`",
            "",
            "This is a short convergence check, not a final paper-scale run.",
        ]),
        encoding="utf-8",
    )
    return {
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "static_rows": static_rows,
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "static_summary": static_summary,
        "report_path": report_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--num-persons", type=int, default=70)
    parser.add_argument("--scenario", default="Best-Static")
    parser.add_argument("--train-seed", type=int, default=10000)
    parser.add_argument("--eval-seed-start", type=int, default=20000)
    parser.add_argument("--eval-seeds", type=int, default=10)
    parser.add_argument("--target-ratio", type=float, default=0.8)
    parser.add_argument("--termination-ratio", type=float, default=0.8)
    parser.add_argument("--exit-service-steps", type=int, default=2)
    parser.add_argument("--friction-mu", type=float, default=0.6)
    parser.add_argument("--no-progress-limit", type=int, default=500)
    parser.add_argument("--max-steps-guard", type=int, default=10000)
    parser.add_argument("--robot-repulsion-cutoff", type=float, default=5.0)
    parser.add_argument("--robot-repulsion-amplitude", type=float, default=0.25)
    parser.add_argument("--robot-friction-beta", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.97)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-dir", default="result/visual")
    parser.add_argument("--model-dir", default="result/models")
    parser.add_argument("--map-path", default="map.json")
    parser.add_argument("--target-area", nargs=4, type=int, default=(3, 32, 2, 7))
    return parser


if __name__ == "__main__":
    train_idqn(build_parser().parse_args())
