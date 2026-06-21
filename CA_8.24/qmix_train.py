from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from robot_env import RobotEnvironment
from rl_common import concat_state, make_config, obs_dict_to_list, plot_training_curve, set_global_seeds, static_policy_summary, summarize_convergence, write_csv


class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, mixing_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_dim = mixing_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixing_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, mixing_dim), nn.ReLU(), nn.Linear(mixing_dim, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch_size = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.n_agents, self.mixing_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.mixing_dim)
        hidden = torch.relu(torch.bmm(agent_qs.view(batch_size, 1, self.n_agents), w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.mixing_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(batch_size)


class QMixLearner:
    def __init__(self, *, n_agents: int, obs_dim: int, action_dim: int, state_dim: int, device: torch.device, hidden_size: int = 128, mixing_dim: int = 32, lr: float = 1e-3, gamma: float = 1.0, batch_size: int = 64, buffer_size: int = 30000):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory: deque = deque(maxlen=buffer_size)
        self.agent_net = AgentQNetwork(obs_dim, action_dim, hidden_size).to(device)
        self.target_agent_net = AgentQNetwork(obs_dim, action_dim, hidden_size).to(device)
        self.mixer = QMixer(n_agents, state_dim, mixing_dim).to(device)
        self.target_mixer = QMixer(n_agents, state_dim, mixing_dim).to(device)
        self.optimizer = optim.Adam(list(self.agent_net.parameters()) + list(self.mixer.parameters()), lr=lr)
        self.update_targets()

    def update_targets(self) -> None:
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def act(self, obs_list: list[np.ndarray], epsilon: float) -> list[int]:
        actions = []
        for obs in obs_list:
            if random.random() < epsilon:
                actions.append(random.randrange(self.action_dim))
                continue
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                actions.append(int(torch.argmax(self.agent_net(obs_t), dim=1).item()))
        return actions

    def remember(self, obs, state, actions, reward, next_obs, next_state, done) -> None:
        self.memory.append((obs, state, actions, reward, next_obs, next_state, done))

    def train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        obs, states, actions, rewards, next_obs, next_states, dones = zip(*batch)
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(np.array(next_obs), dtype=torch.float32, device=self.device)
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        batch_size = obs_t.shape[0]
        flat_obs = obs_t.view(batch_size * self.n_agents, -1)
        q_values = self.agent_net(flat_obs).view(batch_size, self.n_agents, self.action_dim)
        chosen_qs = q_values.gather(2, actions_t.unsqueeze(-1)).squeeze(-1)
        q_total = self.mixer(chosen_qs, states_t)
        with torch.no_grad():
            flat_next_obs = next_obs_t.view(batch_size * self.n_agents, -1)
            next_q_values = self.target_agent_net(flat_next_obs).view(batch_size, self.n_agents, self.action_dim)
            next_agent_qs = next_q_values.max(dim=2).values
            next_q_total = self.target_mixer(next_agent_qs, next_states_t)
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q_total
        loss = nn.functional.mse_loss(q_total, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.agent_net.parameters()) + list(self.mixer.parameters()), 10.0)
        self.optimizer.step()
        return float(loss.item())


def evaluate_qmix(learner: QMixLearner, args: argparse.Namespace, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        env = RobotEnvironment(make_config(args, seed, "QMIX-2R"))
        observations, _ = env.reset(seed=seed)
        done = False
        while not done:
            actions = learner.act(obs_dict_to_list(env, observations), epsilon=0.0)
            observations, _, dones, _ = env.step(actions)
            done = dones["__all__"]
        row = env.episode_summary()
        row["episode"] = len(rows) + 1
        rows.append(row)
    return rows


def train_qmix(args: argparse.Namespace) -> dict:
    set_global_seeds(args.train_seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    probe_env = RobotEnvironment(make_config(args, args.train_seed, "QMIX-2R"))
    obs_dim = probe_env.observation_space.shape[0]
    state_dim = obs_dim * probe_env.num_agents
    learner = QMixLearner(n_agents=probe_env.num_agents, obs_dim=obs_dim, action_dim=probe_env.action_space.n, state_dim=state_dim, device=device, hidden_size=args.hidden_size, mixing_dim=args.mixing_dim, lr=args.lr, gamma=args.gamma, batch_size=args.batch_size, buffer_size=args.buffer_size)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    epsilon = args.epsilon_start
    train_rows = []
    for episode in range(1, args.episodes + 1):
        seed = args.train_seed + episode
        env = RobotEnvironment(make_config(args, seed, "QMIX-2R"))
        observations, _ = env.reset(seed=seed)
        done = False
        losses = []
        while not done:
            obs_list = obs_dict_to_list(env, observations)
            state = concat_state(obs_list)
            actions = learner.act(obs_list, epsilon)
            next_observations, rewards, dones, _ = env.step(actions)
            next_obs_list = obs_dict_to_list(env, next_observations)
            next_state = concat_state(next_obs_list)
            reward = float(sum(rewards.values()) / max(1, len(rewards)))
            done = dones["__all__"]
            learner.remember(obs_list, state, actions, reward, next_obs_list, next_state, float(done))
            loss = learner.train_step()
            if loss is not None:
                losses.append(loss)
            observations = next_observations
        if episode % args.target_update == 0:
            learner.update_targets()
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        row = env.episode_summary()
        row.update({"episode": episode, "epsilon": epsilon, "loss": float(np.mean(losses)) if losses else None, "device": str(device)})
        train_rows.append(row)
        if episode == 1 or episode % args.log_interval == 0:
            print(f"QMIX episode {episode}/{args.episodes}: t80={row['t80']} return={row['mean_episode_return']} invalid={row['invalid_action_count']} eps={epsilon:.3f}")

    torch.save({"agent_net": learner.agent_net.state_dict(), "mixer": learner.mixer.state_dict(), "config": vars(args)}, model_dir / "qmix_model.pth")
    eval_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds))
    eval_rows = evaluate_qmix(learner, args, eval_seeds)
    static_rows = static_policy_summary(args, eval_seeds)
    stem = f"qmix_{args.scenario}_n{args.num_persons}_ep{args.episodes}"
    write_csv(output_dir / f"{stem}_train.csv", train_rows)
    write_csv(output_dir / f"{stem}_eval.csv", eval_rows)
    write_csv(output_dir / f"{stem}_static_eval.csv", static_rows)
    plot_training_curve(output_dir / f"{stem}_curve.png", train_rows, f"QMIX-2R {args.scenario}")
    train_summary = summarize_convergence(train_rows, window=max(5, min(20, args.episodes // 3)))
    eval_summary = summarize_convergence(eval_rows, window=max(1, len(eval_rows)))
    static_summary = summarize_convergence(static_rows, window=max(1, len(static_rows)))
    report_path = output_dir / f"{stem}_report.md"
    report_path.write_text("\n".join([
        "# QMIX-2R Training Report", "", f"- Scenario: `{args.scenario}`", f"- Episodes: `{args.episodes}`", f"- Device: `{device}`", f"- Train summary: `{train_summary}`", f"- Greedy eval summary: `{eval_summary}`", f"- Static same-start eval summary: `{static_summary}`", "", "This is a short convergence check, not a final paper-scale run.",
    ]), encoding="utf-8")
    return {"train_rows": train_rows, "eval_rows": eval_rows, "static_rows": static_rows, "train_summary": train_summary, "eval_summary": eval_summary, "static_summary": static_summary, "report_path": report_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--num-persons", type=int, default=70)
    parser.add_argument("--scenario", default="Best-Static")
    parser.add_argument("--train-seed", type=int, default=11000)
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
    parser.add_argument("--mixing-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=30000)
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
    train_qmix(build_parser().parse_args())
