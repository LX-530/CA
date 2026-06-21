from __future__ import annotations

from typing import Any

import numpy as np

from robot_env import RobotEnvConfig, RobotEnvironment


class QMixRobotEnv:
    """QMIX-facing wrapper around the shared RobotEnvironment.

    The wrapper intentionally delegates all map, pedestrian, exit, friction,
    reward, and termination logic to RobotEnvironment so IDQN and QMIX cannot
    drift into different environments.
    """

    def __init__(self, config: RobotEnvConfig | dict[str, Any] | None = None):
        self.env = RobotEnvironment(config)
        self.n_agents = self.env.num_agents
        self.n_actions = self.env.action_space.n
        self.episode_limit = self.env.config.max_steps_guard

    def reset(self, seed: int | None = None) -> list[np.ndarray]:
        observations, _ = self.env.reset(seed=seed)
        return [observations[agent] for agent in self.env.agents]

    def step(self, actions: list[int] | np.ndarray) -> tuple[float, bool, dict[str, Any]]:
        _, rewards, dones, info = self.env.step(list(actions))
        if rewards:
            reward = float(sum(rewards.values()) / len(rewards))
        else:
            reward = 0.0
        return reward, bool(dones["__all__"]), info

    def get_obs(self) -> list[np.ndarray]:
        observations = self.env._get_observations()
        return [observations[agent] for agent in self.env.agents]

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        return self.get_obs()[agent_id]

    def get_state(self) -> np.ndarray:
        obs = self.get_obs()
        if not obs:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(obs).astype(np.float32)

    def get_avail_actions(self) -> list[list[int]]:
        return [[1] * self.n_actions for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id: int) -> list[int]:
        return self.get_avail_actions()[agent_id]

    def close(self) -> None:
        return None
