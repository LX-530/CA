from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import math
import random
from collections import deque
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from map_loader import MapLoader


Position = tuple[int, int]


ACTION_DELTAS: list[Position] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]
STAY_ACTION = 4


@dataclass(frozen=True)
class PersonState:
    position: Position
    escaped: bool = False


@dataclass
class RobotEnvConfig:
    map_path: str = "map.json"
    target_area: tuple[int, int, int, int] = (3, 32, 2, 7)
    num_persons: int = 70
    robot_start_positions: list[Position] = field(
        default_factory=lambda: [(11, 11), (13, 33)]
    )
    seed: int = 0
    target_ratio: float = 0.8
    termination_ratio: float = 0.8
    exit_service_steps: int = 2
    friction_mu: float = 0.6
    no_progress_limit: int = 500
    max_steps_guard: int = 10000
    robot_repulsion_cutoff: float = 5.0
    robot_repulsion_amplitude: float = 0.25
    robot_friction_beta: float = 1.0
    invalid_action_penalty: float = 0.0
    scenario_id: str = "P0"
    algorithm: str = "unknown"
    train_seed: int | None = None
    eval_seed: int | None = None
    record_step_metrics: bool = True

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None = None) -> "RobotEnvConfig":
        if config is None:
            return cls()
        data = dict(config)
        if "map_file" in data and "map_path" not in data:
            data["map_path"] = data.pop("map_file")
        if "robot_start_positions" in data:
            data["robot_start_positions"] = [
                tuple(pos) for pos in data["robot_start_positions"]
            ]
        if "target_area" in data:
            data["target_area"] = tuple(data["target_area"])
        return cls(**data)


class RobotEnvironment(gym.Env):
    """Shared evacuation environment for Static, IDQN, and QMIX experiments."""

    metadata = {"render_modes": []}

    def __init__(self, config: RobotEnvConfig | dict[str, Any] | None = None):
        self.config = (
            config
            if isinstance(config, RobotEnvConfig)
            else RobotEnvConfig.from_dict(config)
        )
        map_path = Path(self.config.map_path)
        if not map_path.is_absolute():
            map_path = Path(__file__).resolve().parent / map_path
        self.map_path = map_path
        self.map_loader = MapLoader(str(map_path))
        self.rows = self.map_loader.rows
        self.cols = self.map_loader.cols
        self.exit_cells: set[Position] = set(self.map_loader.exits)
        self.fire_cells: set[Position] = set(self.map_loader.fires)
        self.wall_cells: set[Position] = set(self.map_loader.find_positions(1))

        self.num_agents = len(self.config.robot_start_positions)
        self.agents = [f"robot_{i}" for i in range(self.num_agents)]
        self.action_space = spaces.Discrete(len(ACTION_DELTAS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        self.static_distance = self._compute_static_distance()
        self.reset(seed=self.config.seed)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if options:
            if "robot_start_positions" in options:
                self.config.robot_start_positions = [
                    tuple(pos) for pos in options["robot_start_positions"]
                ]
                self.num_agents = len(self.config.robot_start_positions)
                self.agents = [f"robot_{i}" for i in range(self.num_agents)]
            if "algorithm" in options:
                self.config.algorithm = str(options["algorithm"])
            if "scenario_id" in options:
                self.config.scenario_id = str(options["scenario_id"])

        self.seed_value = self.config.seed if seed is None else int(seed)
        self.py_rng = random.Random(self.seed_value)
        self.np_rng = np.random.default_rng(self.seed_value)

        self.current_step = 0
        self.next_exit_available_step = 0
        self.no_progress_steps = 0
        self.failed = False
        self.failure_reason: str | None = None
        self.done = False
        self.invalid_action_count = 0
        self.friction_blocks = 0
        self.normal_conflicts = 0
        self.exit_conflicts = 0
        self.exit_idle_steps = 0
        self.robot_path_lengths = [0 for _ in range(self.num_agents)]
        self.min_robot_distance = math.inf
        self.overlap_steps = 0
        self.episode_return_sum = 0.0

        self.persons = self._initialize_persons()
        self.initial_person_count = len(self.persons)
        self.escaped_persons = 0
        self.target_count = math.ceil(
            self.initial_person_count * self.config.target_ratio
        )
        self.termination_count = math.ceil(
            self.initial_person_count * self.config.termination_ratio
        )
        self.t80: int | None = None
        self.t90: int | None = None
        self.t_all: int | None = None

        self.robot_positions = self._initialize_robots()
        self.initial_robot_positions = list(self.robot_positions)
        self.step_records: list[dict[str, Any]] = []
        self._record_step_metrics(escaped_this_step=0)

        return self._get_observations(), self._build_info()

    def step(
        self,
        actions: dict[str, int] | list[int] | tuple[int, ...] | np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, Any]]:
        if self.done:
            rewards = {agent: 0.0 for agent in self.agents}
            dones = {agent: True for agent in self.agents}
            dones["__all__"] = True
            return self._get_observations(), rewards, dones, self._build_info()

        action_list = self._normalize_actions(actions)
        invalid_this_step = self._move_robots(action_list)
        person_move_count, escaped_this_step = self._move_persons()

        self.current_step += 1
        self._update_target_times()
        self._update_robot_coordination_metrics()

        if person_move_count == 0 and escaped_this_step == 0:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        if self.no_progress_steps >= self.config.no_progress_limit:
            self.failed = True
            self.failure_reason = "deadlock"
            self.done = True
        elif self.current_step >= self.config.max_steps_guard:
            self.failed = True
            self.failure_reason = "max_steps_guard"
            self.done = True
        elif self.escaped_persons >= self.termination_count:
            self.done = True

        if self.escaped_persons == self.initial_person_count and self.t_all is None:
            self.t_all = self.current_step

        reward = -1.0 - self.config.invalid_action_penalty * invalid_this_step
        self.episode_return_sum += reward * max(1, self.num_agents)
        self._record_step_metrics(escaped_this_step=escaped_this_step)

        rewards = {agent: reward for agent in self.agents}
        dones = {agent: self.done for agent in self.agents}
        dones["__all__"] = self.done
        return self._get_observations(), rewards, dones, self._build_info()

    def _is_valid_robot_position(
        self,
        pos: Position,
        occupied_robot_positions: set[Position] | None = None,
    ) -> bool:
        r, c = pos

        if not (0 <= r < self.map_loader.rows and 0 <= c < self.map_loader.cols):
            return False

        if self.map_loader.map_data[r, c] == 1:
            return False

        if pos in self.map_loader.fires:
            return False

        if pos in self.map_loader.exits:
            return False

        person_positions = {
            tuple(p.position)
            for p in getattr(self, "persons", [])
            if not p.escaped
        }
        if pos in person_positions:
            return False

        if occupied_robot_positions and pos in occupied_robot_positions:
            return False

        return True

    def _initialize_persons(self) -> list[PersonState]:
        positions = self._target_area_empty_cells()
        if self.config.num_persons > len(positions):
            raise ValueError(
                f"num_persons={self.config.num_persons} exceeds available "
                f"start cells={len(positions)} in target_area={self.config.target_area}"
            )
        self.py_rng.shuffle(positions)
        return [PersonState(position=pos) for pos in positions[: self.config.num_persons]]

    def _initialize_robots(self) -> list[Position]:
        checked_positions: list[Position] = []
        for pos in self.config.robot_start_positions:
            pos = tuple(pos)
            if not self._is_valid_robot_position(
                pos,
                occupied_robot_positions=set(checked_positions),
            ):
                raise ValueError(f"invalid robot start position: {pos}")
            checked_positions.append(pos)
        return checked_positions.copy()

    def _target_area_empty_cells(self) -> list[Position]:
        r0, r1, c0, c1 = self.config.target_area
        positions: list[Position] = []
        for r in range(r0, r1):
            for c in range(c0, c1):
                if not (0 <= r < self.rows and 0 <= c < self.cols):
                    continue
                if self.map_loader.map_data[r, c] == 0:
                    positions.append((r, c))
        return positions

    def _compute_static_distance(self) -> np.ndarray:
        distance = np.full((self.rows, self.cols), np.inf)
        queue: deque[Position] = deque()
        for exit_cell in self.exit_cells:
            distance[exit_cell] = 0.0
            queue.append(exit_cell)
        while queue:
            r, c = queue.popleft()
            for dr, dc in ACTION_DELTAS:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.map_loader.map_data[nr, nc] in (1, 3):
                    continue
                candidate = distance[r, c] + math.hypot(dr, dc)
                if candidate < distance[nr, nc]:
                    distance[nr, nc] = candidate
                    queue.append((nr, nc))
        return distance

    def _normalize_actions(
        self,
        actions: dict[str, int] | list[int] | tuple[int, ...] | np.ndarray,
    ) -> list[int]:
        if isinstance(actions, dict):
            return [int(actions.get(agent, STAY_ACTION)) for agent in self.agents]
        action_list = [int(action) for action in list(actions)]
        if len(action_list) != self.num_agents:
            raise ValueError(
                f"expected {self.num_agents} robot actions, got {len(action_list)}"
            )
        return action_list

    def _move_robots(self, action_list: list[int]) -> int:
        invalid_count = 0
        new_positions = list(self.robot_positions)
        occupied = set(self.robot_positions)
        for idx, action in enumerate(action_list):
            if not (0 <= action < len(ACTION_DELTAS)):
                invalid_count += 1
                continue
            dr, dc = ACTION_DELTAS[action]
            old_pos = self.robot_positions[idx]
            target = (old_pos[0] + dr, old_pos[1] + dc)
            if target == old_pos:
                continue
            occupied_without_self = set(occupied)
            occupied_without_self.discard(old_pos)
            if self._is_valid_robot_position(target, occupied_without_self):
                new_positions[idx] = target
                occupied.discard(old_pos)
                occupied.add(target)
                self.robot_path_lengths[idx] += 1
            else:
                invalid_count += 1
        self.invalid_action_count += invalid_count
        self.robot_positions = new_positions
        return invalid_count

    def _move_persons(self) -> tuple[int, int]:
        inside_positions = {
            p.position
            for p in self.persons
            if not p.escaped
        }
        proposals: dict[Position, list[int]] = {}
        exit_candidates: list[int] = []
        for pid, person in enumerate(self.persons):
            if person.escaped:
                continue
            target = self._choose_person_target(pid, person.position, inside_positions)
            if target in self.exit_cells:
                exit_candidates.append(pid)
            elif target != person.position:
                proposals.setdefault(target, []).append(pid)

        moved_count = 0
        escaped_this_step = 0
        updated = list(self.persons)

        for target, pids in sorted(proposals.items()):
            if len(pids) == 1:
                pid = pids[0]
                if not updated[pid].escaped:
                    updated[pid] = PersonState(target, escaped=False)
                    moved_count += 1
                continue

            self.normal_conflicts += 1
            mu_eff = self._effective_mu([target])
            if self.py_rng.random() < mu_eff:
                self.friction_blocks += 1
                continue
            winner = self.py_rng.choice(sorted(pids))
            if not updated[winner].escaped:
                updated[winner] = PersonState(target, escaped=False)
                moved_count += 1

        if self.current_step >= self.next_exit_available_step:
            if exit_candidates:
                if len(exit_candidates) > 1:
                    self.exit_conflicts += 1
                    source_cells = [updated[pid].position for pid in exit_candidates]
                    if self.py_rng.random() < self._effective_mu(source_cells):
                        self.friction_blocks += 1
                    else:
                        winner = self.py_rng.choice(sorted(exit_candidates))
                        updated[winner] = PersonState(updated[winner].position, escaped=True)
                        escaped_this_step = 1
                        self.next_exit_available_step = (
                            self.current_step + self.config.exit_service_steps
                        )
                else:
                    winner = exit_candidates[0]
                    updated[winner] = PersonState(updated[winner].position, escaped=True)
                    escaped_this_step = 1
                    self.next_exit_available_step = (
                        self.current_step + self.config.exit_service_steps
                    )
            else:
                self.exit_idle_steps += 1

        self.persons = updated
        self.escaped_persons += escaped_this_step
        self._assert_mass_and_position_invariants()
        return moved_count, escaped_this_step

    def _choose_person_target(
        self,
        pid: int,
        position: Position,
        occupied_positions: set[Position],
    ) -> Position:
        r, c = position
        robot_cells = set(self.robot_positions)
        candidates: list[tuple[float, Position]] = []
        for dr, dc in ACTION_DELTAS:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            target = (nr, nc)
            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue
            if target in self.exit_cells:
                candidates.append((self.static_distance[nr, nc], target))
                continue
            if self.map_loader.map_data[nr, nc] != 0:
                continue
            if target in robot_cells:
                continue
            if target in occupied_positions:
                continue
            score = self.static_distance[nr, nc] + self._robot_repulsion_score(target)
            candidates.append((score, target))
        if not candidates:
            return position
        min_score = min(score for score, _ in candidates)
        best = [cell for score, cell in candidates if abs(score - min_score) < 1e-12]
        return self.py_rng.choice(sorted(best))

    def _robot_repulsion_score(self, cell: Position) -> float:
        if not self.robot_positions:
            return 0.0
        r, c = cell
        score = 0.0
        for rr, rc in self.robot_positions:
            distance = math.hypot(r - rr, c - rc)
            if distance <= self.config.robot_repulsion_cutoff:
                score += (
                    self.config.robot_repulsion_amplitude
                    * (1.0 - distance / self.config.robot_repulsion_cutoff)
                    * self.config.robot_repulsion_cutoff
                )
        return score

    def _robot_influence(self, cell: Position) -> float:
        if not self.robot_positions:
            return 0.0
        r, c = cell
        influence = 0.0
        for rr, rc in self.robot_positions:
            distance = math.hypot(r - rr, c - rc)
            if distance <= self.config.robot_repulsion_cutoff:
                influence += self.config.robot_repulsion_amplitude * (
                    1.0 - distance / self.config.robot_repulsion_cutoff
                )
        return min(influence, self.config.friction_mu)

    def _effective_mu(self, cells: list[Position]) -> float:
        if not cells:
            return self.config.friction_mu
        values = [
            max(
                0.0,
                self.config.friction_mu
                - self.config.robot_friction_beta * self._robot_influence(cell),
            )
            for cell in cells
        ]
        return float(sum(values) / len(values))

    def _update_target_times(self) -> None:
        if self.t80 is None and self.escaped_persons >= self.target_count:
            self.t80 = self.current_step
        n90 = math.ceil(self.initial_person_count * 0.9)
        if self.t90 is None and self.escaped_persons >= n90:
            self.t90 = self.current_step
        if (
            self.t_all is None
            and self.escaped_persons >= self.initial_person_count
        ):
            self.t_all = self.current_step

    def _update_robot_coordination_metrics(self) -> None:
        if self.num_agents < 2:
            return
        d = math.hypot(
            self.robot_positions[0][0] - self.robot_positions[1][0],
            self.robot_positions[0][1] - self.robot_positions[1][1],
        )
        self.min_robot_distance = min(self.min_robot_distance, d)
        if d <= 2.0 * self.config.robot_repulsion_cutoff:
            self.overlap_steps += 1

    def _record_step_metrics(self, escaped_this_step: int) -> None:
        if not self.config.record_step_metrics:
            return
        row: dict[str, Any] = {
            "step": self.current_step,
            "escaped_persons": self.escaped_persons,
            "remaining_persons": self.initial_person_count - self.escaped_persons,
            "escaped_this_step": escaped_this_step,
        }
        for idx, pos in enumerate(self.robot_positions):
            row[f"robot{idx + 1}_row"] = pos[0]
            row[f"robot{idx + 1}_col"] = pos[1]
        self.step_records.append(row)

    def _assert_mass_and_position_invariants(self) -> None:
        inside = [p.position for p in self.persons if not p.escaped]
        if len(inside) != len(set(inside)):
            raise RuntimeError("two pedestrians occupy the same cell")
        for pos in inside:
            r, c = pos
            if self.map_loader.map_data[r, c] in (1, 2, 3):
                raise RuntimeError(f"pedestrian entered invalid cell: {pos}")
        if len(inside) + self.escaped_persons != self.initial_person_count:
            raise RuntimeError("pedestrian mass balance is broken")

    def _get_observations(self) -> dict[str, np.ndarray]:
        return {
            agent: self._get_obs(idx)
            for idx, agent in enumerate(self.agents)
        }

    def _get_obs(self, robot_idx: int) -> np.ndarray:
        if robot_idx >= len(self.robot_positions):
            return np.zeros(10, dtype=np.float32)
        r, c = self.robot_positions[robot_idx]
        local_density = 0
        for person in self.persons:
            if person.escaped:
                continue
            pr, pc = person.position
            if abs(pr - r) <= 2 and abs(pc - c) <= 2:
                local_density += 1
        nearest_exit = min(
            math.hypot(r - er, c - ec)
            for er, ec in self.exit_cells
        )
        escaped_ratio = (
            self.escaped_persons / self.initial_person_count
            if self.initial_person_count
            else 1.0
        )
        return np.array(
            [
                r / max(1, self.rows - 1),
                c / max(1, self.cols - 1),
                local_density / 25.0,
                escaped_ratio,
                min(1.0, self.current_step / max(1, self.config.max_steps_guard)),
                nearest_exit / math.hypot(self.rows, self.cols),
                self.invalid_action_count / max(1, self.current_step + 1),
                self.friction_blocks / max(1, self.current_step + 1),
                self.exit_conflicts / max(1, self.current_step + 1),
                self.no_progress_steps / max(1, self.config.no_progress_limit),
            ],
            dtype=np.float32,
        )

    def _build_info(self) -> dict[str, Any]:
        success_80 = self.t80 is not None and not self.failed
        return {
            "algorithm": self.config.algorithm,
            "train_seed": self.config.train_seed,
            "eval_seed": self.config.eval_seed if self.config.eval_seed is not None else self.seed_value,
            "scenario_id": self.config.scenario_id,
            "num_persons": self.initial_person_count,
            "robot_start_positions": list(self.initial_robot_positions),
            "robot_positions": list(self.robot_positions),
            "target_ratio": self.config.target_ratio,
            "termination_ratio": self.config.termination_ratio,
            "target_count": self.target_count,
            "escaped_persons": self.escaped_persons,
            "final_escaped": self.escaped_persons,
            "t80": self.t80 if success_80 else None,
            "t90": self.t90 if self.t90 is not None and not self.failed else None,
            "t_all": self.t_all if self.t_all is not None and not self.failed else None,
            "success_80": success_80,
            "deadlock": self.failure_reason == "deadlock",
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "invalid_action_count": self.invalid_action_count,
            "normal_conflicts": self.normal_conflicts,
            "exit_conflicts": self.exit_conflicts,
            "friction_blocks": self.friction_blocks,
            "exit_idle_steps": self.exit_idle_steps,
            "robot_path_length": sum(self.robot_path_lengths),
            "robot_path_lengths": list(self.robot_path_lengths),
            "robot_overlap_rate": self._robot_overlap_rate(),
            "min_robot_distance": (
                None if math.isinf(self.min_robot_distance) else self.min_robot_distance
            ),
            "episode_return_sum": self.episode_return_sum,
            "mean_episode_return": self._mean_episode_return(),
            "current_step": self.current_step,
        }

    def _robot_overlap_rate(self) -> float:
        if self.num_agents < 2 or self.current_step <= 0:
            return 0.0
        return self.overlap_steps / self.current_step

    def _mean_episode_return(self) -> float:
        if self.num_agents == 0:
            return 0.0
        return self.episode_return_sum / self.num_agents

    def episode_summary(self) -> dict[str, Any]:
        info = self._build_info()
        robot1 = self.initial_robot_positions[0] if self.num_agents >= 1 else None
        robot2 = self.initial_robot_positions[1] if self.num_agents >= 2 else None
        return {
            "algorithm": info["algorithm"],
            "train_seed": info["train_seed"],
            "eval_seed": info["eval_seed"],
            "scenario_id": info["scenario_id"],
            "num_persons": info["num_persons"],
            "robot1_start": robot1,
            "robot2_start": robot2,
            "t80": info["t80"],
            "t90": info["t90"],
            "t_all": info["t_all"],
            "success_80": info["success_80"],
            "deadlock": info["deadlock"],
            "invalid_action_count": info["invalid_action_count"],
            "final_escaped": info["final_escaped"],
            "robot_path_length": info["robot_path_length"],
            "robot_overlap_rate": info["robot_overlap_rate"],
            "min_robot_distance": info["min_robot_distance"],
            "mean_episode_return": info["mean_episode_return"],
            "failed": info["failed"],
            "failure_reason": info["failure_reason"],
        }

    def write_episode_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = self.episode_summary()
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    def write_step_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.step_records:
            return
        fieldnames = sorted({key for row in self.step_records for key in row})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.step_records)

    def save_config_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(self.config.__dict__)
        data["robot_start_positions"] = [list(pos) for pos in self.config.robot_start_positions]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class EvacuationEnv(RobotEnvironment):
    """Backward-compatible name used by the old DQN scripts."""

    pass


def evacuation_env_creator(env_config: dict[str, Any]) -> RobotEnvironment:
    return RobotEnvironment(env_config)


try:
    from ray.tune.registry import register_env

    register_env("evacuation_env", evacuation_env_creator)
except Exception:
    pass
