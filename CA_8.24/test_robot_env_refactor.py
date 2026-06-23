import unittest

from robot_env import RobotEnvConfig, RobotEnvironment, STAY_ACTION
from qmix_env import QMixRobotEnv


BASE_CONFIG = {
    "num_persons": 70,
    "target_area": (3, 32, 2, 7),
    "robot_start_positions": [(18, 30), (15, 32)],
    "seed": 20001,
    "target_ratio": 0.8,
    "termination_ratio": 0.8,
    "friction_mu": 0.6,
    "exit_service_steps": 2,
}


class RobotEnvironmentRefactorTests(unittest.TestCase):
    def make_env(self, **overrides):
        config = dict(BASE_CONFIG)
        config.update(overrides)
        return RobotEnvironment(RobotEnvConfig.from_dict(config))

    def test_same_seed_reset_reproduces_initial_state(self):
        env = self.make_env()
        env.reset(seed=12345)
        persons_a = [p.position for p in env.persons]
        robots_a = list(env.robot_positions)
        env.reset(seed=12345)
        persons_b = [p.position for p in env.persons]
        robots_b = list(env.robot_positions)
        self.assertEqual(persons_a, persons_b)
        self.assertEqual(robots_a, robots_b)

    def test_static_idqn_qmix_share_same_initial_state(self):
        seed = 20002
        static_env = self.make_env(seed=seed, algorithm="Static-2R")
        idqn_env = self.make_env(seed=seed, algorithm="IDQN-2R")
        qmix_env = QMixRobotEnv(RobotEnvConfig.from_dict({**BASE_CONFIG, "seed": seed, "algorithm": "QMIX-2R"}))

        self.assertEqual(
            [p.position for p in static_env.persons],
            [p.position for p in idqn_env.persons],
        )
        self.assertEqual(
            [p.position for p in static_env.persons],
            [p.position for p in qmix_env.env.persons],
        )
        self.assertEqual(static_env.robot_positions, idqn_env.robot_positions)
        self.assertEqual(static_env.robot_positions, qmix_env.env.robot_positions)

    def test_invalid_robot_start_positions_raise(self):
        invalid_cases = [
            [(0, 0), (18, 30)],
            [(15, 15), (18, 30)],
            [(16, 36), (18, 30)],
            [(18, 30), (18, 30)],
        ]
        for positions in invalid_cases:
            with self.subTest(positions=positions):
                with self.assertRaises(ValueError):
                    self.make_env(robot_start_positions=positions)

    def test_robot_entering_pedestrian_cell_is_invalid(self):
        env = self.make_env(robot_start_positions=[(9, 7), (18, 30)], seed=7)
        env.persons[0] = type(env.persons[0])((9, 6), escaped=False)
        old_position = env.robot_positions[0]
        env.step([3, STAY_ACTION])
        self.assertEqual(env.robot_positions[0], old_position)
        self.assertGreater(env.invalid_action_count, 0)

    def test_t80_records_when_56th_person_escapes_for_70_people(self):
        env = self.make_env()
        self.assertEqual(env.target_count, 56)
        env.escaped_persons = 55
        env.current_step = 17
        env._update_target_times()
        self.assertIsNone(env.t80)
        env.escaped_persons = 56
        env.current_step = 18
        env._update_target_times()
        self.assertEqual(env.t80, 18)

    def test_time_reward_mean_return_equals_negative_t80(self):
        env = self.make_env(
            robot_start_positions=[(15, 32), (18, 32)],
            seed=20003,
            target_ratio=0.8,
            termination_ratio=0.8,
        )
        done = False
        while not done:
            _, _, dones, _ = env.step([STAY_ACTION, STAY_ACTION])
            done = dones["__all__"]
        self.assertTrue(env.t80 is not None)
        self.assertEqual(env.current_step, env.t80)
        self.assertEqual(env.episode_summary()["mean_episode_return"], -env.t80)

    def test_stage0_full_evacuation_keeps_mass_balance(self):
        env = self.make_env(
            robot_start_positions=[(15, 32), (18, 32)],
            seed=20004,
            target_ratio=0.8,
            termination_ratio=1.0,
        )
        done = False
        while not done:
            _, _, dones, _ = env.step([STAY_ACTION, STAY_ACTION])
            inside = sum(1 for p in env.persons if not p.escaped)
            self.assertEqual(inside + env.escaped_persons, env.initial_person_count)
            done = dones["__all__"]
        self.assertEqual(env.escaped_persons, env.initial_person_count)
        self.assertIsNotNone(env.t80)
        self.assertIsNotNone(env.t_all)

    def test_deadlock_guard_marks_failure_without_t80(self):
        env = self.make_env(no_progress_limit=1, termination_ratio=1.0)
        env._choose_person_target = lambda pid, position, occupied: position
        _, _, dones, info = env.step([STAY_ACTION, STAY_ACTION])
        self.assertTrue(dones["__all__"])
        self.assertTrue(info["failed"])
        self.assertTrue(info["deadlock"])
        self.assertIsNone(info["t80"])
        self.assertFalse(info["success_80"])


if __name__ == "__main__":
    unittest.main()
