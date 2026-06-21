# Robot Environment Refactor Validation

## Scope

This stage refactors the evacuation environment for fair Static-2R, IDQN-2R, and QMIX-2R comparison.

Implemented files:

- `robot_env.py`
- `environment.py`
- `qmix_env.py`
- `static_baseline.py`
- `test_robot_env_refactor.py`

## Key Checks

Commands run locally from `CA_8.24`:

```text
python -m unittest test_robot_env_refactor.py
python static_baseline.py --scenario Best-Static --num-persons 70 --seed-start 20000 --seeds 3 --termination-ratio 0.8
python -m py_compile robot_env.py environment.py qmix_env.py static_baseline.py test_robot_env_refactor.py dqn_train.py evaluate_dqn.py main.py
```

Results:

```text
Ran 8 tests in 0.481s
OK

wrote 3 episode rows to result\visual\static_baseline_Best-Static_n70_episodes.csv
```

Compatibility smoke test:

```text
env_agents ['robot_0', 'robot_1']
obs_keys ['robot_0', 'robot_1']
qmix_obs 2 reward -1.0 done False
```

## Verified Requirements

- Same seed reset gives identical pedestrian and robot initial positions.
- Static, IDQN, and QMIX wrappers share the same initial state for the same seed.
- Robot starts on walls, fire cells, exits, or duplicate robot cells raise `ValueError`.
- Robot movement into a pedestrian cell is invalid and leaves the robot in place.
- For 70 pedestrians, the 56th escaped pedestrian records `T80`.
- Mean per-agent episode return equals `-T80` under 80 percent termination.
- Stage-0 full-evacuation mode preserves pedestrian mass balance and records `T_all`.
- Deadlock protection marks failure and does not report a fake `T80`.

## Notes

The local folder is still not a git checkout, so uploads are performed through the GitHub App API. The GitHub branch is `codex/stage0-robot-position-results`.
