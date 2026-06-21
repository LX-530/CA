# RL Baseline Design Judgment

## Basic Judgment

The four-group comparison is appropriate and should be kept:

| Group | Robots | Moving | Learning | Repulsion field | Role |
|---|---:|---|---|---|---|
| No-Robot | 0 | No | No | No | Environment sanity check and lower baseline |
| Static-2R | 2 | No | No | Yes | Same-initial-position static baseline |
| IDQN-2R | 2 | Yes | Yes | Yes | Independent learning baseline |
| QMIX-2R | 2 | Yes | Yes | Yes | Cooperative learning method |

This structure is defensible because it separates three effects: robot presence, robot movement, and cooperative learning. The No-Robot group should be used mainly for environment validation and reference, not as the only baseline for the learning algorithms.

## Position Design

Coordinates use zero-based `(row, col)`. The following candidate positions were checked against the current `map.json`; valid means the cell is inside the map and is an empty cell, not wall, fire, or exit.

| ID | Meaning | Robot 1 | Robot 2 | Validity |
|---|---|---:|---:|---|
| P0-written | GitHub default replacement | (11, 11) | (13, 33) | Valid |
| P1 | Near initial crowd region | (9, 8) | (23, 8) | Valid |
| P2 | Middle evacuation path | (9, 22) | (23, 22) | Valid |
| P3 | Upstream of exit | (13, 29) | (20, 29) | Valid |
| P4 | Unfavorable far-side position | (4, 10) | (28, 10) | Valid |
| Best-Static-current | Best static from current scan | (15, 32) | (18, 32) | Valid |

Important correction: the current code-level default in `environment.py` is `(10,10), (15,15)`. The second cell `(15,15)` is a fire cell in the current map, so this must not be used as a formal P0. Add `_is_valid_position()` before formal experiments and reject robot starts on walls, fire, exits, occupied pedestrian cells, or duplicated robot cells.

## Static Baselines

Two static baselines are needed:

1. `Static-2R-same-start`: static robots placed at the same P0-P5 initial positions used by IDQN and QMIX. This answers whether learning helps from the same start.
2. `Best-Static`: best manually selected static layout from training or validation seeds only. This answers whether the learned moving policy can beat strong human/static placement.

Do not choose `Best-Static` using final test seeds. That would leak test-set information. The final test seeds must be held out and reused identically across No-Robot, Static-2R, IDQN-2R, and QMIX-2R.

The current static scan suggests `(15,32), (18,32)` is the strongest static candidate under `mu=0.6`, with a multi-density mean gain of `28.49` steps versus No-Robot and `138/150` paired runs improved. This does not prove RL will beat it. It means position choice is a strong factor and `Best-Static` is necessary.

## Reward Recommendation

For the first RL version, use the proposed reward:

```text
r_t = alpha * (p_t - p_{t-1}) - beta / H - eta * I_invalid + gamma * I_first_reach_80
```

Recommended first values:

| Parameter | Value |
|---|---:|
| alpha | 1 |
| beta | 1 |
| eta | 0.01 |
| gamma | 1 |

Implementation detail: the `gamma` bonus should be paid once, only when the episode first reaches `p_t >= 0.8`. Otherwise the agent can receive the bonus repeatedly after 80 percent evacuation. The current `environment.py` reward is not aligned with this plan; it still uses a fixed `-4`, an escape bonus, and a terminal time penalty.

## Metrics

The main metric should be `T80`, not only full evacuation time. Full evacuation time is still useful, but only as a secondary metric when most experiments actually complete.

Recommended metrics:

| Metric | Role |
|---|---|
| `T80` | Primary performance metric |
| `P80` | Success rate of reaching 80 percent within horizon |
| `T90` | Late evacuation ability |
| `T_all` | Full evacuation time when completion rate is high |
| Exit throughput | Door capacity usage |
| Exit-area density | Door congestion mechanism |
| Friction/blockage rate | Arch blocking and conflict reduction |
| Robot path length | Avoid meaningless wandering |
| Invalid action rate | Check action feasibility |
| Robot influence overlap | Check duplicated robot work |
| Minimum robot distance | Check cooperation versus clustering |

The proposed overlap metric is appropriate:

```text
O = (1 / T) * sum_t I[d(r1(t), r2(t)) <= 2 r_c]
```

If QMIX has lower `T80` and lower overlap than IDQN, the paper can argue that QMIX improves evacuation by assigning the two robots to less redundant influence regions.

## Recommended Experiment Order

1. Lock the paper environment configuration.
2. Add and verify `_is_valid_position()`.
3. Re-run No-Robot validation.
4. Run static scans using training or validation seeds only.
5. Fix `Best-Static`.
6. Build same-start Static-2R baselines for P0-P5.
7. Train IDQN-2R from P0-P5.
8. Train QMIX-2R from P0-P5.
9. Evaluate all groups on identical held-out test seeds.
10. Compare Static-2R -> IDQN-2R -> QMIX-2R.
11. Analyze trajectories, exit density, friction blocks, overlap, and path length.
12. Run robustness checks for robot influence radius, crowd density, and seeds.

## Final Recommendation

The proposed design is usable, but only after the environment is refactored to match the stage-0 rules and the reward/metric definitions above. The safest thesis-level claim is not "RL must beat the scan." The defensible claim is:

1. `Best-Static` shows the best fixed door-organization effect.
2. `IDQN` tests whether independent moving robots can recover useful positions from different starts.
3. `QMIX` tests whether coordinated learning reduces overlap and improves `T80` compared with IDQN.

If QMIX does not beat `Best-Static`, the result is still interpretable: the optimal static door-control position is very strong in this fixed-map setting. The moving-policy advantage should then be evaluated on different initial robot positions, crowd sizes, and random layouts/seeds rather than only on one fixed easy map.
