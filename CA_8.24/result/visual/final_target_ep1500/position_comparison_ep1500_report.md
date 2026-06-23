# Position Comparison Report, 1500 Episodes

## Setup

- Scenarios: `P0, P1, P2, P3, P4, Best-Static`
- People: `70`
- Eval seeds: `20000` to `20019`
- Training episodes per learned method/scenario: `1500`
- IDQN BC pretrain steps: `0`
- QMIX BC pretrain steps: `1000`
- BC static seeds: `21000` to `21019`
- IDQN stay regularization weight: `0.003`
- QMIX stay regularization weight: `100.0`
- IDQN lr / epsilon: `0.0001`, `1.0->0.05` decay `0.99`
- QMIX lr / epsilon: `0.0001`, `0.0->0.0` decay `1.0`
- Reward/termination: per-agent `-1` per step, terminate at `T80`
- Friction mu: `0.6`
- Target ordering: `QMIX-BC+StayReg < IDQN-StayReg < No-Robot` in mean T80

## Static Placement Ranking

| Rank | Scenario | Robot starts | Static mean T80 | Static std |
|---:|---|---|---:|---:|
| 1 | Best-Static | `[(15, 32), (18, 32)]` | 152.10 | 4.63 |
| 2 | P0 | `[(11, 11), (13, 33)]` | 171.00 | 7.64 |
| 3 | P4 | `[(4, 10), (28, 10)]` | 180.40 | 6.82 |
| 4 | P3 | `[(13, 29), (20, 29)]` | 181.95 | 11.53 |
| 5 | P2 | `[(9, 22), (23, 22)]` | 182.75 | 6.84 |
| 6 | P1 | `[(9, 8), (23, 8)]` | 184.80 | 9.32 |

## Policy Evaluation

| Scenario | Method | Mean T80 | Delta vs No-Robot | Delta vs static | Stay action rate | Path length | Invalid actions |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 | No-Robot | 186.05 | 0.00 | 15.05 | 1.000 | 0.00 | 0.00 |
| P0 | Static-2R | 171.00 | -15.05 | 0.00 | 1.000 | 0.00 | 0.00 |
| P0 | IDQN-StayReg | 162.50 | -23.55 | -8.50 | 0.078 | 128.85 | 170.75 |
| P0 | QMIX-BC+StayReg | 171.00 | -15.05 | 0.00 | 1.000 | 0.00 | 0.00 |
| P1 | No-Robot | 186.05 | 0.00 | 1.25 | 1.000 | 0.00 | 0.00 |
| P1 | Static-2R | 184.80 | -1.25 | 0.00 | 1.000 | 0.00 | 0.00 |
| P1 | IDQN-StayReg | 155.05 | -31.00 | -29.75 | 0.123 | 87.80 | 184.15 |
| P1 | QMIX-BC+StayReg | 184.80 | -1.25 | 0.00 | 1.000 | 0.00 | 0.00 |
| P2 | No-Robot | 186.05 | 0.00 | 3.30 | 1.000 | 0.00 | 0.00 |
| P2 | Static-2R | 182.75 | -3.30 | 0.00 | 1.000 | 0.00 | 0.00 |
| P2 | IDQN-StayReg | 169.05 | -17.00 | -13.70 | 0.557 | 45.55 | 105.75 |
| P2 | QMIX-BC+StayReg | 182.75 | -3.30 | 0.00 | 1.000 | 0.00 | 0.00 |
| P3 | No-Robot | 186.05 | 0.00 | 4.10 | 1.000 | 0.00 | 0.00 |
| P3 | Static-2R | 181.95 | -4.10 | 0.00 | 1.000 | 0.00 | 0.00 |
| P3 | IDQN-StayReg | 160.65 | -25.40 | -21.30 | 0.678 | 45.35 | 57.80 |
| P3 | QMIX-BC+StayReg | 181.95 | -4.10 | 0.00 | 1.000 | 0.00 | 0.00 |
| P4 | No-Robot | 186.05 | 0.00 | 5.65 | 1.000 | 0.00 | 0.00 |
| P4 | Static-2R | 180.40 | -5.65 | 0.00 | 1.000 | 0.00 | 0.00 |
| P4 | IDQN-StayReg | 174.50 | -11.55 | -5.90 | 0.427 | 102.75 | 97.25 |
| P4 | QMIX-BC+StayReg | 180.40 | -5.65 | 0.00 | 1.000 | 0.00 | 0.00 |
| Best-Static | No-Robot | 186.05 | 0.00 | 33.95 | 1.000 | 0.00 | 0.00 |
| Best-Static | Static-2R | 152.10 | -33.95 | 0.00 | 1.000 | 0.00 | 0.00 |
| Best-Static | IDQN-StayReg | 164.05 | -22.00 | 11.95 | 0.178 | 84.05 | 185.80 |
| Best-Static | QMIX-BC+StayReg | 152.10 | -33.95 | 0.00 | 1.000 | 0.00 | 0.00 |

## Target Ordering Check

Lower mean T80 is better. A scenario passes only when `QMIX < IDQN < No-Robot`.

| Scenario | No-Robot | IDQN | QMIX | Pass |
|---|---:|---:|---:|---|
| P0 | 186.05 | 162.50 | 171.00 | no |
| P1 | 186.05 | 155.05 | 184.80 | no |
| P2 | 186.05 | 169.05 | 182.75 | no |
| P3 | 186.05 | 160.65 | 181.95 | no |
| P4 | 186.05 | 174.50 | 180.40 | no |
| Best-Static | 186.05 | 164.05 | 152.10 | yes |

## Interpretation

- Passed target ordering on `1/6` scenarios: `Best-Static`.
- Failed target ordering on `5/6` scenarios: `P0, P1, P2, P3, P4`.
- Positive `improvement_vs_no_robot` means the method reduced T80 relative to no robot.
- Positive `improvement_vs_static` means the learned moving policy reduced T80 relative to staying fixed at the same start.

## Stay-Still Check

The best static placement is `Best-Static` with mean T80 `152.10`.
If learning has enough signal to preserve a good static placement, the greedy policy should show high stay-action rate, low path length, and near-zero delta loss versus Static-2R.

- `IDQN-StayReg` on `Best-Static`: mean T80 `164.05`, delta vs static `11.95`, stay-action rate `0.178`, path length `84.05`.
- `QMIX-BC+StayReg` on `Best-Static`: mean T80 `152.10`, delta vs static `0.00`, stay-action rate `1.000`, path length `0.00`.
