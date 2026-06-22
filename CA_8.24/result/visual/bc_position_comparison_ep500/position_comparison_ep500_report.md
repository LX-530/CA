# Position Comparison Report, 500 Episodes

## Setup

- Scenarios: `P0, P1, P2, P3, P4, Best-Static`
- People: `70`
- Eval seeds: `20000` to `20019`
- Training episodes per learned method/scenario: `500`
- BC pretrain steps: `1000`
- BC static seeds: `21000` to `21019`
- Reward/termination: per-agent `-1` per step, terminate at `T80`
- Friction mu: `0.6`

## Static Placement Ranking

| Rank | Scenario | Robot starts | Static mean T80 | Static std |
|---:|---|---|---:|---:|
| 1 | Best-Static | `[(15, 32), (18, 32)]` | 166.40 | 5.90 |
| 2 | P0 | `[(11, 11), (13, 33)]` | 176.00 | 8.11 |
| 3 | P2 | `[(9, 22), (23, 22)]` | 178.15 | 10.79 |
| 4 | P1 | `[(9, 8), (23, 8)]` | 180.25 | 11.60 |
| 5 | P4 | `[(4, 10), (28, 10)]` | 184.40 | 13.11 |
| 6 | P3 | `[(13, 29), (20, 29)]` | 185.15 | 8.36 |

## Learned Policy Evaluation

| Scenario | Method | Mean T80 | Delta vs static | Stay action rate | Path length | Static-like rate | Invalid actions |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 | Static-2R | 176.00 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P0 | IDQN-BC | 177.30 | 1.30 | 0.177 | 126.80 | 0.000 | 165.30 |
| P0 | QMIX-BC | 184.25 | 8.25 | 0.361 | 25.20 | 0.000 | 209.50 |
| P1 | Static-2R | 180.25 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P1 | IDQN-BC | 182.40 | 2.15 | 0.207 | 162.70 | 0.000 | 126.10 |
| P1 | QMIX-BC | 183.90 | 3.65 | 0.006 | 34.60 | 0.000 | 330.95 |
| P2 | Static-2R | 178.15 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P2 | IDQN-BC | 164.00 | -14.15 | 0.249 | 38.05 | 0.000 | 2165.20 |
| P2 | QMIX-BC | 180.65 | 2.50 | 0.067 | 71.75 | 0.000 | 265.35 |
| P3 | Static-2R | 185.15 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P3 | IDQN-BC | 182.80 | -2.35 | 0.069 | 100.05 | 0.000 | 240.45 |
| P3 | QMIX-BC | 185.30 | 0.15 | 0.303 | 34.65 | 0.000 | 223.95 |
| P4 | Static-2R | 184.40 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P4 | IDQN-BC | 179.55 | -4.85 | 0.045 | 164.70 | 0.000 | 178.40 |
| P4 | QMIX-BC | 186.05 | 1.65 | 0.473 | 32.95 | 0.000 | 163.75 |
| Best-Static | Static-2R | 166.40 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| Best-Static | IDQN-BC | 176.55 | 10.15 | 0.039 | 128.45 | 0.000 | 211.10 |
| Best-Static | QMIX-BC | 180.76 | 14.36 | 0.000 | 87.80 | 0.000 | 3219.50 |

## Interpretation

Positive `improvement_vs_static` means the learned moving policy reduced T80 relative to staying fixed at the same start.

Learned policies improved over static in these cases:
- `IDQN-BC` on `P2`: `14.15` steps faster than static.
- `IDQN-BC` on `P4`: `4.85` steps faster than static.
- `IDQN-BC` on `P3`: `2.35` steps faster than static.

Learned policies did not improve over static in these cases:
- `QMIX-BC` on `Best-Static`: `14.36` steps slower than static.
- `IDQN-BC` on `Best-Static`: `10.15` steps slower than static.
- `QMIX-BC` on `P0`: `8.25` steps slower than static.
- `QMIX-BC` on `P1`: `3.65` steps slower than static.
- `QMIX-BC` on `P2`: `2.50` steps slower than static.
- `IDQN-BC` on `P1`: `2.15` steps slower than static.
- `QMIX-BC` on `P4`: `1.65` steps slower than static.
- `IDQN-BC` on `P0`: `1.30` steps slower than static.
- `QMIX-BC` on `P3`: `0.15` steps slower than static.

## Stay-Still Check

The best static placement is `Best-Static` with mean T80 `166.40`.
If learning has enough signal to preserve a good static placement, the greedy policy should show high stay-action rate, low path length, and near-zero delta loss versus Static-2R.
The current learned policies do not satisfy that stay-still check on the best static placement.

- `IDQN-BC` on `Best-Static`: mean T80 `176.55`, delta vs static `10.15`, stay-action rate `0.039`, path length `128.45`.
- `QMIX-BC` on `Best-Static`: mean T80 `180.76`, delta vs static `14.36`, stay-action rate `0.000`, path length `87.80`.
