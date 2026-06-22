# Position Comparison Report, 0 Episodes

## Setup

- Scenarios: `P0, P1, P2, P3, P4, Best-Static`
- People: `70`
- Eval seeds: `20000` to `20019`
- Training episodes per learned method/scenario: `0`
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
| P0 | IDQN-BC | 176.00 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P0 | QMIX-BC | 176.00 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P1 | Static-2R | 180.25 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P1 | IDQN-BC | 180.25 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P1 | QMIX-BC | 180.25 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P2 | Static-2R | 178.15 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P2 | IDQN-BC | 178.15 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P2 | QMIX-BC | 178.15 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P3 | Static-2R | 185.15 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P3 | IDQN-BC | 185.15 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P3 | QMIX-BC | 185.15 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P4 | Static-2R | 184.40 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P4 | IDQN-BC | 184.40 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P4 | QMIX-BC | 184.40 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| Best-Static | Static-2R | 166.40 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| Best-Static | IDQN-BC | 166.40 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| Best-Static | QMIX-BC | 166.40 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |

## Interpretation

Positive `improvement_vs_static` means the learned moving policy reduced T80 relative to staying fixed at the same start.


Learned policies did not improve over static in these cases:
- `IDQN-BC` on `P0`: `0.00` steps slower than static.
- `QMIX-BC` on `P0`: `0.00` steps slower than static.
- `IDQN-BC` on `P1`: `0.00` steps slower than static.
- `QMIX-BC` on `P1`: `0.00` steps slower than static.
- `IDQN-BC` on `P2`: `0.00` steps slower than static.
- `QMIX-BC` on `P2`: `0.00` steps slower than static.
- `IDQN-BC` on `P3`: `0.00` steps slower than static.
- `QMIX-BC` on `P3`: `0.00` steps slower than static.
- `IDQN-BC` on `P4`: `0.00` steps slower than static.
- `QMIX-BC` on `P4`: `0.00` steps slower than static.
- `IDQN-BC` on `Best-Static`: `0.00` steps slower than static.
- `QMIX-BC` on `Best-Static`: `0.00` steps slower than static.

## Stay-Still Check

The best static placement is `Best-Static` with mean T80 `166.40`.
If learning has enough signal to preserve a good static placement, the greedy policy should show high stay-action rate, low path length, and near-zero delta loss versus Static-2R.
The current learned policies do not satisfy that stay-still check on the best static placement.

- `IDQN-BC` on `Best-Static`: mean T80 `166.40`, delta vs static `0.00`, stay-action rate `1.000`, path length `0.00`.
- `QMIX-BC` on `Best-Static`: mean T80 `166.40`, delta vs static `0.00`, stay-action rate `1.000`, path length `0.00`.
