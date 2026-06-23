# Position Comparison Report, 500 Episodes

## Setup

- Scenarios: `P0, P1, P2, P3, P4, Best-Static`
- People: `70`
- Eval seeds: `20000` to `20019`
- Training episodes per learned method/scenario: `500`
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
| P0 | IDQN-2R | 184.10 | 8.10 | 0.409 | 62.70 | 0.000 | 154.80 |
| P0 | QMIX-2R | 181.90 | 5.90 | 0.009 | 90.40 | 0.000 | 270.35 |
| P1 | Static-2R | 180.25 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P1 | IDQN-2R | 179.00 | -1.25 | 0.009 | 58.30 | 0.000 | 296.50 |
| P1 | QMIX-2R | 185.60 | 5.35 | 0.023 | 64.65 | 0.000 | 298.15 |
| P2 | Static-2R | 178.15 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P2 | IDQN-2R | 181.45 | 3.30 | 0.000 | 71.90 | 0.000 | 291.00 |
| P2 | QMIX-2R | 187.00 | 8.85 | 0.495 | 51.50 | 0.000 | 137.45 |
| P3 | Static-2R | 185.15 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P3 | IDQN-2R | 178.05 | -7.10 | 0.174 | 112.75 | 0.000 | 181.65 |
| P3 | QMIX-2R | 182.10 | -3.05 | 0.086 | 269.65 | 0.000 | 63.50 |
| P4 | Static-2R | 184.40 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| P4 | IDQN-2R | 181.85 | -2.55 | 0.118 | 139.25 | 0.000 | 181.25 |
| P4 | QMIX-2R | 184.50 | 0.10 | 0.254 | 62.45 | 0.000 | 212.75 |
| Best-Static | Static-2R | 166.40 | 0.00 | 1.000 | 0.00 | 1.000 | 0.00 |
| Best-Static | IDQN-2R | 173.50 | 7.10 | 0.000 | 111.60 | 0.000 | 235.40 |
| Best-Static | QMIX-2R | 184.15 | 17.75 | 0.132 | 270.65 | 0.000 | 48.95 |

## Interpretation

Positive `improvement_vs_static` means the learned moving policy reduced T80 relative to staying fixed at the same start.

Learned policies improved over static in these cases:
- `IDQN-2R` on `P3`: `7.10` steps faster than static.
- `QMIX-2R` on `P3`: `3.05` steps faster than static.
- `IDQN-2R` on `P4`: `2.55` steps faster than static.
- `IDQN-2R` on `P1`: `1.25` steps faster than static.

Learned policies did not improve over static in these cases:
- `QMIX-2R` on `Best-Static`: `17.75` steps slower than static.
- `QMIX-2R` on `P2`: `8.85` steps slower than static.
- `IDQN-2R` on `P0`: `8.10` steps slower than static.
- `IDQN-2R` on `Best-Static`: `7.10` steps slower than static.
- `QMIX-2R` on `P0`: `5.90` steps slower than static.
- `QMIX-2R` on `P1`: `5.35` steps slower than static.
- `IDQN-2R` on `P2`: `3.30` steps slower than static.
- `QMIX-2R` on `P4`: `0.10` steps slower than static.

## Stay-Still Check

The best static placement is `Best-Static` with mean T80 `166.40`.
If learning has enough signal to preserve a good static placement, the greedy policy should show high stay-action rate, low path length, and near-zero delta loss versus Static-2R.
The current learned policies do not satisfy that stay-still check on the best static placement.

- `IDQN-2R` on `Best-Static`: mean T80 `173.50`, delta vs static `7.10`, stay-action rate `0.000`, path length `111.60`.
- `QMIX-2R` on `Best-Static`: mean T80 `184.15`, delta vs static `17.75`, stay-action rate `0.132`, path length `270.65`.
