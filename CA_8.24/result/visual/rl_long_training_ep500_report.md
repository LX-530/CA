# RL Long Training Report, 500 Episodes

## Setup

- Conda environment: `base`
- Scenario: `Best-Static`
- Robots: `(15,32)`, `(18,32)`
- People: `70`
- Reward: per-agent `-1` per step
- Termination: `T80`, target escaped count `56`
- Evaluation seeds: `20`, paired with the same Static-2R baseline

## Training Window Summary

| Algorithm | First 50 T80 | Last 50 T80 | Last 50 - First 50 | Last 50 - Previous 50 | Last 50 std |
|---|---:|---:|---:|---:|---:|
| IDQN-2R | 180.96 | 181.38 | 0.42 | 1.46 | 10.87 |
| QMIX-2R | 179.42 | 183.42 | 4.00 | 1.28 | 10.76 |

## Greedy Evaluation After Training

| Method | Mean T80 | Std T80 | Success80 | Mean invalid actions |
|---|---:|---:|---:|---:|
| Static-2R | 166.40 | 5.90 | 1.00 | 0.00 |
| IDQN-2R | 172.50 | 9.10 | 1.00 | 181.35 |
| QMIX-2R | 185.65 | 13.44 | 1.00 | 124.60 |

## Convergence Judgment

The 500-episode run appears stable but not improved beyond the static baseline. IDQN's last-50 moving window is close to its previous-50 window, so it is approximately plateaued. QMIX also remains noisy and does not beat Static-2R.

Because the start position is already `Best-Static`, the best action is often to stay still. A moving RL policy has little room to improve and many ways to hurt the door organization. The current result should be interpreted as: under the strongest static start, long training does not produce a better moving policy with the current sparse `-T80` reward.

Next recommended long-run experiment: train from P0/P1/P2/P3/P4 starts, where movement can actually improve over a weak or remote initial placement.
