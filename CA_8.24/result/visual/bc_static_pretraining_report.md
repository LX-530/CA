# Static Behavior Cloning Position Test

## Setup

- People: `70`
- Friction: `mu=0.6`
- Target: `T80`
- Eval seeds: `20000` to `20019`
- BC expert: static robots executing `STAY_ACTION`
- BC seeds: `21000` to `21019`
- BC pretrain steps: `1000`
- Tested positions: `P0`, `P1`, `P2`, `P3`, `P4`, `Best-Static`

## BC-Only Result

BC pretraining by itself successfully clones the static policy. With `episodes=0`, both IDQN-BC and QMIX-BC match the corresponding static baseline at every tested position:

| Scenario | Static T80 | IDQN-BC T80 | QMIX-BC T80 | Stay rate |
|---|---:|---:|---:|---:|
| Best-Static | 166.40 | 166.40 | 166.40 | 1.000 |
| P0 | 176.00 | 176.00 | 176.00 | 1.000 |
| P1 | 180.25 | 180.25 | 180.25 | 1.000 |
| P2 | 178.15 | 178.15 | 178.15 | 1.000 |
| P3 | 185.15 | 185.15 | 185.15 | 1.000 |
| P4 | 184.40 | 184.40 | 184.40 | 1.000 |

This proves the supervised cloning stage can produce a static-like greedy policy.

## BC + 500 RL Fine-Tuning Result

After 500 RL episodes with `epsilon_start=0.2`, `epsilon_min=0.02`, and `epsilon_decay=0.995`, the learned policies no longer preserve the static expert behavior.

| Scenario | Method | Mean T80 | Delta vs Static | Stay rate | Path length |
|---|---|---:|---:|---:|---:|
| Best-Static | IDQN-BC | 176.55 | +10.15 | 0.039 | 128.45 |
| Best-Static | QMIX-BC | 180.76 | +14.36 | 0.000 | 87.80 |
| P0 | IDQN-BC | 177.30 | +1.30 | 0.177 | 126.80 |
| P0 | QMIX-BC | 184.25 | +8.25 | 0.361 | 25.20 |
| P1 | IDQN-BC | 182.40 | +2.15 | 0.207 | 162.70 |
| P1 | QMIX-BC | 183.90 | +3.65 | 0.006 | 34.60 |
| P2 | IDQN-BC | 164.00 | -14.15 | 0.249 | 38.05 |
| P2 | QMIX-BC | 180.65 | +2.50 | 0.067 | 71.75 |
| P3 | IDQN-BC | 182.80 | -2.35 | 0.069 | 100.05 |
| P3 | QMIX-BC | 185.30 | +0.15 | 0.303 | 34.65 |
| P4 | IDQN-BC | 179.55 | -4.85 | 0.045 | 164.70 |
| P4 | QMIX-BC | 186.05 | +1.65 | 0.473 | 32.95 |

## Interpretation

- Static placement ranking is unchanged: `Best-Static = [(15,32), (18,32)]` remains the strongest fixed placement with mean `T80=166.40`.
- BC-only works exactly as intended: the greedy policy stays still and matches static T80.
- RL fine-tuning overwrites the cloned static policy. On `Best-Static`, both IDQN-BC and QMIX-BC become worse than static, and stay rate collapses.
- IDQN-BC improves over static on weaker placements `P2`, `P3`, and `P4`, but this is not because it learned to preserve a good static placement. It learned moving behavior that sometimes helps from weaker starts.
- If the goal is "learn to keep the best static gate-ordering placement", pretraining alone is insufficient. The next step should add an imitation regularizer or action prior during RL, not only before RL.

## Output Files

- `result/visual/bc_only_position_comparison/position_comparison_ep0_summary.csv`
- `result/visual/bc_only_position_comparison/position_comparison_ep0_report.md`
- `result/visual/bc_only_position_comparison/position_comparison_ep0_t80.png`
- `result/visual/bc_only_position_comparison/position_comparison_ep0_stay_rate.png`
- `result/visual/bc_position_comparison_ep500/position_comparison_ep500_summary.csv`
- `result/visual/bc_position_comparison_ep500/position_comparison_ep500_report.md`
- `result/visual/bc_position_comparison_ep500/position_comparison_ep500_t80.png`
- `result/visual/bc_position_comparison_ep500/position_comparison_ep500_stay_rate.png`
