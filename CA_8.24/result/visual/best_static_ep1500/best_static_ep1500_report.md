# Best-Static Longer Training Report, 1500 Episodes

## Setup

- Scenario: `Best-Static`
- Robots: `(15,32)`, `(18,32)`
- People: `70`
- Friction mu: `0.6`
- Reward/termination: per-agent `-1` per step, terminate at `T80`
- Eval seeds: `20000` to `20019`
- Train seed: `35000`

## Greedy Evaluation

| Training episodes | Method | Mean T80 | Delta vs Static | Stay action rate | Path length | Invalid actions |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Static-2R | 166.40 | 0.00 | 1.000 | 0.00 | 0.00 |
| 500 | Static-2R | 166.40 | 0.00 | 1.000 | 0.00 | 0.00 |
| 500 | IDQN-2R | 173.50 | 7.10 | 0.000 | 111.60 | 235.40 |
| 500 | QMIX-2R | 184.15 | 17.75 | 0.132 | 270.65 | 48.95 |
| 1500 | IDQN-2R | 171.85 | 5.45 | 0.000 | 25.40 | 318.30 |
| 1500 | QMIX-2R | 297.00 | 130.60 | 0.485 | 83.00 | 322.50 |

## Training Windows

| Method | Window | Mean T80 | Std T80 | Mean invalid actions | Best T80 |
|---|---|---:|---:|---:|---:|
| IDQN-2R | first100 | 181.39 | 12.14 | 65.73 | 159 |
| IDQN-2R | mid100_701_800 | 184.95 | 51.95 | 112.51 | 160 |
| IDQN-2R | last100 | 178.84 | 10.57 | 117.22 | 156 |
| IDQN-2R | last300 | 179.06 | 11.34 | 110.84 | 156 |
| QMIX-2R | first100 | 182.84 | 31.16 | 177.58 | 152 |
| QMIX-2R | mid100_701_800 | 182.99 | 11.59 | 70.66 | 160 |
| QMIX-2R | last100 | 320.87 | 376.38 | 497.74 | 155 |
| QMIX-2R | last300 | 311.79 | 350.03 | 510.98 | 151 |

## Judgment

Static remains best: mean T80 `166.40`.
IDQN after 1500 episodes has mean T80 `171.85`, stay-action rate `0.000`, and path length `25.40`.
QMIX after 1500 episodes has mean T80 `297.00`, stay-action rate `0.485`, and path length `83.00`.

- IDQN became slightly closer to Static in T80 than the 500-episode run (`173.50 -> 171.85`), and its path length dropped (`111.60 -> 25.40`), but its stay-action rate is still exactly `0.000`. This is not the desired stay-still policy.
- QMIX became more still than the 500-episode run (`0.132 -> 0.485` stay-action rate), but its T80 became much worse because two eval seeds produced severe delays (`T80=1317` and `T80=1214`). This is not a useful convergence toward Static.

Conclusion: 1500 episodes are still not enough to make the current reinforcement-learning policies learn the desired "stay when already well placed" behavior. IDQN is closer in evacuation time but not in action behavior; QMIX is closer in stay rate but unstable and much slower on average.
