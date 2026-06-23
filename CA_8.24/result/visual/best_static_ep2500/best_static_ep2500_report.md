# Best-Static Longer Training Report, 2500 Episodes

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
| 2500 | IDQN-2R | 172.35 | 5.95 | 0.382 | 79.80 | 133.35 |
| 2500 | QMIX-2R | 178.90 | 12.50 | 0.009 | 279.05 | 75.65 |

## 2500-Episode Training Windows

| Method | Window | Mean T80 | Std T80 | Mean invalid actions | Best T80 | Worst T80 |
|---|---|---:|---:|---:|---:|---:|
| IDQN-2R | first100 | 181.39 | 12.14 | 65.73 | 159 | 226 |
| IDQN-2R | mid100_1201_1300 | 179.89 | 12.35 | 107.88 | 158 | 221 |
| IDQN-2R | last100 | 179.45 | 10.73 | 112.99 | 157 | 218 |
| IDQN-2R | last300 | 177.58 | 11.35 | 124.45 | 153 | 221 |
| IDQN-2R | last500 | 177.27 | 10.99 | 124.41 | 153 | 228 |
| QMIX-2R | first100 | 182.84 | 31.16 | 177.58 | 152 | 382 |
| QMIX-2R | mid100_1201_1300 | 271.36 | 192.53 | 464.75 | 151 | 1515 |
| QMIX-2R | last100 | 182.30 | 10.02 | 56.65 | 162 | 210 |
| QMIX-2R | last300 | 223.85 | 110.92 | 177.55 | 161 | 757 |
| QMIX-2R | last500 | 250.20 | 145.92 | 254.50 | 156 | 930 |

## Worst Eval Seeds After 2500 Episodes

| Method | Eval seed | T80 | Stay action rate | Path length | Invalid actions |
|---|---:|---:|---:|---:|---:|
| IDQN-2R | 20004 | 191 | 0.395 | 86.00 | 145.00 |
| IDQN-2R | 20002 | 190 | 0.387 | 99.00 | 134.00 |
| IDQN-2R | 20010 | 185 | 0.362 | 87.00 | 149.00 |
| QMIX-2R | 20002 | 191 | 0.000 | 285.00 | 97.00 |
| QMIX-2R | 20013 | 189 | 0.011 | 296.00 | 78.00 |
| QMIX-2R | 20009 | 188 | 0.011 | 273.00 | 99.00 |

## Judgment

Static remains best: mean T80 `166.40` and stay-action rate `1.000`.
IDQN after 2500 episodes: mean T80 `172.35`, stay-action rate `0.382`, path length `79.80`.
QMIX after 2500 episodes: mean T80 `178.90`, stay-action rate `0.009`, path length `279.05`.

- IDQN T80 did not improve relative to 1500 episodes (`171.85 -> 172.35`).
- QMIX recovered from the 1500-episode instability in mean T80 (`297.00 -> 178.90`), but the stay-action and path-length metrics determine whether it actually learned to stay still.

Conclusion: even 2500 episodes do not make the current policies learn the desired static/stay-still behavior at the best placement. The learned policies may change movement amount or T80, but they still do not approach the static policy in action space.
