# RL Training Convergence Check

## Environment

- Conda executable: `D:\software\miniconda\Scripts\conda.exe`
- Conda environment used: `base`
- Python: `D:\software\miniconda\python.exe`
- PyTorch: `2.5.1+cu121`
- Gymnasium: `1.0.0`
- Scenario: `Best-Static`, robots start at `(15,32)` and `(18,32)`
- People: `70`
- Termination: `T80`, target count `56`
- Reward: per-agent time reward `-1` per step, so mean per-agent return equals `-T80`

## Commands

```text
conda run -n base python dqn_train.py --episodes 80 --eval-seeds 10 --scenario Best-Static --num-persons 70
conda run -n base python qmix_train.py --episodes 80 --eval-seeds 10 --scenario Best-Static --num-persons 70
```

## Results

| Method | Phase | Mean T80 | Success80 | Mean invalid actions |
|---|---|---:|---:|---:|
| Static-2R | eval | 166.90 | 1.00 | 0.00 |
| IDQN-2R | greedy eval | 172.00 | 1.00 | 284.50 |
| QMIX-2R | greedy eval | 186.40 | 1.00 | 273.30 |

Training trend:

- IDQN first 20 mean T80 = 183.15; last 20 mean T80 = 181.70; change = -1.45.
- QMIX first 20 mean T80 = 187.90; last 20 mean T80 = 185.80; change = -2.10.

## Judgment

This short run does not show reliable convergence beyond the static baseline. IDQN has a very small training-window improvement and evaluates at `T80=172.00`, worse than same-start Static-2R at `T80=166.90`. QMIX also improves slightly in the training window but evaluates worse at `T80=186.40`.

This is expected for the `Best-Static` start: the robots already begin at the strongest fixed position, so moving is often harmful. The current pure time reward is also sparse, so 80 episodes is not enough to prove policy convergence. The code is now ready for longer runs and for P0-P4 start-position experiments.
