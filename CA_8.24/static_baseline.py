from __future__ import annotations

import argparse
import csv
from pathlib import Path

from robot_env import RobotEnvConfig, RobotEnvironment, STAY_ACTION


SCENARIOS = {
    "P0": [(11, 11), (13, 33)],
    "P1": [(9, 8), (23, 8)],
    "P2": [(9, 22), (23, 22)],
    "P3": [(13, 29), (20, 29)],
    "P4": [(4, 10), (28, 10)],
    "Best-Static": [(15, 32), (18, 32)],
}


def run_static_episode(config: RobotEnvConfig) -> tuple[dict, list[dict]]:
    env = RobotEnvironment(config)
    done = False
    actions = [STAY_ACTION for _ in env.agents]
    while not done:
        _, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    return env.episode_summary(), env.step_records


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="Best-Static", choices=sorted(SCENARIOS))
    parser.add_argument("--num-persons", type=int, default=70)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--termination-ratio", type=float, default=0.8)
    parser.add_argument("--target-ratio", type=float, default=0.8)
    parser.add_argument("--output-dir", default="result/visual")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    episode_rows: list[dict] = []
    step_rows: list[dict] = []
    for offset in range(args.seeds):
        seed = args.seed_start + offset
        config = RobotEnvConfig(
            num_persons=args.num_persons,
            robot_start_positions=SCENARIOS[args.scenario],
            seed=seed,
            eval_seed=seed,
            algorithm="Static-2R",
            scenario_id=args.scenario,
            target_ratio=args.target_ratio,
            termination_ratio=args.termination_ratio,
        )
        summary, steps = run_static_episode(config)
        episode_rows.append(summary)
        for row in steps:
            row = dict(row)
            row["episode"] = offset
            row["eval_seed"] = seed
            row["scenario_id"] = args.scenario
            row["algorithm"] = "Static-2R"
            step_rows.append(row)

    stem = f"static_baseline_{args.scenario}_n{args.num_persons}"
    write_rows(output_dir / f"{stem}_episodes.csv", episode_rows)
    write_rows(output_dir / f"{stem}_steps.csv", step_rows)
    print(f"wrote {len(episode_rows)} episode rows to {output_dir / (stem + '_episodes.csv')}")


if __name__ == "__main__":
    main()
