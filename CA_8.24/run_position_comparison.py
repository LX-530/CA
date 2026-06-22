from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_common import SCENARIOS, write_csv


BASE_METHODS = ("Static-2R", "IDQN-2R", "QMIX-2R")


def idqn_method_label(args: argparse.Namespace) -> str:
    return "IDQN-BC" if args.bc_pretrain_steps > 0 else "IDQN-2R"


def qmix_method_label(args: argparse.Namespace) -> str:
    return "QMIX-BC" if args.bc_pretrain_steps > 0 else "QMIX-2R"


def method_order(args: argparse.Namespace) -> tuple[str, str, str]:
    return ("Static-2R", idqn_method_label(args), qmix_method_label(args))


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def command_to_string(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def run_command(cmd: list[str], log_path: Path, *, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(command_to_string(cmd))
        return
    print(f"running: {command_to_string(cmd)}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"command failed with exit code {process.returncode}: {log_path}")


def static_csv(output_dir: Path, scenario: str, num_persons: int) -> Path:
    return output_dir / f"static_baseline_{scenario}_n{num_persons}_episodes.csv"


def idqn_eval_csv(output_dir: Path, scenario: str, num_persons: int, episodes: int) -> Path:
    return output_dir / f"idqn_{scenario}_n{num_persons}_ep{episodes}_eval.csv"


def qmix_eval_csv(output_dir: Path, scenario: str, num_persons: int, episodes: int) -> Path:
    return output_dir / f"qmix_{scenario}_n{num_persons}_ep{episodes}_eval.csv"


def expected_outputs(output_dir: Path, scenario: str, num_persons: int, episodes: int) -> dict[str, Path]:
    return {
        "static": static_csv(output_dir, scenario, num_persons),
        "idqn": idqn_eval_csv(output_dir, scenario, num_persons, episodes),
        "qmix": qmix_eval_csv(output_dir, scenario, num_persons, episodes),
    }


def run_static(args: argparse.Namespace, scenario: str) -> None:
    path = static_csv(args.output_dir, scenario, args.num_persons)
    if args.skip_existing and path.exists():
        print(f"skip existing static: {path}")
        return
    cmd = [
        sys.executable,
        "static_baseline.py",
        "--scenario",
        scenario,
        "--num-persons",
        str(args.num_persons),
        "--seed-start",
        str(args.eval_seed_start),
        "--seeds",
        str(args.eval_seeds),
        "--termination-ratio",
        str(args.termination_ratio),
        "--target-ratio",
        str(args.target_ratio),
        "--output-dir",
        str(args.output_dir),
    ]
    run_command(cmd, args.output_dir / f"static_{scenario}.log", dry_run=args.dry_run)


def run_idqn(args: argparse.Namespace, scenario: str, train_seed: int) -> None:
    path = idqn_eval_csv(args.output_dir, scenario, args.num_persons, args.episodes)
    if args.skip_existing and path.exists():
        print(f"skip existing idqn: {path}")
        return
    model_dir = args.model_dir / scenario / "idqn"
    cmd = [
        sys.executable,
        "dqn_train.py",
        "--scenario",
        scenario,
        "--num-persons",
        str(args.num_persons),
        "--episodes",
        str(args.episodes),
        "--train-seed",
        str(train_seed),
        "--eval-seed-start",
        str(args.eval_seed_start),
        "--eval-seeds",
        str(args.eval_seeds),
        "--target-ratio",
        str(args.target_ratio),
        "--termination-ratio",
        str(args.termination_ratio),
        "--friction-mu",
        str(args.friction_mu),
        "--exit-service-steps",
        str(args.exit_service_steps),
        "--robot-repulsion-cutoff",
        str(args.robot_repulsion_cutoff),
        "--robot-repulsion-amplitude",
        str(args.robot_repulsion_amplitude),
        "--robot-friction-beta",
        str(args.robot_friction_beta),
        "--epsilon-start",
        str(args.epsilon_start),
        "--epsilon-min",
        str(args.epsilon_min),
        "--epsilon-decay",
        str(args.epsilon_decay),
        "--bc-pretrain-steps",
        str(args.bc_pretrain_steps),
        "--bc-seed-start",
        str(args.bc_seed_start),
        "--bc-seeds",
        str(args.bc_seeds),
        "--bc-batch-size",
        str(args.bc_batch_size),
        "--bc-lr",
        str(args.bc_lr),
        "--log-interval",
        str(args.log_interval),
        "--output-dir",
        str(args.output_dir),
        "--model-dir",
        str(model_dir),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    run_command(cmd, args.output_dir / f"idqn_{scenario}_ep{args.episodes}.log", dry_run=args.dry_run)


def run_qmix(args: argparse.Namespace, scenario: str, train_seed: int) -> None:
    path = qmix_eval_csv(args.output_dir, scenario, args.num_persons, args.episodes)
    if args.skip_existing and path.exists():
        print(f"skip existing qmix: {path}")
        return
    model_dir = args.model_dir / scenario / "qmix"
    cmd = [
        sys.executable,
        "qmix_train.py",
        "--scenario",
        scenario,
        "--num-persons",
        str(args.num_persons),
        "--episodes",
        str(args.episodes),
        "--train-seed",
        str(train_seed),
        "--eval-seed-start",
        str(args.eval_seed_start),
        "--eval-seeds",
        str(args.eval_seeds),
        "--target-ratio",
        str(args.target_ratio),
        "--termination-ratio",
        str(args.termination_ratio),
        "--friction-mu",
        str(args.friction_mu),
        "--exit-service-steps",
        str(args.exit_service_steps),
        "--robot-repulsion-cutoff",
        str(args.robot_repulsion_cutoff),
        "--robot-repulsion-amplitude",
        str(args.robot_repulsion_amplitude),
        "--robot-friction-beta",
        str(args.robot_friction_beta),
        "--epsilon-start",
        str(args.epsilon_start),
        "--epsilon-min",
        str(args.epsilon_min),
        "--epsilon-decay",
        str(args.epsilon_decay),
        "--bc-pretrain-steps",
        str(args.bc_pretrain_steps),
        "--bc-seed-start",
        str(args.bc_seed_start),
        "--bc-seeds",
        str(args.bc_seeds),
        "--bc-batch-size",
        str(args.bc_batch_size),
        "--bc-lr",
        str(args.bc_lr),
        "--log-interval",
        str(args.log_interval),
        "--output-dir",
        str(args.output_dir),
        "--model-dir",
        str(model_dir),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    run_command(cmd, args.output_dir / f"qmix_{scenario}_ep{args.episodes}.log", dry_run=args.dry_run)


def load_method_rows(path: Path, method: str, scenario: str) -> list[dict[str, Any]]:
    rows = read_csv(path)
    for row in rows:
        row["method"] = method
        row["algorithm"] = method
        row["scenario_id"] = scenario
        row["robot_start_positions"] = str(SCENARIOS[scenario])
        if method == "Static-2R":
            row["stay_action_rate"] = 1.0
            row["move_action_rate"] = 0.0
            row["static_like_policy"] = True
            row["total_action_count"] = 0
            row["stay_action_count"] = 0
    return rows


def mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(clean)) if clean else None


def std(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0 if clean else None


def summarize_rows(
    rows: list[dict[str, Any]],
    static_mean_by_scenario: dict[str, float],
    methods: tuple[str, str, str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario_id"] == scenario]
        for method in methods:
            method_rows = [row for row in scenario_rows if row["method"] == method]
            if not method_rows:
                continue
            t80_values = [safe_float(row.get("t80")) for row in method_rows]
            valid_t80 = [v for v in t80_values if v is not None]
            success_rate = mean([1.0 if str_to_bool(row.get("success_80")) else 0.0 for row in method_rows])
            mean_t80 = mean(valid_t80)
            static_mean = static_mean_by_scenario.get(scenario)
            summaries.append(
                {
                    "scenario_id": scenario,
                    "method": method,
                    "robot_start_positions": str(SCENARIOS[scenario]),
                    "episodes": len(method_rows),
                    "success_80_rate": success_rate,
                    "mean_t80": mean_t80,
                    "std_t80": std(valid_t80),
                    "median_t80": float(np.median(valid_t80)) if valid_t80 else None,
                    "mean_t80_minus_static": None if mean_t80 is None or static_mean is None else mean_t80 - static_mean,
                    "improvement_vs_static": None if mean_t80 is None or static_mean is None else static_mean - mean_t80,
                    "mean_invalid_actions": mean([safe_float(row.get("invalid_action_count")) or 0.0 for row in method_rows]),
                    "mean_robot_path_length": mean([safe_float(row.get("robot_path_length")) or 0.0 for row in method_rows]),
                    "mean_stay_action_rate": mean([safe_float(row.get("stay_action_rate")) or 0.0 for row in method_rows]),
                    "static_like_episode_rate": mean([1.0 if str_to_bool(row.get("static_like_policy")) else 0.0 for row in method_rows]),
                    "mean_return": mean([safe_float(row.get("mean_episode_return")) for row in method_rows if safe_float(row.get("mean_episode_return")) is not None]),
                }
            )
    return summaries


def collect_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    static_mean_by_scenario: dict[str, float] = {}

    for scenario in args.scenarios:
        outputs = expected_outputs(args.output_dir, scenario, args.num_persons, args.episodes)
        missing = [str(path) for path in outputs.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing expected outputs:\n" + "\n".join(missing))

        static_rows = load_method_rows(outputs["static"], "Static-2R", scenario)
        idqn_rows = load_method_rows(outputs["idqn"], idqn_method_label(args), scenario)
        qmix_rows = load_method_rows(outputs["qmix"], qmix_method_label(args), scenario)
        scenario_rows = static_rows + idqn_rows + qmix_rows
        all_rows.extend(scenario_rows)
        static_t80 = [
            safe_float(row.get("t80"))
            for row in static_rows
            if safe_float(row.get("t80")) is not None
        ]
        static_mean_by_scenario[scenario] = float(np.mean(static_t80))

    summary_rows = summarize_rows(all_rows, static_mean_by_scenario, method_order(args))
    return all_rows, summary_rows


def format_float(value: Any, digits: int = 2) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def write_report(args: argparse.Namespace, summary_rows: list[dict[str, Any]], report_path: Path) -> None:
    static_rows = [row for row in summary_rows if row["method"] == "Static-2R"]
    static_rows = sorted(static_rows, key=lambda row: safe_float(row.get("mean_t80")) or math.inf)

    lines = [
        f"# Position Comparison Report, {args.episodes} Episodes",
        "",
        "## Setup",
        "",
        f"- Scenarios: `{', '.join(args.scenarios)}`",
        f"- People: `{args.num_persons}`",
        f"- Eval seeds: `{args.eval_seed_start}` to `{args.eval_seed_start + args.eval_seeds - 1}`",
        f"- Training episodes per learned method/scenario: `{args.episodes}`",
        f"- BC pretrain steps: `{args.bc_pretrain_steps}`",
        f"- BC static seeds: `{args.bc_seed_start}` to `{args.bc_seed_start + args.bc_seeds - 1}`",
        f"- Reward/termination: per-agent `-1` per step, terminate at `T80`",
        f"- Friction mu: `{args.friction_mu}`",
        "",
        "## Static Placement Ranking",
        "",
        "| Rank | Scenario | Robot starts | Static mean T80 | Static std |",
        "|---:|---|---|---:|---:|",
    ]
    for rank, row in enumerate(static_rows, start=1):
        lines.append(
            f"| {rank} | {row['scenario_id']} | `{row['robot_start_positions']}` | "
            f"{format_float(row['mean_t80'])} | {format_float(row['std_t80'])} |"
        )

    lines.extend([
        "",
        "## Learned Policy Evaluation",
        "",
        "| Scenario | Method | Mean T80 | Delta vs static | Stay action rate | Path length | Static-like rate | Invalid actions |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for scenario in args.scenarios:
        for method in method_order(args):
            matches = [
                row for row in summary_rows
                if row["scenario_id"] == scenario and row["method"] == method
            ]
            if not matches:
                continue
            row = matches[0]
            lines.append(
                f"| {scenario} | {method} | {format_float(row['mean_t80'])} | "
                f"{format_float(row['mean_t80_minus_static'])} | "
                f"{format_float(row['mean_stay_action_rate'], 3)} | "
                f"{format_float(row['mean_robot_path_length'])} | "
                f"{format_float(row['static_like_episode_rate'], 3)} | "
                f"{format_float(row['mean_invalid_actions'])} |"
            )

    learned_rows = [
        row for row in summary_rows
        if row["method"] in {idqn_method_label(args), qmix_method_label(args)}
    ]
    improved = [
        row for row in learned_rows
        if (safe_float(row.get("improvement_vs_static")) or 0.0) > 0.0
    ]
    not_improved = [
        row for row in learned_rows
        if (safe_float(row.get("improvement_vs_static")) or 0.0) <= 0.0
    ]
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Positive `improvement_vs_static` means the learned moving policy reduced T80 relative to staying fixed at the same start.",
        "",
    ])
    if improved:
        lines.append("Learned policies improved over static in these cases:")
        for row in sorted(improved, key=lambda item: safe_float(item["improvement_vs_static"]) or 0.0, reverse=True):
            lines.append(
                f"- `{row['method']}` on `{row['scenario_id']}`: "
                f"`{format_float(row['improvement_vs_static'])}` steps faster than static."
            )
    if not_improved:
        lines.append("")
        lines.append("Learned policies did not improve over static in these cases:")
        for row in sorted(not_improved, key=lambda item: safe_float(item["mean_t80_minus_static"]) or 0.0, reverse=True):
            lines.append(
                f"- `{row['method']}` on `{row['scenario_id']}`: "
                f"`{format_float(row['mean_t80_minus_static'])}` steps slower than static."
            )

    best_static = static_rows[0] if static_rows else None
    if best_static:
        scenario = best_static["scenario_id"]
        learned = [
            row for row in summary_rows
            if row["scenario_id"] == scenario
            and row["method"] in {idqn_method_label(args), qmix_method_label(args)}
        ]
        lines.extend([
            "",
            "## Stay-Still Check",
            "",
            f"The best static placement is `{scenario}` with mean T80 `{format_float(best_static['mean_t80'])}`.",
            "If learning has enough signal to preserve a good static placement, the greedy policy should show high stay-action rate, low path length, and near-zero delta loss versus Static-2R.",
            "The current learned policies do not satisfy that stay-still check on the best static placement.",
            "",
        ])
        for row in learned:
            lines.append(
                f"- `{row['method']}` on `{scenario}`: mean T80 `{format_float(row['mean_t80'])}`, "
                f"delta vs static `{format_float(row['mean_t80_minus_static'])}`, "
                f"stay-action rate `{format_float(row['mean_stay_action_rate'], 3)}`, "
                f"path length `{format_float(row['mean_robot_path_length'])}`."
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(args: argparse.Namespace, summary_rows: list[dict[str, Any]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = list(args.scenarios)
    x = np.arange(len(scenarios))
    width = 0.25
    colors = {
        "Static-2R": "#59A14F",
        "IDQN-2R": "#4E79A7",
        "QMIX-2R": "#F28E2B",
        "IDQN-BC": "#4E79A7",
        "QMIX-BC": "#F28E2B",
    }

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, method in enumerate(method_order(args)):
        values = []
        errors = []
        for scenario in scenarios:
            row = next(
                row for row in summary_rows
                if row["scenario_id"] == scenario and row["method"] == method
            )
            values.append(safe_float(row["mean_t80"]) or np.nan)
            errors.append(safe_float(row["std_t80"]) or 0.0)
        ax.bar(x + (offset - 1) * width, values, width, yerr=errors, label=method, color=colors[method])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Mean T80 on eval seeds")
    ax.set_title(f"Position comparison, {args.episodes} training episodes")
    ax.legend()
    fig.savefig(args.output_dir / f"position_comparison_ep{args.episodes}_t80.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, method in enumerate((idqn_method_label(args), qmix_method_label(args))):
        values = []
        for scenario in scenarios:
            row = next(
                row for row in summary_rows
                if row["scenario_id"] == scenario and row["method"] == method
            )
            values.append(safe_float(row["mean_stay_action_rate"]) or 0.0)
        ax.bar(x + (offset - 0.5) * width, values, width, label=method, color=colors[method])
    ax.axhline(1.0, color="#59A14F", linestyle="--", linewidth=1, label="Static stay rate")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Greedy eval stay-action rate")
    ax.set_title("Does the learned policy stay still?")
    ax.legend()
    fig.savefig(args.output_dir / f"position_comparison_ep{args.episodes}_stay_rate.png")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--num-persons", type=int, default=70)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--train-seed-base", type=int, default=30000)
    parser.add_argument("--eval-seed-start", type=int, default=20000)
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--target-ratio", type=float, default=0.8)
    parser.add_argument("--termination-ratio", type=float, default=0.8)
    parser.add_argument("--exit-service-steps", type=int, default=2)
    parser.add_argument("--friction-mu", type=float, default=0.6)
    parser.add_argument("--robot-repulsion-cutoff", type=float, default=5.0)
    parser.add_argument("--robot-repulsion-amplitude", type=float, default=0.25)
    parser.add_argument("--robot-friction-beta", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.97)
    parser.add_argument("--bc-pretrain-steps", type=int, default=0)
    parser.add_argument("--bc-seed-start", type=int, default=21000)
    parser.add_argument("--bc-seeds", type=int, default=20)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("result/visual/position_comparison_ep500"))
    parser.add_argument("--model-dir", type=Path, default=Path("result/models/position_comparison_ep500"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    unknown = [scenario for scenario in args.scenarios if scenario not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    if not args.collect_only and args.max_workers <= 1:
        for idx, scenario in enumerate(args.scenarios):
            train_seed = args.train_seed_base + idx * 1000
            run_static(args, scenario)
            run_idqn(args, scenario, train_seed)
            run_qmix(args, scenario, train_seed)
    elif not args.collect_only:
        jobs = []
        for idx, scenario in enumerate(args.scenarios):
            train_seed = args.train_seed_base + idx * 1000
            jobs.extend(
                [
                    (f"static:{scenario}", run_static, (args, scenario)),
                    (f"idqn:{scenario}", run_idqn, (args, scenario, train_seed)),
                    (f"qmix:{scenario}", run_qmix, (args, scenario, train_seed)),
                ]
            )
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = {
                executor.submit(func, *params): label
                for label, func, params in jobs
            }
            for future in as_completed(futures):
                label = futures[future]
                future.result()
                print(f"finished: {label}")

    if args.dry_run:
        return

    all_rows, summary_rows = collect_rows(args)
    paired_path = args.output_dir / f"position_comparison_ep{args.episodes}_paired_eval.csv"
    summary_path = args.output_dir / f"position_comparison_ep{args.episodes}_summary.csv"
    report_path = args.output_dir / f"position_comparison_ep{args.episodes}_report.md"
    write_csv(paired_path, all_rows)
    write_csv(summary_path, summary_rows)
    write_report(args, summary_rows, report_path)
    plot_summary(args, summary_rows)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
