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

from robot_env import RobotEnvConfig, RobotEnvironment
from rl_common import SCENARIOS, write_csv


BASE_METHODS = ("Static-2R", "IDQN-2R", "QMIX-2R")


def idqn_method_label(args: argparse.Namespace) -> str:
    has_bc = idqn_bc_steps(args) > 0
    has_reg = idqn_stay_weight(args) > 0.0
    if has_bc and has_reg:
        return "IDQN-BC+StayReg"
    if has_bc:
        return "IDQN-BC"
    if has_reg:
        return "IDQN-StayReg"
    return "IDQN-2R"


def qmix_method_label(args: argparse.Namespace) -> str:
    has_bc = qmix_bc_steps(args) > 0
    has_reg = qmix_stay_weight(args) > 0.0
    if has_bc and has_reg:
        return "QMIX-BC+StayReg"
    if has_bc:
        return "QMIX-BC"
    if has_reg:
        return "QMIX-StayReg"
    return "QMIX-2R"


def method_order(args: argparse.Namespace) -> tuple[str, str, str, str]:
    return ("No-Robot", "Static-2R", idqn_method_label(args), qmix_method_label(args))


def idqn_stay_weight(args: argparse.Namespace) -> float:
    value = getattr(args, "idqn_stay_regularization_weight", None)
    return float(args.stay_regularization_weight if value is None else value)


def qmix_stay_weight(args: argparse.Namespace) -> float:
    value = getattr(args, "qmix_stay_regularization_weight", None)
    return float(args.stay_regularization_weight if value is None else value)


def idqn_bc_steps(args: argparse.Namespace) -> int:
    value = getattr(args, "idqn_bc_pretrain_steps", None)
    return int(args.bc_pretrain_steps if value is None else value)


def qmix_bc_steps(args: argparse.Namespace) -> int:
    value = getattr(args, "qmix_bc_pretrain_steps", None)
    return int(args.bc_pretrain_steps if value is None else value)


def idqn_lr(args: argparse.Namespace) -> float:
    value = getattr(args, "idqn_lr", None)
    return float(args.lr if value is None else value)


def qmix_lr(args: argparse.Namespace) -> float:
    value = getattr(args, "qmix_lr", None)
    return float(args.lr if value is None else value)


def idqn_epsilon_start(args: argparse.Namespace) -> float:
    value = getattr(args, "idqn_epsilon_start", None)
    return float(args.epsilon_start if value is None else value)


def qmix_epsilon_start(args: argparse.Namespace) -> float:
    value = getattr(args, "qmix_epsilon_start", None)
    return float(args.epsilon_start if value is None else value)


def idqn_epsilon_min(args: argparse.Namespace) -> float:
    value = getattr(args, "idqn_epsilon_min", None)
    return float(args.epsilon_min if value is None else value)


def qmix_epsilon_min(args: argparse.Namespace) -> float:
    value = getattr(args, "qmix_epsilon_min", None)
    return float(args.epsilon_min if value is None else value)


def idqn_epsilon_decay(args: argparse.Namespace) -> float:
    value = getattr(args, "idqn_epsilon_decay", None)
    return float(args.epsilon_decay if value is None else value)


def qmix_epsilon_decay(args: argparse.Namespace) -> float:
    value = getattr(args, "qmix_epsilon_decay", None)
    return float(args.epsilon_decay if value is None else value)


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


def no_robot_csv(output_dir: Path, scenario: str, num_persons: int) -> Path:
    return output_dir / f"no_robot_baseline_{scenario}_n{num_persons}_episodes.csv"


def idqn_eval_csv(output_dir: Path, scenario: str, num_persons: int, episodes: int) -> Path:
    return output_dir / f"idqn_{scenario}_n{num_persons}_ep{episodes}_eval.csv"


def qmix_eval_csv(output_dir: Path, scenario: str, num_persons: int, episodes: int) -> Path:
    return output_dir / f"qmix_{scenario}_n{num_persons}_ep{episodes}_eval.csv"


def expected_outputs(output_dir: Path, scenario: str, num_persons: int, episodes: int) -> dict[str, Path]:
    return {
        "no_robot": no_robot_csv(output_dir, scenario, num_persons),
        "static": static_csv(output_dir, scenario, num_persons),
        "idqn": idqn_eval_csv(output_dir, scenario, num_persons, episodes),
        "qmix": qmix_eval_csv(output_dir, scenario, num_persons, episodes),
    }


def run_no_robot(args: argparse.Namespace, scenario: str) -> None:
    path = no_robot_csv(args.output_dir, scenario, args.num_persons)
    if args.skip_existing and path.exists():
        print(f"skip existing no-robot: {path}")
        return
    rows = []
    for offset in range(args.eval_seeds):
        seed = args.eval_seed_start + offset
        config = RobotEnvConfig(
            map_path=getattr(args, "map_path", "map.json"),
            target_area=tuple(getattr(args, "target_area", (3, 32, 2, 7))),
            num_persons=args.num_persons,
            robot_start_positions=[],
            seed=seed,
            eval_seed=seed,
            algorithm="No-Robot",
            scenario_id=scenario,
            target_ratio=args.target_ratio,
            termination_ratio=args.termination_ratio,
            exit_service_steps=args.exit_service_steps,
            friction_mu=args.friction_mu,
            no_progress_limit=500,
            max_steps_guard=10000,
            record_step_metrics=False,
        )
        env = RobotEnvironment(config)
        done = False
        while not done:
            _, _, dones, _ = env.step([])
            done = dones["__all__"]
        row = env.episode_summary()
        row["episode"] = offset + 1
        row["robot1_start"] = None
        row["robot2_start"] = None
        rows.append(row)
    write_csv(path, rows)
    print(f"wrote {len(rows)} no-robot rows to {path}")


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
        "--stay-regularization-weight",
        str(idqn_stay_weight(args)),
        "--lr",
        str(idqn_lr(args)),
        "--epsilon-start",
        str(idqn_epsilon_start(args)),
        "--epsilon-min",
        str(idqn_epsilon_min(args)),
        "--epsilon-decay",
        str(idqn_epsilon_decay(args)),
        "--bc-pretrain-steps",
        str(idqn_bc_steps(args)),
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
        "--stay-regularization-weight",
        str(qmix_stay_weight(args)),
        "--lr",
        str(qmix_lr(args)),
        "--epsilon-start",
        str(qmix_epsilon_start(args)),
        "--epsilon-min",
        str(qmix_epsilon_min(args)),
        "--epsilon-decay",
        str(qmix_epsilon_decay(args)),
        "--bc-pretrain-steps",
        str(qmix_bc_steps(args)),
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
        row["robot_start_positions"] = "[]" if method == "No-Robot" else str(SCENARIOS[scenario])
        if method in {"No-Robot", "Static-2R"}:
            row["stay_action_rate"] = 1.0
            row["move_action_rate"] = 0.0
            row["static_like_policy"] = method == "Static-2R"
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
    no_robot_mean_by_scenario: dict[str, float],
    methods: tuple[str, ...],
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
            no_robot_mean = no_robot_mean_by_scenario.get(scenario)
            summaries.append(
                {
                    "scenario_id": scenario,
                    "method": method,
                    "robot_start_positions": "[]" if method == "No-Robot" else str(SCENARIOS[scenario]),
                    "episodes": len(method_rows),
                    "success_80_rate": success_rate,
                    "mean_t80": mean_t80,
                    "std_t80": std(valid_t80),
                    "median_t80": float(np.median(valid_t80)) if valid_t80 else None,
                    "mean_t80_minus_static": None if mean_t80 is None or static_mean is None else mean_t80 - static_mean,
                    "improvement_vs_static": None if mean_t80 is None or static_mean is None else static_mean - mean_t80,
                    "mean_t80_minus_no_robot": None if mean_t80 is None or no_robot_mean is None else mean_t80 - no_robot_mean,
                    "improvement_vs_no_robot": None if mean_t80 is None or no_robot_mean is None else no_robot_mean - mean_t80,
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
    no_robot_mean_by_scenario: dict[str, float] = {}

    for scenario in args.scenarios:
        outputs = expected_outputs(args.output_dir, scenario, args.num_persons, args.episodes)
        missing = [str(path) for path in outputs.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing expected outputs:\n" + "\n".join(missing))

        no_robot_rows = load_method_rows(outputs["no_robot"], "No-Robot", scenario)
        static_rows = load_method_rows(outputs["static"], "Static-2R", scenario)
        idqn_rows = load_method_rows(outputs["idqn"], idqn_method_label(args), scenario)
        qmix_rows = load_method_rows(outputs["qmix"], qmix_method_label(args), scenario)
        scenario_rows = no_robot_rows + static_rows + idqn_rows + qmix_rows
        all_rows.extend(scenario_rows)
        no_robot_t80 = [
            safe_float(row.get("t80"))
            for row in no_robot_rows
            if safe_float(row.get("t80")) is not None
        ]
        no_robot_mean_by_scenario[scenario] = float(np.mean(no_robot_t80))
        static_t80 = [
            safe_float(row.get("t80"))
            for row in static_rows
            if safe_float(row.get("t80")) is not None
        ]
        static_mean_by_scenario[scenario] = float(np.mean(static_t80))

    summary_rows = summarize_rows(
        all_rows,
        static_mean_by_scenario,
        no_robot_mean_by_scenario,
        method_order(args),
    )
    return all_rows, summary_rows


def format_float(value: Any, digits: int = 2) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def write_report(args: argparse.Namespace, summary_rows: list[dict[str, Any]], report_path: Path) -> None:
    static_rows = [row for row in summary_rows if row["method"] == "Static-2R"]
    static_rows = sorted(static_rows, key=lambda row: safe_float(row.get("mean_t80")) or math.inf)
    idqn_label = idqn_method_label(args)
    qmix_label = qmix_method_label(args)

    lines = [
        f"# Position Comparison Report, {args.episodes} Episodes",
        "",
        "## Setup",
        "",
        f"- Scenarios: `{', '.join(args.scenarios)}`",
        f"- People: `{args.num_persons}`",
        f"- Eval seeds: `{args.eval_seed_start}` to `{args.eval_seed_start + args.eval_seeds - 1}`",
        f"- Training episodes per learned method/scenario: `{args.episodes}`",
        f"- IDQN BC pretrain steps: `{idqn_bc_steps(args)}`",
        f"- QMIX BC pretrain steps: `{qmix_bc_steps(args)}`",
        f"- BC static seeds: `{args.bc_seed_start}` to `{args.bc_seed_start + args.bc_seeds - 1}`",
        f"- IDQN stay regularization weight: `{idqn_stay_weight(args)}`",
        f"- QMIX stay regularization weight: `{qmix_stay_weight(args)}`",
        f"- IDQN lr / epsilon: `{idqn_lr(args)}`, `{idqn_epsilon_start(args)}->{idqn_epsilon_min(args)}` decay `{idqn_epsilon_decay(args)}`",
        f"- QMIX lr / epsilon: `{qmix_lr(args)}`, `{qmix_epsilon_start(args)}->{qmix_epsilon_min(args)}` decay `{qmix_epsilon_decay(args)}`",
        f"- Reward/termination: per-agent `-1` per step, terminate at `T80`",
        f"- Friction mu: `{args.friction_mu}`",
        f"- Target ordering: `{qmix_label} < {idqn_label} < No-Robot` in mean T80",
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
        "## Policy Evaluation",
        "",
        "| Scenario | Method | Mean T80 | Delta vs No-Robot | Delta vs static | Stay action rate | Path length | Invalid actions |",
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
                f"{format_float(row['mean_t80_minus_no_robot'])} | "
                f"{format_float(row['mean_t80_minus_static'])} | "
                f"{format_float(row['mean_stay_action_rate'], 3)} | "
                f"{format_float(row['mean_robot_path_length'])} | "
                f"{format_float(row['mean_invalid_actions'])} |"
            )

    lines.extend([
        "",
        "## Target Ordering Check",
        "",
        "Lower mean T80 is better. A scenario passes only when `QMIX < IDQN < No-Robot`.",
        "",
        "| Scenario | No-Robot | IDQN | QMIX | Pass |",
        "|---|---:|---:|---:|---|",
    ])
    passed = []
    failed = []
    for scenario in args.scenarios:
        rows_by_method = {
            row["method"]: row
            for row in summary_rows
            if row["scenario_id"] == scenario
        }
        no_robot = safe_float(rows_by_method.get("No-Robot", {}).get("mean_t80"))
        idqn = safe_float(rows_by_method.get(idqn_label, {}).get("mean_t80"))
        qmix = safe_float(rows_by_method.get(qmix_label, {}).get("mean_t80"))
        ok = (
            no_robot is not None
            and idqn is not None
            and qmix is not None
            and qmix < idqn < no_robot
        )
        (passed if ok else failed).append(scenario)
        lines.append(
            f"| {scenario} | {format_float(no_robot)} | {format_float(idqn)} | "
            f"{format_float(qmix)} | {'yes' if ok else 'no'} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Passed target ordering on `{len(passed)}/{len(args.scenarios)}` scenarios: `{', '.join(passed) if passed else 'none'}`.",
        f"- Failed target ordering on `{len(failed)}/{len(args.scenarios)}` scenarios: `{', '.join(failed) if failed else 'none'}`.",
        "- Positive `improvement_vs_no_robot` means the method reduced T80 relative to no robot.",
        "- Positive `improvement_vs_static` means the learned moving policy reduced T80 relative to staying fixed at the same start.",
    ])

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
        "No-Robot": "#9C9C9C",
        "Static-2R": "#59A14F",
        "IDQN-2R": "#4E79A7",
        "QMIX-2R": "#F28E2B",
        "IDQN-BC": "#4E79A7",
        "QMIX-BC": "#F28E2B",
        "IDQN-StayReg": "#4E79A7",
        "QMIX-StayReg": "#F28E2B",
        "IDQN-BC+StayReg": "#4E79A7",
        "QMIX-BC+StayReg": "#F28E2B",
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
    methods = method_order(args)
    width = min(0.2, 0.8 / max(1, len(methods)))
    center = (len(methods) - 1) / 2
    for offset, method in enumerate(methods):
        values = []
        errors = []
        for scenario in scenarios:
            row = next(
                row for row in summary_rows
                if row["scenario_id"] == scenario and row["method"] == method
            )
            values.append(safe_float(row["mean_t80"]) or np.nan)
            errors.append(safe_float(row["std_t80"]) or 0.0)
        ax.bar(x + (offset - center) * width, values, width, yerr=errors, label=method, color=colors[method])
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--idqn-lr", type=float, default=None)
    parser.add_argument("--qmix-lr", type=float, default=None)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.97)
    parser.add_argument("--idqn-epsilon-start", type=float, default=None)
    parser.add_argument("--idqn-epsilon-min", type=float, default=None)
    parser.add_argument("--idqn-epsilon-decay", type=float, default=None)
    parser.add_argument("--qmix-epsilon-start", type=float, default=None)
    parser.add_argument("--qmix-epsilon-min", type=float, default=None)
    parser.add_argument("--qmix-epsilon-decay", type=float, default=None)
    parser.add_argument("--stay-regularization-weight", type=float, default=0.0)
    parser.add_argument("--idqn-stay-regularization-weight", type=float, default=None)
    parser.add_argument("--qmix-stay-regularization-weight", type=float, default=None)
    parser.add_argument("--bc-pretrain-steps", type=int, default=0)
    parser.add_argument("--idqn-bc-pretrain-steps", type=int, default=None)
    parser.add_argument("--qmix-bc-pretrain-steps", type=int, default=None)
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
            run_no_robot(args, scenario)
            run_static(args, scenario)
            run_idqn(args, scenario, train_seed)
            run_qmix(args, scenario, train_seed)
    elif not args.collect_only:
        jobs = []
        for idx, scenario in enumerate(args.scenarios):
            train_seed = args.train_seed_base + idx * 1000
            jobs.extend(
                [
                    (f"no-robot:{scenario}", run_no_robot, (args, scenario)),
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
