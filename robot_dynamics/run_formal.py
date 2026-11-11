import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.formal import matrix, validate_completed, write_json, stable_digest


def execute(command, log_path, state, state_path, budget_seconds):
    started = time.monotonic()
    used = state["used_seconds"]
    with log_path.open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        state["active_pid"] = process.pid
        write_json(state_path, state)
        while process.poll() is None:
            remaining = (
                budget_seconds - used - (time.monotonic() - started)
                if budget_seconds is not None
                else None
            )
            if remaining is not None and remaining <= 0:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                state["status"] = "budget_exhausted"
                break
            try:
                process.wait(
                    timeout=min(15, remaining) if remaining is not None else 15
                )
            except subprocess.TimeoutExpired:
                pass
            state["used_seconds"] = used + time.monotonic() - started
            write_json(state_path, state)
    state["used_seconds"] = used + time.monotonic() - started
    state.pop("active_pid", None)
    write_json(state_path, state)
    return process.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/formal_protocol.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--max-hours", type=float)
    budget.add_argument("--complete-matrix", action="store_true")
    args = parser.parse_args()
    if args.max_hours is not None and args.max_hours <= 0:
        raise ValueError("A positive execution budget is required")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = matrix(args.protocol)
    plan_path = args.output / "matrix.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("The frozen matrix changed")
    write_json(plan_path, plan)
    if (args.output / "selection_seal.json").exists():
        for run in plan["runs"]:
            validate_completed(run, args.output / run["run_id"])
    state_path = args.output / "progress.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"used_seconds": 0.0, "completed": [], "events": []}
    )
    if state.get("active_pid"):
        command_file = Path(f"/proc/{state['active_pid']}/cmdline")
        if (
            command_file.exists()
            and str(args.output).encode() in command_file.read_bytes()
        ):
            raise RuntimeError("A previous training child is still running")
        state.pop("active_pid")
    state.update(
        status="training",
        budget_seconds=args.max_hours * 3600 if args.max_hours is not None else None,
        **source_version(),
    )
    if state["working_tree_dirty"]:
        raise ValueError("Commit the formal code before starting the queue")
    (args.output / "configs").mkdir(exist_ok=True)
    for run in plan["runs"]:
        if file_sha256(args.protocol) != plan["protocol_sha256"]:
            raise ValueError("Protocol changed during execution")
        name = run["run_id"]
        directory = args.output / name
        state["active_run"] = name
        if not (directory / "metrics.json").exists():
            if (
                state["budget_seconds"] is not None
                and state["used_seconds"] >= state["budget_seconds"]
            ):
                state["status"] = "budget_exhausted"
                write_json(state_path, state)
                return
            config_path = args.output / "configs" / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(run["config"]))
            command = [
                sys.executable,
                "-m",
                "robot_dynamics.train_pretraining",
                str(args.cache),
                "--config",
                str(config_path),
                "--model",
                run["model"],
                "--method",
                run["method"],
                "--output",
                str(directory),
            ]
            if (directory / "last.pt").exists():
                command += ["--resume", str(directory / "last.pt")]
            state["events"].append(
                {
                    "run": name,
                    "event": "start_or_resume",
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            code = execute(
                command,
                args.output / f"{name}.log",
                state,
                state_path,
                state["budget_seconds"],
            )
            if code:
                if state["status"] != "budget_exhausted":
                    state["status"] = "failed"
                state["events"].append(
                    {"run": name, "event": state["status"], "exit_code": code}
                )
                write_json(state_path, state)
                raise SystemExit(code)
        metrics = validate_completed(run, directory)
        reload_path = args.output / f"{name}_reload.json"
        if not reload_path.exists():
            command = [
                sys.executable,
                "-m",
                "robot_dynamics.evaluate_learning",
                str(args.cache),
                "--checkpoint",
                str(directory / "best.pt"),
                "--output",
                str(reload_path),
            ]
            code = execute(
                command,
                args.output / f"{name}_reload.log",
                state,
                state_path,
                state["budget_seconds"],
            )
            if code:
                state["status"] = (
                    "failed"
                    if state["status"] != "budget_exhausted"
                    else state["status"]
                )
                write_json(state_path, state)
                raise SystemExit(code)
        reloaded = json.loads(reload_path.read_text())
        if (
            reloaded["prediction_sha256"] != metrics["prediction_sha256"]
            or reloaded["summary"] != metrics["summary"]
        ):
            raise ValueError("Independent validation reload differs")
        if name not in state["completed"]:
            state["completed"].append(name)
        write_json(state_path, state)
        print(
            name,
            "complete",
            len(state["completed"]),
            "/",
            len(plan["runs"]),
            flush=True,
        )
    selected = []
    for run in plan["runs"]:
        metrics = validate_completed(run, args.output / run["run_id"])
        selected.append(
            {
                "run_id": run["run_id"],
                "checkpoint_sha256": metrics["best_checkpoint_sha256"],
                "validation_prediction_sha256": metrics["prediction_sha256"],
                "normalization_sha256": metrics["identity"]["normalization_sha256"],
                "best_epoch": metrics["best_epoch"],
            }
        )
    seal = {
        "matrix_sha256": stable_digest(plan),
        "selected": selected,
        "test_metrics_consulted": False,
        **source_version(),
    }
    seal_path = args.output / "selection_seal.json"
    if seal_path.exists():
        previous = json.loads(seal_path.read_text())
        if (
            previous["matrix_sha256"] != seal["matrix_sha256"]
            or previous["selected"] != selected
        ):
            raise ValueError("Existing selection seal differs")
    else:
        write_json(seal_path, seal)
    state.update(status="selected", active_run=None)
    write_json(state_path, state)


if __name__ == "__main__":
    main()
