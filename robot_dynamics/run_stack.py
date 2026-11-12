import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

import yaml

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.formal import write_json
from robot_dynamics.run_formal import execute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/stack_protocol.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = source_version()
    code_files = sorted(Path("robot_dynamics").glob("*.py"))
    training_code = {str(p): file_sha256(p) for p in code_files}
    if source["working_tree_dirty"]:
        raise ValueError("Commit training code before starting the queue")
    protocol = yaml.safe_load(args.protocol.read_text())
    plan = {
        "protocol_sha256": file_sha256(args.protocol),
        "manifest_sha256": file_sha256(args.manifest),
        "runs": [
            {
                "run_id": f"{model}_s{seed}",
                "model": model,
                "config": {**protocol["training"], "seed": seed},
            }
            for seed in protocol["seeds"]
            for model in protocol["models"]
        ],
    }
    if not plan["runs"] or len({r["run_id"] for r in plan["runs"]}) != len(
        plan["runs"]
    ):
        raise ValueError("Empty or duplicated run matrix")
    plan_path = args.output / "matrix.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen matrix changed")
    write_json(plan_path, plan)
    state_path = args.output / "progress.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"used_seconds": 0.0, "completed": [], "events": []}
    )
    if state.get("active_pid"):
        cmd = Path(f"/proc/{state['active_pid']}/cmdline")
        if cmd.exists() and str(args.output).encode() in cmd.read_bytes():
            raise RuntimeError("Previous child is still running")
    state.pop("active_pid", None)
    state.update(status="training", **source)
    configs = args.output / "configs"
    configs.mkdir(exist_ok=True)
    selected = []
    for run in plan["runs"]:
        if (
            file_sha256(args.protocol) != plan["protocol_sha256"]
            or {str(p): file_sha256(p) for p in code_files} != training_code
        ):
            raise ValueError("Code or protocol changed during queue")
        name = run["run_id"]
        directory = args.output / name
        cfg = configs / f"{name}.yaml"
        cfg.write_text(yaml.safe_dump(run["config"]))
        state["active_run"] = name
        if directory.exists() and not (directory / "last.pt").exists():
            archive = (
                args.output
                / "interrupted_before_checkpoint"
                / f"{name}_{time.time_ns()}"
            )
            archive.parent.mkdir(exist_ok=True)
            directory.rename(archive)
        command = [
            sys.executable,
            "-m",
            "robot_dynamics.train_stack",
            str(args.cache),
            "--manifest",
            str(args.manifest),
            "--config",
            str(cfg),
            "--model",
            run["model"],
            "--output",
            str(directory),
        ]
        if not (directory / "metrics.json").exists():
            state["events"].append(
                {
                    "run": name,
                    "event": "resume" if (directory / "last.pt").exists() else "start",
                }
            )
            code = execute(
                command + (["--resume"] if (directory / "last.pt").exists() else []),
                args.output / f"{name}.log",
                state,
                state_path,
                None,
            )
            if code:
                state.update(status="failed", exit_code=code)
                write_json(state_path, state)
                raise RuntimeError(f"Training failed: {name}; inspect its log")
        result = json.loads((directory / "metrics.json").read_text())
        if (
            result["config"] != run["config"]
            or result["model"] != run["model"]
            or result["logs"][-1]["epoch"] != run["config"]["epochs"]
        ):
            raise ValueError("Incomplete or mismatched run")
        code = execute(
            command + ["--reload-only"],
            args.output / f"{name}_reload.log",
            state,
            state_path,
            None,
        )
        if code:
            state.update(status="failed_reload", exit_code=code)
            write_json(state_path, state)
            raise RuntimeError(f"Validation reload failed: {name}")
        selected.append(
            {
                "run_id": name,
                "best_checkpoint_sha256": result["best_checkpoint_sha256"],
                "validation_prediction_sha256": result["prediction_sha256"],
                "normalization_sha256": result["identity"]["normalization_sha256"],
            }
        )
        if name not in state["completed"]:
            state["completed"].append(name)
        write_json(state_path, state)
    write_json(
        args.output / "selection_seal.json",
        {
            "matrix_sha256": file_sha256(plan_path),
            "code_sha256": training_code,
            "runs": selected,
            **source,
        },
    )
    state.pop("active_run", None)
    state.update(status="training_complete", test_evaluation_started=False)
    write_json(state_path, state)
    print(
        "Training and independent validation reloads complete. Test evaluation has not started.",
        flush=True,
    )


if __name__ == "__main__":
    main()
