import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.formal import write_json
from robot_dynamics.train_grail import code_hashes


def wait_for_gpu(state, state_path):
    announced = False
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        pids = result.stdout.strip().splitlines()
        memory = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        if not pids and int(memory.stdout.strip()) >= 8000:
            return
        state.update(status="waiting_for_gpu", other_gpu_pids=pids)
        write_json(state_path, state)
        if not announced:
            print(
                "Waiting for existing GPU work to finish; no other process will be stopped.",
                flush=True,
            )
            announced = True
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/grail_protocol.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = source_version()
    if source["working_tree_dirty"]:
        raise ValueError("Commit code before launching the frozen queue")
    protocol = yaml.safe_load(args.protocol.read_text())
    plan = {
        "protocol_sha256": file_sha256(args.protocol),
        "manifest_sha256": file_sha256(args.manifest),
        "cache_sha256": file_sha256(args.cache),
        "training_code": code_hashes(),
        "runner_sha256": file_sha256(Path(__file__)),
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
    if (
        len({r["run_id"] for r in plan["runs"]}) != len(plan["runs"])
        or not plan["runs"]
    ):
        raise ValueError("Empty or duplicate runs")
    plan_path = args.output / "matrix.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen experiment inputs changed")
    write_json(plan_path, plan)
    state_path = args.output / "progress.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"completed": [], "events": [], "used_seconds": 0.0}
    )
    state.update(source=source, test_evaluation_started=False)
    configs = args.output / "configs"
    configs.mkdir(exist_ok=True)
    selected = []
    for run in plan["runs"]:
        if (
            code_hashes() != plan["training_code"]
            or file_sha256(args.protocol) != plan["protocol_sha256"]
            or file_sha256(args.manifest) != plan["manifest_sha256"]
            or file_sha256(args.cache) != plan["cache_sha256"]
        ):
            raise ValueError("Code or data changed during queue")
        name = run["run_id"]
        directory = args.output / name
        cfg = configs / f"{name}.yaml"
        cfg.write_text(yaml.safe_dump(run["config"]))
        state["active_run"] = name
        if run["config"]["device"] == "cuda":
            wait_for_gpu(state, state_path)
        state.pop("other_gpu_pids", None)
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
            "-u",
            "-m",
            "robot_dynamics.train_grail",
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
        actions = []
        if not (directory / "metrics.json").exists():
            actions.append(
                (
                    "resume" if (directory / "last.pt").exists() else "start",
                    command
                    + (["--resume"] if (directory / "last.pt").exists() else []),
                )
            )
        actions.append(("reload", command + ["--reload-only"]))
        for action, cmd in actions:
            state.update(
                status="training" if action != "reload" else "checking_validation"
            )
            state["events"].append(
                {"run": name, "event": action, "unix_time": time.time()}
            )
            write_json(state_path, state)
            start = time.monotonic()
            with (args.output / f"{name}_{action}.log").open("a") as log:
                # Keep the runner lock held even if the parent process dies first.
                child = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),)
                )
                state["active_pid"] = child.pid
                write_json(state_path, state)
                code = child.wait()
            state["used_seconds"] += time.monotonic() - start
            state.pop("active_pid", None)
            if code:
                state.update(status="failed", exit_code=code)
                write_json(state_path, state)
                raise RuntimeError(
                    f"{name} {action} failed with exit {code}; inspect log"
                )
        result = json.loads((directory / "metrics.json").read_text())
        reloaded = json.loads((directory / "reload.json").read_text())
        if (
            result["config"] != run["config"]
            or result["model"] != run["model"]
            or result["logs"][-1]["epoch"] != run["config"]["epochs"]
            or not reloaded["passed"]
        ):
            raise ValueError("Incomplete or mismatched result")
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
        print(
            f"Completed {name}: {len(state['completed'])}/{len(plan['runs'])}",
            flush=True,
        )
    write_json(
        args.output / "selection_seal.json",
        {
            "matrix_sha256": file_sha256(plan_path),
            "runs": selected,
            "source": source,
            "sealed_unix_time": time.time(),
        },
    )
    state.pop("active_run", None)
    state.update(status="training_complete", test_evaluation_started=False)
    write_json(state_path, state)
    print(
        "All training and validation reloads complete. Test evaluation has not started.",
        flush=True,
    )


if __name__ == "__main__":
    main()
