import hashlib
import json
from pathlib import Path

import yaml

from robot_dynamics.data_audit import file_sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def matrix(protocol_path, manifest_path=Path("reports/stage3/manifest.json")):
    protocol_path, manifest_path = Path(protocol_path), Path(manifest_path)
    protocol = yaml.safe_load(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    digest = file_sha256(protocol_path)
    rows = []
    for fraction in protocol["fractions"]:
        count = len(manifest["d0"]["train_subsets"][fraction])
        if protocol["purpose"] == "workflow_check":
            count = protocol["check_train_episodes"]
        for seed in protocol["seeds"]:
            variants = [
                (model, method, "core")
                for model in protocol["core_models"]
                for method in protocol["core_methods"]
            ]
            variants += [
                (model, "supervised", "non_graph_control")
                for model in protocol["non_graph_models"]
            ]
            if fraction in protocol["state_control_fractions"]:
                variants += [
                    (model, "state", "state_control")
                    for model in protocol["core_models"]
                ]
            for model, method, role in variants:
                config = {
                    **protocol["training"],
                    "seed": seed,
                    "train_fraction": fraction,
                    "train_episodes": count,
                    "formal_protocol_sha256": digest,
                }
                rows.append(
                    {
                        "run_id": f"{model}_{method}_n{count}_s{seed}",
                        "model": model,
                        "method": method,
                        "role": role,
                        "config": config,
                    }
                )
    identifiers = [r["run_id"] for r in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate formal run identifiers")
    return {
        "protocol_sha256": digest,
        "manifest_sha256": file_sha256(manifest_path),
        "runs": rows,
    }


def validate_completed(run, directory):
    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text())
    if (
        metrics["model"] != run["model"]
        or metrics["method"] != run["method"]
        or metrics["config"] != run["config"]
    ):
        raise ValueError(
            f"Completed run differs from frozen configuration: {directory}"
        )
    if (
        metrics["logs"][-1]["epoch"]
        != run["config"]["pretrain_epochs"] + run["config"]["finetune_epochs"]
    ):
        raise ValueError("Incomplete training cannot be sealed")
    if file_sha256(directory / "best.pt") != metrics["best_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    if (
        file_sha256(directory / "normalization.json")
        != metrics["identity"]["normalization_sha256"]
    ):
        raise ValueError("Training normalizer changed")
    return metrics


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
