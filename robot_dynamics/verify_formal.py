import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from robot_dynamics.baselines import mean_interval
from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.evaluate_formal import check_source
from robot_dynamics.formal import matrix, stable_digest, validate_completed, write_json
from robot_dynamics.learning_data import make_arrays
from robot_dynamics.learning_models import DynamicsPredictor
from robot_dynamics.train_learning import setup, tensors, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--d0-cache", type=Path, required=True)
    parser.add_argument("--d1-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.runs
    plan = matrix("configs/formal_protocol.yaml")
    assert json.loads((root / "matrix.json").read_text()) == plan
    assert len(plan["runs"]) == 60
    seal = json.loads((root / "selection_seal.json").read_text())
    assert seal["matrix_sha256"] == stable_digest(plan)
    progress = json.loads((root / "progress.json").read_text())
    assert set(progress["completed"]) == {r["run_id"] for r in plan["runs"]}
    assert len(progress["events"]) == 60
    assert all(e["event"] == "start_or_resume" for e in progress["events"])
    assert (root / "pipeline.exit").read_text().strip() == "0"
    complete = json.loads((root / "evaluation/complete.json").read_text())
    assert complete["prediction_sets"] == 124
    assert not complete["identity"]["validation_only_check"]
    assert complete["identity"]["selection_seal_sha256"] == file_sha256(
        root / "selection_seal.json"
    )
    assert (root / "selection_seal.json").stat().st_mtime <= (
        root / "evaluation/access.json"
    ).stat().st_mtime
    manifest = json.loads(Path("reports/stage3/manifest.json").read_text())
    caches = {"d0_test": args.d0_cache, "d1_final_test": args.d1_cache}
    ids = {
        "d0_test": manifest["d0"]["test"],
        "d1_final_test": manifest["d1_final_test"],
    }
    for split, key in [("d0_test", "square_d0"), ("d1_final_test", "square_d1")]:
        assert file_sha256(caches[split]) == manifest["datasets"][key]["cache_sha256"]
    selected = {r["run_id"]: r for r in seal["selected"]}
    for run in plan["runs"]:
        directory = root / run["run_id"]
        metrics = validate_completed(run, directory)
        assert (
            metrics["train_ids"]
            == manifest["d0"]["train_subsets"][run["config"]["train_fraction"]]
        )
        assert metrics["evaluation_ids"] == manifest["d0"]["validation"]
        assert (
            selected[run["run_id"]]["checkpoint_sha256"]
            == metrics["best_checkpoint_sha256"]
        )
        assert (
            selected[run["run_id"]]["validation_prediction_sha256"]
            == metrics["prediction_sha256"]
        )
        assert (
            file_sha256(directory / "pretrained.pt")
            == metrics["pretrained_checkpoint_sha256"]
        )
        cp = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        check_source(cp)
        assert cp["identity"] == metrics["identity"]
        assert 11 <= cp["epoch"] <= 40
        reload = json.loads((root / f"{run['run_id']}_reload.json").read_text())
        assert (
            reload["prediction_sha256"] == metrics["prediction_sha256"]
            and reload["summary"] == metrics["summary"]
        )
    entries = []
    jobs = [r["run_id"] for r in plan["runs"]] + ["hold", "constant_velocity"]
    for name in jobs:
        for split in caches:
            path = root / "evaluation" / f"{name}_{split}.json"
            result = json.loads(path.read_text())
            assert result["identity"] == complete["identity"]
            assert result["episode_ids"] == ids[split]
            if name in selected:
                assert (
                    result["checkpoint_sha256"] == selected[name]["checkpoint_sha256"]
                )
            rows_path = path.with_name(f"{name}_{split}_episodes.csv")
            assert file_sha256(rows_path) == result["episode_rows_sha256"]
            groups, primary = defaultdict(list), defaultdict(list)
            with rows_path.open() as stream:
                for row in csv.DictReader(stream):
                    key = (row["entity"], int(row["horizon_steps"]), row["segment"])
                    groups[key].append(
                        (
                            row["episode"],
                            int(row["windows"]),
                            float(row["mean_error_cm"]),
                        )
                    )
                    if row["segment"] == "all":
                        primary[row["episode"]].append(float(row["mean_error_cm"]))
            assert set(primary) == set(ids[split]) and all(
                len(v) == 6 for v in primary.values()
            )
            for summary in result["summary"]:
                values = groups[
                    (summary["entity"], summary["horizon_steps"], summary["segment"])
                ]
                assert (
                    len({e for e, _, _ in values}) == len(values) == summary["episodes"]
                )
                assert sum(n for _, n, _ in values) == summary["windows"]
                assert np.isclose(
                    np.mean([v for _, _, v in values]),
                    summary["mean_cm"],
                    rtol=0,
                    atol=1e-12,
                )
            recomputed = mean_interval([np.mean(primary[e]) for e in ids[split]])
            assert recomputed == result["macro"]
            entries.append(
                {
                    "file": str(path.relative_to(root)),
                    "sha256": file_sha256(path),
                    "episodes": len(primary),
                }
            )
    forward_checks = []
    for name, split in [
        ("temporal_gnn_supervised_n80_s0", "d0_test"),
        ("canonical_temporal_gnn_jepa_n800_s2", "d1_final_test"),
    ]:
        run = next(r for r in plan["runs"] if r["run_id"] == name)
        setup(run["config"]["seed"], 4)
        cp = torch.load(root / name / "best.pt", map_location="cpu", weights_only=True)
        stats = json.loads((root / name / "normalization.json").read_text())
        model = DynamicsPredictor(run["model"]).cuda()
        model.load_state_dict(cp["model"])
        digest = hashlib.sha256()
        with h5py.File(caches[split], "r") as cache:
            for episode in ids[split]:
                arrays = make_arrays(cache, [episode], stats)
                digest.update(predict(model, tensors(arrays, "cuda")).tobytes())
        expected = json.loads(
            (root / "evaluation" / f"{name}_{split}.json").read_text()
        )
        assert digest.hexdigest() == expected["prediction_sha256"]
        forward_checks.append(
            {
                "run_id": name,
                "split": split,
                "episodes": len(ids[split]),
                "prediction_sha256": digest.hexdigest(),
                "exact": True,
            }
        )
    analysis = json.loads((root / "analysis/analysis.json").read_text())
    assert analysis["status"] == "complete" and not analysis["flags"]
    assert all(r["seeds"] == [0, 1, 2] for r in analysis["summary"])
    write_json(
        args.output,
        {
            "trained_configurations": 60,
            "prediction_sets": 124,
            "failed_or_retried_training_runs": 0,
            "checkpoint_source_split_normalizer_checks": True,
            "all_episode_summaries_recomputed": True,
            "selection_preceded_test_access": True,
            "analysis_sha256": file_sha256(root / "analysis/analysis.json"),
            "forward_checks": forward_checks,
            "evaluation_artifacts": entries,
            **source_version(),
        },
    )
    print(
        "Verified 60 runs, 124 evaluation sets, and two full forward reproductions",
        flush=True,
    )


if __name__ == "__main__":
    main()
