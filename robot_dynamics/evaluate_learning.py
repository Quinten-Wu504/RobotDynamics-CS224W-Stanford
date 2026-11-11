import argparse
import hashlib
import json
from pathlib import Path

import h5py
import torch

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.learning_data import make_arrays, error_rows
from robot_dynamics.learning_models import DynamicsPredictor
from robot_dynamics.train_learning import setup, tensors, predict, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path("reports/stage3/manifest.json")
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    identity = checkpoint["identity"]
    cfg = identity["config"]
    kind = identity["model"]
    if (
        file_sha256(args.cache) != identity["cache_sha256"]
        or file_sha256(args.manifest) != identity["manifest_sha256"]
    ):
        raise ValueError("Checkpoint data provenance mismatch")
    for name, digest in identity["code"].items():
        if file_sha256(Path("robot_dynamics") / name) != digest:
            raise ValueError("Checkpoint code mismatch")
    setup(cfg["seed"], cfg["cpu_threads"])
    manifest = json.loads(args.manifest.read_text())
    train_ids = manifest["d0"]["train_subsets"][cfg["train_fraction"]][
        : cfg["train_episodes"]
    ]
    ids = train_ids if cfg["mode"] == "fit" else manifest["d0"]["validation"]
    stats = json.loads((args.checkpoint.parent / "normalization.json").read_text())
    if (
        file_sha256(args.checkpoint.parent / "normalization.json")
        != identity["normalization_sha256"]
    ):
        raise ValueError("Checkpoint normalizer checksum mismatch")
    if stats["train_episode_ids"] != train_ids:
        raise ValueError("Incorrect training normalizer")
    with h5py.File(args.cache, "r") as cache:
        arrays = make_arrays(cache, ids, stats, cfg["history"], cfg["horizons"])
    model = DynamicsPredictor(
        kind, arrays["state"].shape[-1], arrays["context"].shape[-1], cfg["history"]
    ).to(cfg["device"])
    model.load_state_dict(checkpoint["model"])
    prediction = predict(model, tensors(arrays, cfg["device"]))
    result = {
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
        "summary": aggregate(
            error_rows(prediction, arrays, stats["position_scale"], kind)
        ),
        **source_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
