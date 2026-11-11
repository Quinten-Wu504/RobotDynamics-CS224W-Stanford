import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import psutil
import torch
import yaml

from robot_dynamics.baselines import mean_interval
from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.learning_data import make_arrays, error_rows
from robot_dynamics.learning_models import DynamicsPredictor
from robot_dynamics.normalization import fit_statistics
from robot_dynamics.splits import validate_split


def setup(seed, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False


def tensors(arrays, device):
    return {
        k: torch.from_numpy(arrays[k]).to(device)
        for k in ("state", "context", "target")
    }


def predict(model, data, batch=4096):
    model.eval()
    with torch.no_grad():
        return (
            torch.cat(
                [
                    model(data["state"][i : i + batch], data["context"][i : i + batch])
                    for i in range(0, len(data["state"]), batch)
                ]
            )
            .cpu()
            .numpy()
        )


def aggregate(rows):
    result = []
    for entity in ("end_effector", "square_nut"):
        for k in (1, 5, 10):
            selected = [
                r for r in rows if r["entity"] == entity and r["horizon_steps"] == k
            ]
            result.append(
                {
                    "entity": entity,
                    "horizon_steps": k,
                    "episodes": len(selected),
                    "windows": sum(r["windows"] for r in selected),
                    **mean_interval([r["mean_error_cm"] for r in selected]),
                }
            )
    return result


def save_checkpoint(path, model, optimizer, epoch, best, logs, generator, identity):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_score": best,
        "logs": logs,
        "generator": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "identity": identity,
        **source_version(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path("reports/stage3/manifest.json")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=[
            "current_mlp",
            "history_mlp",
            "gru",
            "gnn",
            "temporal_gnn",
            "temporal_no_messages",
            "temporal_full_edges",
        ],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    if args.resume and args.resume.resolve().parent != args.output.resolve():
        raise ValueError(
            "Resume in the original run directory to preserve best checkpoint and logs"
        )
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(args.config.read_text())
    if (
        cfg["mode"] not in ("fit", "pilot")
        or cfg["history"] != 4
        or cfg["horizons"] != [1, 5, 10]
    ):
        raise ValueError("Expected the fixed fit/pilot window protocol")
    setup(cfg["seed"], cfg["cpu_threads"])
    device = torch.device(cfg["device"])
    manifest = json.loads(args.manifest.read_text())
    expected = manifest["datasets"]["square_d0"]
    if file_sha256(args.cache) != expected["cache_sha256"]:
        raise ValueError("Cache checksum mismatch")
    pool = manifest["d0"]["train_subsets"][cfg["train_fraction"]]
    if not 0 < cfg["train_episodes"] <= len(pool):
        raise ValueError("Invalid training episode count")
    train_ids = pool[: cfg["train_episodes"]]
    eval_ids = train_ids if cfg["mode"] == "fit" else manifest["d0"]["validation"]
    start = time.perf_counter()
    with h5py.File(args.cache, "r") as cache:
        validate_split(manifest["d0"], list(cache))
        if not cache.attrs["complete"]:
            raise ValueError("Incomplete cache")
        stats = fit_statistics(cache, train_ids)
        train = make_arrays(cache, train_ids, stats, cfg["history"], cfg["horizons"])
        validation = (
            train
            if cfg["mode"] == "fit"
            else make_arrays(cache, eval_ids, stats, cfg["history"], cfg["horizons"])
        )
    loading_seconds = time.perf_counter() - start
    (args.output / "normalization.json").write_text(json.dumps(stats, indent=2) + "\n")
    train_data = tensors(train, device)
    val_data = train_data if cfg["mode"] == "fit" else tensors(validation, device)
    model = DynamicsPredictor(
        args.model, train["state"].shape[-1], train["context"].shape[-1], cfg["history"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    identity = {
        "normalization_sha256": file_sha256(args.output / "normalization.json"),
        "config": cfg,
        "model": args.model,
        "manifest_sha256": file_sha256(args.manifest),
        "cache_sha256": expected["cache_sha256"],
        "code": {
            p: file_sha256(Path("robot_dynamics") / p)
            for p in (
                "learning_data.py",
                "learning_models.py",
                "train_learning.py",
                "windows.py",
                "normalization.py",
                "graph_models.py",
            )
        },
    }
    generator = torch.Generator().manual_seed(cfg["seed"])
    logs = []
    best = float("inf")
    first_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity:
            raise ValueError("Checkpoint configuration, source, or data mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["generator"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        first_epoch = checkpoint["epoch"]
        best = checkpoint["best_score"]
        logs = checkpoint["logs"]
    initial_prediction = predict(model, train_data)
    initial_mse = float(np.mean((initial_prediction - train["target"]) ** 2))
    final_epoch = min(cfg["epochs"], args.stop_after_epoch or cfg["epochs"])
    if final_epoch <= first_epoch:
        raise ValueError("No new epoch requested")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_start = time.perf_counter()
    steps = 0
    max_rss = psutil.Process().memory_info().rss
    for epoch in range(first_epoch + 1, final_epoch + 1):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        order = torch.randperm(len(train["state"]), generator=generator)
        for indices in order.split(cfg["batch_size"]):
            indices = indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(train_data["state"][indices], train_data["context"][indices])
            loss = torch.nn.functional.mse_loss(pred, train_data["target"][indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            steps += 1
        prediction = predict(model, val_data)
        rows = error_rows(prediction, validation, stats["position_scale"], args.model)
        score = float(np.mean([r["mean_error_cm"] for r in rows]))
        logs.append(
            {
                "epoch": epoch,
                "train_mse": loss_sum / len(train["state"]),
                "selection_cm": score,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )
        if score < best:
            best = score
            save_checkpoint(
                args.output / "best.pt",
                model,
                optimizer,
                epoch,
                best,
                logs,
                generator,
                identity,
            )
        save_checkpoint(
            args.output / "last.pt",
            model,
            optimizer,
            epoch,
            best,
            logs,
            generator,
            identity,
        )
        max_rss = max(max_rss, psutil.Process().memory_info().rss)
        print(json.dumps(logs[-1]), flush=True)
    best_checkpoint = torch.load(
        args.output / "best.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(best_checkpoint["model"])
    prediction = predict(model, val_data)
    restored = DynamicsPredictor(
        args.model, train["state"].shape[-1], train["context"].shape[-1], cfg["history"]
    ).to(device)
    restored.load_state_dict(best_checkpoint["model"])
    reloaded = predict(restored, val_data)
    reload_error = float(np.max(np.abs(prediction - reloaded)))
    if reload_error != 0:
        raise ValueError("Reloaded checkpoint predictions differ")
    train_pred = predict(model, train_data)
    rows = error_rows(prediction, validation, stats["position_scale"], args.model)
    with (args.output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "environment": {
            "torch": str(torch.__version__),
            "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            "dtype": "float32",
            "deterministic_algorithms": True,
            "cudnn_enabled": False,
        },
        "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
        "model": args.model,
        "config": cfg,
        "identity": identity,
        "train_ids": train_ids,
        "evaluation_ids": eval_ids,
        "evaluation_role": "training_fit" if cfg["mode"] == "fit" else "d0_validation",
        "train_windows": len(train["state"]),
        "evaluation_windows": len(validation["state"]),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_checkpoint["epoch"],
        "best_score_cm": best,
        "initial_train_mse_this_invocation": initial_mse,
        "best_train_mse": float(np.mean((train_pred - train["target"]) ** 2)),
        "train_summary": aggregate(
            error_rows(train_pred, train, stats["position_scale"], args.model)
        ),
        "summary": aggregate(rows),
        "reload_max_abs_difference": reload_error,
        "best_checkpoint_sha256": file_sha256(args.output / "best.pt"),
        "logs": logs,
        "resources": {
            "loading_seconds": loading_seconds,
            "run_seconds_this_invocation": time.perf_counter() - run_start,
            "optimizer_steps_this_invocation": steps,
            "optimizer_steps_total": final_epoch
            * int(np.ceil(len(train["state"]) / cfg["batch_size"])),
            "max_observed_rss_bytes": max_rss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else 0,
        },
        "resumed_from": str(args.resume) if args.resume else None,
        **source_version(),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
