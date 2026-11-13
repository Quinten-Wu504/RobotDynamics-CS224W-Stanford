import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import psutil
import torch
import yaml

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.formal import write_json
from robot_dynamics.grail_data import validate_manifest
from robot_dynamics.grail_data import fit_statistics, make_arrays, error_rows
from robot_dynamics.grail_models import GrailPredictor
from robot_dynamics.train_learning import setup, tensors, predict
from robot_dynamics.grail_checkpoint import save_checkpoint


def code_hashes():
    return {
        name: file_sha256(Path("robot_dynamics") / name)
        for name in (
            "grail_data.py",
            "grail_models.py",
            "train_grail.py",
            "grail_checkpoint.py",
            "train_learning.py",
            "formal.py",
        )
    }


def load_data(cache_path, manifest_path, cfg):
    manifest = json.loads(manifest_path.read_text())
    if file_sha256(cache_path) != manifest["cache_sha256"]:
        raise ValueError("GRAIL cache differs from audited manifest")
    train_ids = manifest["splits"]["train"][: cfg["train_episodes"]]
    val_ids = (
        train_ids
        if cfg["purpose"] == "fit"
        else manifest["splits"]["validation"][: cfg["validation_episodes"]]
    )
    if len(train_ids) != cfg["train_episodes"] or len(val_ids) != (
        cfg["train_episodes"] if cfg["purpose"] == "fit" else cfg["validation_episodes"]
    ):
        raise ValueError("Requested episode count is unavailable")
    with h5py.File(cache_path, "r") as cache:
        validate_manifest(manifest, cache)
        stats = fit_statistics(cache, train_ids)
        train = make_arrays(cache, train_ids, stats)
        val = train if cfg["purpose"] == "fit" else make_arrays(cache, val_ids, stats)
    return stats, train, val


def reload_validation(cache_path, manifest_path, directory, device):
    result = json.loads((directory / "metrics.json").read_text())
    identity = result["identity"]
    if (
        identity["code"] != code_hashes()
        or file_sha256(manifest_path) != identity["manifest_sha256"]
    ):
        raise ValueError("Source or manifest changed since training")
    if (
        file_sha256(cache_path) != identity["cache_sha256"]
        or file_sha256(directory / "best.pt") != result["best_checkpoint_sha256"]
    ):
        raise ValueError("Data or checkpoint changed")
    if (
        file_sha256(directory / "normalization.json")
        != identity["normalization_sha256"]
    ):
        raise ValueError("Normalizer changed")
    stats = json.loads((directory / "normalization.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    expected_ids = (
        manifest["splits"]["train"][: result["config"]["train_episodes"]]
        if result["config"]["purpose"] == "fit"
        else manifest["splits"]["validation"][: result["config"]["validation_episodes"]]
    )
    if result["evaluation_ids"] != expected_ids:
        raise ValueError("Unexpected validation IDs")
    with h5py.File(cache_path, "r") as cache:
        val = make_arrays(cache, expected_ids, stats)
    checkpoint = torch.load(
        directory / "best.pt", weights_only=True, map_location="cpu"
    )
    if checkpoint["identity"] != identity:
        raise ValueError("Checkpoint identity changed")
    model = GrailPredictor(result["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    prediction = predict(model, tensors(val, device))
    digest = hashlib.sha256(prediction.tobytes()).hexdigest()
    if digest != result["prediction_sha256"]:
        raise ValueError("Independent validation reload differs")
    return {
        "passed": True,
        "prediction_sha256": digest,
        "best_checkpoint_sha256": result["best_checkpoint_sha256"],
        **source_version(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["history_mlp", "gru", "temporal_gnn", "temporal_no_messages"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reload-only", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if (
        cfg["purpose"] not in ("fit", "check", "resource_probe", "formal")
        or not 1 <= cfg["selection_start"] <= cfg["epochs"]
    ):
        raise ValueError("Invalid selection protocol")
    if cfg["purpose"] == "formal" and (
        cfg["train_episodes"],
        cfg["validation_episodes"],
        cfg["epochs"],
        cfg["selection_start"],
    ) != (168, 36, 40, 11):
        raise ValueError("Unexpected formal GRAIL scope")
    setup(cfg["seed"], cfg["threads"])
    device = torch.device(cfg["device"])
    if args.reload_only:
        result = json.loads((args.output / "metrics.json").read_text())
        if result["config"] != cfg or result["model"] != args.model:
            raise ValueError("Reload configuration mismatch")
        write_json(
            args.output / "reload.json",
            reload_validation(args.cache, args.manifest, args.output, device),
        )
        return
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    stats, train, val = load_data(args.cache, args.manifest, cfg)
    stats_path = args.output / "normalization.json"
    if args.resume and json.loads(stats_path.read_text()) != stats:
        raise ValueError("Resume normalizer mismatch")
    if not args.resume:
        write_json(stats_path, stats)
    identity = {
        "model": args.model,
        "config": cfg,
        "manifest_sha256": file_sha256(args.manifest),
        "cache_sha256": file_sha256(args.cache),
        "normalization_sha256": file_sha256(stats_path),
        "code": code_hashes(),
    }
    data, valid = tensors(train, device), tensors(val, device)
    loading = time.perf_counter() - start
    model = GrailPredictor(args.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    generator = torch.Generator().manual_seed(cfg["seed"])
    logs, best, first_epoch = [], float("inf"), 0
    if args.resume:
        checkpoint = torch.load(
            args.output / "last.pt", weights_only=True, map_location="cpu"
        )
        if checkpoint["identity"] != identity:
            raise ValueError("Checkpoint source/data/configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["generator"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        first_epoch, best, logs = (
            checkpoint["epoch"],
            checkpoint["best_score"],
            checkpoint["logs"],
        )
    final_epoch = min(cfg["epochs"], args.stop_after_epoch or cfg["epochs"])
    if final_epoch < first_epoch:
        raise ValueError("Requested stop precedes restored epoch")
    initial_mse = float(np.mean((predict(model, data) - train["target"]) ** 2))
    run_start = time.perf_counter()
    max_rss = psutil.Process().memory_info().rss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch + 1, final_epoch + 1):
        model.train()
        update_start = time.perf_counter()
        loss_sum = 0.0
        for indices in torch.randperm(len(train["state"]), generator=generator).split(
            cfg["batch_size"]
        ):
            indices = indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(data["state"][indices], data["context"][indices])
            loss = torch.nn.functional.mse_loss(pred, data["target"][indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        update_seconds = time.perf_counter() - update_start
        prediction = predict(model, valid)
        rows = error_rows(prediction, val, stats)
        score = float(np.mean([r["mean_error_cm"] for r in rows]))
        logs.append(
            {
                "epoch": epoch,
                "train_mse": loss_sum / len(train["state"]),
                "validation_cm": score,
                "update_seconds": update_seconds,
                "validation_seconds": time.perf_counter()
                - update_start
                - update_seconds,
            }
        )
        if epoch >= cfg["selection_start"] and score < best:
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
    if final_epoch < cfg["epochs"]:
        return
    checkpoint = torch.load(
        args.output / "best.pt", weights_only=True, map_location="cpu"
    )
    model.load_state_dict(checkpoint["model"])
    prediction = predict(model, valid)
    rows = error_rows(prediction, val, stats)
    with (args.output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output / "metrics.json",
        {
            "model": args.model,
            "config": cfg,
            "identity": identity,
            "train_ids": train["ids"],
            "evaluation_ids": val["ids"],
            "evaluation_role": "training_fit"
            if cfg["purpose"] == "fit"
            else "asset_disjoint_validation",
            "train_windows": len(train["state"]),
            "validation_windows": len(val["state"]),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "best_epoch": checkpoint["epoch"],
            "best_score_cm": checkpoint["best_score"],
            "initial_train_mse_this_invocation": initial_mse,
            "best_train_mse": float(
                np.mean((predict(model, data) - train["target"]) ** 2)
            ),
            "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
            "best_checkpoint_sha256": file_sha256(args.output / "best.pt"),
            "logs": logs,
            "optimizer_steps_total": cfg["epochs"]
            * int(np.ceil(len(train["state"]) / cfg["batch_size"])),
            "resources": {
                "loading_seconds": loading,
                "run_seconds_this_invocation": time.perf_counter() - run_start,
                "max_observed_rss_bytes": max_rss,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else 0,
            },
            "environment": {
                "torch": str(torch.__version__),
                "device": str(device),
                "deterministic": True,
                "dtype": "float32",
            },
            **source_version(),
        },
    )


if __name__ == "__main__":
    main()
