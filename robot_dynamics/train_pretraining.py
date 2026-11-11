import argparse
import copy
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
from torch import nn
import yaml

from robot_dynamics.data_audit import file_sha256, source_version
from robot_dynamics.jepa import (
    JointEmbedding,
    graph_backbone,
    encode,
    encoder_inputs,
    target_arrays,
    physical_targets,
    representation_stats,
)
from robot_dynamics.learning_data import make_arrays, error_rows
from robot_dynamics.learning_models import DynamicsPredictor
from robot_dynamics.normalization import fit_statistics
from robot_dynamics.splits import validate_split
from robot_dynamics.train_learning import setup, tensors, predict, aggregate


def optimizer_for(model, auxiliary, cfg, pretraining):
    parameters = list(model.parameters())
    if pretraining and auxiliary is not None:
        parameters += [p for p in auxiliary.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )


def snapshot(
    path,
    model,
    auxiliary,
    optimizer,
    generator,
    initial_head,
    epoch,
    best,
    logs,
    diagnostics,
    identity,
    costs,
):
    state = {
        "model": model.state_dict(),
        "auxiliary": auxiliary.state_dict() if auxiliary else None,
        "optimizer": optimizer.state_dict(),
        "generator": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "initial_head": initial_head,
        "epoch": epoch,
        "best_score": best,
        "logs": logs,
        "diagnostics": diagnostics,
        "identity": identity,
        "costs": costs,
        **source_version(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def diagnose(model, auxiliary, data):
    ids = torch.linspace(
        0,
        len(data["state"]) - 1,
        min(1024, len(data["state"])),
        device=data["state"].device,
    ).long()
    s, c = data["state"][ids], data["context"][ids]
    latent = encode(model, s, c)
    result = {"online": representation_stats(latent)}
    if isinstance(auxiliary, JointEmbedding):
        fs, fc = encoder_inputs(
            model, data["future_state"][ids], data["future_context"][ids]
        )
        target = auxiliary.target.encode(fs, fc)
        result["target"] = representation_stats(target)
        result["online_target_parameter_rms"] = float(
            torch.sqrt(
                torch.cat(
                    [
                        (
                            p - dict(graph_backbone(model).named_parameters())[name]
                        ).flatten()
                        for name, p in auxiliary.target.named_parameters()
                    ]
                )
                .square()
                .mean()
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["temporal_gnn", "canonical_temporal_gnn"], required=True
    )
    parser.add_argument(
        "--method", choices=["jepa", "state", "supervised"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    if args.resume and args.resume.resolve().parent != args.output.resolve():
        raise ValueError("Resume in the original directory")
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(args.config.read_text())
    if (
        cfg["history"] != 4
        or cfg["horizons"] != [1, 5, 10]
        or cfg["mode"] not in ("fit", "pilot")
    ):
        raise ValueError("Expected the fixed window protocol")
    setup(cfg["seed"], cfg["cpu_threads"])
    manifest_path = Path("reports/stage3/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    cache_hash = file_sha256(args.cache)
    if cache_hash != manifest["datasets"]["square_d0"]["cache_sha256"]:
        raise ValueError("Cache mismatch")
    pool = manifest["d0"]["train_subsets"][cfg["train_fraction"]]
    if not 0 < cfg["train_episodes"] <= len(pool):
        raise ValueError("Invalid episode count")
    ids = pool[: cfg["train_episodes"]]
    eval_ids = ids if cfg["mode"] == "fit" else manifest["d0"]["validation"]
    started = time.perf_counter()
    with h5py.File(args.cache, "r") as cache:
        validate_split(manifest["d0"], list(cache))
        stats = fit_statistics(cache, ids)
        train = make_arrays(cache, ids, stats)
        validation = (
            train if cfg["mode"] == "fit" else make_arrays(cache, eval_ids, stats)
        )
        future = (
            target_arrays(cache, train, stats) if args.method != "supervised" else {}
        )
    loading = time.perf_counter() - started
    norm_path = args.output / "normalization.json"
    norm_path.write_text(json.dumps(stats, indent=2) + "\n")
    data = tensors(train, cfg["device"])
    data.update({k: torch.from_numpy(v).to(cfg["device"]) for k, v in future.items()})
    val_data = data if cfg["mode"] == "fit" else tensors(validation, cfg["device"])
    model = DynamicsPredictor(args.model).to(cfg["device"])
    initial_head = copy.deepcopy(graph_backbone(model).head.state_dict())
    auxiliary = None
    if args.method == "jepa":
        auxiliary = JointEmbedding(model).to(cfg["device"])
    elif args.method == "state":
        auxiliary = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 84)).to(
            cfg["device"]
        )
    identity = {
        "model": args.model,
        "method": args.method,
        "config": cfg,
        "cache_sha256": cache_hash,
        "manifest_sha256": file_sha256(manifest_path),
        "normalization_sha256": file_sha256(norm_path),
        "code": {
            name: file_sha256(Path("robot_dynamics") / name)
            for name in (
                "train_pretraining.py",
                "jepa.py",
                "graph_models.py",
                "learning_models.py",
                "equivariant_models.py",
                "learning_data.py",
                "windows.py",
                "normalization.py",
                "train_learning.py",
            )
        },
    }
    generator = torch.Generator().manual_seed(cfg["seed"])
    first, best, logs, diagnostics = 0, float("inf"), [], []
    costs = {
        "initial_phase_update_seconds": 0.0,
        "finetune_phase_update_seconds": 0.0,
        "validation_seconds": 0.0,
    }
    optimizer = optimizer_for(model, auxiliary, cfg, True)
    if args.resume:
        cp = torch.load(args.resume, map_location="cpu", weights_only=True)
        if cp["identity"] != identity:
            raise ValueError("Resume provenance mismatch")
        model.load_state_dict(cp["model"])
        if auxiliary:
            auxiliary.load_state_dict(cp["auxiliary"])
        first, best, logs, diagnostics, costs = (
            cp["epoch"],
            cp["best_score"],
            cp["logs"],
            cp["diagnostics"],
            cp["costs"],
        )
        initial_head = cp["initial_head"]
        optimizer = optimizer_for(
            model, auxiliary, cfg, first <= cfg["pretrain_epochs"]
        )
        optimizer.load_state_dict(cp["optimizer"])
        generator.set_state(cp["generator"])
        torch.set_rng_state(cp["torch_rng"])
        torch.cuda.set_rng_state_all(cp["cuda_rng"])
    else:
        diagnostics.append({"epoch": 0, **diagnose(model, auxiliary, data)})
    total_epochs = cfg["pretrain_epochs"] + cfg["finetune_epochs"]
    end = min(total_epochs, args.stop_after_epoch or total_epochs)
    if end <= first:
        raise ValueError("No new epoch requested")
    torch.cuda.reset_peak_memory_stats()
    run_start = time.perf_counter()
    max_rss = psutil.Process().memory_info().rss
    for epoch in range(first + 1, end + 1):
        pre = epoch <= cfg["pretrain_epochs"]
        if epoch == cfg["pretrain_epochs"] + 1 and args.method != "supervised":
            graph_backbone(model).head.load_state_dict(initial_head)
            optimizer = optimizer_for(model, auxiliary, cfg, False)
        model.train()
        order = torch.randperm(len(train["state"]), generator=generator)
        sums = {"loss": 0.0}
        update_start = time.perf_counter()
        for indices in order.split(cfg["batch_size"]):
            indices = indices.to(cfg["device"])
            s, c = data["state"][indices], data["context"][indices]
            optimizer.zero_grad(set_to_none=True)
            details = {}
            if pre and args.method == "jepa":
                loss, details = auxiliary.loss(
                    model,
                    s,
                    c,
                    data["future_state"][indices],
                    data["future_context"][indices],
                )
            elif pre and args.method == "state":
                target, mask = physical_targets(
                    model,
                    data["future_state"][indices],
                    data["future_context"][indices],
                )
                prediction = auxiliary(encode(model, s, c)[:, :2])
                loss = ((prediction - target).square() * mask).sum() / (
                    len(indices) * mask.sum()
                )
            else:
                loss = (model(s, c) - data["target"][indices]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            parameters = [
                p for group in optimizer.param_groups for p in group["params"]
            ]
            nn.utils.clip_grad_norm_(
                parameters, cfg["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            if pre and args.method == "jepa":
                auxiliary.update_target(model, cfg["ema_momentum"])
            for key, value in {"loss": float(loss.detach()), **details}.items():
                sums[key] = sums.get(key, 0.0) + value * len(indices)
        elapsed = time.perf_counter() - update_start
        phase = "pretrain" if pre else "downstream"
        costs[
            "initial_phase_update_seconds" if pre else "finetune_phase_update_seconds"
        ] += elapsed
        log = {
            "epoch": epoch,
            "phase": "supervised_warmup"
            if pre and args.method == "supervised"
            else phase,
            "update_seconds": elapsed,
            **{k: v / len(train["state"]) for k, v in sums.items()},
        }
        if not pre:
            val_start = time.perf_counter()
            prediction = predict(model, val_data)
            score = float(
                np.mean(
                    [
                        r["mean_error_cm"]
                        for r in error_rows(
                            prediction, validation, stats["position_scale"], args.model
                        )
                    ]
                )
            )
            costs["validation_seconds"] += time.perf_counter() - val_start
            log["selection_cm"] = score
        logs.append(log)
        if pre or epoch == end:
            diagnostics.append({"epoch": epoch, **diagnose(model, auxiliary, data)})
        if not pre and score < best:
            best = score
            snapshot(
                args.output / "best.pt",
                model,
                auxiliary,
                optimizer,
                generator,
                initial_head,
                epoch,
                best,
                logs,
                diagnostics,
                identity,
                costs,
            )
        snapshot(
            args.output / "last.pt",
            model,
            auxiliary,
            optimizer,
            generator,
            initial_head,
            epoch,
            best,
            logs,
            diagnostics,
            identity,
            costs,
        )
        if epoch == cfg["pretrain_epochs"]:
            snapshot(
                args.output / "pretrained.pt",
                model,
                auxiliary,
                optimizer,
                generator,
                initial_head,
                epoch,
                best,
                logs,
                diagnostics,
                identity,
                costs,
            )
        max_rss = max(max_rss, psutil.Process().memory_info().rss)
        print(json.dumps(log), flush=True)
    if end != total_epochs:
        return
    checkpoint = torch.load(
        args.output / "best.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    prediction = predict(model, val_data)
    fresh = DynamicsPredictor(args.model).to(cfg["device"])
    fresh.load_state_dict(checkpoint["model"])
    if not np.array_equal(prediction, predict(fresh, val_data)):
        raise ValueError("Reload mismatch")
    rows = error_rows(prediction, validation, stats["position_scale"], args.model)
    with (args.output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    batch_steps = int(np.ceil(len(train["state"]) / cfg["batch_size"]))
    result = {
        "model": args.model,
        "method": args.method,
        "config": cfg,
        "identity": identity,
        "train_ids": ids,
        "evaluation_ids": eval_ids,
        "train_windows": len(train["state"]),
        "evaluation_windows": len(validation["state"]),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "auxiliary_parameter_count": sum(p.numel() for p in auxiliary.parameters())
        if auxiliary
        else 0,
        "pretrain_steps": cfg["pretrain_epochs"] * batch_steps
        if args.method != "supervised"
        else 0,
        "downstream_steps": (
            cfg["finetune_epochs"] if args.method != "supervised" else total_epochs
        )
        * batch_steps,
        "supervised_updates_total": total_epochs * batch_steps
        if args.method == "supervised"
        else cfg["finetune_epochs"] * batch_steps,
        "best_epoch": checkpoint["epoch"],
        "best_score_cm": best,
        "summary": aggregate(rows),
        "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
        "best_checkpoint_sha256": file_sha256(args.output / "best.pt"),
        "pretrained_checkpoint_sha256": file_sha256(args.output / "pretrained.pt"),
        "logs": logs,
        "diagnostics": diagnostics,
        "resources": {
            "loading_seconds": loading,
            "run_seconds_this_invocation": time.perf_counter() - run_start,
            "max_observed_rss_bytes": max_rss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            **costs,
        },
        **source_version(),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
