import argparse
import json
import math
from pathlib import Path

import yaml


def horizon_steps(seconds, frequency):
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("Control frequency must be finite and positive")
    steps = []
    for duration in seconds:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Prediction durations must be finite and positive")
        count = duration * frequency
        if not math.isclose(count, round(count), rel_tol=0, abs_tol=1e-8):
            raise ValueError(
                f"{duration} s is not a whole number of steps at {frequency} Hz"
            )
        if round(count) < 1:
            raise ValueError("Prediction horizons must span at least one step")
        steps.append(round(count))
    return steps


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["history_steps"] < 1 or not isinstance(config["history_steps"], int):
        raise ValueError("history_steps must be a positive integer")
    if config["action_conditioning"] != "planned_sequence":
        raise ValueError("This protocol requires the planned action sequence")
    if config["position_unit"] != "m" or config["coordinate_frame"] != "world":
        raise ValueError("Canonical positions must use world coordinates in metres")
    split = config["split"]
    if split["unit"] != "episode" or split["d1_role"] != "final_test_only":
        raise ValueError("Use episode splits and reserve D1 for final testing")
    fractions = split["d0_fractions"]
    if len(fractions) != 3 or any(not 0 < f < 1 for f in fractions):
        raise ValueError("D0 needs positive train, validation, and test fractions")
    if not math.isclose(sum(fractions), 1):
        raise ValueError("D0 split fractions must sum to one")
    if config["control_frequency_hz"] is not None:
        horizon_steps(config["horizons_seconds"], config["control_frequency_hz"])
    return config


def unresolved_audit(config):
    required = (
        "observation_fields",
        "controller_semantics",
        "time_alignment",
        "fixture_world_poses",
    )
    pending = [name for name in required if config["audit"].get(name) is not True]
    if config["control_frequency_hz"] is None:
        pending.append("control_frequency_hz")
    return pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/square.yaml"))
    parser.add_argument("--require-audited", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    pending = unresolved_audit(config)
    print(json.dumps({"config": config, "pending_audit": pending}, indent=2))
    if args.require_audited and pending:
        parser.exit(1, "Data audit is incomplete: " + ", ".join(pending) + "\n")


if __name__ == "__main__":
    main()
