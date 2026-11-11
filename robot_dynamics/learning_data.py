import numpy as np

from robot_dynamics.windows import (
    HISTORY_FIELDS,
    FIXTURE_FIELDS,
    anchors,
    read_episode,
    window,
    normalize_window,
)


def encode_sample(sample, stats, horizon, horizons=(1, 5, 10)):
    normalized = normalize_window(sample, stats)
    history = normalized["history"]
    relative = [
        history["nut_position"] - history["eef_position"],
        normalized["fixtures"]["peg1/position"] - history["nut_position"],
        normalized["fixtures"]["peg1/position"] - history["eef_position"],
    ]
    state = np.concatenate(
        [history[k].reshape(len(history[k]), -1) for k in HISTORY_FIELDS] + relative,
        axis=-1,
    )
    actions = np.zeros((max(horizons), 7), dtype=np.float32)
    actions[:horizon] = sample["actions"]
    code = np.array([horizon == k for k in horizons], dtype=np.float32)
    context = np.concatenate(
        [normalized["fixtures"][k].ravel() for k in FIXTURE_FIELDS]
        + [actions.ravel(), code]
    )
    return (
        state.astype(np.float32),
        context.astype(np.float32),
        normalized["target_delta"].ravel().astype(np.float32),
    )


def make_arrays(cache, ids, stats, history=4, horizons=(1, 5, 10)):
    states, contexts, targets, episodes, steps, times = [], [], [], [], [], []
    for i, name in enumerate(ids):
        episode = read_episode(cache[name])
        for k in horizons:
            for t in anchors(len(episode["time"]), history, k):
                s, c, y = encode_sample(
                    window(episode, int(t), history, k), stats, k, horizons
                )
                states.append(s)
                contexts.append(c)
                targets.append(y)
                episodes.append(i)
                steps.append(k)
                times.append(int(t))
    return {
        "state": np.stack(states),
        "context": np.stack(contexts),
        "target": np.stack(targets),
        "episode": np.array(episodes),
        "horizon": np.array(steps),
        "anchor": np.array(times),
        "ids": ids,
    }


def error_rows(predictions, arrays, scale, model):
    error = (
        100
        * scale
        * np.linalg.norm((predictions - arrays["target"]).reshape(-1, 2, 3), axis=-1)
    )
    rows = []
    for i, name in enumerate(arrays["ids"]):
        for k in sorted(set(arrays["horizon"])):
            mask = (arrays["episode"] == i) & (arrays["horizon"] == k)
            for j, entity in enumerate(("end_effector", "square_nut")):
                rows.append(
                    {
                        "episode": name,
                        "entity": entity,
                        "horizon_steps": int(k),
                        "model": model,
                        "windows": int(mask.sum()),
                        "mean_error_cm": float(error[mask, j].mean()),
                    }
                )
    return rows
