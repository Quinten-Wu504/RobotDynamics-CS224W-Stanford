import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.with_suffix(".png").exists()
        or args.output.with_suffix(".pdf").exists()
    ):
        raise FileExistsError(args.output)
    checks = json.loads(args.verification.read_text())
    with h5py.File(args.cache, "r") as cache:
        if (
            not cache.attrs["complete"]
            or cache.attrs["source_sha256"] != checks["source_sha256"]
        ):
            raise ValueError("Incomplete cache or mismatched source dataset")
        fig, axes = plt.subplots(
            len(checks["episodes"]), 2, figsize=(10, 9), squeeze=False
        )
        for row, item in enumerate(checks["episodes"]):
            group = cache[item["episode"]]
            xy, height = axes[row]
            time = group["time"][:] - group["time"][0]
            for entity, color, label in [
                ("eef", "#286fa0", "End effector"),
                ("nut", "#c56827", "Square nut"),
            ]:
                observed = group[f"{entity}_position"][:]
                restored = np.asarray(item[f"restored_{entity}_position"])
                xy.plot(observed[:, 0], observed[:, 1], color=color, label=label)
                xy.scatter(
                    restored[::5, 0], restored[::5, 1], s=10, marker="x", color=color
                )
                xy.scatter(
                    observed[0, 0],
                    observed[0, 1],
                    s=30,
                    facecolors="white",
                    edgecolors=color,
                    zorder=5,
                )
                height.plot(time, observed[:, 2], color=color, label=label)
                height.scatter(
                    time[::5], restored[::5, 2], color=color, marker="x", s=10
                )
            center = group["peg1/geom_position"][:]
            size = group["peg1/half_size"][:]
            rotation = group["peg1/geom_rotation"][:]
            corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * size
            xy.add_patch(
                Polygon(
                    (corners @ rotation.T + center)[:, :2],
                    color="#696969",
                    alpha=0.45,
                    label="Target peg",
                )
            )
            height.axhline(
                group["table/surface_position"][2],
                color="0.5",
                ls="--",
                label="Table surface",
            )
            xy.set(xlabel="World x (m)", ylabel="World y (m)", title=item["episode"])
            xy.set_aspect("equal", adjustable="datalim")
            height.set(
                xlabel="Time (s)",
                ylabel="World z (m)",
                title=f"{item['restored_frames']} recorded states",
            )
            for ax in (xy, height):
                ax.grid(alpha=0.2)
                ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].legend(fontsize=8)
        fig.suptitle(
            f"{checks['dataset'].removesuffix('.hdf5')}: observation and state recovery",
            fontsize=14,
        )
        fig.text(
            0.5,
            0.015,
            f"Lines: recorded observations   ×: restored states (every fifth frame)   ○: start\nSource: {checks['source_commit'][:12]}; world coordinates in metres",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.96))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output.with_suffix(".png"), dpi=180)
        fig.savefig(args.output.with_suffix(".pdf"))
        plt.close(fig)


if __name__ == "__main__":
    main()
