import json
from pathlib import Path

import h5py
import pytest

from robot_dynamics.demo_square import prepare


def test_demo_uses_only_original_train_and_validation(tmp_path):
    original = json.loads(Path("reports/stage3/manifest.json").read_text())
    selected = original["d0"]["train"][:8] + original["d0"]["validation"][:2]
    source = tmp_path / "cache.h5"
    with h5py.File(source, "w") as cache:
        cache.attrs["complete"] = True
        cache.attrs["source_sha256"] = original["datasets"]["square_d0"][
            "source_sha256"
        ]
        for name in selected + original["d0"]["test"][:1]:
            cache.create_dataset(f"{name}/time", data=[0, 0.05])
    output = tmp_path / "demo"
    prepare(source, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["d0"]["train"] == original["d0"]["train"][:8]
    assert manifest["d0"]["validation"] == original["d0"]["validation"][:2]
    assert manifest["d0"]["test"] == []
    with h5py.File(output / "square_demo.hdf5") as cache:
        assert set(cache) == set(selected)
    with pytest.raises(FileExistsError):
        prepare(source, output)
    with h5py.File(source, "a") as cache:
        cache.attrs["source_sha256"] = "wrong-source"
    with pytest.raises(ValueError, match="pinned Square"):
        prepare(source, tmp_path / "invalid")
