import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


REPOSITORY = "amandlek/mimicgen_datasets"
REVISION = "33016f8a62c02334f929f2913af8fdd2a8a129e1"
FILES = {
    "square_d0": (
        1621351476,
        "41fc24bce0f88343099c0b1b5bf6eee08cbc35851e71276d4509a01c9b75481c",
    ),
    "square_d1": (
        1659441082,
        "bdb88ebb36f791645ed4b42512b5cf92e337c1673a71ddf4a7a6295c446fabdf",
    ),
}


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=list(FILES), default=list(FILES))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        size, digest = FILES[task]
        destination = args.directory / f"{task}.hdf5"
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/core/{task}.hdf5"
        if not destination.exists():
            partial = destination.with_suffix(".partial")
            # An interrupted transfer stays visible; never overwrite it silently.
            with partial.open("xb") as output, urlopen(url, timeout=120) as response:
                for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    output.write(block)
            if partial.stat().st_size != size or checksum(partial) != digest:
                raise ValueError(f"Download checksum mismatch: {partial}")
            partial.rename(destination)
        elif destination.stat().st_size != size or checksum(destination) != digest:
            raise ValueError(f"Existing file checksum mismatch: {destination}")
        print(
            json.dumps({"task": task, "url": url, "bytes": size, "sha256": digest}),
            flush=True,
        )


if __name__ == "__main__":
    main()
