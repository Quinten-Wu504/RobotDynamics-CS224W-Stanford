import argparse
from pathlib import Path
from urllib.request import urlopen

from robot_dynamics.data_audit import file_sha256
from robot_dynamics.stack_data import RELEASE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mirror", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    host = "hf-mirror.com" if args.mirror else "huggingface.co"
    revision = "33016f8a62c02334f929f2913af8fdd2a8a129e1"
    for task, (size, sha) in RELEASE.items():
        destination = args.directory / f"{task}.hdf5"
        if not destination.exists():
            partial = destination.with_suffix(".partial")
            url = f"https://{host}/datasets/amandlek/mimicgen_datasets/resolve/{revision}/core/{task}.hdf5"
            with partial.open("xb") as output, urlopen(url, timeout=120) as response:
                for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    output.write(block)
            if partial.stat().st_size != size or file_sha256(partial) != sha:
                raise ValueError(f"Invalid download: {partial}")
            partial.rename(destination)
        if destination.stat().st_size != size or file_sha256(destination) != sha:
            raise ValueError(f"Invalid existing release: {destination}")
        print(f"{task}: size and SHA-256 match the pinned official release", flush=True)


if __name__ == "__main__":
    main()
