#!/usr/bin/env bash
set -euo pipefail
cache=${1:?GRAIL cache required}
manifest=${2:?GRAIL manifest required}
output=${3:?Output directory required}
protocol=${4:-configs/grail_protocol.yaml}
mkdir -p "$output"
exec 9>"$output/driver.lock"
flock -n 9 || { echo 'GRAIL driver is already running'; exit 1; }
trap 'code=$?; printf "%s\n" "$code" > "$output/training.exit"; date -u +%FT%TZ > "$output/finished.txt"' EXIT
rm -f "$output/training.exit" "$output/finished.txt"
date -u +%FT%TZ >> "$output/started.txt"
echo $$ > "$output/driver.pid"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -u -m robot_dynamics.run_grail "$cache" --manifest "$manifest" --protocol "$protocol" --output "$output"
