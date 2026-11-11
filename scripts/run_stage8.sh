#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo 'Usage: bash scripts/run_stage8.sh D0_CACHE D1_CACHE OUTPUT [PROTOCOL] [--validation-only-check]' >&2
  exit 2
fi
d0_cache=$1
d1_cache=$2
output=$3
protocol=${4:-configs/formal_protocol.yaml}
mkdir -p "$output"
exec 9>"$output/pipeline.lock"
flock -n 9 || { echo 'This pipeline is already running' >&2; exit 3; }
finish() {
  code=$?
  printf '%s\n' "$code" > "$output/pipeline.exit"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$output/pipeline.finished"
}
trap finish EXIT
rm -f "$output/pipeline.exit" "$output/pipeline.finished"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$output/pipeline.started"
echo "Pipeline PID $$"
echo $$ > "$output/pipeline.pid"
.venv/bin/python -m robot_dynamics.run_formal "$d0_cache" --protocol "$protocol" --output "$output" --complete-matrix
extra=()
if [[ ${5:-} == '--validation-only-check' ]]; then
  extra+=(--validation-only-check)
elif [[ $# == 5 ]]; then
  echo 'Unknown fifth argument' >&2
  exit 2
fi
.venv/bin/python -m robot_dynamics.evaluate_formal "$d0_cache" --d1-cache "$d1_cache" --protocol "$protocol" --runs "$output" "${extra[@]}"
.venv/bin/python -m robot_dynamics.analyze_formal --runs "$output"
.venv/bin/python -m robot_dynamics.plot_formal "$output/analysis/analysis.json"
echo 'Pipeline complete. Inspect analysis/README.md and analysis/analysis.json, including diagnostic flags.'
