#!/usr/bin/env bash
set -euo pipefail

# pipe help from Python script
# (it shows with bash run_training.sh --help)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    exec uv run mlflow_train.py --help
fi

# Usage message if required argument missing
usage="Missing arguments. Usage: bash $0 /path/to/exported/sleap/training/job.zip [+ optional args, see 'bash $0 --help']"
zip_path="${1:?$usage}"

# Check file before `uv run` (uv run installs dependencies first)
if [ ! -f "$zip_path" ]; then
    echo "Error: file not found: $zip_path" >&2
    exit 1
fi

# Run script
uv run mlflow_train.py "$@"
