#!/usr/bin/env bash
set -euo pipefail

# The MLflow server only needs mlflow itself, so it runs in an ephemeral uv
# environment. Keep the same major version as the `mlflow` dependency in
# mlflow_train.py so the server's DB schema matches the one written by training.
mlflow_spec='mlflow>=3.13,<4'

# Wrapper-only help, followed by the Python script's own help
# (it shows with bash run_mlflow_training.sh --help)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<'EOF'
-------------------------------
Options handled by this script 
-------------------------------
  --no-dashboard        Do not launch the MLflow dashboard (use for batch jobs).
  --mlflow-port PORT    Port for the MLflow dashboard (default: a free port).

**All other arguments are passed to mlflow_train.py (see options below).**

While the dashboard is running, this script stays alive after training ends so
you can inspect the results. Press Ctrl-C to stop the dashboard and exit.

---------------------------
mlflow_train.py arguments
---------------------------

EOF
    exec uv run mlflow_train.py --help
fi

# Parse the arguments this script acts on, and collect the rest for the
# Python script. Note that --mlflow-tracking-uri is both read here (the
# dashboard must point at the same database) and passed through.
tracking_uri="sqlite:///mlflow.db" # default
launch_dashboard=true # default
port=""
train_args=()

while [ $# -gt 0 ]; do
    case "$1" in
        --no-dashboard)
            launch_dashboard=false
            shift
            ;;
        --mlflow-port)
            port="${2:?Error: --mlflow-port requires a value}"
            shift 2
            ;;
        --mlflow-port=*)
            port="${1#*=}"
            shift
            ;;
        --mlflow-tracking-uri)
            tracking_uri="${2:?Error: --mlflow-tracking-uri requires a value}"
            train_args+=("$1" "$2")
            shift 2
            ;;
        --mlflow-tracking-uri=*)
            tracking_uri="${1#*=}"
            train_args+=("$1")
            shift
            ;;
        *)
            train_args+=("$1")
            shift
            ;;
    esac
done

# Usage message if required argument missing
usage="Missing arguments. Usage: bash $0 /path/to/exported/sleap/training/job.zip [+ optional args, see 'bash $0 --help']"
zip_path="${train_args[0]:?$usage}"

# Check file before `uv run` (uv run installs dependencies first)
if [ ! -f "$zip_path" ]; then
    echo "Error: file not found: $zip_path" >&2
    exit 1
fi

# Launch the dashboard in the background, so it can be used to monitor
# training as it runs
if [ "$launch_dashboard" = true ]; then
    # Ask the OS for a free port, to avoid clashing with other MLflow servers
    # (e.g. other users on a shared interactive node)
    if [ -z "$port" ]; then
        port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
    fi
    url="http://localhost:$port/#/experiments?workflowType=machine_learning"
    server_log=$(mktemp -t mlflow-server-XXXXXX.log)

    # Build the ephemeral mlflow environment in the foreground, so uv's
    # installation progress is visible. The first run can take a few minutes;
    # afterwards this is a fast cache hit and the server below starts quickly.
    echo "Preparing MLflow environment ($mlflow_spec)..."
    uvx --python 3.13 "$mlflow_spec" --version

    uvx --python 3.13 "$mlflow_spec" server \
        --backend-store-uri "$tracking_uri" \
        --port "$port" >"$server_log" 2>&1 &
    server_pid=$!
    trap 'kill "$server_pid" 2>/dev/null || true' EXIT

    # Wait for the server to accept connections before opening the browser
    echo "Starting MLflow dashboard on port $port (log: $server_log)..."
    for _ in $(seq 60); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Error: MLflow dashboard failed to start, see $server_log" >&2
            exit 1
        fi
        if python3 -c "import socket,sys; sys.exit(0 if socket.socket().connect_ex(('127.0.0.1', $port)) == 0 else 1)"; then
            break
        fi
        sleep 1
    done

    # Open the browser on the experiments tab (a no-op on a headless machine)
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
        open "$url" >/dev/null 2>&1 || true
    fi
    echo "MLflow dashboard: $url"
fi

# Run training (without exiting on failure, so the dashboard stays available)
set +e
uv run mlflow_train.py "${train_args[@]}"
status=$?
set -e

# Keep the dashboard alive until the user stops it
if [ "$launch_dashboard" = true ]; then
    echo
    echo "Training finished (exit code $status)."
    echo "MLflow dashboard still running at $url - press Ctrl-C to stop it."
    wait "$server_pid" || true
fi

exit "$status"
