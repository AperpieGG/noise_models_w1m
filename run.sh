#!/usr/bin/env bash

# You will run this on the observing night directory, e.g. 20250806
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA_CONFIG_ARG="${1:-qhy600.json}"
NIGHT="${2:-}"

if [[ -f "$CAMERA_CONFIG_ARG" ]]; then
    CAMERA_CONFIG_PATH="$(cd "$(dirname "$CAMERA_CONFIG_ARG")" && pwd)/$(basename "$CAMERA_CONFIG_ARG")"
elif [[ -f "$SCRIPT_DIR/configs/$CAMERA_CONFIG_ARG" ]]; then
    CAMERA_CONFIG_PATH="$SCRIPT_DIR/configs/$CAMERA_CONFIG_ARG"
elif [[ -f "$SCRIPT_DIR/configs/$CAMERA_CONFIG_ARG.json" ]]; then
    CAMERA_CONFIG_PATH="$SCRIPT_DIR/configs/$CAMERA_CONFIG_ARG.json"
elif [[ -f "$SCRIPT_DIR/$CAMERA_CONFIG_ARG" ]]; then
    CAMERA_CONFIG_PATH="$SCRIPT_DIR/$CAMERA_CONFIG_ARG"
else
    echo "Camera config does not exist: $CAMERA_CONFIG_ARG" >&2
    exit 1
fi

CAMERA_NAME="$(python -c 'import json,sys,os; p=sys.argv[1]; d=json.load(open(p)); print(d.get("name") or os.path.splitext(os.path.basename(p))[0])' "$CAMERA_CONFIG_PATH")"
CONFIG_WORKERS="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("workers", 1))' "$CAMERA_CONFIG_PATH")"
WORKERS="${PIPELINE_WORKERS:-$CONFIG_WORKERS}"

if [[ -n "$NIGHT" ]]; then
    WORK_DIR="$SCRIPT_DIR/$NIGHT"
else
    WORK_DIR="$PWD"
fi

if [[ ! -d "$WORK_DIR" ]]; then
    echo "Observing-night directory does not exist: $WORK_DIR" >&2
    exit 1
fi

cd "$WORK_DIR"

LOG_DIR="${PIPELINE_LOG_DIR:-$WORK_DIR/logs}"
mkdir -p "$LOG_DIR"
export PIPELINE_LOG_DIR="$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_${CAMERA_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'status=$?; echo "Pipeline failed at line ${LINENO} with exit code ${status}"; exit ${status}' ERR

# Record the start time
start_time=$(date +%s)

echo "Starting processing for camera: ${CAMERA_NAME}"
echo "Camera config: ${CAMERA_CONFIG_PATH}"
echo "Working directory: ${WORK_DIR}"
echo "Pipeline workers: ${WORKERS}"
echo "Pipeline log: ${LOG_FILE}"

# Run the Python scripts
python -u "$SCRIPT_DIR/simple_wrapper_W1m.py" --camera "$CAMERA_CONFIG_PATH"
python -u "$SCRIPT_DIR/check_cmos_W1m.py" --camera "$CAMERA_CONFIG_PATH" --workers "$WORKERS"
python -u "$SCRIPT_DIR/adding_header_W1m.py" --camera "$CAMERA_CONFIG_PATH"
python -u "$SCRIPT_DIR/process_cmos_W1m.py" --camera "$CAMERA_CONFIG_PATH" --workers "$WORKERS"

if [[ -f "$SCRIPT_DIR/relative_phot_dev_W1m.py" ]]; then
    python -u "$SCRIPT_DIR/relative_phot_dev_W1m.py"
elif [[ -f "$SCRIPT_DIR/relative_process_W1m.py" ]]; then
    python -u "$SCRIPT_DIR/relative_process_W1m.py" --cam "$CAMERA_CONFIG_PATH" --workers "$WORKERS"
fi

echo "Finishing processing!"

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
elapsed_time=$((end_time - start_time))

# Print the total time taken
echo "Total time taken: $elapsed_time seconds"
