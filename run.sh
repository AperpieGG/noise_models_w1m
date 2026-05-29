#!/usr/bin/env bash

# You will run this on the observing night directory, e.g. 20250806
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA="${1:-QHY600}"
LOG_DIR="${PIPELINE_LOG_DIR:-$PWD/logs}"
mkdir -p "$LOG_DIR"
export PIPELINE_LOG_DIR="$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_${CAMERA}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'status=$?; echo "Pipeline failed at line ${LINENO} with exit code ${status}"; exit ${status}' ERR

# Record the start time
start_time=$(date +%s)

echo "Starting processing for camera: ${CAMERA}"
echo "Pipeline log: ${LOG_FILE}"

# Run the Python scripts
python -u "$SCRIPT_DIR/simple_wrapper_W1m.py" --camera "$CAMERA"
python -u "$SCRIPT_DIR/check_cmos_W1m.py" --camera "$CAMERA"
python -u "$SCRIPT_DIR/adding_header_W1m.py"
python -u "$SCRIPT_DIR/process_cmos_W1m.py" --camera "$CAMERA"

if [[ -f "$SCRIPT_DIR/relative_phot_dev_W1m.py" ]]; then
    python -u "$SCRIPT_DIR/relative_phot_dev_W1m.py"
fi

echo "Finishing processing!"

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
elapsed_time=$((end_time - start_time))

# Print the total time taken
echo "Total time taken: $elapsed_time seconds"
