#!/bin/bash

# You will run this on the observing night directory, e.g. 20250806

# Define base path
BASE_DIR="/home/ops/Apergis/W1m_stuff"

# Record the start time
start_time=$(date +%s)

echo "Starting processing..."

# Run the Python scripts
python "$BASE_DIR/simple_wrapper_W1m.py" --camera QHY600
python "$BASE_DIR/check_cmos_W1m.py"
python "$BASE_DIR/adding_headers_W1m.py"
python "$BASE_DIR/process_cmos_W1m.py"
python "$BASE_DIR/relative_phot_dev_W1m.py"

echo "Finishing processing!"

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
elapsed_time=$((end_time - start_time))

# Print the total time taken
echo "Total time taken: $elapsed_time seconds"