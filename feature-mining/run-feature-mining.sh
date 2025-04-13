#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Specify the Python script name
PYTHON_SCRIPT="feature-mining.py"

# Run the Python script
python "$SCRIPT_DIR/$PYTHON_SCRIPT"
