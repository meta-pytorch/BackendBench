#!/bin/bash
# Run KernelAgent on the 77 core TorchBench operators using the Python script

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Please set OPENAI_API_KEY environment variable"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Set Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run the Python script with all arguments passed through
python scripts/run_kernel_agent.py "$@"