#!/bin/bash
# Helper script to run the Python example
# Usage: ./run.sh [device_id]

DEVICE_ID=${1:-9}

# Set up environment
export PYTHONPATH=../../python:$PYTHONPATH
export PTO_ISA_ROOT=../../_deps/pto-isa-src

echo "Running Python example on device $DEVICE_ID..."
echo "PYTHONPATH: $PYTHONPATH"
echo "PTO_ISA_ROOT: $PTO_ISA_ROOT"
echo ""

python3 graphbuilder.py $DEVICE_ID
