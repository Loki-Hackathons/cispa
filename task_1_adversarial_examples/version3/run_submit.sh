#!/bin/bash
# Wrapper script to submit to leaderboard
# Usage: ./run_submit.sh <submission_file.npz>

set -e

cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3

module load GCC CUDA PyTorch torchvision

if [ $# -eq 0 ]; then
    echo "Usage: $0 <submission_file.npz>"
    echo "Example: $0 output/submission_ram_v4.npz"
    exit 1
fi

SUBMISSION_FILE="$1"

if [ ! -f "$SUBMISSION_FILE" ]; then
    echo "ERROR: File not found: $SUBMISSION_FILE"
    exit 1
fi

echo "Submitting: $SUBMISSION_FILE"
python3 submit.py "$SUBMISSION_FILE" --action submit

