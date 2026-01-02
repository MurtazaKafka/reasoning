#!/bin/bash
# Run evaluation with nohup so it continues after SSH disconnect
#
# Usage:
#   ./run_eval_nohup.sh              # Run all models
#   ./run_eval_nohup.sh --resume     # Resume from checkpoint
#
# Monitor progress:
#   tail -f nohup_eval.log
#
# Check if running:
#   ps aux | grep eval_all_models

cd "$(dirname "$0")"

# Activate conda/venv if needed (uncomment and modify as needed)
# source ~/.bashrc
# conda activate your_env

# Set HuggingFace token if needed
# export HF_TOKEN="your_token_here"

# Default arguments
ARGS="--output-dir ./experiment_outputs/evals --runs-dir ./experiment_outputs/runs --max-samples 500"

# Add any extra arguments passed to script
if [ "$1" == "--resume" ]; then
    ARGS="$ARGS --resume"
fi

echo "Starting evaluation at $(date)"
echo "Log file: nohup_eval.log"
echo "To monitor: tail -f nohup_eval.log"
echo ""

# Run with nohup, redirect all output to log file
nohup python -u scripts/eval_all_models.py $ARGS > nohup_eval.log 2>&1 &

# Save the process ID
PID=$!
echo $PID > eval_pid.txt
echo "Started with PID: $PID"
echo "PID saved to: eval_pid.txt"
echo ""
echo "To stop: kill $PID"
echo "To check status: ps -p $PID"
