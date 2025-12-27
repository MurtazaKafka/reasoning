#!/bin/bash
#SBATCH --job-name=fb-reasoning-dpo
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL

# Forward-Backward Reasoning with DPO - Full Experiment Pipeline
# Submit with: sbatch run_experiments.slurm

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Create log directory
mkdir -p slurm_logs

# Load required modules (adjust for your cluster)
# module load python/3.10
# module load cuda/12.1

# Activate conda/virtual environment (adjust path)
# source /path/to/your/venv/bin/activate
# conda activate reasoning

# Set environment variables
export HF_HOME="${SCRATCH:-$HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export OUTPUT_DIR="./experiment_outputs"
export WANDB_MODE="offline"  # Set to "online" if you have wandb configured

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the experiment pipeline
python run_all_experiments.py \
    --output-dir "$OUTPUT_DIR" \
    --data-limit 2000 \
    --eval-samples 500 \
    --configs forward_only backward_only hybrid_60_40 hybrid_50_50 hybrid_80_20 \
    --resume

exit_code=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

# Copy results to a safe location (optional, adjust path)
# if [ $exit_code -eq 0 ]; then
#     cp -r "$OUTPUT_DIR/reports" /path/to/permanent/storage/
# fi

exit $exit_code
