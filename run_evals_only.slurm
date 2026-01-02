#!/bin/bash
#SBATCH --job-name=fb-reasoning-eval
#SBATCH --output=slurm_logs/%j_eval.out
#SBATCH --error=slurm_logs/%j_eval.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=munikzad@davidson.edu

# Evaluation-Only Pipeline for Forward-Backward Reasoning
# Models are already trained - this only runs evaluation
#
# Submit with: sbatch run_evals_only.slurm
# Check status: squeue -u munikzad
# View output:  cat slurm_logs/<JOB_ID>_eval.out

echo "=========================================="
echo "EVALUATION-ONLY JOB"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Create log directory
mkdir -p slurm_logs

# Navigate to project directory
cd /home/DAVIDSON/munikzad/csc224n/reasoning

# Activate conda environment (adjust if different)
source ~/.bashrc
conda activate reasoning

# Set environment variables
export HF_HOME="${SCRATCH:-$HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export OUTPUT_DIR="./experiment_outputs"
export WANDB_MODE="offline"

# Verify GPU is available
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Verify trained models exist
echo "Checking trained models..."
for model in forward_only_dpo backward_only_dpo hybrid_60_40_dpo; do
    if [ -d "$OUTPUT_DIR/runs/$model" ]; then
        echo "  ✓ Found: $model"
    else
        echo "  ✗ Missing: $model"
    fi
done
echo ""

# Run ONLY evaluation (skip data generation and training)
echo "Starting evaluation pipeline..."
python run_all_experiments.py \
    --output-dir "$OUTPUT_DIR" \
    --eval-samples 500 \
    --only-eval \
    --configs forward_only backward_only hybrid_60_40

exit_code=$?

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

# Show results summary if successful
if [ $exit_code -eq 0 ]; then
    echo ""
    echo "Results saved to:"
    ls -la "$OUTPUT_DIR/evals/"
    echo ""
    if [ -f "$OUTPUT_DIR/reports/EXPERIMENT_SUMMARY.txt" ]; then
        echo "Summary:"
        cat "$OUTPUT_DIR/reports/EXPERIMENT_SUMMARY.txt"
    fi
fi

exit $exit_code
