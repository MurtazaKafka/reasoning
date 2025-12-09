#!/bin/bash
#
# Run all reasoning experiments: baseline, forward-only, backward-only, and hybrid
#
# Usage:
#   ./scripts/run_experiments.sh [--eval-only] [--train-only] [--quick]
#
# Options:
#   --eval-only   Skip training, only run evaluations
#   --train-only  Only run training, skip evaluations
#   --quick       Use small sample size for quick testing (8 samples)
#

set -e  # Exit on error

# Parse arguments
EVAL_ONLY=false
TRAIN_ONLY=false
QUICK=false

for arg in "$@"; do
    case $arg in
        --eval-only)
            EVAL_ONLY=true
            ;;
        --train-only)
            TRAIN_ONLY=true
            ;;
        --quick)
            QUICK=true
            ;;
    esac
done

echo "========================================"
echo "  Reasoning Experiments Runner"
echo "========================================"
echo ""
echo "Settings:"
echo "  EVAL_ONLY:  $EVAL_ONLY"
echo "  TRAIN_ONLY: $TRAIN_ONLY"
echo "  QUICK:      $QUICK"
echo ""

# Create output directories
mkdir -p outputs/evals
mkdir -p outputs/figures

# ----------------------------------------
# Step 1: Baseline Evaluation
# ----------------------------------------
if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "========================================"
    echo "  Step 1: Baseline Evaluation"
    echo "========================================"
    echo ""
    
    if [ "$QUICK" = true ]; then
        # Modify config for quick test (8 samples)
        python scripts/eval_reasoning.py configs/eval_baseline.yaml
    else
        python scripts/eval_reasoning.py configs/eval_baseline.yaml
    fi
fi

# ----------------------------------------
# Step 2: Train Forward-Only Model
# ----------------------------------------
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "========================================"
    echo "  Step 2: Train Forward-Only Model"
    echo "========================================"
    echo ""
    
    python scripts/train_dpo.py configs/dpo_forward_only.yaml
    
    # Fix adapter config if needed
    FORWARD_CKPT="./outputs/runs/llama31_8b_forward_only_dpo/checkpoint-200"
    if [ -d "$FORWARD_CKPT" ]; then
        if [ ! -s "$FORWARD_CKPT/adapter_config.json" ]; then
            echo "Fixing adapter config..."
            python scripts/fix_adapter_config.py "$FORWARD_CKPT"
        fi
    fi
fi

# ----------------------------------------
# Step 3: Evaluate Forward-Only Model
# ----------------------------------------
if [ "$TRAIN_ONLY" = false ]; then
    FORWARD_CKPT="./outputs/runs/llama31_8b_forward_only_dpo/checkpoint-200"
    if [ -d "$FORWARD_CKPT" ]; then
        echo ""
        echo "========================================"
        echo "  Step 3: Evaluate Forward-Only Model"
        echo "========================================"
        echo ""
        
        python scripts/eval_reasoning.py configs/eval_forward_only.yaml
    else
        echo "Skipping forward-only eval: checkpoint not found at $FORWARD_CKPT"
    fi
fi

# ----------------------------------------
# Step 4: Train Backward-Only Model
# ----------------------------------------
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "========================================"
    echo "  Step 4: Train Backward-Only Model"
    echo "========================================"
    echo ""
    
    python scripts/train_dpo.py configs/dpo_backward_only.yaml
    
    # Fix adapter config if needed
    BACKWARD_CKPT="./outputs/runs/llama31_8b_backward_only_dpo/checkpoint-200"
    if [ -d "$BACKWARD_CKPT" ]; then
        if [ ! -s "$BACKWARD_CKPT/adapter_config.json" ]; then
            echo "Fixing adapter config..."
            python scripts/fix_adapter_config.py "$BACKWARD_CKPT"
        fi
    fi
fi

# ----------------------------------------
# Step 5: Evaluate Backward-Only Model
# ----------------------------------------
if [ "$TRAIN_ONLY" = false ]; then
    BACKWARD_CKPT="./outputs/runs/llama31_8b_backward_only_dpo/checkpoint-200"
    if [ -d "$BACKWARD_CKPT" ]; then
        echo ""
        echo "========================================"
        echo "  Step 5: Evaluate Backward-Only Model"
        echo "========================================"
        echo ""
        
        python scripts/eval_reasoning.py configs/eval_backward_only.yaml
    else
        echo "Skipping backward-only eval: checkpoint not found at $BACKWARD_CKPT"
    fi
fi

# ----------------------------------------
# Step 6: Train Hybrid Model (if not already trained)
# ----------------------------------------
if [ "$EVAL_ONLY" = false ]; then
    HYBRID_CKPT="./outputs/runs/llama31_8b_forward_backward_dpo/checkpoint-200"
    if [ ! -d "$HYBRID_CKPT" ]; then
        echo ""
        echo "========================================"
        echo "  Step 6: Train Hybrid Model"
        echo "========================================"
        echo ""
        
        python scripts/train_dpo.py configs/dpo_hybrid.yaml
        
        # Fix adapter config if needed
        if [ -d "$HYBRID_CKPT" ]; then
            if [ ! -s "$HYBRID_CKPT/adapter_config.json" ]; then
                echo "Fixing adapter config..."
                python scripts/fix_adapter_config.py "$HYBRID_CKPT"
            fi
        fi
    else
        echo "Hybrid model already trained at $HYBRID_CKPT"
    fi
fi

# ----------------------------------------
# Step 7: Evaluate Hybrid Model
# ----------------------------------------
if [ "$TRAIN_ONLY" = false ]; then
    HYBRID_CKPT="./outputs/runs/llama31_8b_forward_backward_dpo/checkpoint-200"
    if [ -d "$HYBRID_CKPT" ]; then
        echo ""
        echo "========================================"
        echo "  Step 7: Evaluate Hybrid Model"
        echo "========================================"
        echo ""
        
        python scripts/eval_reasoning.py configs/eval_hybrid.yaml
    else
        echo "Skipping hybrid eval: checkpoint not found at $HYBRID_CKPT"
    fi
fi

# ----------------------------------------
# Step 8: Generate Visualizations
# ----------------------------------------
if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "========================================"
    echo "  Step 8: Generate Visualizations"
    echo "========================================"
    echo ""
    
    python scripts/visualize_results.py --results-dir outputs/evals --output-dir outputs/figures
fi

echo ""
echo "========================================"
echo "  Experiments Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - outputs/evals/*.json      (raw metrics)"
echo "  - outputs/figures/*.png     (charts)"
echo "  - outputs/figures/results_summary.md"
echo ""
