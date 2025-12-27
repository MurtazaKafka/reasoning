# Experiment Configurations

This directory contains YAML configurations for the complete experimental study.

## Experimental Design

### Main Hypotheses

1. **H1**: Hybrid forward+backward DPO improves accuracy over forward-only DPO
2. **H2**: Backward training increases acknowledgement rate (hallucination awareness)
3. **H3**: The combination provides better calibration than either alone

### Experimental Conditions

| Condition | Forward Weight | Backward Weight | Description |
|-----------|---------------|-----------------|-------------|
| `baseline` | - | - | No fine-tuning (base LLaMA 3.1 8B) |
| `forward_only` | 1.0 | 0.0 | Standard DPO on reasoning traces |
| `backward_only` | 0.0 | 1.0 | DPO only on verification |
| `hybrid_60_40` | 0.6 | 0.4 | Main hybrid configuration |
| `hybrid_50_50` | 0.5 | 0.5 | Equal weight ablation |
| `hybrid_80_20` | 0.8 | 0.2 | Forward-heavy ablation |

### Ablation Studies

1. **Weight Ratio Ablation**: 80/20, 60/40, 50/50, 40/60
2. **LoRA Rank Ablation**: r=8, r=16, r=32, r=64
3. **Data Quality Ablation**: With/without rejection sampling
4. **Reference Model Ablation**: reference_free vs frozen reference

### Evaluation Benchmarks

| Benchmark | Type | Samples | Purpose |
|-----------|------|---------|---------|
| GSM8K | Math | 1319 test | Primary reasoning benchmark |
| MATH | Math | 5000 test | Hard math problems |
| ARC-Challenge | Science | 1172 test | Multi-hop reasoning |
| TruthfulQA | Factual | 817 test | Hallucination detection |

### Metrics

| Metric | Definition | What it Measures |
|--------|------------|------------------|
| `accuracy` | % correct final answers | Reasoning ability |
| `acknowledgement_rate` | % of wrong answers flagged FAIL | Hallucination awareness |
| `false_positive_rate` | % of correct answers flagged FAIL | Over-conservatism |
| `verification_calibration` | F1 of verification vs reality | Overall calibration |
| `self_consistency` | % agreement across samples | Uncertainty estimation |

### Running All Experiments

```bash
# 1. Generate high-quality training data
python scripts/generate_dpo_pairs.py --limit 5000 --rejection-sampling

# 2. Run all training conditions
for config in configs/experiments/train_*.yaml; do
    python scripts/train_dpo.py $config
done

# 3. Evaluate all models
for config in configs/experiments/eval_*.yaml; do
    python scripts/eval_reasoning.py $config
done

# 4. Analyze results
python scripts/analyze_results.py outputs/evals/
```
