# Phase A: LOSO Cross-Validation Results

## Summary

Leave-One-Subject-Out (LOSO) evaluation on BCI Competition IV-2a dataset (9 subjects: A01-A09).

## Results

| Model | Accuracy | Macro F1 | Time (M4 MPS) |
|-------|----------|----------|---------------|
| ShallowConvNet | 50.1% ± 12.9% | 0.448 ± 0.174 | 24 min |
| EEGNet | 49.7% ± 13.1% | 0.477 ± 0.146 | 13 min |

**Target:** 74%  
**Current:** ~50%  
**Random baseline:** 33% (3-class)

## Files

- `loso_shallow_bci.json` - ShallowConvNet full results with per-subject breakdown
- `loso_eegnet_bci.json` - EEGNet full results with per-subject breakdown

## How to reproduce

```bash
# ShallowConvNet
python -m ml_core.experiments.loso_eval \
  --model shallowconvnet \
  --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
  --filter-version bp_8_30_v1 \
  --holdout-dataset bci_iv_2a \
  --out-dir artifacts/reports/loso_shallow_bci \
  --epochs 50 \
  --patience 10

# EEGNet
python -m ml_core.experiments.loso_eval \
  --model eegnet \
  --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
  --filter-version bp_8_30_v1 \
  --holdout-dataset bci_iv_2a \
  --out-dir artifacts/reports/loso_eegnet_bci \
  --epochs 50 \
  --patience 10
```

## Per-Subject Performance

### ShallowConvNet
- **Best:** A09 (72.5%), A03 (68.2%)
- **Worst:** A02 (35.0%), A07 (36.8%), A05 (38.2%)

### EEGNet
- **Best:** A09 (73.0%), A03 (68.6%)
- **Worst:** A07 (35.0%), A02 (37.3%), A06 (40.5%)

## Key Findings

1. **High variance** (12-13%) indicates poor cross-subject generalization
2. **Consistent weak subjects** (A02, A05, A07) suggest subject-specific challenges
3. **EEGNet faster** (2× speed) with slightly better F1
4. **Both models plateau** around 50% - need better approaches for 74% target
