# Data Leakage Fixes & Proper Metrics Evaluation

## Changes Made

### 1. **New 3-Way Data Split** (`training/prep_lc25000_3way.py`)
- **Before**: Only 2-way split (train/val)
- **After**: Proper 3-way split (60% train / 20% val / 20% test)
- **Why**: Prevents validation set from being reused for final metrics, ensuring true generalization assessment

```bash
python training/prep_lc25000_3way.py
```

Output: `data/lc25000_3way/` with train/val/test splits

### 2. **Fixed Training Script** (`training/train_resnet18.py`)
- **Data Leakage Fixed**:
  - ✅ Normalization statistics computed FROM TRAINING DATA ONLY
  - ✅ Validation data normalized using train statistics (no look-ahead)
  - ❌ Removed ImageNet statistics hardcoding

- **How to use**:
  ```bash
  python training/train_resnet18.py --data_dir data/lc25000_3way --epochs 20 --batch_size 32
  ```

### 3. **Fixed Evaluation Script** (`training/eval_and_plots.py`)
- **Evaluation Improvements**:
  - ✅ Evaluates on HELD-OUT TEST SET by default (`--eval_split test`)
  - ✅ Normalization statistics recomputed from training data (no test data used)
  - ✅ Outputs clearly labeled metrics: `metrics_summary_test.json` (final) vs `metrics_summary_val.json` (tuning only)
  - ✅ Stores normalization mean/std in metrics for reproducibility

- **How to use** (final metrics on test set):
  ```bash
  python training/eval_and_plots.py --data_dir data/lc25000_3way --weights artifacts/resnet18_histopath.pt
  ```

- **Optional**: Evaluate on validation set (for hyperparameter selection only):
  ```bash
  python training/eval_and_plots.py --data_dir data/lc25000_3way --eval_split val
  ```

### Output Files
- `artifacts/metrics_summary_test.json` - **FINAL METRICS** (unbiased)
- `artifacts/confusion_matrix_test.png`
- `artifacts/roc_curve_test.png`
- `artifacts/pr_curve_test.png`

## ML Best Practices Applied

| Issue | Before | After |
|-------|--------|-------|
| **Train/Val/Test Split** | 2-way (leakage risk) | 3-way (90/10 → 60/20/20) |
| **Normalization Computed From** | Hardcoded ImageNet values | Training data only |
| **Validation Purpose** | Dual-use (training + final metrics) | Hyperparameter tuning only |
| **Final Metrics From** | Validation set (biased) | Test set (unbiased) |
| **Gradient Computation** | Training & inference mixed | Inference-only (no adaptation) |

## Reproducing Fixed Metrics

```bash
# 1. Prepare data with proper 3-way split
python training/prep_lc25000_3way.py

# 2. Train with train-only normalization
python training/train_resnet18.py --data_dir data/lc25000_3way --epochs 20

# 3. Evaluate on held-out test set (FINAL METRICS)
python training/eval_and_plots.py --data_dir data/lc25000_3way
```

## Impact on Reported Metrics

Expected changes (metrics may decrease slightly due to reduced training data):
- Metrics are now **representative of true generalization**
- No information leakage from validation/test into training
- Reproducible with proper normalization statistics documented

## Technical Details

### Normalization Leakage Issue (FIXED)
**The Problem**: Using hardcoded ImageNet statistics or computing stats from the full dataset:
- Test data influences normalization → inflates test performance
- Validation data sees "future" statistics → overly optimistic during training

**The Solution**: 
- Compute mean/std from training data ONLY
- Apply same statistics to val/test sets
- Store statistics in metrics JSON for reproducibility

### Data Contamination (FIXED)
**The Problem**: Using validation set for both:
1. Hyperparameter tuning during training
2. Final performance reporting

**The Solution**:
- Train/tune on 60% (train) + 20% (val)
- Report final metrics on 20% (test) only
- Test set never touched during training

### File Organization
```
data/lc25000_3way/
├── train/          (60% of data - used for normalization + training)
│   ├── cancer/
│   └── normal/
├── val/            (20% of data - used for hyperparameter tuning)
│   ├── cancer/
│   └── normal/
└── test/           (20% of data - held-out, only for FINAL metrics)
    ├── cancer/
    └── normal/
```
