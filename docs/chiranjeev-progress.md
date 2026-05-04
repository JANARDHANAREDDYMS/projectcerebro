# Chiranjeev Progress — Week 2 ML Core

**Branch:** `chiranjeev/ml-core-week2-v2`
**Owner:** Chiranjeev Kumar
**Reference architecture:** `projectcerebro_architecture_v2-2.drawio`
**Preprocessed data:** [Google Drive folder](https://drive.google.com/drive/folders/1eWYfso6ILulAiVoFZAmbB5eADBGRGxPb?usp=sharing)

## Why this branch exists

The earlier `codex/ml-architecture-plan` PR (merged as #1) and the follow-up PR #2 (`chiranjeev/ml-core-week2`) were rolled back because the implementation did not match the validated plan in `PLAN.md`. PR #2 was closed; PR #1 was reverted on `main` via this branch's first commit. We then implemented the ML core from scratch on top of clean `main`, faithful to the plan.

## Current State (project-wide)

Already done before this branch:

- Data sources downloaded: PhysioNet EEGMMIDB, BCI Competition IV-2a.
- Batch ingestion to MongoDB (subject metadata) and Cassandra (raw epochs).
- Stage 1 ICA cleaning.
- Stage 2 Spark preprocessing → Delta Lake at `delta_lake/epochs_mi_v1_ch5_sr128_bp8_30/` (primary, 8–30 Hz) and `…_bp4_38/` (ablation, 4–38 Hz). Schema is fixed at `(5 channels × 512 samples) → features list of length 2560`, sampling rate 128 Hz, channels FZ/C3/CZ/C4/PZ.

What this branch adds:

- `ml_core/` Python package with data loader, models, trainer, evaluation, and pgvector embedding export.
- Tests under `tests/`.
- Updated `requirements.txt` (mlflow, pytest, pytest-cov, torchmetrics, psycopg v3).
- Configs under `ml_core/configs/`.
- Artifact directories under `artifacts/` (gitignored, contains only `.gitkeep`).

## Compute strategy: hybrid M4 + Colab T4

| Workload | Device | Why |
| --- | --- | --- |
| Loader / split / model unit tests | M4 MPS or CPU | Fast iteration, no Colab session limit. |
| ShallowConvNet smoke (`train_smoke.py`) | M4 MPS | Finishes in <2 min. |
| ShallowConvNet baseline | M4 MPS or T4 | Either works; T4 ≈3× faster. |
| EEGNet PhysioNet pretrain | T4 (Colab) | ~9k epochs × 50–80 training epochs, MPS is slow on Conv2d-heavy stacks. |
| EEGNet BCI fine-tune | T4 (Colab) | Already on Colab from pretrain; reuse session. |
| Conformer + Ray Tune (later) | T4 (Colab) | Heavier. |

The trainer auto-picks the best device: CUDA → MPS → CPU. Same scripts run on both environments. A Colab notebook for mounting Drive and running the pretrain/fine-tune pipeline is planned as a follow-up PR.

## Setup checklist

- [x] New branch off main: `chiranjeev/ml-core-week2-v2`.
- [x] Closed PR #2 + deleted remote branch `chiranjeev/ml-core-week2`.
- [x] Reverted PR #1 merge commit `931ff82` on this branch (clean slate for ML core).
- [x] Implemented `ml_core/` package per validated plan.
- [x] Added pytest test suite under `tests/`.
- [x] Updated `requirements.txt` (mlflow, pytest, pytest-cov, torchmetrics, psycopg).
- [x] Download Drive `delta_lake/` into `delta_lake/`.
- [x] `pip install -r requirements.txt` in venv.
- [x] `pytest tests/` — 21 pass, 2 skip (real Delta + pgvector gated).
- [x] `python -m ml_core.experiments.train_smoke` — 5 epochs, val F1 0.343.
- [x] ShallowConvNet baseline trained — 52.9% acc, 0.523 macro F1.
- [x] EEGNet from scratch trained — 52.9% acc, 0.495 macro F1.
- [ ] EEGNet pretrain on PhysioNet (needs separate dataset).
- [ ] pgvector embedding export (script implemented, not run).
- [ ] Colab notebook (follow-up PR).

## Training Results (2026-05-04)

All training runs completed locally on M4 MPS (MLflow disabled due to file store race condition).

### ShallowConvNet Baseline

**Command:**
```bash
python -m ml_core.experiments.train_shallow_baseline \
  --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
  --filter-version bp_8_30_v1 \
  --out-dir artifacts/checkpoints/shallow_baseline
```

**Results (test set, 4668 samples, all datasets):**
- Accuracy: **52.9%**
- Macro F1: **0.523**
- Balanced accuracy: **53.2%**
- Best val F1: 0.485 @ epoch 19, stopped @ epoch 34

**Per-class F1:**
- Class 0: 0.493
- Class 1: 0.464 (weakest, 41.9% recall)
- Class 2: 0.613 (best, 71% recall)

**Outputs:** `artifacts/checkpoints/shallow_baseline/{best.pt, test_overall.json, test_by_subject.json, norm_stats.json, split_manifest.json}`

### EEGNet Baseline (from scratch)

**Command:**
```bash
python -m ml_core.experiments.finetune_eegnet_bci \
  --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
  --filter-version bp_8_30_v1 \
  --out-dir artifacts/checkpoints/eegnet_scratch
```

**Results (test set, 888 samples, BCI IV-2a only):**
- Accuracy: **52.9%**
- Macro F1: **0.495**
- Balanced accuracy: **51.9%**
- Best val F1: 0.394 @ epoch 15, stopped @ epoch 30

**Per-class F1:**
- Class 0: 0.398 (75% precision, 27% recall — too conservative)
- Class 1: 0.467
- Class 2: 0.619 (47% precision, 89.7% recall — too aggressive)

**Outputs:** `artifacts/checkpoints/eegnet_scratch/{best.pt, test_overall.json, test_by_subject.json, norm_stats.json, split_manifest.json}`

### Comparison

| Model | Test Samples | Accuracy | Macro F1 | Bal. Acc | Notes |
|-------|-------------|----------|----------|----------|-------|
| ShallowConvNet | 4668 (all) | 52.9% | **0.523** | **53.2%** | Better balanced performance |
| EEGNet | 888 (BCI only) | 52.9% | 0.495 | 51.9% | Biased toward class 2 |

**Note:** ShallowConvNet trained on all datasets; EEGNet filtered to BCI IV-2a only (transfer learning setup). Not directly comparable.

## Validation of `PLAN.md`

`PLAN.md` was reviewed against the actual Stage 2 schema in `scripts/stage2_spark_preprocess.py`. The data contract, split policy, training defaults, and milestone order all align with what is implemented here. Clarifications that were folded into the implementation:

- `filter_version` value stored in Delta is `"bp_8_30_v1"` / `"bp_4_38_v1"` (string). The directory name uses `bp8_30`. Loader filter argument uses the stored string.
- `is_rest_synthetic` rows can be excluded via `read_epochs(drop_synthetic_rest=True)` — useful for honest baselines.
- `session_id` is preserved for stratified analysis but not used for splitting yet.
- Conformer + Ray Tune are deferred until the EEGNet path is stable.

## Module layout

```
ml_core/
├── data/          delta_loader, schema, splits, normalize, dataset
├── models/        shallowconv, eegnet
├── training/      trainer, checkpoint, callbacks
├── evaluation/    metrics, subject_eval
├── embeddings/    export_pgvector
├── experiments/   train_smoke, train_shallow_baseline,
│                  pretrain_eegnet_physionet, finetune_eegnet_bci
└── configs/       default.yaml, shallow_baseline.yaml,
                   eegnet_pretrain.yaml, eegnet_finetune.yaml
```

## Next steps

1. Pull `delta_lake/` from Drive.
2. Run `pytest tests/`.
3. Run smoke trainer on M4 to confirm loss decreases.
4. Run pretrain + fine-tune on Colab T4 with `--mlflow-experiment cerebro_week2`.
5. Export 128-d embeddings to pgvector.
6. Open follow-up PR with the Colab notebook.
