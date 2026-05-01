# Chiranjeev Progress — ML Core and Training Plan

**Branch:** `codex/ml-architecture-plan`
**Owner:** Chiranjeev Kumar
**Current date:** May 1, 2026
**Reference architecture:** `projectcerebro_architecture_v2-2.drawio` and `Untitled Diagram.png`
**Preprocessed data:** [Google Drive folder](https://drive.google.com/drive/folders/1eWYfso6ILulAiVoFZAmbB5eADBGRGxPb?usp=sharing)

## Current State

Completed through the Week 1 batch path:

- Data sources are identified and downloaded or staged: PhysioNet EEGMMIDB and BCI Competition IV-2a.
- Batch ingestion is complete for metadata/time-series stores: MongoDB and Cassandra.
- Batch preprocessing is complete conceptually and supported by the repo through Stage 1 ICA cleaning and Stage 2 Spark/Delta generation.
- Storage for Week 2 ML should use Delta Lake as the source of truth for training-ready epochs.

The local repository currently has scripts and docs, but no local `delta_lake/` directory. The expected setup path is to download `delta_lake/` from the shared Drive and place it at the project root.

## Setup Checklist

- [x] Created working branch for ML planning: `codex/ml-architecture-plan`.
- [x] Updated the Week 1 setup guide with the shared Google Drive link.
- [x] Added an `ml_core/` package scaffold for Week 2 implementation.
- [x] Added this progress and architecture planning document.
- [ ] Download Drive `delta_lake/` into `projectcerebro/delta_lake/`.
- [ ] Verify the primary Delta table exists at `delta_lake/epochs_mi_v1_ch5_sr128_bp8_30`.
- [ ] Run a quick Delta read smoke test before implementing model training.

## Training Data Contract

Use the Stage 2 Delta output as the only training source for Week 2.

Primary path:

```text
delta_lake/epochs_mi_v1_ch5_sr128_bp8_30
```

Ablation path:

```text
delta_lake/epochs_mi_v1_ch5_sr128_bp4_38
```

Each epoch row contains a flattened `features` array that must be reshaped to:

```text
(channels=5, samples=512)
```

The fixed channel order is:

```text
FZ, C3, CZ, C4, PZ
```

Labels:

```text
0 = left
1 = right
2 = rest
```

## ML Architecture Plan

Week 2 should be implemented as a reproducible batch training pipeline with four layers.

### 1. Data Layer

Build a Delta-backed dataset loader that:

- Reads one Delta table at a time using Spark or Delta Lake APIs.
- Validates `features` length is exactly `2560`.
- Reshapes each epoch to `(5, 512)` and stores labels as integer class IDs.
- Keeps metadata columns beside every sample: `dataset`, `subject_id`, `run_id`, `filter_version`, and `preprocessing_version`.
- Applies training-only normalization after the split to avoid leakage.
- Creates deterministic splits by subject:
  - PhysioNet: pretrain/train/validation/test by subject.
  - BCI IV-2a: fine-tune/evaluate by subject and session where available.

Subject-level splitting is critical. Epoch-level random splitting would leak subject-specific EEG signatures and inflate accuracy.

### 2. Model Layer

Implement three models in this order:

1. **ShallowConvNet baseline**
   - Fastest to train and useful for validating the data loader.
   - Should become the first sanity-check model.

2. **EEGNet**
   - Main Week 2 backbone.
   - Use depthwise temporal/spatial convolutions for compact EEG feature extraction.
   - Export a 128-dimensional penultimate embedding for pgvector.
   - Train first on PhysioNet, then fine-tune on BCI IV-2a.

3. **EEG Conformer**
   - Higher-capacity model for local spatial plus global temporal attention.
   - Use after EEGNet establishes a reliable baseline.
   - Compare against EEGNet on macro F1, balanced accuracy, latency, and calibration.

### 3. Training Layer

Use PyTorch as the core training stack.

Training phases:

1. **Smoke test**
   - Load 100-500 epochs.
   - Overfit a small subset.
   - Confirm loss decreases and class labels align.

2. **Baseline training**
   - Train ShallowConvNet on `bp8_30`.
   - Record metrics and confusion matrix.

3. **EEGNet pretraining**
   - Train on PhysioNet subjects.
   - Save checkpoint, training config, preprocessing metadata, and label map.

4. **BCI fine-tuning**
   - Load the PhysioNet EEGNet checkpoint.
   - Fine-tune on BCI IV-2a.
   - Freeze early layers for the first short run, then compare against full fine-tuning.

5. **Conformer comparison**
   - Train only after the loader/training loop is stable.
   - Keep the same splits so metrics are comparable.

Core metrics:

- Accuracy
- Macro F1
- Per-class precision and recall
- Balanced accuracy
- Confusion matrix
- Expected calibration error or reliability curve
- Subject-wise performance table

### 4. Experiment and Storage Layer

Use the existing architecture services this way:

- **Delta Lake:** immutable source of training epochs.
- **MLflow:** experiment params, metrics, artifacts, checkpoints, and model registry.
- **pgvector:** 128-dimensional trial embeddings from the best EEGNet encoder.
- **Redis:** later inference cache, not needed for first training implementation.
- **Ray Tune:** later hyperparameter search after one stable training run exists.

Recommended artifact layout:

```text
artifacts/
  mlruns/                 # MLflow local tracking, ignored by git
  checkpoints/            # model checkpoints, ignored by git
  reports/                # generated metric summaries
```

## Implementation Sequence

1. Add `ml_core/data/delta_dataset.py`
   - Read Delta rows.
   - Validate schema.
   - Reshape features.
   - Build deterministic subject splits.

2. Add `ml_core/models/shallow_convnet.py`
   - Implement the fast baseline first.

3. Add `ml_core/training/train_baseline.py`
   - Train/evaluate loop, metrics, checkpoint save.
   - Use CLI args for table path, batch size, epochs, split seed, and output dir.

4. Add `ml_core/models/eegnet.py`
   - Include an encoder method that returns a 128-dimensional embedding.

5. Add `ml_core/training/train_eegnet.py`
   - PhysioNet pretraining and BCI fine-tuning modes.

6. Add `ml_core/evaluation/metrics.py`
   - Confusion matrix, macro F1, balanced accuracy, per-subject report.

7. Add `ml_core/embeddings/export_pgvector.py`
   - Load best EEGNet checkpoint.
   - Generate embeddings from Delta epochs.
   - Insert into `trial_embeddings` in PostgreSQL/pgvector.

8. Add MLflow integration
   - Start local tracking.
   - Log params, metrics, model artifacts, and dataset metadata.

9. Add Ray Tune only after the single-run training path is reliable.

## Risks and Decisions

- **Decision:** Start with 3-class classification: left, right, rest.
- **Decision:** Treat BCI feet/tongue labels as future work because current preprocessing emits left/right/rest only.
- **Decision:** Use `bp8_30` first because it is the standard motor imagery band.
- **Risk:** Five-channel training is efficient and consistent, but may limit accuracy compared with full-channel models.
- **Risk:** Rest class distribution differs between PhysioNet explicit T0 rest and BCI synthetic gap-mined rest.
- **Risk:** Without subject-level splits, metrics will look better than true deployment performance.
- **Risk:** Local training may be CPU-bound on macOS unless batch size/model size is tuned.

## Next Implementation Target

The next coding milestone should be:

```text
Delta loader + subject split + ShallowConvNet smoke training
```

That gives the project a verified ML path before adding EEGNet, EEG Conformer, pgvector export, or Ray Tune.
