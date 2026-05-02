# Chiranjeev Progress

## Week 2 — ML Core

Branch: `chiranjeev/ml-core-week2`

### Status

- Implementing the Week 2 ML training path on top of the Stage 2 Delta Lake epoch output.
- The primary dataset path is `delta_lake/epochs_mi_v1_ch5_sr128_bp8_30`.
- The ablation dataset path is `delta_lake/epochs_mi_v1_ch5_sr128_bp4_38`.
- The Delta path suffix uses `bp8_30`, while the stored `filter_version` column uses `bp_8_30_v1`.

### Locked Contract

- Input features are flattened arrays of length `2560`.
- Training tensors are shaped `(1, 5, 512)` for CNN models.
- Channel order is `FZ, C3, CZ, C4, PZ`.
- Labels are `0=left`, `1=right`, `2=rest`.
- Splits are deterministic by `subject_id`, never random by epoch.
- EEGNet embeddings are 128-dimensional and exported to existing pgvector storage.

### Milestones

- [x] Add requirements for MLflow, pytest, pytest-cov, torchmetrics, and tqdm.
- [x] Add data schema validation, Delta loading, subject splits, normalization, and Torch dataset adapters.
- [x] Add ShallowConvNet and EEGNet models.
- [x] Add reusable trainer, checkpointing, MLflow callback, metrics, and subject evaluation helpers.
- [x] Add smoke/baseline/pretrain/fine-tune experiment entrypoints.
- [x] Add pgvector embedding export.
- [x] Add unit tests for loader, splits, models, training, metrics, and optional DB export.
- [ ] Download shared Drive `delta_lake/` locally before real training runs.
- [ ] Run full baseline and EEGNet training on real data.
