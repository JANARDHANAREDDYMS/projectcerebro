# ProjectCerebro ML Core

This package is the Week 2 implementation surface for the batch-training path.
It should consume the Stage 2 Delta Lake outputs and produce versioned model
artifacts, metrics, embeddings, and experiment records.

## Input Contract

The current preprocessing output is a Delta Lake table under one of:

- `delta_lake/epochs_mi_v1_ch5_sr128_bp8_30`
- `delta_lake/epochs_mi_v1_ch5_sr128_bp4_38`

Each row is one motor-imagery epoch with:

- `features`: flattened `float32` signal of shape `(5, 512)`
- `label_code`: `0=left`, `1=right`, `2=rest`
- `label_name`: `left`, `right`, or `rest`
- `dataset`: `physionet` or `bci_iv_2a`
- `subject_id`, `run_id`, `filter_version`, `preprocessing_version`
- `channel_names`: fixed order `FZ,C3,CZ,C4,PZ`
- `sampling_rate_hz`: `128.0`

## Planned Modules

- `data`: Delta Lake reader, deterministic splits, normalization, and PyTorch datasets.
- `models`: EEGNet, EEG Conformer, ShallowConvNet baseline, and embedding heads.
- `training`: train/evaluate loops, checkpointing, early stopping, and calibration.
- `evaluation`: subject-level metrics, confusion matrices, calibration curves, and reports.
- `experiments`: MLflow/Ray Tune configuration entrypoints.
- `embeddings`: pgvector export for trial similarity and prediction explanations.

## Implementation Rules

- Split by subject, not by epoch, for honest generalization estimates.
- Keep PhysioNet pretraining and BCI fine-tuning as separate experiment phases.
- Treat `bp8_30` as the primary signal path and `bp4_38` as an ablation.
- Preserve preprocessing metadata in every model artifact and metric record.
- Export a 128-dimensional embedding from the best model for pgvector.
