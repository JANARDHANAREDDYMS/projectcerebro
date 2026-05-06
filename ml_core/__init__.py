"""ml_core: Week 2 EEG ML training package.

Modules:
    data        - Delta Lake reader, schema, splits, normalization, datasets.
    models      - ShallowConvNet, EEGNet (and Conformer later).
    training    - Trainer loop, checkpointing, MLflow callbacks.
    evaluation  - Metrics + per-subject aggregation.
    embeddings  - 128-d trial embedding export to pgvector.
    experiments - Entrypoint scripts (smoke, baseline, pretrain, finetune).
"""

__version__ = "0.1.0"
