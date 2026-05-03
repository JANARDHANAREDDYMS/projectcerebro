- This week, I am doing implemenation of EEGNet and EEG Conformer.p

ML Layer Research part : 1 May : 10:08 pm

EEGNet
A compact EEG-specific CNN that uses temporal, depthwise, and separable convolutions to learn robust features from limited EEG data. In this project, it serves as the lightweight baseline and the main transfer-learning candidate for pre-train on PhysioNet, then fine-tune on BCI. Source: original EEGNet paper.  ￼

EEG Conformer
A hybrid convolution + Transformer architecture for EEG decoding that combines local convolutional feature extraction with self-attention for longer-range temporal modeling. In this project, it is the higher-capacity model used to test whether attention improves motor imagery decoding and cross-hardware generalization. Source: EEG Conformer paper / Braindecode implementation reference.  ￼

ShallowConvNet
A classic EEG deep learning architecture designed for decoding oscillatory EEG patterns efficiently, especially in motor imagery settings. In this project, it acts as a strong frequency-sensitive baseline and fast-training benchmark against which the more modern models can be compared. Source: original Schirrmeister et al. architecture, also exposed in Braindecode as ShallowFBCSPNet.  ￼

Ensemble + Platt Scaling
A post-training layer that combines predictions from multiple base models and then calibrates confidence scores so predicted probabilities are better aligned with true correctness likelihood. In this project, it is used to improve robustness and make the final 3-class output more trustworthy for downstream use. Source: Platt’s original probability calibration method.  ￼

MLflow
An experiment tracking and model lifecycle platform used to store parameters, metrics, artifacts, and registered model versions. In this project, it provides reproducibility, experiment management, and traceability across preprocessing versions, models, and hyperparameter runs. Source: official MLflow docs.  ￼

Ray Tune
A distributed hyperparameter tuning framework for running and managing many training trials at scale. In this project, it is used to search model and training hyperparameters systematically across EEGNet, EEG Conformer, and ShallowConvNet. Source: official Ray Tune docs.  ￼

Sources:
1. EEGNet
    https://arxiv.org/abs/1611.08024
2. EEG Conformer
    https://www.semanticscholar.org/paper/EEG-Conformer%3A-Convolutional-Transformer-for-EEG-Song-Zheng/db10636e62862c9a4bd3e75012ae0273492ec125
3. ShallowConvNet
    https://arxiv.org/abs/1703.05051
4. Platt Scaling
    https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods
5. MLflow
    https://mlflow.org/
6. Ray Tune
    https://docs.ray.io/en/latest/tune/index.html


    



DOne today for ML:

Done so far:

  1. Merged/available ML core locally
      - Chiranjeev’s ml_core/ structure is present locally.
      - Models, trainer, evaluation, experiment scripts, tests, configs are present.
  2. Implemented missing ml_core/data/ locally
     Added:
      - schema.py
      - delta_loader.py
      - splits.py
      - normalize.py
      - dataset.py
      - __init__.py

     This includes:
      - Delta/Parquet loading
      - schema validation
      - subject-level train/val/test split
      - Euclidean Alignment
      - z-score normalization
      - PyTorch EpochDataset
  3. Integrated EA into experiment loader
      - Added --use-ea
      - EA fitted on training split only
      - z-score fitted on training split only
      - val/test use train-fitted transforms
  4. Installed pytest
     Installed into cerebro_env:
      - pytest==8.2.0
      - pytest-cov==5.0.0
  5. Ran tests
     Full test suite result:

     22 passed, 1 skipped
  6. Ran smoke training
     One-epoch smoke run with EA completed successfully.
  7. Cleaned push-blocking history state
      - Your branch is now back aligned with origin/janardhan.
      - The bad secret-containing commits are no longer ahead of remote.
      - Added ignores for:
          - credentials.json
          - token.json
          - temporary ml_pipeline/
          - temporary scripts/train_ml_pipeline.py