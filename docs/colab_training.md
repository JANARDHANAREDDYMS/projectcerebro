# ProjectCerebro Colab Training Guide

Use this guide when training the Week 2 ML core in Google Colab with the
preprocessed `ProjectCerebro-Shared` data from Google Drive.

## One-Time Google Drive Setup

The data is shared through **Shared with me**, so add a shortcut into **My Drive**
before opening Colab:

1. Open Google Drive.
2. Go to **Shared with me**.
3. Find `ProjectCerebro-Shared`.
4. Right click, then choose **Organize** or **Add shortcut to Drive**.
5. Place the shortcut at:

```text
My Drive/projectcerebro/ProjectCerebro-Shared
```

After mounting Drive in Colab, the expected path is:

```text
/content/drive/MyDrive/projectcerebro/ProjectCerebro-Shared
```

## Colab Runtime

In Colab, select:

```text
Runtime -> Change runtime type -> GPU -> T4
```

Then run the notebook cells in:

```text
notebooks/projectcerebro_colab_training.ipynb
```

## Manual Command Sequence

Mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone the repo and switch to the ML branch:

```bash
%cd /content
!git clone https://github.com/JANARDHANAREDDYMS/projectcerebro.git
%cd /content/projectcerebro
!git fetch origin
!git switch chiranjeev/ml-core-week2
```

Set paths for both Python and shell cells:

```python
import os

DATA_ROOT = "/content/drive/MyDrive/projectcerebro/ProjectCerebro-Shared/delta_lake"
BP8_30 = f"{DATA_ROOT}/epochs_mi_v1_ch5_sr128_bp8_30"
BP4_38 = f"{DATA_ROOT}/epochs_mi_v1_ch5_sr128_bp4_38"

os.environ["DATA_ROOT"] = DATA_ROOT
os.environ["BP8_30"] = BP8_30
os.environ["BP4_38"] = BP4_38
```

Verify the Delta Lake folders:

```bash
!ls "$BP8_30"
!ls "$BP8_30/_delta_log"
```

Install dependencies without replacing Colab CUDA PyTorch:

```bash
!grep -v "^torch==" requirements.txt > requirements_colab.txt
!pip install -r requirements_colab.txt
```

Verify GPU:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

Run tests:

```bash
!pytest tests -v
```

Run training in order:

```bash
!python -m ml_core.experiments.train_smoke --filter bp8_30 --delta-path "$BP8_30"

!python -m ml_core.experiments.train_shallow_baseline --filter bp8_30 --delta-path "$BP8_30"

!python -m ml_core.experiments.pretrain_eegnet_physionet --filter bp8_30 --delta-path "$BP8_30"

!python -m ml_core.experiments.finetune_eegnet_bci \
  --filter bp8_30 \
  --delta-path "$BP8_30" \
  --pretrained artifacts/checkpoints/eegnet_pretrain_bp8_30/best.pt
```

Persist artifacts back to Drive before disconnecting Colab:

```bash
!mkdir -p "/content/drive/MyDrive/projectcerebro/training_artifacts"
!cp -r artifacts/checkpoints artifacts/reports artifacts/mlruns "/content/drive/MyDrive/projectcerebro/training_artifacts/"
```

## Expected T4 Runtime

- Setup and dependency install: 5-15 minutes
- Tests: under 1 minute
- Smoke training: 2-5 minutes
- ShallowConvNet baseline: 3-10 minutes
- EEGNet PhysioNet pretrain: 8-20 minutes
- EEGNet BCI fine-tune: 3-10 minutes

Expected first full run: **30-60 minutes**.

## Accounts Needed

- Required: Google account with Colab and Drive access.
- Not required: Databricks, Snowflake, S3, GCS, paid data lake, MLflow cloud, or pgvector cloud.
