# Chiranjeev — Next-Iteration Plan

**Branch (current):** `chiranjeev/ml-core-week2-v2`
**New work branch:** `chiranjeev/loso-and-cho2017`

## Context

Baselines on the bp8_30 Delta table are now in (`docs/chiranjeev-progress.md`):

- ShallowConvNet: acc 52.9 %, macro F1 0.523 on a single 70/15/15 subject split.
- EEGNet from scratch on BCI only: acc 52.9 %, macro F1 0.495.

Two new requests came in from teammates:

1. **Run leave-one-subject-out (LOSO) on the baselines.** Single random split is noisy; LOSO is the standard honest evaluation for motor-imagery EEG. Janardhan reports 74 % per-subject mean accuracy after fixes; we need a comparable LOSO number.
2. **Add the Cho 2017 (GigaDB 100542) dataset to the pretrain pool.** PhysioNet alone is too sparse per subject for EEGNet to converge well. Cho 2017 adds 52 subjects × 240 trials = ~12 480 trials. Janardhan wrote `docs/new_dataset_addition_steps.md` (on branch `janardhan`) with the recipe.

Neither change is implemented yet. Cho 2017 is two-class (left/right) only, so we mine rest from inter-trial gaps the same way Stage 2 already mines rest for PhysioNet.

## Phase A — LOSO evaluation harness (no new data)

Goal: report per-subject accuracy and macro F1 averaged across all held-out subjects, on both baselines, against the existing bp8_30 table.

**New code:**

- `ml_core/data/splits.py`: add `loso_iter(df, holdout_dataset)` — yields `(train_df, val_df, test_df, manifest)` for each subject. `val_df` = one random training subject (seed-fixed); `test_df` = the held-out subject; `train_df` = the remainder. Reuse `compute_norm_stats` + `apply_norm_stats` per fold (no leakage).
- `ml_core/experiments/loso_eval.py`: argparse entrypoint — model name (`shallowconvnet|eegnet`), Delta path, filter version, holdout dataset (default `bci_iv_2a`), n_epochs, patience. Trains a fresh model per fold. Aggregates per-subject metrics into `artifacts/reports/loso_<model>_<dataset>.json` with mean ± std for accuracy / macro F1 / balanced acc.
- `ml_core/evaluation/subject_eval.py`: extend with `aggregate_loso(per_subject_dict) -> {mean, std, per_subject}` so reports are easy to scan.
- Optional: a thin `--max-subjects` flag for fast smoke runs (3 folds × 5 epochs).

**Tests:**

- `tests/test_loso.py` over `synthetic_epochs_df`: `loso_iter` yields one fold per held-out subject, no overlap, deterministic given seed.

**Expected runtime on M4 MPS:**

- ShallowConvNet (12 k params): 9 BCI subjects × ~2 min = ~18 min.
- EEGNet (34 k params): 9 × ~5 min = ~45 min.
- PhysioNet LOSO (109 subjects) is too slow on MPS — restrict LOSO to BCI IV-2a in this iteration; keep PhysioNet on the random subject split.

**Deliverable:** `artifacts/reports/loso_shallow_bci.json` and `loso_eegnet_bci.json` with per-subject metrics + means.

## Phase B — Cho 2017 ingest

Follow `docs/new_dataset_addition_steps.md` (on `janardhan` branch). Concretely:

1. **Download** the 52 `.mat` files into `data/cho2017/` (manual; ~5–7 GB; gitignored).
2. **Add `pymatreader`** to `requirements.txt` (Janardhan's reader uses `scipy.io.loadmat` directly, which is fine — `pymatreader` is optional). Also add `mne-icalabel` if not already pinned.
3. **Implement `scripts/stage1_cho2017_ingest.py`** following Janardhan's draft. Notes the draft does not call out:
   - Verify the 5 common channels (FZ, C3, CZ, C4, PZ) exist in Cho 2017's 64-channel montage before processing — fail fast otherwise. Cho channel names may differ in case (`Fz` vs `FZ`); upper-case before comparing.
   - Reuse the same `LABEL_LEFT/RIGHT/REST` constants from `scripts/stage2_spark_preprocess.py` to avoid drift.
   - Manifest path `data_cleaned/cho2017_stage1_manifest.jsonl`. Append-only, one JSON per subject.
   - Resume-friendly: skip subjects whose `*_cleaned_raw.fif` already exists.
4. **Patch `scripts/stage2_spark_preprocess.py`:**
   - Add `iter_cho2017_jobs(filter_key, test_mode)` modeled on `iter_physionet_jobs`.
   - Add `extract_cho2017_epochs(...)` (Janardhan supplies a draft) — left/right from annotations, then `extract_rest_epochs` for synthetic rest from inter-trial gaps with `is_rest_synthetic=True`.
   - Wire both into `process_one_file`'s dataset switch.
   - Keep the schema identical: `n_channels=5`, `n_samples=512`, `sampling_rate_hz=128.0`, `channel_names=("FZ","C3","CZ","C4","PZ")`, `filter_version` strings unchanged.
5. **Re-run Stage 2** on the merged dataset:
   ```bash
   python scripts/stage2_spark_preprocess.py --test --filter bp8_30
   python scripts/stage2_spark_preprocess.py --filter both
   ```
6. **Verify Delta Lake** in a small notebook cell:
   ```python
   from deltalake import DeltaTable
   df = DeltaTable("delta_lake/epochs_mi_v1_ch5_sr128_bp8_30").to_pandas()
   df.groupby(["dataset", "label_name"]).size()
   ```
   Expect ~14 k physionet, ~12.5 k cho2017, ~1.3 k bci_iv_2a per filter.

**Loader change:** none. `read_epochs(datasets=["physionet","cho2017"])` already filters by `dataset` and works as soon as the new rows land.

**Time:** Stage 1 takes ~2–3 hours on the 52 Cho subjects. Stage 2 patch + re-run is fast (~20 min for full bp8_30 + bp4_38).

## Phase C — Retrain EEGNet on the expanded pretrain pool

Once Phase B finishes:

```bash
python -m ml_core.experiments.pretrain_eegnet_physionet \
    --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
    --filter-version bp_8_30_v1 \
    --datasets physionet cho2017 \
    --out-dir artifacts/checkpoints/eegnet_pretrain_phys_cho \
    --epochs 80 --batch-size 64 --lr 1e-3 --patience 20 --seed 42
```

Then fine-tune on BCI:

```bash
python -m ml_core.experiments.finetune_eegnet_bci \
    --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
    --filter-version bp_8_30_v1 \
    --pretrained artifacts/checkpoints/eegnet_pretrain_phys_cho/best.pt \
    --out-dir artifacts/checkpoints/eegnet_finetune_phys_cho \
    --epochs 60 --lr 5e-4
```

Run pretrain on Colab T4 (heavy); fine-tune on either device.

## Phase D — Euclidean Alignment (per-subject covariance whitening)

Janardhan's command had `--use-ea`. EA dramatically helps cross-subject EEG models by zero-meaning each subject's epoch covariance matrix:

For each subject s:
$$R_s = \frac{1}{N_s}\sum_i X_i^{(s)} {X_i^{(s)}}^\top, \quad \tilde{X} = R_s^{-1/2} X$$

**New code:**

- `ml_core/data/euclidean_alignment.py`: `compute_subject_R(df) -> dict[sid -> (5,5)]`, `apply_ea(x_cnt, R_inv_sqrt) -> x_cnt`. Uses `scipy.linalg.fractional_matrix_power(R, -0.5)` plus a small Tikhonov regulariser (`R += eps*I`).
- Plumb a `--use-ea` flag through `_common.build_loaders` so EA is applied after z-score normalisation. Persist the per-subject `R_inv_sqrt` next to `norm_stats.json` so inference reuses identical transforms.
- Test: `tests/test_ea.py` confirms whitened covariance ≈ I per subject within 1e-3.

EA is cheap and well-supported in literature for motor imagery — a typical single-fix that buys 5–10 percentage points of accuracy on BCI IV-2a.

## Phase E — LOSO once more

Re-run Phase A's `loso_eval.py` against the new EA + Cho-pretrained EEGNet. This is the apples-to-apples comparison we report:

| Variant | Per-subject acc (mean ± std) |
| --- | --- |
| ShallowConvNet (no pretrain, no EA) | from Phase A |
| EEGNet from scratch (no EA) | from Phase A |
| EEGNet pretrain (PhysioNet only) + fine-tune | new |
| EEGNet pretrain (PhysioNet + Cho) + fine-tune | new |
| EEGNet pretrain (PhysioNet + Cho) + EA + fine-tune | target ≥ 74 % |

Save to `artifacts/reports/loso_summary.md`.

## Risks / decisions

- **Disk:** Cho 2017 is ~5–7 GB raw + ~2× cleaned `.fif`. Confirm laptop has the headroom before downloading. Drive folder is fine for backup.
- **Channel intersection:** if Cho's 64-channel set is missing one of FZ/C3/CZ/C4/PZ (unlikely — these are 10–20 standard), Phase B is blocked until we either widen the common set or interpolate. Verify in Step 4 of Janardhan's doc before running Stage 1 on all 52.
- **Class imbalance:** Cho is left/right only. Synthetic-rest mining keeps rest at ~ trials/2; this matches the existing PhysioNet rest synthesis policy. EEGNet already gets per-class weights via `train_label_array` in the trainer.
- **Reproducibility:** `seed=42` everywhere; persist `split_manifest.json` per fold so any LOSO run can be exactly replayed.
- **MLflow file-store race:** the existing baseline run hit it. Either bump to `--mlflow-uri sqlite:///artifacts/mlruns/mlflow.db` or keep MLflow disabled and rely on JSON artifacts. The simpler fix is sqlite — change the default in `_common.add_common_args`.

## Sequencing / handoff

1. **Start Phase A immediately** (LOSO harness) on the existing bp8_30 table — no data dependency.
2. **Phase B in parallel** if disk + bandwidth allow (Cho 2017 download is the long pole).
3. Phases C and D after B lands.
4. Phase E gates the writeup.

If Cho 2017 download stalls, Janardhan's pretrain plan is blocked, but LOSO + EA work is still useful. Open one PR per phase to keep reviews small:

- PR ① `loso eval harness + reports`
- PR ② `cho2017 ingest (stage1 + stage2 patch)`
- PR ③ `eegnet pretrain on phys+cho + finetune`
- PR ④ `euclidean alignment + final loso writeup`

## Verification

- Phase A: `pytest tests/test_loso.py -v` green; `python -m ml_core.experiments.loso_eval --model shallowconvnet --max-subjects 3 --epochs 3` finishes <10 min and writes a report JSON.
- Phase B: `python scripts/stage2_spark_preprocess.py --test --filter bp8_30` shows `[cho2017] s1/s1 …` lines; Delta verification snippet shows three datasets with expected counts.
- Phase C: pretrain checkpoint exists; fine-tune test report `test_overall.json` shows acc > random (>34 %).
- Phase D: `tests/test_ea.py` green; whitened covariance close to identity per subject.
- Phase E: `loso_summary.md` checked in with all five variants.
