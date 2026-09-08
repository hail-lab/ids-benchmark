# Heliyon revision (HELIYON-D-26-03019)

Experiments added during major revision, answering the reviewers' methodological
comments. Each script is self-contained, resumable, and writes one CSV per
experiment into `results_r1/tables/`.

## Experiments and the comments they answer

| Script | Experiment | Reviewer comments |
|---|---|---|
| `e1_splits.py` | Stratified vs host/day-grouped vs temporal cross-validation | R6.1, R6.2, R3.1 |
| `e2_transfer.py` | Cross-dataset transfer on a common 13-feature schema | R6.3 |
| `e3_nested_fs.py` | Feature selection nested inside each training fold | R6.4, R1.8 |
| `e4_lgbm_sensitivity.py` | Seven configurations for the LightGBM multi-class collapse, incl. SMOTE | R6.6, R7.2, R1.17 |
| `e5_fairness_gpu.py` | Subsample sensitivity curve; matched imbalance handling | R1.15, R1.16, R3.4, R6.5 |
| `e67_dl_colab.py` | FT-Transformer benchmark; hyperparameter search for 1D-CNN and BiLSTM | R3.2, R7.1 |
| `e8a_latency.py` | Inference latency for every trained model | R1.11, R1.18 |
| `e8b_shap_multi.py` | Per-class SHAP attribution and fold-to-fold ranking stability | R3.3, R6.12, R1.10 |
| `e8c_effect_sizes.py` | Kendall's W, pairwise effect sizes, per-class per-fold counts | R1.19, R6.7, R6.8, R7.3 |

## Data preparation

`prep_groups.py` rebuilds the datasets retaining capture-day and source-host
metadata, which the grouped and temporal splits of `e1_splits.py` require.

`fix_labels.py` writes label-corrected copies of the cleaned datasets. Two
defects in the submitted pipeline are corrected there and documented in the
manuscript: UNSW-NB15 carried both `Backdoor` and `Backdoors` for one attack
category (merged; 11 classes to 10), and CICIDS2017's web-attack labels were
mis-decoded from Latin-1 as UTF-8. Corrected copies are written to `data_r1/`
rather than overwriting the originals, so the submitted results remain
reproducible; `utils.dataset_path()` resolves to the corrected copy when one
exists.

`recompute_perclass.py` regenerates the per-class F1 table on the corrected
labels.

## Running

Datasets are not included (see the repository `.gitignore`); obtain them from
the providers cited in the paper and run the pipeline in `../src` to produce
`data/clean/`. Then:

```bash
cd experiments/src
python fix_labels.py          # label corrections
python prep_groups.py         # grouped/temporal split metadata
python run_remaining.py       # E1-E5, E8 in sequence, resumable
python run_e7_local.py        # FT-Transformer  (GPU, several hours)
python run_e6_local.py        # deep-model tuning (GPU)
```

`experiments/notebooks/revision_dl_colab.ipynb` runs the GPU experiments on
Google Colab instead, for machines without a local CUDA device.

## Reproducibility notes

All experiments use a fixed seed (42). Every script skips work already recorded
in its output CSV, so interrupted runs resume rather than restart. Fits that
depend on an iteration budget record whether they converged, after an earlier
version of the subsample experiment produced a result that turned out to be
iteration starvation rather than an effect of training-set size.
