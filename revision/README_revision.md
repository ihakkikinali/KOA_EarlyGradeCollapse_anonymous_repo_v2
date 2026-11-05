# Revision package (BPEX-106665, major revision)

This folder contains everything added for the revision. The training code itself is unchanged:
`notebooks/04_revision_experiments_kaggle.ipynb` embeds the original training functions verbatim
(`koa_core.py`) and only adds an optional train/validation exclusion filter, an extended prediction
export, and resumable job bookkeeping.

## Contents

| Path | Content |
|---|---|
| `../notebooks/04_revision_experiments_kaggle.ipynb` | Kaggle notebook for the 30 revision runs (2x2 imbalance factorial, extra seeds, pHash-clean retraining, ordinal-weight sensitivity). Two T4 GPUs, resumable across sessions. |
| `revision_runs/` | Outputs of the 30 revision runs: per-seed `val/test_predictions_ext.csv` (file name, subject ID, side, softmax and ordinal-head probabilities, blended probabilities), `history.csv`, `config_used.json`, `run_meta.json`, file lists actually used for training/validation. |
| `legacy_runs/` | Stored predictions of the original runs reused in the analyses (WCE + sampler and moderate ordinal, seeds 42/7/123; single-seed exploratory runs). |
| `analysis/revision_analysis.py` | Reproduces all tables and figures of the revised manuscript and supplement from the stored predictions (no training). |
| `analysis/supp_extra.py`, `analysis/fig_pr.py`, `analysis/make_supp_tables.py` | Supplementary table S3, the final Grade-1 PR figure, and LaTeX table bodies. |
| `analysis/results/` | Generated tables (`tables/*.csv`), figures (`figures/*.pdf|png`) and `numbers.json`. |

## Reproduce the analyses

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
cd revision/analysis
python revision_analysis.py      # ~5 min on CPU
python supp_extra.py
python fig_pr.py
python make_supp_tables.py
```

## Subject-level structure (conditional inference)

File names have the form `<7-digit identifier><L|R>.png`, a pattern consistent with participant
identifiers. All 4130 identifiers occur exactly twice (L and R), and no identifier occurs in more than
one official split or in any perceptual-hash candidate pair. We read the identifiers as subject
identifiers; this is an inference from the naming pattern, because the public files contain no
documentation linking file names to clinical records. The check is part of the notebook pre-flight
cell and of `revision_analysis.py` (`numbers.json` -> `subjects`). The pHash-clean retraining does not
depend on this inference.

## Visual review of candidate pairs

`../notebooks/05_phash_visual_review_kaggle.ipynb` (CPU) renders all 223 candidate pairs side by side
with difference images and SSIM, compares their SSIM with reference distributions (same identifier
L vs mirrored R; random same-grade cross-split pairs), and writes a review template CSV.
Result of the manual review: all 223 pairs
classified as different knees (`../outputs/phash_audit/visual_review/visual_review_completed.csv`).

## pHash-clean split

`revision_runs/phash_exclusion_list.json` lists the 254 training/validation images (182 train, 72 val)
that took part in any cross-split perceptual-hash candidate pair (Hamming distance <= 5). They were
removed for the `rev_CLEAN_*` runs; the test split was not changed.

## Run-to-run non-determinism

cuDNN autotuning was enabled (`cudnn.benchmark=True`), so executions with the same seed are not
bitwise identical. Every run analysed is a single execution whose predictions are stored here.
