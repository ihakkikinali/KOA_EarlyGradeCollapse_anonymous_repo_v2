# Revision update (BPEX-106665)

The revised manuscript is supported by the material in [`revision/`](revision/README_revision.md):
30 additional training runs under the unchanged protocol (5-seed 2x2 imbalance factorial, 5-seed
ordinal comparison, 3-seed ordinal-weight sensitivity, 3-seed retraining on a pHash-cleaned split),
per-seed predictions with subject identifiers, and a script that regenerates every table and figure.
Earlier material below is kept unchanged for traceability.

# Anonymous reproducibility package

This package contains code, configuration files, selected de-identified outputs, a perceptual-hash split audit, and pHash-clean test-sensitivity outputs for the knee osteoarthritis early-grade collapse study.

It intentionally does **not** include manuscript PDFs, LaTeX files, cover letters, author names, affiliations, journal correspondence, trained model checkpoints, raw medical images, or unnecessary training logs.

## Contents

- `notebooks/`: reproducibility notebooks with outputs cleared for anonymous review.
- `scripts/`: utility scripts, including the perceptual-hash cross-split audit.
- `configs/`: experiment configuration summaries.
- `outputs/metrics/`: selected final metrics needed to reproduce manuscript tables.
- `outputs/predictions/`: de-identified test-set prediction tables used for ensemble/confusion summaries.
- `outputs/phash_audit/`: generated perceptual-hash audit outputs with relative public dataset paths only.
- `outputs/phash_clean_sensitivity/`: official-test versus pHash-clean-test sensitivity outputs generated from prediction CSV files.
- `data/`: dataset layout instructions.
- `docs/`: reproducibility and audit interpretation notes.

## Dataset

The raw Knee Osteoarthritis Dataset with Severity is not redistributed here. Place the Kaggle dataset in the expected train/validation/test folder structure described in `data/README_dataset.md`.

## pHash audit

The primary audit identified 223 cross-split pHash near-duplicate candidate pairs at Hamming distance <= 5. These included 129 train-test, 72 train-validation, and 22 validation-test candidate pairs. Threshold-sensitivity outputs are also included: threshold <= 3 identified 10 pairs, whereas threshold <= 8 identified 22480 pairs and was considered too permissive for clean-test evaluation. No candidate pair shares a filename identifier. The file names follow a seven-digit-identifier-plus-side pattern (for example `9003430L.png`) consistent with participant identifiers; each of the 4130 identifiers occurs once as L and once as R, and no identifier occurs in more than one split. Read as subject identifiers, they indicate that the official split is subject-disjoint and that the candidates link different subjects. This is an inference from the naming pattern: the public files contain no documentation linking file names to clinical records, and acquisition-session identifiers are unavailable. The cleaned-split retraining in `revision/` does not depend on this inference. See `outputs/phash_audit/`, `docs/phash_audit_interpretation.md`, and `revision/README_revision.md`.

## pHash-clean sensitivity analysis

Notebook `notebooks/03_phash_clean_test_sensitivity_analysis.ipynb` recomputes ensemble metrics after excluding test images that have train-test pHash near-duplicate candidate pairs. It also reports a stricter any-cross-split clean subset. This analysis does not retrain models; it filters existing test prediction CSV files and recomputes metrics.

Key generated files are in `outputs/phash_clean_sensitivity/`:

- `phash_clean_sensitivity_metrics.csv`
- `phash_clean_sensitivity_delta_vs_official.csv`
- `phash_clean_sensitivity_per_class.csv`
- `phash_clean_summary_table.csv`
- `../phash_audit/phash_threshold_sensitivity_summary.csv`
- `affected_test_images_train_test.csv`
- `affected_test_images_any_cross_split.csv`

Summary of the pHash-clean sensitivity results:

| Model output | Test subset | n | Accuracy | QWK | MAE | Macro-F1 | Grade-1 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation-bias ensemble | Official | 1656 | 0.7065 | 0.8368 | 0.3557 | 0.6920 | 0.2973 |
| Validation-bias ensemble | Train-test pHash-clean | 1565 | 0.7093 | 0.8389 | 0.3534 | 0.6929 | 0.2989 |
| Validation-bias ensemble | Any-cross-split pHash-clean | 1554 | 0.7091 | 0.8382 | 0.3539 | 0.6926 | 0.3011 |
| Uncalibrated ensemble | Official | 1656 | 0.7035 | 0.8376 | 0.3563 | 0.6883 | 0.2804 |
| Uncalibrated ensemble | Train-test pHash-clean | 1565 | 0.7061 | 0.8410 | 0.3527 | 0.6889 | 0.2811 |
| Uncalibrated ensemble | Any-cross-split pHash-clean | 1554 | 0.7059 | 0.8403 | 0.3533 | 0.6886 | 0.2832 |

The clean-subset metrics remain stable relative to the official test set, indicating that the detected pHash near-duplicate candidate relationships do not drive the ensemble results.

## Anonymity note

This repository is prepared for anonymous peer review. Please do not add manuscript source files, local paths, personal Kaggle usernames, or author-identifying metadata before submission.
