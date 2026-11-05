# Perceptual-hash audit interpretation

The audit computes a 64-bit perceptual hash for each image. The primary audit flags cross-split pHash near-duplicate candidate pairs whose Hamming distance is at most 5. The repository also includes a threshold-sensitivity summary for stricter (<=3) and more permissive (<=8) cutoffs.

The provided primary-threshold run found:

- 8260 total images
- 223 cross-split candidate pairs
- 129 train-test candidate pairs
- 72 train-validation candidate pairs
- 22 validation-test candidate pairs
- 78 same-label candidate pairs
- 145 different-label candidate pairs

Unique images involved:

- Train: 182 / 5778 (3.15%)
- Validation: 72 / 826 (8.72%)
- Test: 102 / 1656 (6.16%)

These pairs should be interpreted as pHash near-duplicate candidate pairs, not as confirmed duplicates.

**Update for the revision (supersedes the earlier statement that no subject information is available).** The file names follow a seven-digit-identifier-plus-side pattern (for example `9003430L.png`) consistent with participant identifiers. Each of the 4130 identifiers occurs exactly twice (once L, once R), no identifier occurs in more than one split, and no candidate pair shares an identifier. Read as subject identifiers, this indicates that the official split is subject-disjoint, that each knee is represented by one image, and that all 223 candidates link different subjects. This reading is an inference from the naming pattern: the public files contain no documentation linking file names to clinical records, and acquisition-session identifiers are unavailable. The conclusions about the candidates therefore do not rest on it alone: retraining the key models after removing the 182 training and 72 validation candidate images (test split unchanged) did not change test performance or the accuracy on the 102 affected test images (`revision/`).


## Threshold sensitivity

The threshold-sensitivity summary reports:

| Hamming threshold | Overall pairs | Train-test | Train-validation | Validation-test | Train-test affected test images | Any-cross-split affected test images |
|---:|---:|---:|---:|---:|---:|---:|
| <=3 | 10 | 5 | 4 | 1 | 5 | 6 |
| <=5 | 223 | 129 | 72 | 22 | 91 | 102 |
| <=8 | 22480 | 13799 | 6702 | 1979 | 1260 | 1278 |

The <=3 threshold is very strict and detects only a small number of pairs. The <=8 threshold is deliberately permissive and flags a large fraction of the test split; therefore, the primary pHash-clean sensitivity analysis uses the intermediate <=5 threshold.

The fact that 145 of 223 primary-threshold candidates have different KL labels supports interpreting many candidates as visually similar but distinct radiographs, follow-up/contralateral images, or grading-ambiguous cases rather than literal same-image duplicates. This interpretation is consistent with the filename-identifier check above, which remains a naming-pattern inference. The manual visual review below classified all 223 pairs as different knees.

## Pixel-level duplicate check (revision)

`notebooks/05_phash_visual_review_kaggle.ipynb` computed the SSIM (256 x 256 grey level) of every
candidate pair and of two reference sets (`outputs/phash_audit/visual_review/`):

| Pair set | n | 5% | median | 95% | max |
|---|---:|---:|---:|---:|---:|
| Candidate pairs | 223 | 0.639 | 0.728 | 0.875 | 0.941 |
| Same identifier, L vs mirrored R | 223 | 0.635 | 0.783 | 0.901 | 0.941 |
| Random cross-split, same grade | 223 | 0.516 | 0.698 | 0.812 | 0.908 |

No candidate pair exceeded SSIM 0.95, so none is a pixel-level duplicate. The notebook also renders
review sheets (not redistributed here because they contain dataset images) and a per-pair review
template.

## Manual visual review (revision)

All 223 pairs were reviewed visually by the author on contrast-stretched
side-by-side renderings with difference images, comparing femoral/tibial contours, tibial spines,
joint-space shape and fibular anatomy. Codes: D = same image, S = probably the same knee, N = different
knees, U = undecidable. Result: N = 223, D = 0, S = 0, U = 0. Per-pair verdicts:
`outputs/phash_audit/visual_review/visual_review_completed.csv`.
