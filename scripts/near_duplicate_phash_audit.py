#!/usr/bin/env python3
"""
Image-level near-duplicate audit for the public Knee Osteoarthritis Dataset with Severity.

This script checks whether visually similar images occur across train/validation/test folders.
It is an image-level perceptual-hash safeguard; it cannot prove patient-level independence because
the public Kaggle release does not provide patient identifiers.

Expected layouts supported
--------------------------
DATA_ROOT/
  train/0/*.png
  train/1/*.png
  ...
  valid/0/*.png     or val/0/*.png or validation/0/*.png
  test/0/*.png

The script also searches one directory level below DATA_ROOT if the dataset is nested inside an
archive folder.

Example
-------
python scripts/near_duplicate_phash_audit.py \\
  --data_root /kaggle/input/knee-osteoarthritis-dataset-with-severity \\
  --out_dir /kaggle/working/phash_audit \\
  --hash_size 8 \\
  --threshold 5

Outputs
-------
phash_all_images.csv
phash_cross_split_near_duplicate_candidates.csv
phash_audit_summary.csv
phash_counts_by_split_pair.csv
phash_same_grade_summary.csv
phash_unique_images_involved_by_split.csv
phash_threshold_sensitivity_summary.csv
phash_cross_split_candidates_threshold_max.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES: Dict[str, List[str]] = {
    "train": ["train", "training", "Train", "Training"],
    "val": ["val", "valid", "validation", "Val", "Valid", "Validation"],
    "test": ["test", "testing", "Test", "Testing"],
}


def _dct2(x: np.ndarray) -> np.ndarray:
    """2-D DCT using OpenCV if available, otherwise SciPy."""
    try:
        import cv2  # type: ignore
        return cv2.dct(x.astype(np.float32))
    except Exception:
        try:
            from scipy.fftpack import dct  # type: ignore
            return dct(dct(x.astype(np.float32), axis=0, norm="ortho"), axis=1, norm="ortho")
        except Exception as exc:
            raise RuntimeError(
                "This script needs either OpenCV (cv2) or SciPy for the DCT step. "
                "On Kaggle, OpenCV is usually available by default. Otherwise install opencv-python or scipy."
            ) from exc


def phash_hex(path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Compute a pHash-like hexadecimal string using grayscale resize + DCT median thresholding."""
    img_size = hash_size * highfreq_factor
    with Image.open(path) as im:
        im = im.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(im, dtype=np.float32)
    dct = _dct2(pixels)
    low = dct[:hash_size, :hash_size].copy()
    # Exclude the DC component when computing the median threshold.
    vals = low.flatten()
    med = np.median(vals[1:]) if vals.size > 1 else np.median(vals)
    bits = vals > med
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    hex_len = (hash_size * hash_size + 3) // 4
    return f"{value:0{hex_len}x}"


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def find_split_dir(root: Path, split: str) -> Path:
    candidates = []
    for base in [root] + [p for p in root.iterdir() if p.is_dir()]:
        for name in SPLIT_ALIASES[split]:
            p = base / name
            if p.exists() and p.is_dir():
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"Could not find a {split!r} directory under {root}")
    # Prefer the shortest path to avoid accidentally picking nested duplicates.
    return sorted(candidates, key=lambda p: len(str(p)))[0]


def collect_images(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    split_dirs = {split: find_split_dir(root, split) for split in ["train", "val", "test"]}
    for split, split_dir in split_dirs.items():
        for path in sorted(split_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            rows.append({
                "split": split,
                "grade": path.parent.name,
                "path": str(path),
                "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            })
    if not rows:
        raise RuntimeError(f"No image files found under {root}")
    return pd.DataFrame(rows)


def compute_hashes(df: pd.DataFrame, hash_size: int) -> pd.DataFrame:
    hashes: List[str] = []
    for p in tqdm(df["path"].tolist(), desc="Computing pHash"):
        hashes.append(phash_hex(Path(p), hash_size=hash_size))
    out = df.copy()
    out["phash"] = hashes
    return out


def cross_split_pairs(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    records = df.to_dict("records")
    for i in tqdm(range(len(records)), desc="Checking cross-split pairs"):
        r1 = records[i]
        for j in range(i + 1, len(records)):
            r2 = records[j]
            if r1["split"] == r2["split"]:
                continue
            dist = hamming_hex(str(r1["phash"]), str(r2["phash"]))
            if dist <= threshold:
                rows.append({
                    "split_a": r1["split"],
                    "grade_a": r1["grade"],
                    "relative_path_a": r1["relative_path"],
                    "phash_a": r1["phash"],
                    "split_b": r2["split"],
                    "grade_b": r2["grade"],
                    "relative_path_b": r2["relative_path"],
                    "phash_b": r2["phash"],
                    "hamming_distance": dist,
                })
    if not rows:
        return pd.DataFrame(columns=[
            "split_a", "grade_a", "relative_path_a", "phash_a",
            "split_b", "grade_b", "relative_path_b", "phash_b", "hamming_distance"
        ])
    return pd.DataFrame(rows).sort_values(["hamming_distance", "split_a", "split_b"]).reset_index(drop=True)



def summarize_pairs(pairs: pd.DataFrame, df_hash: pd.DataFrame, threshold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return split-pair counts, same-grade summary, and unique-image involvement summary."""
    if pairs.empty:
        counts = pd.DataFrame(columns=["split_a", "split_b", "candidate_pairs"])
        same_grade = pd.DataFrame(columns=["same_grade", "candidate_pairs"])
        unique = pd.DataFrame({
            "split": ["train", "val", "test"],
            "unique_images_involved": [0, 0, 0],
        })
    else:
        p = pairs.copy()
        p["same_grade"] = p["grade_a"].astype(str) == p["grade_b"].astype(str)
        counts = p.groupby(["split_a", "split_b"]).size().reset_index(name="candidate_pairs")
        same_grade = p.groupby("same_grade").size().reset_index(name="candidate_pairs")
        rows = []
        for split in ["train", "val", "test"]:
            rels = set(p.loc[p["split_a"] == split, "relative_path_a"].astype(str))
            rels |= set(p.loc[p["split_b"] == split, "relative_path_b"].astype(str))
            rows.append({"split": split, "unique_images_involved": len(rels)})
        unique = pd.DataFrame(rows)
    split_sizes = df_hash["split"].value_counts().to_dict()
    unique["split_size"] = unique["split"].map(split_sizes).fillna(0).astype(int)
    unique["percent_of_split"] = (100 * unique["unique_images_involved"] / unique["split_size"].replace(0, np.nan)).round(2)
    return counts, same_grade, unique


def threshold_sensitivity_summary(all_pairs: pd.DataFrame, thresholds: List[int]) -> pd.DataFrame:
    """Summarize candidate-pair counts and affected test images at multiple thresholds."""
    rows: List[Dict[str, object]] = []
    for th in thresholds:
        p = all_pairs[all_pairs["hamming_distance"] <= th].copy()
        pair_counts = p.groupby(["split_a", "split_b"]).size().to_dict() if not p.empty else {}
        same = int((p["grade_a"].astype(str) == p["grade_b"].astype(str)).sum()) if not p.empty else 0
        diff = int((p["grade_a"].astype(str) != p["grade_b"].astype(str)).sum()) if not p.empty else 0
        tt = p[((p["split_a"] == "train") & (p["split_b"] == "test")) | ((p["split_a"] == "test") & (p["split_b"] == "train"))]
        tt_test = set(tt.loc[tt["split_a"] == "test", "relative_path_a"].astype(str))
        tt_test |= set(tt.loc[tt["split_b"] == "test", "relative_path_b"].astype(str))
        any_test = set(p.loc[p["split_a"] == "test", "relative_path_a"].astype(str))
        any_test |= set(p.loc[p["split_b"] == "test", "relative_path_b"].astype(str))
        rows.append({
            "threshold": int(th),
            "candidate_pairs": int(len(p)),
            "train_test_pairs": int(pair_counts.get(("train", "test"), 0) + pair_counts.get(("test", "train"), 0)),
            "train_val_pairs": int(pair_counts.get(("train", "val"), 0) + pair_counts.get(("val", "train"), 0)),
            "val_test_pairs": int(pair_counts.get(("val", "test"), 0) + pair_counts.get(("test", "val"), 0)),
            "same_label_pairs": same,
            "different_label_pairs": diff,
            "affected_test_images_train_test": int(len(tt_test)),
            "affected_test_images_any_cross_split": int(len(any_test)),
        })
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root folder containing train/val/test image folders")
    parser.add_argument("--out_dir", default="phash_audit", help="Directory where audit outputs will be saved")
    parser.add_argument("--hash_size", type=int, default=8, help="pHash size; 8 gives 64-bit hashes")
    parser.add_argument("--threshold", type=int, default=5, help="Primary Hamming distance threshold for candidate near-duplicates")
    parser.add_argument("--threshold_sensitivity", default="3,5,8", help="Comma-separated Hamming thresholds for sensitivity summary")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_images(root)
    df_hash = compute_hashes(df, hash_size=args.hash_size)
    pairs = cross_split_pairs(df_hash, threshold=args.threshold)

    all_images_csv = out_dir / "phash_all_images.csv"
    pairs_csv = out_dir / "phash_cross_split_near_duplicate_candidates.csv"
    summary_csv = out_dir / "phash_audit_summary.csv"

    df_hash.to_csv(all_images_csv, index=False)
    pairs.to_csv(pairs_csv, index=False)

    split_counts = df_hash["split"].value_counts().to_dict()
    summary = {
        "n_images": int(len(df_hash)),
        "n_train": int(split_counts.get("train", 0)),
        "n_val": int(split_counts.get("val", 0)),
        "n_test": int(split_counts.get("test", 0)),
        "hash_size": int(args.hash_size),
        "threshold": int(args.threshold),
        "n_cross_split_near_duplicate_candidates": int(len(pairs)),
        "all_images_csv": str(all_images_csv),
        "candidate_pairs_csv": str(pairs_csv),
    }
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print("\\nNear-duplicate pHash audit summary")
    print("----------------------------------")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
