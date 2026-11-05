"""
Revision analyses for BPEX-106665 (knee OA early-grade collapse).

Inputs (paths set below or via environment variables):
  LEGACY_DIR : outputs of the original runs (per-seed val/test predictions, bias vectors)
  NEW_DIR    : outputs of the revision runs (koa_revision_runs/)
  HASH_CSV   : phash_all_image_hashes_sanitized.csv (file list with split/grade)
  AFFECTED   : affected_test_images_any_cross_split.csv (pHash audit)
Outputs: results/tables/*.csv, results/figures/*.pdf|png, results/numbers.json
No model is trained here; everything is computed from saved predictions.
"""
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, recall_score,
                             precision_score, confusion_matrix, precision_recall_curve,
                             average_precision_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO = HERE.parent.parent
LEGACY_DIR = Path(os.environ.get("LEGACY_DIR", REPO / "revision" / "legacy_runs"))
NEW_DIR = Path(os.environ.get("NEW_DIR", REPO / "revision" / "revision_runs"))
HASH_CSV = Path(os.environ.get("HASH_CSV", REPO / "outputs" / "phash_audit" / "phash_all_image_hashes_sanitized.csv"))
AFFECTED = Path(os.environ.get("AFFECTED", REPO / "outputs" / "phash_clean_sensitivity" / "affected_test_images_any_cross_split.csv"))
OUT = Path(os.environ.get("OUT_DIR", HERE / "results"))
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
NUM = {}
RNG = np.random.default_rng(20261105)
B = 2000
SEEDS5 = [42, 7, 123, 2024, 2025]
SEEDS3 = [42, 7, 123]
TRAIN_COUNTS = np.array([2286, 1046, 1516, 757, 173])

LEGACY = {
    ("rev_F_WCE_sampler", 42): "ce_baseline_efficientnet_b4_380/seed_42",
    ("rev_F_WCE_sampler", 7): "ce_baseline_efficientnet_b4_380/seed_7",
    ("rev_F_WCE_sampler", 123): "ce_baseline_efficientnet_b4_380/seed_123",
    ("rev_ORD_main", 42): "oa_ordnet_full_efficientnet_b4_380/seed_42",
    ("rev_ORD_main", 7): "oa_ordnet_full_efficientnet_b4_380/seed_7",
    ("rev_ORD_main", 123): "oa_ordnet_full_efficientnet_b4_380/seed_123",
    ("rev_F_CE_nosampler", 42): "ablation_SE_CE_no_sampler_efficientnet_b4_380_SE_ce_nosampler_ord0p0_rank0p0/seed_42",
}
FACT = {"CE": "rev_F_CE_nosampler", "WCE only": "rev_F_WCE_nosampler",
        "Sampler only": "rev_F_CE_sampler", "WCE + sampler": "rev_F_WCE_sampler"}
ORD = "rev_ORD_main"
METRICS = ["qwk", "acc", "mae", "macro_f1", "g1_recall", "g1_precision", "g1_f1", "g1_spec"]


# ----------------------------------------------------------------- helpers
def metrics(y, p):
    y1, p1 = (y == 1), (p == 1)
    return dict(
        qwk=cohen_kappa_score(y, p, weights="quadratic"), acc=accuracy_score(y, p),
        mae=float(np.abs(y - p).mean()),
        macro_f1=f1_score(y, p, average="macro", labels=range(5), zero_division=0),
        g1_recall=recall_score(y1, p1, zero_division=0), g1_precision=precision_score(y1, p1, zero_division=0),
        g1_f1=f1_score(y1, p1, zero_division=0), g1_spec=float(((~p1) & (~y1)).sum() / (~y1).sum()))


def load(exp, seed, split="test", field="prob"):
    p = NEW_DIR / exp / f"seed_{seed}" / f"{split}_predictions_ext.csv"
    if p.exists():
        d = pd.read_csv(p, dtype={"subject_id": str}).sort_values("idx")
        return d.y_true.values, d[[f"{field}_{c}" for c in range(5)]].values
    d = pd.read_csv(LEGACY_DIR / LEGACY[(exp, seed)] / f"{split}_predictions.csv").sort_values("idx")
    assert field == "prob"
    return d.y_true.values, d[[f"prob_{c}" for c in range(5)]].values


def file_table(split):
    h = pd.read_csv(HASH_CSV)
    h = h[h.split == split].sort_values("relative_path").reset_index(drop=True)
    h["filename"] = h.relative_path.str.split("/").str[-1]
    h["subject_id"] = h.filename.str[:7]
    return h


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); s = x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return m, s, m - h, m + h


def ece(y, P, nb=15):
    c, pr = P.max(1), P.argmax(1); e = 0.0
    for lo, hi in zip(np.linspace(0, 1, nb + 1)[:-1], np.linspace(0, 1, nb + 1)[1:]):
        m = (c > lo) & (c <= hi)
        if m.any():
            e += m.mean() * abs((pr[m] == y[m]).mean() - c[m].mean())
    return e


def brier(y, P):
    return float(((P - np.eye(5)[y]) ** 2).sum(1).mean())


def fmt(m, s=None, d=3):
    return f"{m:.{d}f}" if s is None else f"{m:.{d}f} ± {s:.{d}f}"


# ----------------------------------------------------------------- 0. alignment + subject structure
test_files, val_files = file_table("test"), file_table("val")
y_test, y_val = test_files.grade.values, val_files.grade.values
for (e, s) in LEGACY:
    assert (load(e, s)[0] == y_test).all() and (load(e, s, "val")[0] == y_val).all()
h = pd.read_csv(HASH_CSV); h["sid"] = h.relative_path.str.split("/").str[-1].str[:7]
spl = h.groupby("sid").split.nunique()
NUM["subjects"] = {"n_subjects": int(h.sid.nunique()), "subjects_in_multiple_splits": int((spl > 1).sum()),
                   "per_split": h.groupby("split").sid.nunique().to_dict(),
                   "images_per_subject": h.groupby("sid").size().value_counts().to_dict()}
lr = h.assign(side=h.relative_path.str.split("/").str[-1].str[7]).pivot_table(index="sid", columns="side", values="grade", aggfunc="first")
NUM["subjects"]["LR_same_grade_fraction"] = float((lr.L == lr.R).mean())
NUM["subjects"]["LR_grade_spearman"] = float(stats.spearmanr(lr.L, lr.R).correlation)
subj = test_files.subject_id.values
uniq = np.unique(subj)
groups = [np.where(subj == u)[0] for u in uniq]


def cluster_boot(fn, B=B):
    out = []
    for _ in range(B):
        ii = np.concatenate([groups[j] for j in RNG.integers(0, len(groups), len(groups))])
        out.append(fn(ii))
    return np.array(out)


# ----------------------------------------------------------------- 1. per-seed table of all arms
ARMS = {**FACT, "Ordinal (moderate)": ORD,
        "pHash-clean WCE + sampler": "rev_CLEAN_WCE_sampler", "pHash-clean ordinal": "rev_CLEAN_ORD_main",
        "Ordinal λ 0.05/0.02": "rev_SENS_l005_r002", "Ordinal λ 0.35/0.20": "rev_SENS_l035_r020"}
rows = []
for arm, exp in ARMS.items():
    for s in SEEDS5:
        if not ((NEW_DIR / exp / f"seed_{s}" / "DONE.json").exists() or (exp, s) in LEGACY):
            continue
        y, P = load(exp, s); yv, Pv = load(exp, s, "val")
        m = metrics(y, P.argmax(1)); mv = metrics(yv, Pv.argmax(1))
        rows.append({"arm": arm, "experiment": exp, "seed": s, **m, "ece": ece(y, P), "brier": brier(y, P),
                     "val_acc": mv["acc"], "val_qwk": mv["qwk"], "val_score": 0.85 * mv["acc"] + 0.15 * mv["qwk"]})
per_seed = pd.DataFrame(rows)
per_seed.to_csv(OUT / "tables" / "per_seed_all_arms.csv", index=False)
summ = per_seed.groupby("arm", sort=False)[METRICS + ["ece", "brier"]].agg(["mean", "std", "count"])
summ.to_csv(OUT / "tables" / "summary_all_arms.csv")

# ----------------------------------------------------------------- 2. 2x2 factorial (5 seeds)
f = per_seed[per_seed.arm.isin(FACT)].copy()
f["w"] = np.where(f.arm.isin(["WCE only", "WCE + sampler"]), 1, -1)
f["s"] = np.where(f.arm.isin(["Sampler only", "WCE + sampler"]), 1, -1)
f["seedc"] = f.seed.astype(str); f["neg_mae"] = -f.mae
eff = []
for m in ["qwk", "acc", "neg_mae", "macro_f1", "g1_recall", "g1_precision", "g1_f1", "g1_spec"]:
    fit = smf.ols(f"{m} ~ w*s + C(seedc)", data=f).fit()
    for term, lab, k in [("w", "Weighted CE (main effect)", 2), ("s", "Sampler (main effect)", 2), ("w:s", "Interaction", 4)]:
        ci = fit.conf_int().loc[term] * k
        eff.append({"metric": m, "term": lab, "estimate": fit.params[term] * k, "ci_lo": ci[0], "ci_hi": ci[1], "p": fit.pvalues[term]})
eff = pd.DataFrame(eff)
eff["p_holm"] = np.nan
for t in eff.term.unique():
    i = eff.term == t
    eff.loc[i, "p_holm"] = multipletests(eff.loc[i, "p"], method="holm")[1]
eff.to_csv(OUT / "tables" / "factorial_effects.csv", index=False)

# factorial ensembles (5 seeds) + subject-cluster bootstrap vs CE
ens = {a: np.mean([load(e, s)[1] for s in SEEDS5], 0) for a, e in {**FACT, "Ordinal (moderate)": ORD}.items()}
ens_pred = {a: P.argmax(1) for a, P in ens.items()}
ens_rows = []
for a, p in ens_pred.items():
    ens_rows.append({"arm": a, **metrics(y_test, p), "ece": ece(y_test, ens[a]), "brier": brier(y_test, ens[a])})
pd.DataFrame(ens_rows).to_csv(OUT / "tables" / "ensemble5_metrics.csv", index=False)
cm_rows = {a: confusion_matrix(y_test, p, labels=range(5)) for a, p in ens_pred.items()}
json.dump({a: c.tolist() for a, c in cm_rows.items()}, open(OUT / "tables" / "ensemble5_confusion.json", "w"), indent=1)

# ----------------------------------------------------------------- 3. Ordinal vs WCE+sampler (5 paired seeds)
def paired(armA, armB, seeds, tag):
    A = per_seed[(per_seed.arm == armA) & per_seed.seed.isin(seeds)].set_index("seed").sort_index()
    Bq = per_seed[(per_seed.arm == armB) & per_seed.seed.isin(seeds)].set_index("seed").sort_index()
    r = []
    for m in METRICS:
        d = (Bq[m] - A[m]) * (-1 if m == "mae" else 1)
        mu, sd, lo, hi = tci(d)
        r.append({"metric": ("-" if m == "mae" else "") + m, "mean_diff": mu, "sd": sd, "ci_lo": lo, "ci_hi": hi,
                  "p": stats.ttest_rel(Bq[m], A[m]).pvalue, "n_seeds": len(d)})
    r = pd.DataFrame(r); r["p_holm"] = multipletests(r.p, method="holm")[1]
    r.to_csv(OUT / "tables" / f"paired_{tag}.csv", index=False)
    return r

ord_vs = paired("WCE + sampler", "Ordinal (moderate)", SEEDS5, "ordinal_minus_wce_5seeds")
ord_vs_3 = paired("WCE + sampler", "Ordinal (moderate)", SEEDS3, "ordinal_minus_wce_original3seeds")
# ensemble-level cluster bootstrap (5 seeds each)
yv = y_test
bt = cluster_boot(lambda ii: [metrics(yv[ii], ens_pred["Ordinal (moderate)"][ii])[k] - metrics(yv[ii], ens_pred["WCE + sampler"][ii])[k]
                              for k in ["qwk", "acc", "macro_f1", "g1_recall", "g1_spec"]], B=1000)
NUM["ordinal_vs_wce_ensemble5_cluster_bootstrap"] = {k: [float(bt[:, i].mean()), *map(float, np.percentile(bt[:, i], [2.5, 97.5]))]
                                                     for i, k in enumerate(["qwk", "acc", "macro_f1", "g1_recall", "g1_spec"])}

# ----------------------------------------------------------------- 4. pHash-clean retraining
aff = pd.read_csv(AFFECTED); aff_col = [c for c in aff.columns if "path" in c][0]
affected_mask = test_files.relative_path.isin(set(aff[aff_col])).values
NUM["phash"] = {"affected_test_images": int(affected_mask.sum())}
ph = []
for lab, exp in [("Official WCE + sampler", "rev_F_WCE_sampler"), ("pHash-clean WCE + sampler", "rev_CLEAN_WCE_sampler"),
                 ("Official ordinal", ORD), ("pHash-clean ordinal", "rev_CLEAN_ORD_main")]:
    for s in SEEDS3:
        y, P = load(exp, s); p = P.argmax(1)
        ph.append({"model": lab, "seed": s, **{f"full_{k}": v for k, v in metrics(y, p).items()},
                   "acc_affected102": float((p[affected_mask] == y[affected_mask]).mean()),
                   "acc_unaffected": float((p[~affected_mask] == y[~affected_mask]).mean())})
ph = pd.DataFrame(ph); ph.to_csv(OUT / "tables" / "phash_clean_retraining_per_seed.csv", index=False)
phs = ph.groupby("model", sort=False).agg(["mean", "std"])
phs.to_csv(OUT / "tables" / "phash_clean_retraining_summary.csv")
p_wce = paired("WCE + sampler", "pHash-clean WCE + sampler", SEEDS3, "phashclean_minus_official_wce")
p_ord = paired("Ordinal (moderate)", "pHash-clean ordinal", SEEDS3, "phashclean_minus_official_ordinal")

# ----------------------------------------------------------------- 5. Lambda sensitivity (3 seeds, blend fixed 0.10)
LAM = {"0 / 0": "WCE + sampler", "0.05 / 0.02": "Ordinal λ 0.05/0.02", "0.15 / 0.08": "Ordinal (moderate)", "0.35 / 0.20": "Ordinal λ 0.35/0.20"}
sens = per_seed[per_seed.arm.isin(LAM.values()) & per_seed.seed.isin(SEEDS3)].copy()
sens["lam"] = sens.arm.map({v: k for k, v in LAM.items()})
sens["lamnum"] = sens.lam.str.split(" ").str[0].astype(float); sens["seedc"] = sens.seed.astype(str)
sens_tests = []
from statsmodels.stats.anova import anova_lm
for m in ["qwk", "acc", "mae", "macro_f1", "g1_recall", "g1_spec"]:
    a = anova_lm(smf.ols(f"{m} ~ C(lam) + C(seedc)", data=sens).fit())
    tr = smf.ols(f"{m} ~ lamnum + C(seedc)", data=sens).fit()
    sens_tests.append({"metric": m, "omnibus_p": a.loc["C(lam)", "PR(>F)"], "trend_per_0.1": tr.params["lamnum"] * 0.1, "trend_p": tr.pvalues["lamnum"]})
pd.DataFrame(sens_tests).to_csv(OUT / "tables" / "lambda_sensitivity_tests.csv", index=False)
sens.groupby("lam", sort=False)[METRICS + ["val_score"]].agg(["mean", "std"]).to_csv(OUT / "tables" / "lambda_sensitivity_summary.csv")
sel = sens.loc[sens.groupby("seed").val_score.idxmax(), ["seed", "lam", "val_score", "qwk", "acc", "g1_recall"]]
sel.to_csv(OUT / "tables" / "lambda_validation_selection.csv", index=False)
NUM["lambda_selected_by_mean_val_score"] = sens.groupby("lam").val_score.mean().idxmax()

# ----------------------------------------------------------------- 6. Inference blend and ordinal decoding
dec = []
blend_runs = [(ORD, 2024), (ORD, 2025)] + [("rev_CLEAN_ORD_main", s) for s in SEEDS3] + \
             [(e, s) for e in ["rev_SENS_l005_r002", "rev_SENS_l035_r020"] for s in SEEDS3]
for e, s in blend_runs:
    y, C = load(e, s, field="cls_prob"); _, O = load(e, s, field="ord_prob")
    cum = np.stack([O[:, t + 1:].sum(1) for t in range(4)], 1)
    for b in [0, 0.05, 0.10, 0.25, 0.5, 1.0]:
        dec.append({"run": f"{e}/{s}", "decoder": f"blend {b:.2f}", "blend": b, **metrics(y, ((1 - b) * C + b * O).argmax(1))})
    dec.append({"run": f"{e}/{s}", "decoder": "ordinal head, threshold counting", "blend": np.nan, **metrics(y, (cum > 0.5).sum(1))})
    dec.append({"run": f"{e}/{s}", "decoder": "ordinal head, max P(y=1)", "blend": np.nan, "maxP1": float(O[:, 1].max()), **metrics(y, O.argmax(1))})
dec = pd.DataFrame(dec); dec.to_csv(OUT / "tables" / "blend_and_decoding_per_run.csv", index=False)
dec.groupby("decoder", sort=False)[METRICS].agg(["mean", "std"]).to_csv(OUT / "tables" / "blend_and_decoding_summary.csv")
NUM["ordinal_head_max_P_grade1"] = float(dec["maxP1"].max())

# ----------------------------------------------------------------- 7. Validation-bias calibration audit (original runs)
def bias_probs(P, b):
    l = np.log(np.clip(P, 1e-12, 1)) + b; e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
LEG_EXP = {"WCE + sampler": "ce_baseline_efficientnet_b4_380", "Ordinal (moderate)": "oa_ordnet_full_efficientnet_b4_380"}
ba = []
for arm, d in LEG_EXP.items():
    exp = FACT["WCE + sampler"] if arm == "WCE + sampler" else ORD
    for s in SEEDS3:
        b = np.load(LEGACY_DIR / d / f"seed_{s}" / "best_class_bias.npy")
        row = {"arm": arm, "seed": s, **{f"bias_{c}": float(b[c]) for c in range(5)}}
        for split in ["val", "test"]:
            y, P = load(exp, s, split)
            m0, m1 = metrics(y, P.argmax(1)), metrics(y, bias_probs(P, b).argmax(1))
            for k in ["acc", "qwk", "macro_f1", "g1_recall"]:
                row[f"{split}_delta_{k}"] = m1[k] - m0[k]
        ba.append(row)
ba = pd.DataFrame(ba); ba.to_csv(OUT / "tables" / "validation_bias_audit.csv", index=False)
NUM["bias_audit"] = {"val_delta_acc_mean": float(ba.val_delta_acc.mean()), "test_delta_acc_mean": float(ba.test_delta_acc.mean()),
                     "val_improved_runs": int((ba.val_delta_acc > 0).sum()), "test_improved_runs": int((ba.test_delta_acc > 0).sum())}

# ----------------------------------------------------------------- 8. A priori logit adjustment of plain CE
la = []
for tau in [0, 0.25, 0.5, 0.75, 1.0]:
    for s in SEEDS5:
        y, P = load(FACT["CE"], s)
        la.append({"tau": tau, "seed": s, **metrics(y, (np.log(np.clip(P, 1e-12, 1)) - tau * np.log(TRAIN_COUNTS / TRAIN_COUNTS.sum())).argmax(1))})
la = pd.DataFrame(la); la.to_csv(OUT / "tables" / "logit_adjustment_ce_per_seed.csv", index=False)
la.groupby("tau")[METRICS].agg(["mean", "std"]).to_csv(OUT / "tables" / "logit_adjustment_ce_summary.csv")

# ----------------------------------------------------------------- 9. Grade-1 operating points (validation-selected) and PR
op_rows, pr_curves = [], {}
for arm in ["WCE + sampler", "Ordinal (moderate)", "CE"]:
    exp = {**FACT, "Ordinal (moderate)": ORD}[arm]
    Pv = np.mean([load(exp, s, "val")[1] for s in SEEDS5], 0); Pt = ens[arm]
    prec, rec, _ = precision_recall_curve(y_test == 1, Pt[:, 1])
    pr_curves[arm] = (rec, prec, average_precision_score(y_test == 1, Pt[:, 1]))
    offs = np.linspace(-1, 3, 401)
    def shift(P, o):
        L = np.log(np.clip(P, 1e-12, 1)); L[:, 1] += o; return L.argmax(1)
    vrec = np.array([recall_score(y_val == 1, shift(Pv, o) == 1) for o in offs])
    for target in ["default", 0.40, 0.50, 0.60, 0.70]:
        o = 0.0 if target == "default" else offs[np.argmax(vrec >= target)]
        pt = shift(Pt, o)
        m = metrics(y_test, pt)
        bt = cluster_boot(lambda ii: [metrics(y_test[ii], pt[ii])[k] for k in ["g1_recall", "g1_precision", "g1_spec", "qwk"]], B=500)
        op_rows.append({"arm": arm, "val_target_g1_recall": target, "grade1_log_offset": o, **m,
                        **{f"{k}_ci": f"[{np.percentile(bt[:, i], 2.5):.3f}, {np.percentile(bt[:, i], 97.5):.3f}]"
                           for i, k in enumerate(["g1_recall", "g1_precision", "g1_spec", "qwk"])},
                        "g1_to_g0": int(((y_test == 1) & (pt == 0)).sum()), "g0_to_g1": int(((y_test == 0) & (pt == 1)).sum())})
ops = pd.DataFrame(op_rows); ops.to_csv(OUT / "tables" / "grade1_operating_points.csv", index=False)
NUM["grade1_average_precision"] = {a: float(v[2]) for a, v in pr_curves.items()}

# Grade-1 metrics with CIs, per arm (per-seed t-intervals)
g1 = []
for arm in ["CE", "WCE only", "Sampler only", "WCE + sampler", "Ordinal (moderate)"]:
    d = per_seed[per_seed.arm == arm]
    for k in ["g1_recall", "g1_precision", "g1_f1", "g1_spec"]:
        mu, sd, lo, hi = tci(d[k]); g1.append({"arm": arm, "metric": k, "mean": mu, "sd": sd, "ci_lo": lo, "ci_hi": hi})
pd.DataFrame(g1).to_csv(OUT / "tables" / "grade1_metrics_with_ci.csv", index=False)

# ----------------------------------------------------------------- 10. Calibration (5 seeds, both main arms)
cal = per_seed[per_seed.arm.isin(["CE", "WCE only", "Sampler only", "WCE + sampler", "Ordinal (moderate)"])].groupby("arm", sort=False)[["ece", "brier"]].agg(["mean", "std"])
cal.to_csv(OUT / "tables" / "calibration_summary.csv")

# ================================================================= FIGURES
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
COL = {"CE": "#7f7f7f", "WCE only": "#1f77b4", "Sampler only": "#2ca02c", "WCE + sampler": "#d62728", "Ordinal (moderate)": "#9467bd"}

def save(fig, name):
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / "figures" / f"{name}.png", bbox_inches="tight"); plt.close(fig)

# Fig: factorial per-seed dot plot
fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.6))
order = ["CE", "WCE only", "Sampler only", "WCE + sampler", "Ordinal (moderate)"]
lab = ["CE", "WCE", "Sampler", "WCE+\nsampler", "Ordinal"]
for ax, (m, t) in zip(axes, [("qwk", "QWK (primary)"), ("macro_f1", "Macro-F1"), ("g1_recall", "Grade-1 recall"), ("g1_spec", "Grade-1 specificity")]):
    for i, a in enumerate(order):
        v = per_seed[per_seed.arm == a][m].values
        ax.scatter(np.full(len(v), i) + RNG.uniform(-0.08, 0.08, len(v)), v, s=14, color=COL[a], alpha=0.8, zorder=3)
        mu, sd, lo, hi = tci(v); ax.errorbar(i + 0.22, mu, yerr=[[mu - lo], [hi - mu]], fmt="o", color="k", ms=3.5, capsize=2, lw=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, fontsize=7); ax.set_title(t, fontsize=9)
    ax.axvline(3.5, color="0.8", lw=0.8, ls="--")
fig.tight_layout(); save(fig, "fig_factorial_seed_level")

# Fig: confusion matrices (5-seed ensembles) for CE and WCE+sampler
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
for ax, a in zip(axes, ["CE", "WCE + sampler", "Ordinal (moderate)"]):
    c = cm_rows[a]; r = c / c.sum(1, keepdims=True)
    ax.imshow(r, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{c[i, j]}\n{100 * r[i, j]:.0f}%", ha="center", va="center", fontsize=6.5, color="white" if r[i, j] > 0.55 else "black")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xlabel("Predicted KL grade"); ax.set_ylabel("True KL grade")
    ax.set_title(f"{a} (5-seed ensemble)", fontsize=8.5)
fig.tight_layout(); save(fig, "fig_confusion_ensembles")

# Fig: Grade-1 PR curves + validation-selected operating points
fig, ax = plt.subplots(figsize=(4.2, 3.4))
for a, (rc, pc, ap) in pr_curves.items():
    ax.plot(rc, pc, color=COL[a], lw=1.3, label=f"{a} (AP {ap:.3f})")
for _, r in ops[ops.arm != "CE"].iterrows():
    ax.scatter(r.g1_recall, r.g1_precision, color=COL[r.arm], s=22, marker="o" if r.val_target_g1_recall == "default" else "s", edgecolor="k", lw=0.5, zorder=4)
ax.axhline((y_test == 1).mean(), color="0.6", ls=":", lw=0.8)
ax.set_xlabel("Grade-1 recall (test)"); ax.set_ylabel("Grade-1 precision (test)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=7, frameon=False); fig.tight_layout(); save(fig, "fig_grade1_pr_operating_points")

# Fig: reliability diagrams (5-seed ensembles)
fig, ax = plt.subplots(figsize=(3.6, 3.4))
bins = np.linspace(0, 1, 11)
for a in ["CE", "WCE + sampler", "Ordinal (moderate)"]:
    P = ens[a]; c = P.max(1); ok = P.argmax(1) == y_test; xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (c > lo) & (c <= hi)
        if m.sum() >= 10:
            xs.append(c[m].mean()); ys.append(ok[m].mean())
    ax.plot(xs, ys, "o-", color=COL[a], ms=3, lw=1.1, label=f"{a} (ECE {ece(y_test, P):.3f})")
ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=0.8); ax.set_xlabel("Mean top-class confidence"); ax.set_ylabel("Observed accuracy")
ax.legend(fontsize=7, frameon=False); fig.tight_layout(); save(fig, "fig_reliability")

# Fig: lambda sensitivity and blend
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
lams = list(LAM.keys())
for k, (m, c, t) in enumerate([("qwk", "#1f77b4", "QWK"), ("g1_recall", "#d62728", "Grade-1 recall")]):
    ax = axes[0]
    for i, l in enumerate(lams):
        v = sens[sens.lam == l][m].values
        ax.scatter(np.full(len(v), i) + (k - 0.5) * 0.18, v, color=c, s=12, alpha=0.7)
        ax.errorbar(i + (k - 0.5) * 0.18, v.mean(), yerr=v.std(ddof=1), fmt="_", color=c, capsize=3, ms=10)
    ax.plot([], [], "o", color=c, label=t)
axes[0].set_xticks(range(4)); axes[0].set_xticklabels(lams, fontsize=7.5); axes[0].set_xlabel(r"$\lambda_{CORAL}$ / $\lambda_{rank}$ (blend 0.10, $\delta$ 1.0)")
axes[0].legend(fontsize=7, frameon=False); axes[0].set_title("Ordinal weight (3 seeds)", fontsize=9)
bs = dec[dec.blend.notna()].groupby("blend")[["qwk", "acc", "g1_recall", "g1_spec"]].agg(["mean", "std"])
for m, c, t in [("qwk", "#1f77b4", "QWK"), ("acc", "#2ca02c", "Accuracy"), ("g1_recall", "#d62728", "Grade-1 recall")]:
    axes[1].errorbar(bs.index, bs[(m, "mean")], yerr=bs[(m, "std")], color=c, marker="o", ms=3, lw=1.1, capsize=2, label=t)
axes[1].set_xlabel("Inference blend weight of ordinal-head probabilities"); axes[1].set_title("Inference blend (11 ordinal runs)", fontsize=9)
axes[1].legend(fontsize=7, frameon=False); fig.tight_layout(); save(fig, "fig_lambda_blend")

# Fig: validation-bias audit
fig, ax = plt.subplots(figsize=(3.6, 3.0))
for arm, mk in [("WCE + sampler", "o"), ("Ordinal (moderate)", "s")]:
    d = ba[ba.arm == arm]; ax.scatter(d.val_delta_acc, d.test_delta_acc, marker=mk, color=COL[arm], s=28, label=arm, edgecolor="k", lw=0.4)
ax.axhline(0, color="0.6", lw=0.8); ax.axvline(0, color="0.6", lw=0.8)
ax.set_xlabel("Δ accuracy on validation (tuned)"); ax.set_ylabel("Δ accuracy on test"); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); save(fig, "fig_validation_bias_audit")

json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1, default=str)
print("done")
