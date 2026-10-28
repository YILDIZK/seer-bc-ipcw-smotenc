"""Post hoc recalibration analysis (Online Resource 1, analysis (v); planned in writing before it was run).

Every saved model prediction is recalibrated with a logistic map logit(p~) = a + b*logit(p^) (primary) and an
intercept-only map (secondary), fitted by IPCW-weighted maximum likelihood on the determinable inner-validation
patients of the same split and horizon, and evaluated on the untouched test fold exactly as in the main analysis.
No model is refitted; seer_modeling.py is not modified.
Exploratory, not in the plan (Section 3.5): Zero and Discard minus IPCW before and after logistic recalibration.

Run from this folder after notebooks 01-04 (README.md):  python 05_recalibration.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import seer_modeling as sm

ROOT = config.WORK_DIR
MODEL_DIR = ROOT / "outputs_modeling"
OUT_DIR = ROOT / "outputs_recalibration"
NB_THRESHOLDS = np.array([0.05, 0.10, 0.20])
EPS = 1e-6
METHODS = ("original", "logistic", "intercept")


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_recalibration(p, y, w, slope: bool, iters: int = 100, tol: float = 1e-10):
    """IPCW-weighted logistic regression of y on logit(p): returns (a, b). Newton-Raphson; b fixed at 1 if not slope."""
    z = logit(p)
    a, b = 0.0, 1.0
    for _ in range(iters):
        mu = expit(a + b * z)
        r, v = w * (y - mu), w * mu * (1 - mu)
        if slope:
            g = np.array([r.sum(), (r * z).sum()])
            H = np.array([[v.sum(), (v * z).sum()], [(v * z).sum(), (v * z * z).sum()]])
            step = np.linalg.solve(H, g)
            a, b = a + step[0], b + step[1]
        else:
            step = np.array([r.sum() / v.sum(), 0.0])
            a = a + step[0]
        if np.max(np.abs(step)) < tol:
            break
    return float(a), float(b)


def apply_recalibration(p, a, b):
    return expit(a + b * logit(p))


def evaluate(split: pd.DataFrame, data, r: int, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    V, status = data.cohort["V"].to_numpy(), data.cohort["status"].to_numpy()
    iv, te = split["role"].eq("inner_val").to_numpy(), split["role"].eq("test").to_numpy()
    rows_iv, rows_te = split["row"].to_numpy()[iv], split["row"].to_numpy()[te]
    n_test = int(te.sum())
    recs, fits = [], []
    for tau in sm.HORIZONS:
        y_all = data.labels[f"y_{tau}"].to_numpy()
        y_iv, y_te = y_all[rows_iv], y_all[rows_te]
        w_iv, w_te = split[f"w_{tau}"].to_numpy(float)[iv], split[f"w_{tau}"].to_numpy(float)[te]
        dv, dt = y_iv >= 0, y_te >= 0
        yv, wv = (y_iv[dv] == 1).astype(int), w_iv[dv]          # IPCW-weighted natural outcome, every cell alike
        yt, wt = (y_te[dt] == 1).astype(int), w_te[dt]
        cif_aj = sm.aalen_johansen_cif1(V[rows_te], status[rows_te], tau)
        for col in [c for c in split.columns if c.endswith(f"|{tau}")]:
            model, cell, _ = col.split("|")
            rule = "ipcw" if cell == "-" else cell.split("+")[0]
            s_rows, ys, ws = sm.cell_rows(rule, y_iv, w_iv)            # threshold rule of the main analysis
            p_iv_all = split[col].to_numpy(float)[iv]
            p_te_all = split[col].to_numpy(float)[te]
            for method in METHODS:
                if method == "original":
                    a, b = 0.0, 1.0
                    q_iv, q_te = p_iv_all, p_te_all
                else:
                    a, b = fit_recalibration(p_iv_all[dv], yv, wv, slope=(method == "logistic"))
                    q_iv, q_te = apply_recalibration(p_iv_all, a, b), apply_recalibration(p_te_all, a, b)
                    fits.append({"repeat": r, "fold": k, "tau": tau, "model": model, "cell": cell,
                                 "method": method, "a": a, "b": b})
                qt = q_te[dt]
                gamma = sm.select_threshold(q_iv[s_rows], ys, ws)      # TPR >= 0.80 on inner-val, as in the main analysis
                cm = sm.classification_metrics(qt, yt, wt, gamma)
                nb = sm.net_benefit(qt, yt, wt, n_test, NB_THRESHOLDS)
                recs.append({
                    "repeat": r, "fold": k, "tau": tau, "model": model, "cell": cell, "method": method,
                    "brier": float(np.sum(wt * (yt - qt) ** 2) / n_test),
                    "auc": float(roc_auc_score(yt, qt, sample_weight=wt)),
                    "oe_ratio": cif_aj / float(q_te.mean()),
                    "sensitivity": cm["sensitivity"], "ppv": cm["precision"],
                    **{f"nb_{pt:.2f}": v for pt, v in zip(NB_THRESHOLDS, nb)},
                })
    return pd.DataFrame(recs), pd.DataFrame(fits)


def contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    """Oversampled minus non-oversampled among recalibrated models; recalibrated minus original per cell."""
    key = ["repeat", "fold"]
    out = []
    piv = metrics.set_index(key + ["tau", "model", "cell", "method"])
    for metric in ["brier", "oe_ratio", "nb_0.05", "nb_0.10", "nb_0.20", "auc", "ppv"]:
        for method in ("logistic", "intercept", "original"):
            for tau in sm.HORIZONS:
                for model in sm.CLASSIFIERS:
                    for rule in sm.CENSORING_RULES:
                        a = piv.xs((tau, model, f"{rule}+smote", method), level=["tau", "model", "cell", "method"])[metric]
                        b = piv.xs((tau, model, rule, method), level=["tau", "model", "cell", "method"])[metric]
                        d = (a - b).sort_index().to_numpy()
                        t = sm.corrected_ttest(d)
                        out.append({"family": "smote_minus_none", "metric": metric, "method": method, "tau": tau,
                                    "model": model, "contrast": f"{rule}+smote - {rule}", **t})
        for method in ("logistic", "intercept"):
            for tau in sm.HORIZONS:
                for model in list(sm.CLASSIFIERS) + list(sm.REFERENCES):
                    cells = [c for c in metrics.loc[metrics["model"].eq(model), "cell"].unique()]
                    for cell in cells:
                        a = piv.xs((tau, model, cell, method), level=["tau", "model", "cell", "method"])[metric]
                        b = piv.xs((tau, model, cell, "original"), level=["tau", "model", "cell", "method"])[metric]
                        d = (a - b).sort_index().to_numpy()
                        out.append({"family": "recal_minus_original", "metric": metric, "method": method, "tau": tau,
                                    "model": model, "contrast": cell, **sm.corrected_ttest(d)})
    res = pd.DataFrame(out)
    fam = res["family"].eq("smote_minus_none")
    res["p_holm"] = np.nan
    for (metric, method), idx in res[fam].groupby(["metric", "method"]).groups.items():
        res.loc[idx, "p_holm"] = sm.holm_adjust(res.loc[idx, "p_value"].to_numpy())
    return res


def exploratory_censoring_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    """Exploratory, not in the plan: Zero and Discard minus IPCW, before ("original") and after logistic recalibration."""
    rows = []
    for method in ("original", "logistic"):
        for tau in sm.HORIZONS:
            for model in sm.CLASSIFIERS:
                sel = metrics[metrics["method"].eq(method) & metrics["tau"].eq(tau) & metrics["model"].eq(model)]
                for rule in ("zero", "discard"):
                    for metric in ("brier", "nb_0.20", "oe_ratio"):
                        a = sel[sel["cell"].eq(rule)].sort_values(["repeat", "fold"])[metric].to_numpy()
                        b = sel[sel["cell"].eq("ipcw")].sort_values(["repeat", "fold"])[metric].to_numpy()
                        rows.append({"method": method, "tau": tau, "model": model, "contrast": f"{rule} - ipcw",
                                     "metric": metric, **sm.corrected_ttest(a - b)})
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = sm.load_prep(ROOT / "outputs_model_prep")
    splits = sorted((int(m[1]), int(m[2])) for m in
                    (re.match(r"split_r(\d+)_k(\d+)\.parquet", p.name) for p in MODEL_DIR.glob("split_r*_k*.parquet")) if m)
    assert len(splits) == 15, splits
    parts, fit_parts = [], []
    for r, k in splits:
        m, f = evaluate(sm.load_split(MODEL_DIR, r, k), data, r, k)
        parts.append(m)
        fit_parts.append(f)
        print(f"split r{r} k{k} done", flush=True)
    metrics = pd.concat(parts, ignore_index=True)
    fits = pd.concat(fit_parts, ignore_index=True)
    metrics.to_csv(OUT_DIR / "recalibration_metrics_by_split.csv", index=False)
    fits.to_csv(OUT_DIR / "recalibration_parameters_by_split.csv", index=False)

    summary = (metrics.groupby(["tau", "model", "cell", "method"], sort=False)
               .agg(["mean", "std"]).drop(columns=["repeat", "fold"], level=0))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(OUT_DIR / "recalibration_summary.csv", index=False)
    contrasts(metrics).to_csv(OUT_DIR / "recalibration_contrasts.csv", index=False)
    exploratory_censoring_contrasts(metrics).to_csv(OUT_DIR / "exploratory_censoring_contrasts_after_recalibration.csv",
                                                    index=False)

    # Consistency checks: original metrics must reproduce the main analysis; ranking measures must be unchanged.
    main_m = pd.read_csv(ROOT / "outputs_evaluation" / "metrics_by_split.csv")
    orig = metrics[metrics["method"].eq("original")].merge(main_m, on=["repeat", "fold", "tau", "model", "cell"],
                                                           suffixes=("", "_main"))
    print("max |Brier - main|:", float((orig["brier"] - orig["brier_main"]).abs().max()))
    print("max |AUC - main|:", float((orig["auc"] - orig["auc_main"]).abs().max()))
    print("max |O/E - main|:", float((orig["oe_ratio"] - orig["oe_ratio_main"]).abs().max()))
    print("max |PPV - main|:", float((orig["ppv"] - orig["precision"]).abs().max()))
    v = metrics.pivot_table(index=["repeat", "fold", "tau", "model", "cell"], columns="method", values="ppv")
    print("max |PPV logistic - original|:", float((v["logistic"] - v["original"]).abs().max()))
    w = metrics.pivot_table(index=["repeat", "fold", "tau", "model", "cell"], columns="method", values="auc")
    print("max |AUC logistic - original|:", float((w["logistic"] - w["original"]).abs().max()))
    print("max |AUC intercept - original|:", float((w["intercept"] - w["original"]).abs().max()))
    print("slopes b <= 0:", int((fits.loc[fits["method"].eq("logistic"), "b"] <= 0).sum()), "of",
          int(fits["method"].eq("logistic").sum()))


if __name__ == "__main__":
    main()
