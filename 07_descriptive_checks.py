"""Common threshold rule and descriptive checks (Online Resource 1, analysis (vii) and Tables S20-S22). Saved outputs only; no model is refitted.

N2  common threshold rule: gamma selected on inner-val with IPCW labels and weights for every cell (main analysis: own rule)
N3  follow-up support by year of diagnosis and horizon
N4  computational cost from the saved run diagnostics
N5  deaths shortly after diagnosis

Run from this folder after notebooks 01-04 (README.md):  python 07_descriptive_checks.py
"""
import json
import platform
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
import seer_modeling as sm

ROOT = config.WORK_DIR
MODEL_DIR = ROOT / "outputs_modeling"
OUT = ROOT / "outputs_descriptive"


def n2_common_threshold(data, splits):
    main_m = pd.read_csv(ROOT / "outputs_evaluation" / "metrics_by_split.csv")
    recs = []
    for r, k in splits:
        split = sm.load_split(MODEL_DIR, r, k)
        iv, te = split["role"].eq("inner_val").to_numpy(), split["role"].eq("test").to_numpy()
        rows_iv, rows_te = split["row"].to_numpy()[iv], split["row"].to_numpy()[te]
        for tau in sm.HORIZONS:
            y_all = data.labels[f"y_{tau}"].to_numpy()
            y_iv, y_te = y_all[rows_iv], y_all[rows_te]
            w_iv, w_te = split[f"w_{tau}"].to_numpy(float)[iv], split[f"w_{tau}"].to_numpy(float)[te]
            v_rows, yv, wv = sm.cell_rows("ipcw", y_iv, w_iv)
            det = y_te >= 0
            yt, wt = (y_te[det] == 1).astype(int), w_te[det]
            for col in [c for c in split.columns if c.endswith(f"|{tau}")]:
                model, cell, _ = col.split("|")
                p_iv, p_te = split[col].to_numpy(float)[iv], split[col].to_numpy(float)[te]
                gamma = sm.select_threshold(p_iv[v_rows], yv, wv)
                cm = sm.classification_metrics(p_te[det], yt, wt, gamma)
                recs.append({"repeat": r, "fold": k, "tau": tau, "model": model, "cell": cell, "gamma_common": gamma,
                             "sensitivity_common": cm["sensitivity"], "fpr_common": cm["fpr"],
                             "ppv_common": cm["precision"]})
    m = pd.DataFrame(recs).merge(main_m[["repeat", "fold", "tau", "model", "cell", "gamma", "sensitivity", "fpr",
                                         "precision"]], on=["repeat", "fold", "tau", "model", "cell"])
    m.to_csv(OUT / "threshold_common_rule_by_split.csv", index=False)
    rows = []
    for (tau, model, cell), g in m.groupby(["tau", "model", "cell"], sort=False):
        g = g.sort_values(["repeat", "fold"])
        rec = {"tau": tau, "model": model, "cell": cell,
               "sens_own": g["sensitivity"].mean(), "sens_common": g["sensitivity_common"].mean(),
               "fpr_own": g["fpr"].mean(), "fpr_common": g["fpr_common"].mean(),
               "ppv_own": g["precision"].mean(), "ppv_common": g["ppv_common"].mean()}
        for met, a, b in [("sens", "sensitivity_common", "sensitivity"), ("ppv", "ppv_common", "precision")]:
            t = sm.corrected_ttest((g[a] - g[b]).to_numpy())
            rec.update({f"{met}_diff": t["mean"], f"{met}_ci_low": t["ci_low"], f"{met}_ci_high": t["ci_high"],
                        f"{met}_p": t["p_value"]})
        rows.append(rec)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "threshold_common_rule_summary.csv", index=False)
    return s


def n3_followup_support(data):
    c = data.cohort
    rows = []
    for year in sorted(c["year_dx"].unique()):
        sel = c["year_dx"].eq(year).to_numpy()
        for tau in sm.HORIZONS:
            y = data.labels[f"y_{tau}"].to_numpy()[sel]
            rows.append({"year_dx": int(year), "tau": tau, "n": int(sel.sum()), "determinable": int((y >= 0).sum()),
                         "censored_before_tau": int((y < 0).sum()), "pct_censored": 100 * float((y < 0).mean()),
                         "max_followup_months": int(c.loc[sel, "V"].max())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "followup_support_by_year.csv", index=False)
    return t


def n4_compute(splits):
    recs, cells = [], []
    for r, k in splits:
        j = json.loads((MODEL_DIR / f"split_r{r}_k{k}.json").read_text(encoding="utf-8"))
        recs.append({"repeat": r, "fold": k, "censoring_model_s": j["censoring_model_seconds"],
                     "cs_cox_s": j["references"]["cs_cox"]["fit_predict_seconds"],
                     "cs_rsf_s": j["references"]["cs_rsf"]["fit_predict_seconds"], "total_s": j["total_seconds"]})
        for key, v in j["cells"].items():
            model, cell, tau = key.split("|")
            cells.append({"repeat": r, "fold": k, "model": model, "cell": cell, "tau": int(tau),
                          "fit_s": v["fit_seconds"], "n_rows": v.get("n_rows_after", v["n_rows"])})
    per_split = pd.DataFrame(recs)
    per_cell = pd.DataFrame(cells)
    summary = (per_cell.groupby(["cell", "model"])[["fit_s", "n_rows"]].mean().round(1).reset_index())
    per_split.to_csv(OUT / "compute_times_per_split.csv", index=False)
    summary.to_csv(OUT / "compute_times_by_cell.csv", index=False)
    machine = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()}
    (OUT / "machine_now.json").write_text(json.dumps(machine, indent=1), encoding="utf-8")
    return per_split, summary, machine


def n5_early_deaths(data):
    c = data.cohort
    rows = []
    for months in (1, 2, 3):
        sel = c["V"].le(months)
        rows.append({"followup_months_at_most": months, "deaths_all_causes": int((sel & c["status"].gt(0)).sum()),
                     "breast_cancer_deaths": int((sel & c["status"].eq(1)).sum())})
    t = pd.DataFrame(rows)
    t["breast_cancer_deaths_total"] = int(c["status"].eq(1).sum())
    t.to_csv(OUT / "early_deaths.csv", index=False)
    return t


def main():
    OUT.mkdir(exist_ok=True)
    data = sm.load_prep(ROOT / "outputs_model_prep")
    splits = sorted((int(m[1]), int(m[2])) for m in
                    (re.match(r"split_r(\d+)_k(\d+)\.parquet", p.name) for p in MODEL_DIR.glob("split_r*_k*.parquet")) if m)
    pd.set_option("display.width", 220)
    s = n2_common_threshold(data, splits)
    print("N2 (10 years, sensitivity and PPV, own rule vs common IPCW rule):")
    print(s[s.tau.eq(120)][["model", "cell", "sens_own", "sens_common", "sens_diff", "sens_p", "ppv_own", "ppv_common",
                            "ppv_diff", "ppv_p"]].round(4).to_string(index=False))
    print("\nN3:"); print(n3_followup_support(data).to_string(index=False))
    per_split, summary, machine = n4_compute(splits)
    print("\nN4 per split (s):", per_split[["censoring_model_s", "cs_cox_s", "cs_rsf_s", "total_s"]].describe().round(0).to_string())
    print(summary.to_string(index=False)); print(machine)
    print("\nN5:"); print(n5_early_deaths(data).to_string(index=False))


if __name__ == "__main__":
    main()
