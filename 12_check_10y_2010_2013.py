"""10-year check in the test patients diagnosed in 2010-2013 (Online Resource 1, analysis (iii), Table S13).

Patients diagnosed in 2010-2013 had almost complete 10-year follow-up. The saved test-fold predictions of the main
analysis are evaluated in this subgroup without IPCW weights: the determinable patients (label >= 0) for the Brier score,
AUC and sensitivity at gamma*, and O/E as the Aalen-Johansen cumulative incidence in all subgroup patients divided by
their mean predicted risk. gamma* is the threshold of the main analysis (selected on the inner validation set with the
cell's own rule). No model is refitted.

The saved main-run outputs are read from, and the results written to, the working folder (config.WORK_DIR).

Run from this folder after notebooks 01-04 (README.md):  python 12_check_10y_2010_2013.py
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import seer_modeling as sm

DATA_ROOT = config.WORK_DIR
MODEL_DIR = DATA_ROOT / "outputs_modeling"
OUT_DIR = config.WORK_DIR / "outputs_evaluation"
TAU = 120
LAST_YEAR = 2013


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = sm.load_prep(DATA_ROOT / "outputs_model_prep")
    year = pd.to_numeric(data.cohort["year_dx"]).to_numpy()
    V, status = data.cohort["V"].to_numpy(), data.cohort["status"].to_numpy()
    y_all = data.labels[f"y_{TAU}"].to_numpy()

    recs = []
    for r in range(3):
        for k in range(5):
            split = sm.load_split(MODEL_DIR, r, k)
            iv, te = split["role"].eq("inner_val").to_numpy(), split["role"].eq("test").to_numpy()
            rows_iv, rows_te = split["row"].to_numpy()[iv], split["row"].to_numpy()[te]
            y_iv, w_iv = y_all[rows_iv], split[f"w_{TAU}"].to_numpy()[iv]
            sub = year[rows_te] <= LAST_YEAR
            rows_sub = rows_te[sub]
            y_sub = y_all[rows_sub]
            det = y_sub >= 0
            yt = (y_sub[det] == 1).astype(int)
            cif = sm.aalen_johansen_cif1(V[rows_sub], status[rows_sub], TAU)
            for col in [c for c in split.columns if c.endswith(f"|{TAU}")]:
                model, cell, _ = col.split("|")
                rule = "ipcw" if cell == "-" else cell.split("+")[0]
                p_iv = split[col].to_numpy()[iv].astype(float)
                v_rows, yv, wv = sm.cell_rows(rule, y_iv, w_iv)
                gamma = sm.select_threshold(p_iv[v_rows], yv, wv)            # same gamma* as the main analysis
                p_sub = split[col].to_numpy()[te][sub].astype(float)
                pt = p_sub[det]
                recs.append({"repeat": r, "fold": k, "model": model, "cell": cell,
                             "n_sub": int(sub.sum()), "share_censored": float((~det).mean()),
                             "brier": float(np.mean((yt - pt) ** 2)), "auc": float(roc_auc_score(yt, pt)),
                             "oe_ratio": cif / float(p_sub.mean()), "sensitivity": float(np.mean(pt[yt == 1] >= gamma))})
    m = pd.DataFrame(recs)
    m.to_csv(OUT_DIR / "metrics_10y_diagnosed_2010_2013_by_split.csv", index=False)

    agg = m.groupby(["model", "cell"], sort=False).agg(
        brier_mean=("brier", "mean"), brier_sd=("brier", "std"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
        oe_mean=("oe_ratio", "mean"), oe_sd=("oe_ratio", "std"),
        sens_mean=("sensitivity", "mean"), sens_sd=("sensitivity", "std")).reset_index()
    agg.to_csv(OUT_DIR / "table_S16_10y_diagnosed_2010_2013.csv", index=False)

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"subgroup per test fold: censored before {TAU} months = "
          f"{100 * m.share_censored.min():.1f}-{100 * m.share_censored.max():.1f}%")
    pd.set_option("display.width", 200)
    print(agg.round(4).to_string())
    # paired contrasts (corrected t-test) quoted in the text
    for metric in ("brier", "oe_ratio", "sensitivity"):
        for mod in sm.CLASSIFIERS:
            piv = m[m.model == mod].pivot_table(index=["repeat", "fold"], columns="cell", values=metric)
            for a, b in (("ipcw", "zero"), ("ipcw", "discard")):
                t = sm.corrected_ttest((piv[a] - piv[b]).to_numpy())
                print(f"{metric:12s} {mod:4s} {a}-{b:8s} {t['mean']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] "
                      f"p={t['p_value']:.2g}")


if __name__ == "__main__":
    main()
