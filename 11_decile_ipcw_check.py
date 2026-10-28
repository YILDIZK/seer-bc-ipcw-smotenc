"""Decile-level IPCW check of the calibration curves (Online Resource 1, analysis (ix); planned in writing before it was run).

The calibration curves estimate the observed risk in each decile of predicted risk with the unweighted Aalen-Johansen
estimator, which assumes independent censoring within the decile. Here the observed risk of each decile is also computed with
the truncated IPCW weights of the main analysis, sum(w * y) / sum(w) over the determinable patients of the decile (the
estimator of the overall check in Table S17), and the two are compared. Deciles are formed as in Fig. 3: within each repeat,
from the pooled test-fold predictions; values are averaged over the three repeats. No model is refitted.

The saved main-run outputs are read from, and the results written to, the working folder (config.WORK_DIR).

Run from this folder after notebooks 01-04 (README.md):  python 11_decile_ipcw_check.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
import seer_modeling as sm

DATA_ROOT = config.WORK_DIR
MODEL_DIR = DATA_ROOT / "outputs_modeling"
OUT_DIR = config.WORK_DIR / "outputs_decile_check"
BINS = 10


def pooled_test(repeat, splits):
    frames = [sm.load_split(MODEL_DIR, r, k) for r, k in splits if r == repeat]
    df = pd.concat(frames, ignore_index=True)
    return df[df["role"] == "test"].reset_index(drop=True)


def decile_points(df, col, tau, V, S, y_all):
    """Per decile: mean prediction, Aalen-Johansen observed risk, IPCW observed risk (as in 08_jdsa_figures.calibration_points)."""
    p = df[col].to_numpy(float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, BINS + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = df["row"].to_numpy()
    w = df[f"w_{tau}"].to_numpy(float)
    y = y_all[rows]
    out = []
    for b in range(len(edges) - 1):
        m = which == b
        if not m.any():
            continue
        det = m & (y >= 0)
        out.append({"decile": b + 1, "n": int(m.sum()), "pred": float(p[m].mean()),
                    "obs_aj": sm.aalen_johansen_cif1(V[rows[m]], S[rows[m]], tau),
                    "obs_ipcw": float(np.sum(w[det] * (y[det] == 1)) / np.sum(w[det]))})
    return pd.DataFrame(out)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = sm.load_prep(DATA_ROOT / "outputs_model_prep")
    splits = sorted((int(m[1]), int(m[2])) for m in
                    (re.match(r"split_r(\d+)_k(\d+)\.parquet", p.name) for p in MODEL_DIR.glob("split_r*_k*.parquet")) if m)
    assert len(splits) == 15
    V, S = data.cohort["V"].to_numpy(), data.cohort["status"].to_numpy()
    repeats = sorted({r for r, _ in splits})
    pooled = {r: pooled_test(r, splits) for r in repeats}
    cols = [c for c in pooled[repeats[0]].columns if re.match(r"^[a-z_]+\|[^|]+\|\d+$", c)]
    rows = []
    for col in cols:
        model, cell, tau = col.split("|")
        tau = int(tau)
        y_all = data.labels[f"y_{tau}"].to_numpy()
        for r in repeats:
            pts = decile_points(pooled[r], col, tau, V, S, y_all)
            rows.append(pts.assign(repeat=r, model=model, cell=cell, tau=tau))
    by_repeat = pd.concat(rows, ignore_index=True)
    by_repeat.to_csv(OUT_DIR / "decile_points_by_repeat.csv", index=False)

    pts = (by_repeat.groupby(["tau", "model", "cell", "decile"])[["pred", "obs_aj", "obs_ipcw"]].mean().reset_index())
    pts["diff"] = pts["obs_ipcw"] - pts["obs_aj"]
    pts.to_csv(OUT_DIR / "decile_points_mean.csv", index=False)
    # per decile: |IPCW - AJ|, calibration error under each estimator, and whether both put observed risk on the same side
    pts = pts.assign(abs_diff=pts["diff"].abs(), err_aj=(pts["obs_aj"] - pts["pred"]).abs(),
                     err_ipcw=(pts["obs_ipcw"] - pts["pred"]).abs(),
                     same_dir=np.sign(pts["obs_aj"] - pts["pred"]) == np.sign(pts["obs_ipcw"] - pts["pred"]))
    summary = (pts.groupby(["tau", "model", "cell"])
               .agg(max_abs_diff=("abs_diff", "max"), mean_abs_diff=("abs_diff", "mean"),
                    max_err_aj=("err_aj", "max"), max_err_ipcw=("err_ipcw", "max"),
                    deciles_same_direction=("same_dir", "sum"))
               .reset_index())
    summary.to_csv(OUT_DIR / "decile_check_summary.csv", index=False)

    claimed = summary[summary["cell"].isin(["ipcw", "-"])]
    print("Largest |IPCW - AJ| over deciles, IPCW cells and reference models:")
    print(claimed.pivot_table(index=["model", "cell"], columns="tau", values="max_abs_diff").round(4).to_string())
    print("\nAgreement (all <= 0.01):", bool((claimed["max_abs_diff"] <= 0.01).all()))
    print("\nLargest |IPCW - AJ| over deciles, all cells, by horizon:")
    print(summary.groupby("tau")["max_abs_diff"].agg(["min", "max"]).round(4).to_string())
    print("\nDeciles with the same direction of miscalibration (of 10), Zero and Discard cells:")
    print(summary[summary["cell"].isin(["zero", "discard"])].pivot_table(index=["model", "cell"], columns="tau",
                                                                         values="deciles_same_direction").to_string())


if __name__ == "__main__":
    main()
