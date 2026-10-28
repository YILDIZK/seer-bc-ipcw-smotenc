"""Weighted-bootstrap control: the IPCW-weighted bootstrap but without SMOTE-NC (Online Resource 1, analysis (vi)).

For every split, horizon and classifier, the IPCW + SMOTE-NC cell is rebuilt exactly as in seer_modeling.run_split up to the
SMOTE-NC step: the same weighted bootstrap sample (same random-number stream), and the stream is advanced through SMOTE-NC so
that the classifier seeds are identical; the synthetic patients are then discarded and the classifier is trained on the
bootstrap sample alone with unit weights. seer_modeling.py is not modified.

Run from this folder after notebooks 01-04 (README.md):  python 06_bootstrap_control.py
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import seer_modeling as sm

ROOT = config.WORK_DIR
MODEL_DIR = ROOT / "outputs_modeling"
OUT_DIR = ROOT / "outputs_bootstrap_control"
CELL_INDEX = sm.CELLS.index(("ipcw", True))          # = 5, the random-number stream of the IPCW + SMOTE-NC cell
NEW_CELL = "ipcw+boot"


def rebuild_split(data, r, k):
    """Encoder, design matrices and truncated IPCW weights exactly as in run_split."""
    df, roles = data.cohort, sm.split_roles(data, r, k)
    idx = {role: np.flatnonzero(roles == role) for role in ("inner_train", "inner_val", "test")}
    outer = np.flatnonzero(roles != "test")
    V = df["V"].to_numpy().astype(int)
    status = df["status"].to_numpy().astype(int)
    enc = sm.Encoder(data, df.iloc[outer])
    X = {role: enc.full(df.iloc[rows]) for role, rows in idx.items()}
    x_outer = enc.full(df.iloc[outer])
    x_num_it = enc.numeric(df.iloc[idx["inner_train"]])
    codes_it = enc.cat_codes(df.iloc[idx["inner_train"]])
    G = sm.fit_censoring_model(x_outer, V[outer], status[outer])
    pos = {role: np.searchsorted(outer, rows) for role, rows in idx.items() if role != "test"}
    w = {}
    for tau in sm.HORIZONS:
        y_all = data.labels[f"y_{tau}"].to_numpy()
        w_outer = sm.ipcw_weights(G, x_outer, V[outer], y_all[outer], tau)
        cap = float(np.percentile(w_outer[w_outer > 0], sm.WEIGHT_TRUNCATION_PCT))
        w_test = sm.ipcw_weights(G, X["test"], V[idx["test"]], y_all[idx["test"]], tau)
        w[tau] = {"inner_train": np.minimum(w_outer[pos["inner_train"]], cap),
                  "inner_val": np.minimum(w_outer[pos["inner_val"]], cap),
                  "test": np.minimum(w_test, cap)}
    return enc, idx, X, x_num_it, codes_it, w


def run(data, r, k, check_reproduction=False):
    saved = sm.load_split(MODEL_DIR, r, k)
    enc, idx, X, x_num_it, codes_it, w = rebuild_split(data, r, k)
    out_rows = np.concatenate([idx["inner_val"], idx["test"]])
    assert np.array_equal(out_rows, saved["row"].to_numpy()), "row order differs from the saved split"
    out = saved[["row", "role", "w_36", "w_60", "w_120"]].copy()
    for tau in sm.HORIZONS:                                 # recomputed weights must equal the saved ones
        rec = np.concatenate([w[tau]["inner_val"], w[tau]["test"]]).astype(np.float32)
        assert np.array_equal(rec, saved[f"w_{tau}"].to_numpy()), f"weights differ, split ({r},{k}) tau={tau}"
    x_out = np.vstack([X["inner_val"], X["test"]])
    diag = {}
    for tau in sm.HORIZONS:
        y_all = data.labels[f"y_{tau}"].to_numpy()
        y_it, y_iv = y_all[idx["inner_train"]], y_all[idx["inner_val"]]
        rng = np.random.default_rng([sm.SEED, r, k, tau, CELL_INDEX])
        rows, _, _ = sm.cell_rows("ipcw", y_it, w[tau]["inner_train"])
        boot = sm.weighted_bootstrap(rows, w[tau]["inner_train"][rows], rng)
        labels = (y_it[boot] == 1).astype(int)
        new_num, new_cat = sm.smote_nc(x_num_it[boot], codes_it[boot], labels, boot, rng)   # advances the stream
        v_rows, y_val, w_val = sm.cell_rows("ipcw", y_iv, w[tau]["inner_val"])
        for m in sm.CLASSIFIERS:
            seed = int(rng.integers(2**31 - 1))
            if check_reproduction and m == "lr":                 # full IPCW + SMOTE-NC refit must match the saved column
                Xs = np.vstack([X["inner_train"][boot], np.hstack([new_num, enc.onehot_from_codes(new_cat)])])
                ys = np.concatenate([labels, np.ones(new_num.shape[0], dtype=int)])
                ms, _ = sm.fit_classifier(m, Xs, ys, np.ones(ys.size), X["inner_val"][v_rows], y_val, w_val, seed)
                ref = saved[sm.pred_column(m, "ipcw+smote", tau)].to_numpy()
                diff = float(np.max(np.abs(ms.predict_proba(x_out)[:, 1].astype(np.float32) - ref)))
                print(f"  reproduction check split ({r},{k}) tau={tau} lr ipcw+smote: max |diff| = {diff:.2e}", flush=True)
                assert diff < 1e-5, "the IPCW + SMOTE-NC cell is not reproduced exactly; stopping"
                diag[f"repro_lr_{tau}"] = diff
            t0 = time.time()
            model, extra = sm.fit_classifier(m, X["inner_train"][boot], labels, np.ones(labels.size),
                                             X["inner_val"][v_rows], y_val, w_val, seed)
            out[sm.pred_column(m, NEW_CELL, tau)] = model.predict_proba(x_out)[:, 1].astype(np.float32)
            diag[sm.pred_column(m, NEW_CELL, tau)] = {"n_rows": int(boot.size), "n_pos": int(labels.sum()),
                                                      **extra, "fit_seconds": round(time.time() - t0, 1)}
    return out, diag


def evaluate(out, data, r, k):
    """Main-analysis metrics for the new column and the two comparators, plus net benefit at 0.20."""
    saved = sm.load_split(MODEL_DIR, r, k)
    cols = [c for c in saved.columns if re.match(r"^(lr|rf|xgb)\|(ipcw|ipcw\+smote)\|", c)]
    frame = pd.concat([out, saved[cols]], axis=1)
    met = sm.evaluate_split(frame, data, r, k)
    te = frame["role"].eq("test").to_numpy()
    rows_te = frame["row"].to_numpy()[te]
    nb = []
    for tau in sm.HORIZONS:
        y = data.labels[f"y_{tau}"].to_numpy()[rows_te]
        det = y >= 0
        yt, wt = (y[det] == 1).astype(int), frame[f"w_{tau}"].to_numpy(float)[te][det]
        for col in [c for c in frame.columns if c.endswith(f"|{tau}")]:
            model, cell, _ = col.split("|")
            p = frame[col].to_numpy(float)[te][det]
            nb.append({"tau": tau, "model": model, "cell": cell,
                       "nb_0.20": float(sm.net_benefit(p, yt, wt, int(te.sum()), np.array([0.20]))[0])})
    return met.merge(pd.DataFrame(nb), on=["tau", "model", "cell"])


def contrasts(metrics):
    rows = []
    for metric in ["brier", "auc", "precision", "sensitivity", "oe_ratio", "nb_0.20"]:
        for name, a, b in [("ipcw+boot - ipcw", NEW_CELL, "ipcw"), ("ipcw+smote - ipcw+boot", "ipcw+smote", NEW_CELL)]:
            for tau in sm.HORIZONS:
                for model in sm.CLASSIFIERS:
                    sel = metrics[(metrics.tau == tau) & (metrics.model == model)]
                    x = sel[sel.cell == a].sort_values(["repeat", "fold"])[metric].to_numpy()
                    y = sel[sel.cell == b].sort_values(["repeat", "fold"])[metric].to_numpy()
                    rows.append({"metric": metric, "contrast": name, "tau": tau, "model": model,
                                 **sm.corrected_ttest(x - y)})
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for metric, g in res.groupby("metric").groups.items():
        res.loc[g, "p_holm"] = sm.holm_adjust(res.loc[g, "p_value"].to_numpy())
    return res


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = sm.load_prep(ROOT / "outputs_model_prep")
    splits = sorted((int(m[1]), int(m[2])) for m in
                    (re.match(r"split_r(\d+)_k(\d+)\.parquet", p.name) for p in MODEL_DIR.glob("split_r*_k*.parquet")) if m)
    assert len(splits) == 15
    all_metrics, all_diag = [], {}
    for r, k in splits:
        t0 = time.time()
        out, diag = run(data, r, k, check_reproduction=(r, k) == (0, 0))
        out.to_parquet(OUT_DIR / f"split_r{r}_k{k}.parquet", index=False)
        all_diag[f"{r}_{k}"] = {**diag, "seconds": round(time.time() - t0, 1)}
        all_metrics.append(evaluate(out, data, r, k))
        print(f"split ({r},{k}) done in {time.time() - t0:.0f}s", flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(OUT_DIR / "bootstrap_control_metrics_by_split.csv", index=False)
    (OUT_DIR / "bootstrap_control_diagnostics.json").write_text(json.dumps(all_diag, indent=1), encoding="utf-8")
    summary = metrics.groupby(["tau", "model", "cell"]).agg(["mean", "std"]).drop(columns=["repeat", "fold"], level=0)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(OUT_DIR / "bootstrap_control_summary.csv", index=False)
    contrasts(metrics).to_csv(OUT_DIR / "bootstrap_control_contrasts.csv", index=False)
    main_m = pd.read_csv(ROOT / "outputs_evaluation" / "metrics_by_split.csv")
    chk = metrics[metrics.cell.isin(["ipcw", "ipcw+smote"])].merge(
        main_m, on=["repeat", "fold", "tau", "model", "cell"], suffixes=("", "_main"))
    print("comparator metrics reproduce main analysis: max |Brier diff| =",
          float((chk["brier"] - chk["brier_main"]).abs().max()))


if __name__ == "__main__":
    main()
