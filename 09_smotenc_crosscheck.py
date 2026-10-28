"""SMOTE-NC implementation check against imbalanced-learn (Online Resource 1, analysis (viii); planned in writing before it was run).

For every split, horizon and classifier, the Zero + SMOTE-NC cell is rebuilt exactly as in seer_modeling.run_split: the same
inner-training rows and encoder, and the main run's random stream advanced through our SMOTE-NC, so that the classifier
seeds are identical. The synthetic patients are then generated a second time with imbalanced-learn's SMOTENC from the same
standardized numeric predictors and categorical codes, one-hot encoded in the same way, and each classifier is refitted on
them. Evaluation uses the saved weights and the main-analysis functions. seer_modeling.py is not modified.

The saved main-run outputs are read from, and the results written to, the working folder (config.WORK_DIR).

Run from this folder after notebooks 01-04 (README.md):  python 09_smotenc_crosscheck.py
"""
import json
import re
import time
from pathlib import Path

import imblearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

import config
import seer_modeling as sm

DATA_ROOT = config.WORK_DIR
MODEL_DIR = DATA_ROOT / "outputs_modeling"
OUT_DIR = config.WORK_DIR / "outputs_smotenc_crosscheck"
CELL_INDEX = sm.CELLS.index(("zero", True))          # = 3, the random-number stream of the Zero + SMOTE-NC cell
NEW_CELL = "zero+imblearn"
METRICS = ["brier", "auc", "oe_ratio", "sensitivity", "precision", "nb_0.20"]


def rebuild_split(data, r, k):
    """Encoder and design matrices exactly as in run_split (the Zero cell needs no censoring weights for training)."""
    df, roles = data.cohort, sm.split_roles(data, r, k)
    idx = {role: np.flatnonzero(roles == role) for role in ("inner_train", "inner_val", "test")}
    outer = np.flatnonzero(roles != "test")
    enc = sm.Encoder(data, df.iloc[outer])
    X = {role: enc.full(df.iloc[rows]) for role, rows in idx.items()}
    x_num_it = enc.numeric(df.iloc[idx["inner_train"]])
    codes_it = enc.cat_codes(df.iloc[idx["inner_train"]])
    return enc, idx, X, x_num_it, codes_it


def imblearn_smotenc(x_num, codes, labels, seed):
    """Synthetic minority patients from imbalanced-learn SMOTENC (1:1); returns (numeric, codes) of the new rows only."""
    n, n_num = labels.size, x_num.shape[1]
    x_in = np.hstack([x_num, codes.astype(np.float64)])
    smote = SMOTENC(categorical_features=list(range(n_num, x_in.shape[1])), k_neighbors=sm.SMOTE_K,
                    sampling_strategy=1.0, random_state=seed)
    x_res, y_res = smote.fit_resample(x_in, labels)
    assert np.array_equal(x_res[:n], x_in) and np.array_equal(y_res[:n], labels), "original rows changed"
    assert (y_res[n:] == 1).all() and (y_res == 0).sum() == (y_res == 1).sum(), "not a 1:1 minority resample"
    return x_res[n:, :n_num], np.rint(x_res[n:, n_num:]).astype(codes.dtype)


def synthetic_comparison(num_a, cat_a, num_b, cat_b):
    """Distributional agreement of two synthetic samples: numeric means and SDs, categorical total variation distance."""
    tv = []
    for j in range(cat_a.shape[1]):
        m = int(max(cat_a[:, j].max(), cat_b[:, j].max())) + 1
        pa = np.bincount(cat_a[:, j], minlength=m) / cat_a.shape[0]
        pb = np.bincount(cat_b[:, j], minlength=m) / cat_b.shape[0]
        tv.append(0.5 * float(np.abs(pa - pb).sum()))
    return {"n_ours": int(num_a.shape[0]), "n_imblearn": int(num_b.shape[0]),
            "max_abs_mean_diff_numeric": float(np.max(np.abs(num_a.mean(0) - num_b.mean(0)))),
            "max_abs_sd_diff_numeric": float(np.max(np.abs(num_a.std(0) - num_b.std(0)))),
            "max_tv_categorical": float(max(tv)), "mean_tv_categorical": float(np.mean(tv))}


def run(data, r, k, check_reproduction=False):
    saved = sm.load_split(MODEL_DIR, r, k)
    enc, idx, X, x_num_it, codes_it = rebuild_split(data, r, k)
    out_rows = np.concatenate([idx["inner_val"], idx["test"]])
    assert np.array_equal(out_rows, saved["row"].to_numpy()), "row order differs from the saved split"
    out = saved[["row", "role", "w_36", "w_60", "w_120"]].copy()
    x_out = np.vstack([X["inner_val"], X["test"]])
    diag = {}
    for tau in sm.HORIZONS:
        y_all = data.labels[f"y_{tau}"].to_numpy()
        y_it, y_iv = y_all[idx["inner_train"]], y_all[idx["inner_val"]]
        rng = np.random.default_rng([sm.SEED, r, k, tau, CELL_INDEX])
        Xs, ys, ws, info = sm.build_training_set(enc, X["inner_train"], x_num_it, codes_it, y_it,
                                                 np.ones(y_it.size), "zero", True, rng)   # advances the stream
        labels = (y_it == 1).astype(int)
        n = labels.size
        imb_seed = int(np.random.default_rng([sm.SEED, r, k, tau, CELL_INDEX, 1]).integers(2**31 - 1))
        t0 = time.time()
        new_num, new_cat = imblearn_smotenc(x_num_it, codes_it, labels, imb_seed)
        smote_seconds = round(time.time() - t0, 1)
        Xi = np.vstack([X["inner_train"], np.hstack([new_num, enc.onehot_from_codes(new_cat)])])
        yi = np.concatenate([labels, np.ones(new_num.shape[0], dtype=int)])
        assert Xi.shape == Xs.shape and np.array_equal(yi, ys), "imbalanced-learn and ours differ in size or labels"
        ours_num, ours_cat = Xs[n:, :x_num_it.shape[1]], None
        # recover our synthetic categorical codes from the one-hot columns (code 0 = reference level, all zeros)
        oh = Xs[n:, x_num_it.shape[1]:]
        ours_cat = np.zeros((oh.shape[0], len(enc.levels)), dtype=codes_it.dtype)
        for j in range(len(enc.levels)):
            block = oh[:, enc.offsets[j]:enc.offsets[j + 1]]
            hit = block.sum(axis=1) > 0
            ours_cat[hit, j] = block[hit].argmax(axis=1) + 1
        diag[f"synthetic|{tau}"] = {**synthetic_comparison(ours_num, ours_cat, new_num, new_cat),
                                    "imblearn_seconds": smote_seconds, "n_synthetic_main": info["n_synthetic"]}
        v_rows, y_val, w_val = sm.cell_rows("zero", y_iv, np.ones(y_iv.size))
        for m in sm.CLASSIFIERS:
            seed = int(rng.integers(2**31 - 1))
            if check_reproduction and m == "lr":                 # our Zero + SMOTE-NC refit must match the saved column
                ms, _ = sm.fit_classifier(m, Xs, ys, ws, X["inner_val"][v_rows], y_val, w_val, seed)
                ref = saved[sm.pred_column(m, "zero+smote", tau)].to_numpy()
                diff = float(np.max(np.abs(ms.predict_proba(x_out)[:, 1].astype(np.float32) - ref)))
                print(f"  reproduction check split ({r},{k}) tau={tau} lr zero+smote: max |diff| = {diff:.2e}", flush=True)
                assert diff < 1e-5, "the Zero + SMOTE-NC cell is not reproduced exactly; stopping"
                diag[f"repro_lr_{tau}"] = diff
            t0 = time.time()
            model, extra = sm.fit_classifier(m, Xi, yi, np.ones(yi.size), X["inner_val"][v_rows], y_val, w_val, seed)
            out[sm.pred_column(m, NEW_CELL, tau)] = model.predict_proba(x_out)[:, 1].astype(np.float32)
            diag[sm.pred_column(m, NEW_CELL, tau)] = {"n_rows": int(yi.size), **extra,
                                                      "fit_seconds": round(time.time() - t0, 1)}
    return out, diag


def evaluate(out, data, r, k):
    """Main-analysis metrics for the new column and the two comparators, plus net benefit at 0.20."""
    saved = sm.load_split(MODEL_DIR, r, k)
    cols = [c for c in saved.columns if re.match(r"^(lr|rf|xgb)\|(zero|zero\+smote)\|", c)]
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
    families = {"crosscheck": [("imblearn - ours", NEW_CELL, "zero+smote"), ("imblearn - zero", NEW_CELL, "zero")],
                "main": [("ours - zero", "zero+smote", "zero")]}
    rows = []
    for family, pairs in families.items():
        for metric in METRICS:
            for name, a, b in pairs:
                for tau in sm.HORIZONS:
                    for model in sm.CLASSIFIERS:
                        sel = metrics[(metrics.tau == tau) & (metrics.model == model)]
                        x = sel[sel.cell == a].sort_values(["repeat", "fold"])[metric].to_numpy()
                        y = sel[sel.cell == b].sort_values(["repeat", "fold"])[metric].to_numpy()
                        rows.append({"family": family, "metric": metric, "contrast": name, "tau": tau, "model": model,
                                     **sm.corrected_ttest(x - y)})
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for _, g in res.groupby(["family", "metric"]).groups.items():
        res.loc[g, "p_holm"] = sm.holm_adjust(res.loc[g, "p_value"].to_numpy())
    return res


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"imbalanced-learn {imblearn.__version__}", flush=True)
    data = sm.load_prep(DATA_ROOT / "outputs_model_prep")
    splits = sorted((int(m[1]), int(m[2])) for m in
                    (re.match(r"split_r(\d+)_k(\d+)\.parquet", p.name) for p in MODEL_DIR.glob("split_r*_k*.parquet")) if m)
    assert len(splits) == 15
    all_metrics, all_diag = [], {"imblearn_version": imblearn.__version__}
    for r, k in splits:
        t0 = time.time()
        out, diag = run(data, r, k, check_reproduction=(r, k) == (0, 0))
        out.to_parquet(OUT_DIR / f"split_r{r}_k{k}.parquet", index=False)
        all_diag[f"{r}_{k}"] = {**diag, "seconds": round(time.time() - t0, 1)}
        all_metrics.append(evaluate(out, data, r, k))
        print(f"split ({r},{k}) done in {time.time() - t0:.0f}s", flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(OUT_DIR / "crosscheck_metrics_by_split.csv", index=False)
    (OUT_DIR / "crosscheck_diagnostics.json").write_text(json.dumps(all_diag, indent=1), encoding="utf-8")
    summary = metrics.groupby(["tau", "model", "cell"]).agg(["mean", "std"]).drop(columns=["repeat", "fold"], level=0)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(OUT_DIR / "crosscheck_summary.csv", index=False)
    contrasts(metrics).to_csv(OUT_DIR / "crosscheck_contrasts.csv", index=False)
    main_m = pd.read_csv(DATA_ROOT / "outputs_evaluation" / "metrics_by_split.csv")
    chk = metrics[metrics.cell.isin(["zero", "zero+smote"])].merge(
        main_m, on=["repeat", "fold", "tau", "model", "cell"], suffixes=("", "_main"))
    print("comparator metrics reproduce main analysis: max |Brier diff| =",
          float((chk["brier"] - chk["brier_main"]).abs().max()), flush=True)


if __name__ == "__main__":
    main()
