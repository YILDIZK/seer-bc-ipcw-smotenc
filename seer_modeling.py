"""Shared modeling functions for the SEER breast-cancer factorial benchmark.

Implements the design in Documents/Methods_Modeling.md and Documents/SEER_BC_Revision_Spec_v4.md:
censoring model and IPCW weights, the six cells (Zero / Discard / IPCW x none / SMOTE-NC 1:1),
three grid classifiers, and two cause-specific reference survival models.

Used by 03_modeling.ipynb (fitting, one file per split) and 04_evaluation.ipynb.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Prespecified settings (Methods_Modeling.md, Sections 3.3-3.8)
# ---------------------------------------------------------------------------
HORIZONS = (36, 60, 120)
CENSORING_RULES = ("zero", "discard", "ipcw")
CELLS = [(rule, smote) for smote in (False, True) for rule in CENSORING_RULES]   # Table 1 order: cells 1-6
CLASSIFIERS = ("lr", "rf", "xgb")
REFERENCES = ("cs_cox", "cs_rsf")
TPR_TARGET = 0.80
WEIGHT_TRUNCATION_PCT = 99
WEIGHT_MEAN_TOL = 0.05        # mean untruncated training weight must lie in 1 +/- this
SMOTE_K = 5
COX_ALPHA = 1.0               # ridge on the partial log-likelihood; negligible vs >= 18,000 events, stabilizes IV/M1
SEED = 42
TEST_TRAIN_RATIO = 0.20 / 0.64   # corrected t-test: test fold (20%) / inner-training set (64%) used to fit the classifiers


def cell_name(rule: str, smote: bool) -> str:
    return f"{rule}+smote" if smote else rule


def pred_column(model: str, cell: str, tau: int) -> str:
    """Column name of a prediction in the per-split output file."""
    return f"{model}|{cell}|{tau}"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PrepData:
    cohort: pd.DataFrame
    labels: pd.DataFrame
    folds: pd.DataFrame
    template: object          # unfitted ColumnTransformer from model-prep
    spec: dict
    manifest: dict
    x_cols: list
    num_cols: list            # in transformer output order
    cat_cols: list


def load_prep(prep_dir: Path) -> PrepData:
    """Load the model-prep outputs after verifying every checksum in prep_manifest.json."""
    prep_dir = Path(prep_dir)
    manifest = json.loads((prep_dir / "prep_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        if sha256(prep_dir / name) != digest:
            raise RuntimeError(f"{name} does not match prep_manifest.json; re-run model-prep")

    cohort = pd.read_parquet(prep_dir / "cohort.parquet")
    labels = pd.read_parquet(prep_dir / "labels.parquet")
    folds = pd.read_parquet(prep_dir / "folds.parquet")
    if not cohort["patient_id"].equals(labels["patient_id"]):
        raise RuntimeError("cohort and labels are not in the same patient order")

    template = joblib.load(prep_dir / "design_transformer.joblib")
    spec = json.loads((prep_dir / "transformer_spec.json").read_text(encoding="utf-8"))
    parts = {name: cols for name, _, cols in template.transformers}
    if set(parts) - {"num_log", "num", "cat"}:
        raise RuntimeError(f"unexpected transformer parts: {sorted(parts)}")
    num_cols = list(parts["num_log"]) + list(parts["num"])
    cat_cols = list(parts["cat"])
    return PrepData(cohort, labels, folds, template, spec, manifest, num_cols + cat_cols, num_cols, cat_cols)


def split_roles(data: PrepData, r: int, k: int) -> np.ndarray:
    """inner_role ('inner_train', 'inner_val', 'test') of every cohort row in split (r, k)."""
    sel = data.folds[(data.folds["repeat"] == r) & (data.folds["outer_fold"] == k)]
    roles = data.cohort["patient_id"].map(sel.set_index("patient_id")["inner_role"].astype(str))
    if roles.isna().any():
        raise RuntimeError(f"split ({r}, {k}): patients without a role")
    return roles.to_numpy()


# ---------------------------------------------------------------------------
# Design matrix (Section 3.1), split into numeric and categorical parts for SMOTE-NC
# ---------------------------------------------------------------------------
class Encoder:
    """The model-prep transformer fitted on outer-train, plus code-level access for SMOTE-NC."""

    def __init__(self, data: PrepData, outer_train: pd.DataFrame):
        self.ct = clone(data.template).fit(outer_train[data.x_cols])
        fitted = {name: (est, cols) for name, est, cols in self.ct.transformers_ if name != "remainder"}
        self.num_parts = [fitted["num_log"], fitted["num"]]
        self.cat_est, self.cat_cols = fitted["cat"]
        self.levels = [list(c) for c in self.cat_est.categories_]   # reference level first (dropped)
        self.offsets = np.concatenate([[0], np.cumsum([len(l) - 1 for l in self.levels])])
        self.feature_names = list(self.ct.get_feature_names_out())

    def full(self, df: pd.DataFrame) -> np.ndarray:
        return self.ct.transform(df).astype(np.float64)

    def numeric(self, df: pd.DataFrame) -> np.ndarray:
        return np.hstack([est.transform(df[cols]) for est, cols in self.num_parts])

    def cat_codes(self, df: pd.DataFrame) -> np.ndarray:
        codes = np.column_stack([
            pd.Categorical(df[col], categories=levels).codes
            for col, levels in zip(self.cat_cols, self.levels)
        ])
        if (codes < 0).any():
            raise RuntimeError("categorical value outside the prespecified levels")
        return codes.astype(np.int16)

    def onehot_from_codes(self, codes: np.ndarray) -> np.ndarray:
        """Same columns as the fitted one-hot encoder (reference level = code 0 dropped)."""
        out = np.zeros((codes.shape[0], int(self.offsets[-1])))
        rows = np.arange(codes.shape[0])
        for j in range(codes.shape[1]):
            hit = codes[:, j] > 0
            out[rows[hit], self.offsets[j] + codes[hit, j] - 1] = 1.0
        return out


# ---------------------------------------------------------------------------
# Cox model with a Breslow baseline on the integer-month grid
# ---------------------------------------------------------------------------
class CoxModel:
    """Ridge Cox regression (scikit-survival, Breslow ties); Breslow cumulative baseline hazard on months 0..T."""

    def __init__(self, alpha: float = COX_ALPHA):
        self.alpha = alpha

    def fit(self, X: np.ndarray, time_: np.ndarray, event: np.ndarray) -> "CoxModel":
        time_ = np.asarray(time_)
        if not np.all(time_ == np.round(time_)):
            raise ValueError("follow-up must be in integer months")
        event = np.asarray(event, bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph = CoxPHSurvivalAnalysis(alpha=self.alpha, ties="breslow", n_iter=100)
            cph.fit(X, Surv.from_arrays(event=event, time=time_.astype(float)))
        self.beta = np.asarray(cph.coef_, dtype=float)
        lp = X @ self.beta
        self.center = float(lp.mean())
        risk = np.exp(lp - self.center)

        t = time_.astype(int)
        self.t_max = int(t.max())
        at_risk = np.cumsum(np.bincount(t, weights=risk, minlength=self.t_max + 1)[::-1])[::-1]
        deaths = np.bincount(t[np.asarray(event, bool)], minlength=self.t_max + 1)
        increments = np.divide(deaths, at_risk, out=np.zeros(self.t_max + 1), where=at_risk > 0)
        self.h0 = np.cumsum(increments)          # h0[m] = baseline cumulative hazard at month m
        return self

    def relative_risk(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X @ self.beta - self.center)

    def cumhaz_at(self, X: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Cumulative hazard of each row at its own time s (step function, jumps at integer months)."""
        m = np.clip(np.floor(s).astype(int), 0, self.t_max)
        return self.h0[m] * self.relative_risk(X)

    def cumhaz_grid(self, X: np.ndarray, months: int) -> np.ndarray:
        """n x (months + 1) cumulative hazards at months 0..months."""
        grid = self.h0[np.minimum(np.arange(months + 1), self.t_max)]
        return self.relative_risk(X)[:, None] * grid[None, :]


# ---------------------------------------------------------------------------
# Censoring model and IPCW weights (Sections 3.2-3.3)
# ---------------------------------------------------------------------------
def fit_censoring_model(X: np.ndarray, V: np.ndarray, status: np.ndarray) -> CoxModel:
    """G(t | X): event = censoring (status 0); both death types censored."""
    return CoxModel().fit(X, V, status == 0)


def ipcw_weights(G: CoxModel, X: np.ndarray, V: np.ndarray, y: np.ndarray, tau: int) -> np.ndarray:
    """omega = 1 / G(min(V, tau) - 0.5 | X) for determinable rows (y >= 0), 0 otherwise."""
    s = np.minimum(V, tau) - 0.5
    g = np.exp(-G.cumhaz_at(X, s))
    return np.where(y >= 0, 1.0 / g, 0.0)


def kish_ess(w: np.ndarray) -> float:
    w = w[w > 0]
    return float(w.sum() ** 2 / (w**2).sum()) if w.size else 0.0


def weight_diagnostics(w_raw: np.ndarray, cap: float) -> dict:
    det = w_raw > 0
    return {
        "mean_untruncated_all": float(w_raw.mean()),
        "n_determinable": int(det.sum()),
        "cap_p99": float(cap),
        "max_untruncated": float(w_raw.max()),
        "share_truncated": float((w_raw[det] > cap).mean()),
        "kish_ess_truncated": kish_ess(np.minimum(w_raw, cap)),
        "quantiles_determinable": {
            q: float(v) for q, v in zip(("p01", "p25", "p50", "p75", "p99"),
                                        np.percentile(w_raw[det], [1, 25, 50, 75, 99]))
        },
    }


# ---------------------------------------------------------------------------
# Cell construction (Sections 3.4-3.5)
# ---------------------------------------------------------------------------
def cell_rows(rule: str, y: np.ndarray, w: np.ndarray):
    """Rows, binary labels and weights of one censoring rule (y in {1, 0, -1})."""
    if rule == "zero":
        rows = np.arange(y.size)
        return rows, (y[rows] == 1).astype(int), np.ones(rows.size)
    rows = np.flatnonzero(y >= 0)
    weights = w[rows] if rule == "ipcw" else np.ones(rows.size)
    return rows, (y[rows] == 1).astype(int), weights


def weighted_bootstrap(rows: np.ndarray, w: np.ndarray, rng) -> np.ndarray:
    """Same-size resample with probability proportional to w (IPCW pseudo-population)."""
    return rng.choice(rows, size=rows.size, replace=True, p=w / w.sum())


def smote_nc(x_num: np.ndarray, cat_codes: np.ndarray, y: np.ndarray, patient: np.ndarray,
             rng, k: int = SMOTE_K):
    """SMOTE-NC (Chawla et al. 2002) to a 1:1 class ratio.

    Neighbors are searched among distinct minority patients, so bootstrap duplicates
    are never their own neighbors. Base rows are drawn from all minority rows, i.e.
    proportionally to how often each patient appears.
    """
    minority = np.flatnonzero(y == 1)
    n_new = int((y == 0).sum() - minority.size)
    if n_new <= 0:
        return np.empty((0, x_num.shape[1])), np.empty((0, cat_codes.shape[1]), dtype=cat_codes.dtype)

    _, first, inverse = np.unique(patient[minority], return_index=True, return_inverse=True)
    rep = minority[first]                     # one row per distinct minority patient
    u_num, u_cat = x_num[rep], cat_codes[rep]

    # Mixed distance: Euclidean on numerics + median minority SD per categorical mismatch
    med = float(np.median(u_num.std(axis=0)))
    n_levels = cat_codes.max(axis=0).astype(int) + 1
    offsets = np.concatenate([[0], np.cumsum(n_levels)])
    cat_space = np.zeros((rep.size, int(offsets[-1])))
    for j in range(u_cat.shape[1]):
        cat_space[np.arange(rep.size), offsets[j] + u_cat[:, j]] = med / np.sqrt(2.0)
    space = np.hstack([u_num, cat_space])

    n_nb = min(k, rep.size - 1)
    idx = NearestNeighbors(n_neighbors=n_nb + 1).fit(space).kneighbors(space, return_distance=False)
    not_self = idx != np.arange(rep.size)[:, None]
    order = np.argsort(~not_self, axis=1, kind="stable")
    neighbors = np.take_along_axis(idx, order, axis=1)[:, :n_nb]

    # Categorical values of a synthetic patient: most frequent value among the base's neighbors
    modes = stats.mode(u_cat[neighbors], axis=1, keepdims=False).mode.astype(cat_codes.dtype)

    base = inverse[rng.integers(minority.size, size=n_new)]
    nb = neighbors[base, rng.integers(n_nb, size=n_new)]
    gap = rng.random((n_new, 1))
    new_num = u_num[base] + gap * (u_num[nb] - u_num[base])
    return new_num, modes[base]


def build_training_set(enc: Encoder, x_full: np.ndarray, x_num: np.ndarray, codes: np.ndarray,
                       y: np.ndarray, w: np.ndarray, rule: str, smote: bool, rng):
    """Training matrix of one cell from the inner-training set; returns (X, y, weights, info)."""
    rows, labels, weights = cell_rows(rule, y, w)
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    info = {
        "n_rows": int(rows.size), "n_pos": n_pos, "n_neg": n_neg,
        "ir_before": n_neg / n_pos, "kish_ess": kish_ess(weights), "n_synthetic": 0,
    }
    if not smote:
        info["ir_after"] = info["ir_before"]
        return x_full[rows], labels, weights, info

    if rule == "ipcw":
        rows = weighted_bootstrap(rows, w[rows], rng)
        labels = (y[rows] == 1).astype(int)
        info["n_pos_after_bootstrap"] = int(labels.sum())
    new_num, new_cat = smote_nc(x_num[rows], codes[rows], labels, rows, rng)
    X = np.vstack([x_full[rows], np.hstack([new_num, enc.onehot_from_codes(new_cat)])])
    labels = np.concatenate([labels, np.ones(new_num.shape[0], dtype=int)])
    info.update(n_synthetic=int(new_num.shape[0]), ir_after=float((labels == 0).sum() / labels.sum()),
                n_rows_after=int(labels.size))
    return X, labels, np.ones(labels.size), info


# ---------------------------------------------------------------------------
# Classifiers (Section 3.6): fixed hyperparameters, XGBoost early stopping on inner-val
# ---------------------------------------------------------------------------
def fit_classifier(name: str, X, y, w, X_val, y_val, w_val, seed: int):
    if name == "lr":
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        model.fit(X, y, sample_weight=w)
        return model, {}
    if name == "rf":
        model = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=50,
                                       n_jobs=-1, random_state=seed)
        model.fit(X, y, sample_weight=w)
        return model, {}
    if name == "xgb":
        model = XGBClassifier(
            n_estimators=10000, learning_rate=0.05,     # ceiling only; early stopping decides max_depth=4, min_child_weight=1,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
            objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=50,
            n_jobs=-1, random_state=seed,
        )
        model.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val],
                  verbose=False)
        return model, {"best_iteration": int(model.best_iteration)}
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Reference models (Section 3.7): cause-specific hazards combined into the CIF
# ---------------------------------------------------------------------------
def cif_from_cumhaz(H1: np.ndarray, H2: np.ndarray, horizons=HORIZONS) -> dict:
    """Cumulative incidence of cause 1 from two cause-specific cumulative hazards on months 0..T.

    Hazards are constant within each month, so the probability of a cause-1 death in month t is
    S(t-1) * [1 - exp(-(dH1 + dH2))] * dH1 / (dH1 + dH2); F1 + F2 + S = 1 holds exactly.
    """
    dH1, dH2 = np.diff(H1, axis=1), np.diff(H2, axis=1)
    dH = dH1 + dH2
    S_prev = np.exp(-(H1 + H2))[:, :-1]
    share = np.divide(dH1, dH, out=np.zeros_like(dH1), where=dH > 0)
    F1 = np.cumsum(S_prev * -np.expm1(-dH) * share, axis=1)    # column m-1 = month m
    return {tau: F1[:, tau - 1] for tau in horizons}


def fit_cs_cox(X, V, status):
    return [CoxModel().fit(X, V, status == cause) for cause in (1, 2)]


def predict_cs_cox(models, X, horizons=HORIZONS) -> dict:
    T = max(horizons)
    return cif_from_cumhaz(models[0].cumhaz_grid(X, T), models[1].cumhaz_grid(X, T), horizons)


def fit_cs_rsf(X, V, status, seed: int):
    forests = []
    for cause in (1, 2):
        rsf = RandomSurvivalForest(n_estimators=300, max_features="sqrt", min_samples_leaf=100,
                                   max_samples=0.5, n_jobs=-1, random_state=seed + cause)
        rsf.fit(X, Surv.from_arrays(event=status == cause, time=V.astype(float)))
        forests.append(rsf)
    return forests


def _rsf_cumhaz_grid(rsf, X, months: int) -> np.ndarray:
    chf = rsf.predict_cumulative_hazard_function(X, return_array=True)
    pos = np.searchsorted(rsf.unique_times_, np.arange(months + 1), side="right") - 1
    out = np.zeros((X.shape[0], months + 1))
    ok = pos >= 0
    out[:, ok] = chf[:, pos[ok]]
    return out


def predict_cs_rsf(forests, X, horizons=HORIZONS, batch: int = 20000) -> dict:
    T = max(horizons)
    parts = {tau: [] for tau in horizons}
    for start in range(0, X.shape[0], batch):
        xb = X[start:start + batch]
        cif = cif_from_cumhaz(_rsf_cumhaz_grid(forests[0], xb, T), _rsf_cumhaz_grid(forests[1], xb, T), horizons)
        for tau in horizons:
            parts[tau].append(cif[tau])
    return {tau: np.concatenate(parts[tau]) for tau in horizons}


# ---------------------------------------------------------------------------
# One split: fit everything, save predictions for inner-val and test
# ---------------------------------------------------------------------------
def run_split(data: PrepData, r: int, k: int, out_dir: Path, models=CLASSIFIERS, references=REFERENCES,
              horizons=HORIZONS, log=print) -> Path:
    """Fit the censoring model, 6 cells x classifiers x horizons, and the references for split (r, k).

    Writes split_r{r}_k{k}.parquet (one row per inner-val / test patient: weights and predictions)
    and split_r{r}_k{k}.json (diagnostics). Returns the parquet path; skips finished splits.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path, diag_path = out_dir / f"split_r{r}_k{k}.parquet", out_dir / f"split_r{r}_k{k}.json"
    if pred_path.exists() and diag_path.exists():
        log(f"split ({r}, {k}): already done, skipped")
        return pred_path

    t_split = time.time()
    df, roles = data.cohort, split_roles(data, r, k)
    idx = {role: np.flatnonzero(roles == role) for role in ("inner_train", "inner_val", "test")}
    outer = np.flatnonzero(roles != "test")
    V = df["V"].to_numpy().astype(int)
    status = df["status"].to_numpy().astype(int)

    # Design matrix: encoder fitted on outer-train only
    enc = Encoder(data, df.iloc[outer])
    X = {role: enc.full(df.iloc[rows]) for role, rows in idx.items()}
    x_outer = enc.full(df.iloc[outer])
    x_num_it = enc.numeric(df.iloc[idx["inner_train"]])
    codes_it = enc.cat_codes(df.iloc[idx["inner_train"]])
    if not np.allclose(np.hstack([x_num_it, enc.onehot_from_codes(codes_it)]), X["inner_train"]):
        raise RuntimeError("numeric + one-hot parts do not reproduce the fitted transformer")

    # Censoring model on outer-train; weights for every partition (training-fold G)
    t0 = time.time()
    G = fit_censoring_model(x_outer, V[outer], status[outer])
    diag = {"split": [r, k], "censoring_model_seconds": round(time.time() - t0, 1), "weights": {}, "cells": {},
            "references": {}}
    pos_in_outer = {role: np.searchsorted(outer, rows) for role, rows in idx.items() if role != "test"}
    w = {}
    for tau in horizons:
        y_all = data.labels[f"y_{tau}"].to_numpy()
        w_outer = ipcw_weights(G, x_outer, V[outer], y_all[outer], tau)
        cap = float(np.percentile(w_outer[w_outer > 0], WEIGHT_TRUNCATION_PCT))
        d = weight_diagnostics(w_outer, cap)
        diag["weights"][str(tau)] = d
        if abs(d["mean_untruncated_all"] - 1) > WEIGHT_MEAN_TOL:
            raise RuntimeError(f"split ({r}, {k}), tau={tau}: mean training weight "
                               f"{d['mean_untruncated_all']:.3f} is not ~1; G is misestimated")
        w_test = ipcw_weights(G, X["test"], V[idx["test"]], y_all[idx["test"]], tau)
        w[tau] = {
            "inner_train": np.minimum(w_outer[pos_in_outer["inner_train"]], cap),
            "inner_val": np.minimum(w_outer[pos_in_outer["inner_val"]], cap),
            "test": np.minimum(w_test, cap),
        }

    # Output frame: inner-val and test patients
    out_rows = np.concatenate([idx["inner_val"], idx["test"]])
    out = pd.DataFrame({
        "row": out_rows.astype(np.int32),
        "role": np.repeat(["inner_val", "test"], [idx["inner_val"].size, idx["test"].size]),
    })
    for tau in horizons:
        out[f"w_{tau}"] = np.concatenate([w[tau]["inner_val"], w[tau]["test"]]).astype(np.float32)
    x_out = np.vstack([X["inner_val"], X["test"]])

    # Grid: 6 cells x classifiers x horizons
    for tau in horizons:
        y_all = data.labels[f"y_{tau}"].to_numpy()
        y_it, y_iv = y_all[idx["inner_train"]], y_all[idx["inner_val"]]
        for c, (rule, smote) in enumerate(CELLS):
            cname = cell_name(rule, smote)
            rng = np.random.default_rng([SEED, r, k, tau, c])
            Xtr, ytr, wtr, info = build_training_set(enc, X["inner_train"], x_num_it, codes_it,
                                                     y_it, w[tau]["inner_train"], rule, smote, rng)
            v_rows, y_val, w_val = cell_rows(rule, y_iv, w[tau]["inner_val"])
            for m in models:
                t0 = time.time()
                model, extra = fit_classifier(m, Xtr, ytr, wtr, X["inner_val"][v_rows], y_val, w_val,
                                              seed=int(rng.integers(2**31 - 1)))
                out[pred_column(m, cname, tau)] = model.predict_proba(x_out)[:, 1].astype(np.float32)
                diag["cells"][pred_column(m, cname, tau)] = {**info, **extra,
                                                              "fit_seconds": round(time.time() - t0, 1)}
                log(f"  tau={tau:>3} {cname:<14} {m:<4} {time.time() - t0:6.1f}s")

    # Reference models on inner-train (all patients; censoring handled internally)
    V_it, s_it = V[idx["inner_train"]], status[idx["inner_train"]]
    for ref in references:
        t0 = time.time()
        if ref == "cs_cox":
            preds = predict_cs_cox(fit_cs_cox(X["inner_train"], V_it, s_it), x_out, horizons)
        else:
            preds = predict_cs_rsf(fit_cs_rsf(X["inner_train"], V_it, s_it, SEED + 100 * r + k), x_out, horizons)
        for tau in horizons:
            out[pred_column(ref, "-", tau)] = preds[tau].astype(np.float32)
        diag["references"][ref] = {"fit_predict_seconds": round(time.time() - t0, 1)}
        log(f"  reference {ref:<7} {time.time() - t0:6.1f}s")

    diag["total_seconds"] = round(time.time() - t_split, 1)
    out.to_parquet(pred_path, index=False)
    diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    log(f"split ({r}, {k}) done in {diag['total_seconds'] / 60:.1f} min")
    return pred_path


# ---------------------------------------------------------------------------
# Evaluation (Methods_Modeling.md, Sections 3.8, 4 and 5)
# ---------------------------------------------------------------------------
def load_split(out_dir: Path, r: int, k: int) -> pd.DataFrame:
    return pd.read_parquet(Path(out_dir) / f"split_r{r}_k{k}.parquet")


def select_threshold(p: np.ndarray, y: np.ndarray, w: np.ndarray, target: float = TPR_TARGET) -> float:
    """gamma* = argmin FPR s.t. weighted TPR >= target.

    FPR and TPR are both non-increasing in gamma, so gamma* is the largest candidate threshold
    (a unique predicted probability) whose TPR still reaches the target. Always feasible.
    """
    order = np.argsort(-p, kind="stable")
    ps, tp = p[order], np.cumsum((w * y)[order])
    last = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])      # last position of each tied value
    tpr = tp[last] / tp[-1]
    return float(ps[last[np.argmax(tpr >= target - 1e-12)]])


def aalen_johansen_cif1(V: np.ndarray, status: np.ndarray, tau: float) -> float:
    """Aalen-Johansen cumulative incidence of cause 1 at tau (status 0 censored, 1, 2)."""
    V = np.asarray(V, float)
    status = np.asarray(status, int)
    times = np.unique(V[(status > 0) & (V <= tau)])
    if times.size == 0:
        return 0.0
    at_risk = (V[None, :] >= times[:, None]).sum(axis=1)
    d_all = np.array([((V == t) & (status > 0)).sum() for t in times])
    d_1 = np.array([((V == t) & (status == 1)).sum() for t in times])
    s_left = np.r_[1.0, np.cumprod(1.0 - d_all / at_risk)[:-1]]
    return float(np.sum(s_left * d_1 / at_risk))


def classification_metrics(p, y, w, gamma) -> dict:
    """IPCW-weighted confusion-matrix metrics at threshold gamma (determinable rows)."""
    flag = p >= gamma
    tp, fp = float(np.sum(w * y * flag)), float(np.sum(w * (1 - y) * flag))
    fn, tn = float(np.sum(w * y * ~flag)), float(np.sum(w * (1 - y) * ~flag))
    sens = tp / (tp + fn) if tp + fn > 0 else np.nan
    spec = tn / (tn + fp) if tn + fp > 0 else np.nan
    prec = tp / (tp + fp) if tp + fp > 0 else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec and sens else np.nan
    return {"sensitivity": sens, "specificity": spec, "fpr": 1 - spec, "precision": prec, "f1": f1,
            "n_fp": fp, "n_fn": fn}


def net_benefit(p, y, w, n_total: int, thresholds: np.ndarray) -> np.ndarray:
    """IPCW-weighted decision-curve net benefit at each threshold probability."""
    out = np.empty(thresholds.size)
    for i, pt in enumerate(thresholds):
        flag = p >= pt
        tp, fp = np.sum(w * y * flag), np.sum(w * (1 - y) * flag)
        out[i] = (tp - fp * pt / (1 - pt)) / n_total
    return out


def evaluate_split(split: pd.DataFrame, data: PrepData, r: int, k: int, horizons=HORIZONS) -> pd.DataFrame:
    """Threshold selection on inner-val and test-fold metrics for every model column of one split."""
    from sklearn.metrics import roc_auc_score

    V = data.cohort["V"].to_numpy()
    status = data.cohort["status"].to_numpy()
    iv, te = split["role"].eq("inner_val").to_numpy(), split["role"].eq("test").to_numpy()
    rows_iv, rows_te = split["row"].to_numpy()[iv], split["row"].to_numpy()[te]
    n_test = int(te.sum())
    records = []
    for tau in horizons:
        y_all = data.labels[f"y_{tau}"].to_numpy()
        y_iv, y_te = y_all[rows_iv], y_all[rows_te]
        w_iv, w_te = split[f"w_{tau}"].to_numpy()[iv], split[f"w_{tau}"].to_numpy()[te]
        det = y_te >= 0
        yt, wt = (y_te[det] == 1).astype(int), w_te[det]
        cif_aj = aalen_johansen_cif1(V[rows_te], status[rows_te], tau)
        for col in [c for c in split.columns if c.endswith(f"|{tau}")]:
            model, cell, _ = col.split("|")
            rule = "ipcw" if cell == "-" else cell.split("+")[0]
            p_iv, p_te = split[col].to_numpy()[iv].astype(float), split[col].to_numpy()[te].astype(float)
            v_rows, yv, wv = cell_rows(rule, y_iv, w_iv)
            gamma = select_threshold(p_iv[v_rows], yv, wv)
            pt = p_te[det]
            records.append({
                "repeat": r, "fold": k, "tau": tau, "model": model, "cell": cell,
                "brier": float(np.sum(wt * (yt - pt) ** 2) / n_test),
                "auc": float(roc_auc_score(yt, pt, sample_weight=wt)),
                "gamma": gamma,
                **classification_metrics(pt, yt, wt, gamma),
                "mean_pred": float(p_te.mean()), "cif_aj": cif_aj,
                "oe_ratio": cif_aj / float(p_te.mean()),
            })
    return pd.DataFrame(records)


def corrected_ttest(diffs: np.ndarray, test_train_ratio: float = TEST_TRAIN_RATIO, alpha: float = 0.05) -> dict:
    """Corrected repeated k-fold CV t-test (Nadeau & Bengio 2003; Bouckaert & Frank 2004).

    Variance inflated by 1/J + n_test/n_train, with n_train the 64% inner-training set on which the
    classifiers were fitted (more conservative than the 80% outer-training set).
    """
    diffs = np.asarray(diffs, float)
    J = diffs.size
    mean, var = diffs.mean(), diffs.var(ddof=1)
    se = np.sqrt((1 / J + test_train_ratio) * var)
    t = mean / se if se > 0 else np.nan
    q = stats.t.ppf(1 - alpha / 2, J - 1)
    return {"mean": mean, "ci_low": mean - q * se, "ci_high": mean + q * se,
            "t": t, "p_value": float(2 * stats.t.sf(abs(t), J - 1)) if se > 0 else np.nan}


def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error rate)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((p.size - np.arange(p.size)) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def test_weights_untruncated(data: PrepData, r: int, k: int, horizons=HORIZONS) -> dict:
    """Untruncated IPCW weights of the test fold of split (r, k), for the truncation sensitivity analysis.

    Refits the outer-train encoder and censoring model exactly as run_split does; no classifier is refitted.
    """
    df, roles = data.cohort, split_roles(data, r, k)
    outer, test = np.flatnonzero(roles != "test"), np.flatnonzero(roles == "test")
    V = df["V"].to_numpy().astype(int)
    status = df["status"].to_numpy().astype(int)
    enc = Encoder(data, df.iloc[outer])
    G = fit_censoring_model(enc.full(df.iloc[outer]), V[outer], status[outer])
    x_te = enc.full(df.iloc[test])
    return {tau: ipcw_weights(G, x_te, V[test], data.labels[f"y_{tau}"].to_numpy()[test], tau)
            for tau in horizons}
