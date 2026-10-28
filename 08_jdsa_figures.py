"""JDSA Figs. 3 and 4 and Online Resource Figs. S1-S3.

The saved predictions of the main analysis are read from the working folder (config.WORK_DIR), and the figures are
written to its Figures subfolder. The computations are those of 04_evaluation.ipynb, cells 15
(calibration), 16 (decision curves) and 26 (Fig. S1); only the presentation differs.

Reproduction check: the same data are also drawn in the original PLOS style and compared pixel by pixel with the
PLOS figure files, which shows that only the presentation changed.

Run from this folder after notebook 04 (README.md):  python 08_jdsa_figures.py
"""
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageChops

matplotlib.rcParams.update({"ps.fonttype": 42, "pdf.fonttype": 42})   # TrueType fonts embedded in EPS and PDF

import config
import seer_modeling as sm

DATA_ROOT = config.WORK_DIR
MODEL_DIR = DATA_ROOT / "outputs_modeling"
EVAL_SRC = DATA_ROOT / "outputs_evaluation"
OUT = config.WORK_DIR / "Figures"
REPRO = OUT / "reproduction_check"
CHECKS = OUT / "checks"
for d in (OUT, REPRO, CHECKS):
    d.mkdir(exist_ok=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------- data (04_evaluation.ipynb, cells 1 and 15)
data = sm.load_prep(DATA_ROOT / "outputs_model_prep")
SPLITS = sorted((int(m[1]), int(m[2])) for m in
                (re.match(r"split_r(\d+)_k(\d+)\.parquet", p.name) for p in MODEL_DIR.glob("split_r*_k*.parquet")) if m)
assert len(SPLITS) == 15, SPLITS
CELL_ORDER = [sm.cell_name(rule, smote) for rule, smote in sm.CELLS]
LABELS = {"lr": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost",
          "cs_cox": "Cause-specific Cox", "cs_rsf": "Cause-specific RSF"}


def pooled_test(repeat: int) -> pd.DataFrame:
    frames = [sm.load_split(MODEL_DIR, r, k) for r, k in SPLITS if r == repeat]
    df = pd.concat(frames, ignore_index=True)
    return df[df["role"] == "test"].reset_index(drop=True)


REPEATS = sorted({r for r, _ in SPLITS})
POOLED = {r: pooled_test(r) for r in REPEATS}
V_ALL, S_ALL = data.cohort["V"].to_numpy(), data.cohort["status"].to_numpy()


def calibration_points(df: pd.DataFrame, col: str, tau: int, bins: int = 10) -> pd.DataFrame:
    p = df[col].to_numpy(float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = df["row"].to_numpy()
    return pd.DataFrame([
        {"pred": p[which == b].mean(), "obs": sm.aalen_johansen_cif1(V_ALL[rows[which == b]], S_ALL[rows[which == b]], tau)}
        for b in range(len(edges) - 1) if (which == b).any()
    ])


PANELS = [("lr", "Logistic regression"), ("rf", "Random forest"), ("xgb", "XGBoost"),
          ("ref", "Reference survival models")]


def panel_keys(panel):
    return ["cs_cox", "cs_rsf"] if panel == "ref" else CELL_ORDER


def col_for(panel, key, tau):
    return sm.pred_column(key, "-", tau) if panel == "ref" else sm.pred_column(panel, key, tau)


def calibration_data(tau):
    pts = {(panel, key): pd.concat([calibration_points(POOLED[r], col_for(panel, key, tau), tau) for r in REPEATS])
           .groupby(level=0).mean() for panel, _ in PANELS for key in panel_keys(panel)}
    top = max(max(d["pred"].max(), d["obs"].max()) for d in pts.values())
    return pts, np.ceil(top * 10) / 10


PT = np.round(np.arange(0.01, 0.51, 0.01), 2)


def dca_data(tau):
    nb_all = []
    for r in REPEATS:
        rows = POOLED[r]["row"].to_numpy()
        cif = sm.aalen_johansen_cif1(V_ALL[rows], S_ALL[rows], tau)
        nb_all.append(cif - (1 - cif) * PT / (1 - PT))
    treat_all = np.mean(nb_all, axis=0)
    curves = {}
    for panel, _ in PANELS:
        for key in panel_keys(panel):
            col, cs = col_for(panel, key, tau), []
            for r in REPEATS:
                df = POOLED[r]
                y = data.labels[f"y_{tau}"].to_numpy()[df["row"].to_numpy()]
                det = y >= 0
                cs.append(sm.net_benefit(df[col].to_numpy(float)[det], (y[det] == 1).astype(int),
                                         df[f"w_{tau}"].to_numpy(float)[det], len(df), PT))
            curves[(panel, key)] = np.mean(cs, axis=0)
    return curves, treat_all


# ---------------------------------------------------------------- shared colors (04_evaluation.ipynb, cell 15)
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e6e5e0", "#9a9993"
COLOR = {"zero": "#2a78d6", "discard": "#eb6834", "ipcw": "#1baf7a", "cs_cox": "#4a3aa7", "cs_rsf": "#008300"}


class Style:
    """Presentation only. PLOS: the original style of the PLOS figures. JDSA: Springer artwork rules."""

    def __init__(self, jdsa: bool):
        self.jdsa = jdsa
        self.name = {"zero": "Zero" if jdsa else "None (Zero)", "discard": "Discard", "ipcw": "IPCW",
                     "zero+smote": "Zero + SMOTE-NC" if jdsa else "SMOTE-NC (Zero + SMOTE-NC)",
                     "discard+smote": "Discard + SMOTE-NC", "ipcw+smote": "IPCW + SMOTE-NC",
                     "cs_cox": "Cause-specific Cox model", "cs_rsf": "Cause-specific random survival forest"}
        self.letters = "abcd" if jdsa else "ABCD"
        self.size = (6.85, 5.9) if jdsa else (7.5, 6.9)           # 174 x 150 mm (JDSA) / PLOS original
        base = {"axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
                "text.color": INK, "font.family": "Arial"}
        if jdsa:   # all lettering 8-9 pt; lines >= 0.5 pt; TrueType fonts embedded in EPS and PDF
            self.rc = {**base, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                       "ytick.labelsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                       "ps.fonttype": 42, "pdf.fonttype": 42}
        else:
            self.rc = {**base, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                       "ytick.labelsize": 7.5}
        self.legend_fs = 8 if jdsa else 7.5

    MARKER = {"zero": "o", "discard": "s", "ipcw": "^", "cs_cox": "D", "cs_rsf": "X"}
    OFFSET = {"zero": 0, "discard": 1, "ipcw": 2, "cs_cox": 3, "cs_rsf": 1}

    def line(self, key, markers=True, dca=False):
        b, smote = key.split("+")[0], key.endswith("+smote")
        if not self.jdsa:   # original: circles for every strategy; decision curves without markers
            return dict(color=COLOR[b], ls="--" if smote else "-", lw=1.3, marker="o" if markers else None,
                        ms=3.2, mfc="white" if smote else COLOR[b], mec=COLOR[b], mew=0.9)
        style = dict(color=COLOR[b], ls="--" if smote else "-", lw=1.2, marker=self.MARKER[b],
                     ms=4.2 if b == "ipcw" else 3.6, mfc="white" if smote else COLOR[b], mec=COLOR[b], mew=0.9)
        if dca:   # a marker every 0.05 of threshold probability, staggered by strategy so overlapping curves stay visible
            style["markevery"] = slice(4 + self.OFFSET[b], None, 5)
        return style

    def frame(self, axes, xlabel, ylabel):
        for ax, (_, title), letter in zip(axes.flat, PANELS, self.letters):
            ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", color=INK)
            ax.grid(color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[1, :]:
            ax.set_xlabel(xlabel)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)

    def legend(self, fig, extra, markers=True):
        """Legend columns fill top-down; this order reads row-wise: no oversampling / SMOTE-NC / references."""
        def h(k):
            return Line2D([], [], **{k2: v for k2, v in self.line(k, markers or self.jdsa).items() if k2 != "markevery"},
                          label=self.name[k])
        col1 = [h("zero"), h("zero+smote"), h("cs_cox")] + extra[:-1]
        col2 = [h("discard"), h("discard+smote"), h("cs_rsf")]
        col3 = [h("ipcw"), h("ipcw+smote"), extra[-1]]
        fig.legend(handles=col1 + col2 + col3, loc="lower center", ncol=3, frameon=False, fontsize=self.legend_fs,
                   handlelength=2.6, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))


def calibration_figure(style, pts, lim):
    with plt.rc_context(style.rc):
        fig, axes = plt.subplots(2, 2, figsize=style.size, sharex=True, sharey=True)
        for ax, (panel, _) in zip(axes.flat, PANELS):
            for key in panel_keys(panel):
                ax.plot(pts[(panel, key)]["pred"], pts[(panel, key)]["obs"], **style.line(key))
        for ax in axes.flat:
            ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls=":", zorder=0)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            if style.jdsa:   # ticks every 0.1 on both axes
                ax.xaxis.set_major_locator(MultipleLocator(0.1))
                ax.yaxis.set_major_locator(MultipleLocator(0.1))
        style.frame(axes, "Mean predicted risk", "Observed risk (Aalen–Johansen)")
        style.legend(fig, [Line2D([], [], color=INK2, lw=0.8, ls=":", label="Perfect calibration")])
        fig.tight_layout(rect=(0, 0.12 if style.jdsa else 0.10, 1, 1), h_pad=1.2, w_pad=1.0)
        if not style.jdsa:
            # The PLOS files have x ticks every 0.1 at the final drawing; run as a script, matplotlib's automatic
            # spacing chooses 0.2, the only difference found in the reproduction check (layout itself unchanged).
            for ax in axes.flat:
                ax.xaxis.set_major_locator(MultipleLocator(0.1))
    return fig


def dca_figure(style, curves, treat_all):
    with plt.rc_context(style.rc):
        fig, axes = plt.subplots(2, 2, figsize=style.size, sharex=True, sharey=True)
        for ax, (panel, _) in zip(axes.flat, PANELS):
            for key in panel_keys(panel):
                ax.plot(PT, curves[(panel, key)], **style.line(key, markers=False, dca=True))
            ax.plot(PT, treat_all, color=INK2, lw=0.9, ls="-.", zorder=1)
            ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
            ax.set_xlim(0, 0.5)
            ax.set_ylim(-0.01, max(0.02, treat_all.max() * 1.15))
        style.frame(axes, "Threshold probability", "Net benefit")
        style.legend(fig, [Line2D([], [], color=INK2, lw=0.9, ls="-.", label="Treat all"),
                           Line2D([], [], color=AXIS, lw=0.9, label="Treat none")], markers=False)
        fig.tight_layout(rect=(0, 0.14 if style.jdsa else 0.12, 1, 1), h_pad=1.2, w_pad=1.0)
    return fig


# ---------------------------------------------------------------- Fig. S1 (04_evaluation.ipynb, cell 26)
table_s6 = pd.read_csv(EVAL_SRC / "table_S6_ppv_matched_sensitivity.csv")


def figure_s1_plos():
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for i, tau in enumerate(sm.HORIZONS):
        for j, model in enumerate(sm.CLASSIFIERS):
            ax = axes[i, j]
            for rule in sm.CENSORING_RULES:
                g = table_s6[(table_s6["tau"] == tau) & (table_s6["model"] == model) & (table_s6["rule"] == rule)]
                ax.plot(g["sensitivity_target"], g["mean"], marker="o", ms=3, color=COLOR[rule],
                        label={"zero": "Zero", "discard": "Discard", "ipcw": "IPCW"}[rule])
                ax.fill_between(g["sensitivity_target"], g["ci_low"], g["ci_high"], color=COLOR[rule], alpha=0.15)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"{LABELS[model]}, {tau} months", fontsize=10)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("PPV(SMOTE-NC) − PPV(no oversampling)")
            if i == 2:
                ax.set_xlabel("Matched sensitivity")
    axes[0, 0].legend(title="Censoring strategy", fontsize=8)
    plt.tight_layout()
    return fig


def figure_s1_jdsa(style):
    """Sized for the landscape page of Online Resource 1 (9.0 x 5.9 in), so the lettering stays >= 8 pt."""
    with plt.rc_context(style.rc):
        fig, axes = plt.subplots(3, 3, figsize=(9.0, 5.9), sharex=True)
        letters = "abcdefghi"
        for i, tau in enumerate(sm.HORIZONS):
            for j, model in enumerate(sm.CLASSIFIERS):
                ax = axes[i, j]
                for rule in sm.CENSORING_RULES:
                    g = table_s6[(table_s6["tau"] == tau) & (table_s6["model"] == model) & (table_s6["rule"] == rule)]
                    ax.fill_between(g["sensitivity_target"], g["ci_low"], g["ci_high"], color=COLOR[rule], alpha=0.15, lw=0)
                    ax.plot(g["sensitivity_target"], g["mean"], color=COLOR[rule], lw=1.1, marker=Style.MARKER[rule],
                            ms=4.0 if rule == "ipcw" else 3.4, mfc=COLOR[rule], mec=COLOR[rule])
                ax.axhline(0, color=INK, lw=0.7)
                ax.set_title(f"{letters[3 * i + j]}   {LABELS[model]}, {tau} months", loc="left", fontweight="bold")
                ax.grid(color=GRID, lw=0.5)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                if i == 2:
                    ax.set_xlabel("Matched sensitivity")
        fig.supylabel("PPV with SMOTE-NC − PPV without oversampling", fontsize=8.5, color=INK)
        handles = [Line2D([], [], color=COLOR[r], lw=1.1, marker=Style.MARKER[r], ms=3.8, label=n)
                   for r, n in (("zero", "Zero"), ("discard", "Discard"), ("ipcw", "IPCW"))]
        handles.append(Patch(facecolor=INK2, alpha=0.15, label="95% confidence interval"))
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0.01, 0.06, 1, 1), h_pad=0.8, w_pad=1.0)
    return fig


# ---------------------------------------------------------------- saving and checks
report = {"reproduction": {}, "figures": {}}


def min_font(fig):
    return min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip())


def save_jdsa(fig, stem, eps=False):
    w, h = (x * 25.4 for x in fig.get_size_inches())
    info = {"width_mm": round(w, 1), "height_mm": round(h, 1), "min_font_pt": min_font(fig)}
    assert info["min_font_pt"] >= 8, (stem, info)
    if eps:
        assert w <= 174.5 and h <= 234, (stem, info)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.savefig(OUT / f"{stem}.eps", facecolor="white")
        eps_text = (OUT / f"{stem}.eps").read_bytes()
        info["eps_warnings"] = [str(c.message) for c in caught]
        info["eps_type42_fonts"] = eps_text.count(b"/FontType 42")
        info["eps_header"] = eps_text[:30].decode("latin-1").strip()
        fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=600, facecolor="white")
    im = Image.open(OUT / f"{stem}.png").convert("RGB")
    im.save(OUT / f"{stem}.png", dpi=(600, 600))
    gray = im.convert("L")
    gray.thumbnail((1400, 1400))
    gray.save(CHECKS / f"{stem}_grayscale.png")
    report["figures"][stem] = info
    plt.close(fig)


def compare(fig, name, dpi, facecolor="white"):
    fig.savefig(REPRO / name, dpi=dpi, **({"facecolor": facecolor} if facecolor else {}))
    plt.close(fig)
    a, b = Image.open(REPRO / name).convert("RGBA"), Image.open(EVAL_SRC / name).convert("RGBA")
    if a.size != b.size:
        report["reproduction"][name] = {"identical": False, "size_new": a.size, "size_plos": b.size}
        return
    diff = np.asarray(ImageChops.difference(a, b))
    report["reproduction"][name] = {"identical": bool(diff.max() == 0), "differing_pixels": int((diff.max(axis=2) > 0).sum()),
                                    "max_difference": int(diff.max())}


PLOS, JDSA = Style(False), Style(True)
for tau in sm.HORIZONS:
    pts, lim = calibration_data(tau)
    compare(calibration_figure(PLOS, pts, lim), f"figure_calibration_{tau}m.png", 300)
    stem = {120: "Fig3", 36: "FigS2a", 60: "FigS2b"}[tau]
    save_jdsa(calibration_figure(JDSA, pts, lim), stem, eps=tau == 120)

    curves, treat_all = dca_data(tau)
    compare(dca_figure(PLOS, curves, treat_all), f"figure_dca_{tau}m.png", 300)
    stem = {120: "Fig4", 36: "FigS3a", 60: "FigS3b"}[tau]
    save_jdsa(dca_figure(JDSA, curves, treat_all), stem, eps=tau == 120)

compare(figure_s1_plos(), "figure_S1_ppv_matched_sensitivity.png", 200, facecolor=None)
save_jdsa(figure_s1_jdsa(JDSA), "FigS1")

(CHECKS / "figure_checks.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
print(json.dumps(report, indent=1, default=str))
