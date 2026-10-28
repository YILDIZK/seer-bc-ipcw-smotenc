"""Fig 2 (modeling and evaluation workflow) for JDSA, same style as Fig 1: EPS (vector, fonts embedded),
PNG for Word (600 dpi), PDF preview. Shows the data partition used by each step and the post hoc
recalibration (Figures_plan_v1.md, F4-F5). 174 mm wide, at most 234 mm high, lettering >= 8 pt Arial."""
from pathlib import Path

import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

OUT = config.WORK_DIR / "Figures"
OUT.mkdir(exist_ok=True)

INK, INK2, LINE, SHADE = "#0b0b0b", "#52514e", "#3a3a38", "#eeeeec"
W, FS, LH, PAD, GAP = 6.85, 8.4, 0.155, 0.075, 0.28
plt.rcParams.update({"font.family": "Arial", "ps.fonttype": 42, "pdf.fonttype": 42, "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})

G, F1, GAM, TAU = r"$\hat{G}$", r"$F_1(\tau\,|\,X)$", "γ*", r"$\tau$"
# each box: (title, [content lines], shaded)
B = {
    "IN": ("Inputs fixed before modeling",
           ["Cohort: n = 257,462 · 28 predictors",
            f"Labels: {TAU} = 36, 60 and 120 months · Partition: 3 repeats × 5 folds"], False),
    "OT": ("Outer training set (80%)", [], False),
    "ENC": ("Encoder", [r"log(1 + $x$) · median imputation", "standardization · one-hot encoding"], False),
    "GH": (r"Censoring model $\hat{G}(t\,|\,X)$",
           ["Cox regression; event = censoring",
            r"IPCW weights $\omega = 1\,/\,\hat{G}(\min(V,\tau)^{-}\,|\,X)$,",
            "truncated at the 99th percentile, for",
            "training, validation and test patients"], False),
    "IT": ("Inner training set (64%)",
           ["Classifiers and reference models fitted · the only data oversampled"], False),
    "GRID": (f"Factorial grid (each horizon {TAU}; Algorithm 1)",
             ["Censoring: Zero · Discard · IPCW", "× Imbalance: none · SMOTE-NC (1:1)*",
              "× Classifier: logistic regression ·", "random forest · XGBoost"], True),
    "REF": ("Reference survival models",
            ["Cause-specific Cox model", "Cause-specific random survival forest",
             f"→ predicted cumulative incidence {F1}"], False),
    "IV": ("Inner validation set (16%), never oversampled",
           [f"XGBoost early stopping · threshold {GAM}: minimum FPR subject to TPR ≥ 0.80",
            "Post hoc: recalibration maps fitted with IPCW-weighted outcomes"], False),
    "TS": ("Test fold (20%)", [f"Predictions from frozen models and {GAM}"], False),
    "EV": (f"Evaluation against {F1} with IPCW weights",
           [f"Brier score (primary) · AUC · sensitivity, PPV and F1 at {GAM}",
            "Calibration (Aalen–Johansen), O/E · decision curves"], True),
    "ST": ("Statistical analysis over the 15 splits",
           ["Mean ± SD · corrected repeated cross-validation t-test",
            "Paired contrasts between approaches · factorial effects"], False),
}
FULL, HALF, NARROW = 5.6, 3.15, 3.2
CX, LX, RX = W / 2, 0.22 + HALF / 2, W - 0.22 - HALF / 2
ROWS = [  # (boxes in the row as (key, x_center, width)), inside the per-split container?
    ([("IN", CX, FULL)], False),
    ([("OT", CX, NARROW)], True),
    ([("ENC", LX, HALF), ("GH", RX, HALF)], True),
    ([("IT", CX, FULL)], True),
    ([("GRID", LX, HALF), ("REF", RX, HALF)], True),
    ([("IV", CX, FULL)], True),
    ([("TS", CX, NARROW)], True),
    ([("EV", CX, FULL)], False),
    ([("ST", CX, FULL)], False),
]


def h_of(key):
    return (1 + len(B[key][1])) * LH + 2 * PAD


row_h = [max(h_of(k) for k, _, _ in boxes) for boxes, _ in ROWS]
CONTAINER_LABEL = 0.22                         # space for the container label
H = sum(row_h) + GAP * (len(ROWS) - 1) + CONTAINER_LABEL + 0.2 + 0.08

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                 color=LINE, shrinkA=0, shrinkB=0))


def seg(x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color=LINE, lw=0.8, solid_capstyle="butt")


pos = {}                                       # key -> (x_center, y_top, y_bottom)
y = H - 0.04
for i, (boxes, inside) in enumerate(ROWS):
    if inside and not ROWS[i - 1][1]:
        y_container_top = y + GAP / 2
        y -= CONTAINER_LABEL                  # leave room for the label under the container top
    rh = row_h[i]
    for key, xc, w in boxes:
        title, lines, shaded = B[key]
        h = h_of(key)
        top = y - (rh - h) / 2
        ax.add_patch(Rectangle((xc - w / 2, top - h), w, h, facecolor=SHADE if shaded else "white",
                               edgecolor=LINE, lw=1.1 if shaded else 0.8))
        ty = top - PAD - LH / 2
        ax.text(xc, ty, title, ha="center", va="center", fontsize=FS, fontweight="bold", color=INK)
        for t in lines:
            ty -= LH
            ax.text(xc, ty, t, ha="center", va="center", fontsize=FS - 0.3, color=INK)
        pos[key] = (xc, top, top - h)
    y -= rh
    if inside and (i + 1 == len(ROWS) or not ROWS[i + 1][1]):
        y_container_bottom = y - GAP / 2 + 0.06
    y -= GAP

# per-split container
ax.add_patch(Rectangle((0.08, y_container_bottom), W - 0.16, y_container_top - y_container_bottom - 0.04,
                       facecolor="#fafaf9", edgecolor=INK2, lw=0.8, ls=(0, (4, 3)), zorder=0))
ax.text(0.18, y_container_top - 0.04 - 0.13, "Repeated in each of the 15 splits", ha="left", va="center",
        fontsize=FS - 0.3, style="italic", color=INK2)


def down(a, b):                                # straight arrow from box a to box b (same column)
    arrow(pos[a][0], pos[a][2], pos[b][0], pos[b][1])


def fork(a, targets):                          # one box splitting into several boxes below
    xa, _, ya = pos[a]
    ym = (ya + max(pos[t][1] for t in targets)) / 2
    seg(xa, ya, xa, ym)
    seg(min(pos[t][0] for t in targets), ym, max(pos[t][0] for t in targets), ym)
    for t in targets:
        arrow(pos[t][0], ym, pos[t][0], pos[t][1])


def join(sources, b):                          # several boxes merging into one box below
    xb, yb, _ = pos[b]
    ym = (min(pos[s][2] for s in sources) + yb) / 2
    for s in sources:
        seg(pos[s][0], pos[s][2], pos[s][0], ym)
    seg(min(pos[s][0] for s in sources), ym, max(pos[s][0] for s in sources), ym)
    arrow(xb, ym, xb, yb)


down("IN", "OT")
fork("OT", ["ENC", "GH"])
join(["ENC", "GH"], "IT")
fork("IT", ["GRID", "REF"])
join(["GRID", "REF"], "IV")
down("IV", "TS")
down("TS", "EV")
down("EV", "ST")

assert H * 25.4 <= 234, f"Fig2 is {H * 25.4:.0f} mm high (JDSA maximum 234 mm)"
assert min(t.get_fontsize() for t in ax.texts) >= 8, "lettering below 8 pt"
fig.savefig(OUT / "Fig2.eps", facecolor="white")
fig.savefig(OUT / "Fig2.png", dpi=600, facecolor="white")
Image.open(OUT / "Fig2.png").convert("RGB").save(OUT / "Fig2.png", dpi=(600, 600))
fig.savefig(OUT / "Fig2.pdf", facecolor="white")
print(f"Fig2: {W * 25.4:.0f} x {H * 25.4:.0f} mm; smallest text {min(t.get_fontsize() for t in ax.texts):.1f} pt")
im = Image.open(OUT / "Fig2.png")
print("Fig2.png", im.size, im.mode, im.info.get("dpi"))
