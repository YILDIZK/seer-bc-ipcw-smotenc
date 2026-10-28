"""Fig 1 (cohort selection) for JDSA: EPS (vector, fonts embedded), PNG for Word (600 dpi), PDF preview.
All counts are read from figure_data/cohort_cascade.csv (Online Resource 1, Table S1).
JDSA artwork rules: 174 mm wide, at most 234 mm high, lettering 8-12 pt Arial (Figures_plan_v1.md, F4)."""
import re
from pathlib import Path

import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import pandas as pd
from PIL import Image

OUT = config.WORK_DIR / "Figures"
OUT.mkdir(exist_ok=True)

# ---- counts from Table S1 (step -> excluded, remaining), as published in Online Resource 1
cascade = pd.read_csv(config.REPO_DIR / "figure_data" / "cohort_cascade.csv")
steps = {int(r.step): (None if pd.isna(r.excluded) else int(r.excluded), int(r.remaining)) for r in cascade.itertuples()}
n = {k: v[1] for k, v in steps.items()}
x = {k: v[0] for k, v in steps.items()}
for k in range(1, 14):                                   # arithmetic check of the whole cascade
    assert n[k - 1] - x[k] == n[k], k
bc, oc, alive = 30_713, 28_268, 198_481                  # outcome split of the analytic cohort (Fig 1, S1 Table)
assert bc + oc + alive == n[13]
f = lambda v: f"{v:,}"

MAIN = [
    (["Malignant breast tumors, women aged ≥20 years,", "diagnosed 2010–2017, SEER-17 (Nov 2025 submission)"],
     f"n = {f(n[0])} tumor records"),
    (["Microscopically confirmed"], f"n = {f(n[1])}"),
    (["First primary tumor per patient (SEER*Stat extract)"], f"n = {f(n[2])} patients"),
    (["Reporting source other than autopsy only"], f"n = {f(n[3])}"),
    (["One primary only"], f"n = {f(n[4])}"),
    (["Complete eligibility-defining covariates"], f"n = {f(n[11])}"),
    (["Analytic cohort"], f"n = {f(n[13])}"),
]
COHORT_NOTE = [f"Breast cancer deaths {f(bc)} · Other-cause deaths {f(oc)}", f"Alive at last contact {f(alive)}"]
EXCL = [   # (title, [(label, count), ...]) between consecutive main boxes
    (f"Excluded: not microscopically confirmed, n = {f(x[1])}", []),
    (f"Excluded: second or later primary tumors, n = {f(x[2])}", []),
    (f"Excluded: autopsy only, n = {f(x[3])}", []),
    (f"Excluded: first of ≥2 primary tumors, n = {f(x[4])}", []),
    (f"Excluded (sequential), n = {f(sum(x[k] for k in range(5, 12)))}",
     [("Race or marital status unknown", x[5]), ("ER/PR/HER2 unknown or borderline", x[6]),
      ("Laterality unknown or bilateral", x[7]), ("AJCC T/N/M or stage unknown", x[8]),
      ("In situ only (Tis / stage 0)", x[9]), ("Metastatic site unknown", x[10]),
      ("Grade or histologic type unknown", x[11])]),
    (f"Excluded: outcome not ascertainable, n = {f(x[12] + x[13])}",
     [("Unknown cause of death", x[12]), ("Unknown or zero survival time", x[13])]),
]

# ---- geometry (inches); 6.85 in = 174 mm, the JDSA full-width size; smallest text FS - 0.4 = 8.0 pt
W, FS, LH, PAD = 6.85, 8.4, 0.155, 0.075
MX0, MW = 0.05, 3.45                      # main column
EX0, EW = 3.75, 3.05                      # exclusion column
INK, LINE, SHADE = "#0b0b0b", "#3a3a38", "#eeeeec"
plt.rcParams.update({"font.family": "Arial", "ps.fonttype": 42, "pdf.fonttype": 42})


def main_h(i):
    lines = len(MAIN[i][0]) + 1 + (len(COHORT_NOTE) if i == len(MAIN) - 1 else 0)
    return lines * LH + 2 * PAD


def excl_h(j):
    title_lines = 2 if EXCL[j][1] else 2       # titles wrap to two lines in the exclusion column
    return (title_lines + len(EXCL[j][1])) * LH + 2 * PAD


gaps = [max(0.30, excl_h(j) + 0.14) for j in range(len(EXCL))]
H = sum(main_h(i) for i in range(len(MAIN))) + sum(gaps) + 0.08
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x0, y_top, w, h, fill="white", lw=0.8):
    ax.add_patch(Rectangle((x0, y_top - h), w, h, facecolor=fill, edgecolor=LINE, lw=lw))


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                 color=LINE, shrinkA=0, shrinkB=0))


def wrap_title(t):
    if ", n = " in t:
        a, b = t.split(", n = ")
        return [a, f"n = {b}"]
    return [t]


y = H - 0.04
cx = MX0 + MW / 2
for i, (text, count) in enumerate(MAIN):
    h = main_h(i)
    last = i == len(MAIN) - 1
    box(MX0, y, MW, h, fill=SHADE if last else "white", lw=1.1 if last else 0.8)
    ty = y - PAD - LH / 2
    for t in text:
        ax.text(cx, ty, t, ha="center", va="center", fontsize=FS, color=INK, fontweight="bold" if last else "normal")
        ty -= LH
    ax.text(cx, ty, count, ha="center", va="center", fontsize=FS, color=INK, fontweight="bold")
    if last:
        for k, note in enumerate(COHORT_NOTE, 1):
            ax.text(cx, ty - k * LH, note, ha="center", va="center", fontsize=FS - 0.4, color=INK)
    y -= h
    if i < len(EXCL):
        g = gaps[i]
        arrow(cx, y, cx, y - g)                          # main flow
        eh = excl_h(i)
        ey_mid = y - g / 2
        arrow(cx, ey_mid, EX0, ey_mid)                   # exclusion branch
        etop = ey_mid + eh / 2
        box(EX0, etop, EW, eh)
        ty = etop - PAD - LH / 2
        for t in wrap_title(EXCL[i][0]):
            ax.text(EX0 + PAD, ty, t, ha="left", va="center", fontsize=FS, color=INK)
            ty -= LH
        for label, cnt in EXCL[i][1]:
            ax.text(EX0 + PAD + 0.1, ty, label, ha="left", va="center", fontsize=FS - 0.4, color=INK)
            ax.text(EX0 + EW - PAD, ty, f(cnt), ha="right", va="center", fontsize=FS - 0.4, color=INK)
            ty -= LH
        y -= g

assert H * 25.4 <= 234, f"Fig1 is {H * 25.4:.0f} mm high (JDSA maximum 234 mm)"
assert min(t.get_fontsize() for t in ax.texts) >= 8, "lettering below 8 pt"
fig.savefig(OUT / "Fig1.eps", facecolor="white")
fig.savefig(OUT / "Fig1.png", dpi=600, facecolor="white")
Image.open(OUT / "Fig1.png").convert("RGB").save(OUT / "Fig1.png", dpi=(600, 600))
fig.savefig(OUT / "Fig1.pdf", facecolor="white")
print(f"Fig1: {W * 25.4:.0f} x {H * 25.4:.0f} mm; smallest text {min(t.get_fontsize() for t in ax.texts):.1f} pt")
im = Image.open(OUT / "Fig1.png")
print("Fig1.png", im.size, im.mode, im.info.get("dpi"))
