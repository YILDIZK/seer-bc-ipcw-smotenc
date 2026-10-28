"""Write an artificial SEER-like case listing, so that the whole pipeline can be run without SEER access.

Every record is invented. The label strings are those expected by the recoding rules of 01_preprocess.ipynb and
02_model_prep.ipynb (SEER*Stat label format; see seer_stat/export.csv.dic for the columns); category probabilities and the
survival model are set by hand in this script. No SEER record, count or frequency is used. Survival times follow a simple
simulation with administrative censoring on 31 December 2023 and some loss to follow-up, so that the censoring, competing-
risks and imbalance steps all have work to do. The results have no clinical meaning and will not match the article.

Usage (from the repository folder):  python synthetic/make_synthetic_extract.py [n_records] [output.csv]
Defaults: 16,000 records, written to synthetic/synthetic_case_listing.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16_000
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "synthetic_case_listing.csv"
rng = np.random.default_rng(2026)

# Columns of the SEER*Stat export, in order (seer_stat/export.csv.dic)
COLUMNS = [
    "Age recode with <1 year olds and 90+", "Age recode with single ages and 90+", "Patient ID", "Sequence number",
    "Type of Reporting Source", "Total number of in situ/malignant tumors for patient",
    "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Marital status at diagnosis", "Year of diagnosis",
    "Median household income inflation adj to 2024", "Rural-Urban Continuum Code", "Histologic Type ICD-O-3",
    "Histology recode - broad groupings", "ICD-O-3 Hist/behav, malignant", "Grade Recode (thru 2017)", "Laterality",
    "Primary Site", "Primary Site - labeled", "Diagnostic Confirmation", "Tumor Size Over Time Recode (1988+)",
    "Regional nodes examined (1988+)", "Regional nodes positive (1988+)", "LN Positive Axillary Level I-II Recode (2010+)",
    "Derived AJCC Stage Group, 7th ed (2010-2015)", "Derived AJCC T, 7th ed (2010-2015)", "Derived AJCC N, 7th ed (2010-2015)",
    "Derived AJCC M, 7th ed (2010-2015)", "7th Edition Stage Group Recode (2016-2017)", "Derived SEER Cmb Stg Grp (2016-2017)",
    "Derived SEER Combined T (2016-2017)", "Derived SEER Combined N (2016-2017)", "Derived SEER Combined M (2016-2017)",
    "Derived SEER Combined T Src (2016-2017)", "Derived SEER Combined N Src (2016-2017)", "Derived SEER Combined M Src (2016-2017)",
    "SEER Combined Mets at DX-bone (2010+)", "SEER Combined Mets at DX-brain (2010+)", "SEER Combined Mets at DX-liver (2010+)",
    "SEER Combined Mets at DX-lung (2010+)", "Breast Subtype (2010+)", "ER Status Recode Breast Cancer (1990+)",
    "PR Status Recode Breast Cancer (1990+)", "Derived HER2 Recode (2010+)",
    "CS site-specific factor 1 (2004-2017 varying by schema)", "CS site-specific factor 2 (2004-2017 varying by schema)",
    "CS site-specific factor 3 (2004-2017 varying by schema)", "CS site-specific factor 4 (2004-2017 varying by schema)",
    "CS site-specific factor 5 (2004-2017 varying by schema)", "CS site-specific factor 6 (2004-2017 varying by schema)",
    "Adjusted CS site-specific factor 7 (2004-2017 varying by schema)", "CS site-specific factor 15 (2004-2017 varying by schema)",
    "RX Summ--Surg Prim Site (1998-2022)", "RX Summ--Surg/Rad Seq", "Radiation recode", "Chemotherapy recode (yes, no/unk)",
    "RX Summ--Systemic/Sur Seq (2007+)", "Time from diagnosis to treatment in days recode",
    "SEER cause-specific death classification", "SEER other cause of death classification", "Survival months", "Survival Days",
    "Survival months flag", "Vital status recode (study cutoff used)", "Year of follow-up recode",
    "COD to site recode ICD-O-3 2023 Revision Expanded (1999+)",
]


def pick(options, size=None):
    """Draw labels from {label: probability} (probabilities are normalized)."""
    labels = list(options)
    p = np.array([options[k] for k in labels], dtype=float)
    return rng.choice(labels, size=size, p=p / p.sum())


# ------------------------------------------------------------------ patient characteristics
year = rng.integers(2010, 2018, N)
month = rng.integers(1, 13, N)
age = np.clip(np.round(rng.normal(60, 13, N)), 20, 90).astype(int)
race = pick({"Non-Hispanic White": 66, "Hispanic (All Races)": 13, "Non-Hispanic Black": 10,
             "Non-Hispanic Asian or Pacific Islander": 9, "Non-Hispanic American Indian/Alaska Native": 1,
             "Non-Hispanic Unknown Race": 1}, N)
marital = pick({"Married (including common law)": 52, "Single (never married)": 15, "Widowed": 13, "Divorced": 12,
                "Separated": 2, "Unmarried or Domestic Partner": 1, "Unknown": 5}, N)
INCOME = ["< $40,000"] + [f"${lo:,} - ${lo + 4_999:,}" for lo in range(40_000, 120_000, 5_000)] + ["$120,000+"]
income = rng.choice(INCOME, N)
rural = pick({"Counties in metropolitan areas ge 1 million pop": 55,
              "Counties in metropolitan areas of 250,000 to 1 million pop": 20,
              "Counties in metropolitan areas of lt 250 thousand pop": 10,
              "Nonmetropolitan counties adjacent to a metropolitan area": 9,
              "Nonmetropolitan counties not adjacent to a metropolitan area": 5,
              "Unknown/missing/no match/Not 1990-2024": 1}, N)
sequence = pick({"One primary only": 85, "1st of 2 or more primaries": 15}, N)
source = pick({"Hospital inpatient/outpatient or clinic": 97, "Physicians office/private medical practitioner (LMD)": 2.9,
               "Autopsy only": 0.1}, N)
SITES = {"C50.0-Nipple": 1, "C50.1-Central portion of breast": 5, "C50.2-Upper-inner quadrant of breast": 12,
         "C50.3-Lower-inner quadrant of breast": 6, "C50.4-Upper-outer quadrant of breast": 35,
         "C50.5-Lower-outer quadrant of breast": 8, "C50.6-Axillary tail of breast": 1,
         "C50.8-Overlapping lesion of breast": 16, "C50.9-Breast, NOS": 16}
site = pick(SITES, N)
laterality = pick({"Left - origin of primary": 50, "Right - origin of primary": 49, "Bilateral, single primary": 0.5,
                   "Paired site, but no information concerning laterality": 0.5}, N)
grade_n = rng.choice([1, 2, 3, 4], N, p=[0.22, 0.44, 0.33, 0.01])
grade_unknown = rng.random(N) < 0.04
GRADE = {1: "Well differentiated; Grade I", 2: "Moderately differentiated; Grade II", 3: "Poorly differentiated; Grade III",
         4: "Undifferentiated; anaplastic; Grade IV"}
HIST = {  # code: (probability, ICD-O-3 description, broad grouping)
    "8500": (68, "Infiltrating duct carcinoma, NOS", "8500-8549: ductal and lobular neoplasms"),
    "8520": (9, "Lobular carcinoma, NOS", "8500-8549: ductal and lobular neoplasms"),
    "8522": (6, "Infiltrating duct and lobular carcinoma", "8500-8549: ductal and lobular neoplasms"),
    "8523": (4, "Infiltrating duct mixed with other types of carcinoma", "8500-8549: ductal and lobular neoplasms"),
    "8480": (3, "Mucinous adenocarcinoma", "8480-8499: cystic, mucinous and serous neoplasms"),
    "8211": (2, "Tubular adenocarcinoma", "8140-8389: adenomas and adenocarcinomas"),
    "8501": (3, "Comedocarcinoma, NOS", "8500-8549: ductal and lobular neoplasms"),
    "8510": (3, "Medullary carcinoma, NOS", "8500-8549: ductal and lobular neoplasms"),
    "8524": (1, "Infiltrating lobular mixed with other types of carcinoma", "8500-8549: ductal and lobular neoplasms"),
    "8000": (1, "Neoplasm, malignant", "8000-8009: unspecified neoplasms"),
}
hist = pick({k: v[0] for k, v in HIST.items()}, N)
subtype = pick({"HR+/HER2-": 66, "HR+/HER2+": 12, "HR-/HER2-": 12, "HR-/HER2+": 6, "Unknown": 4}, N)
hr_pos = np.char.startswith(subtype.astype(str), "HR+")
her2_pos = np.char.endswith(subtype.astype(str), "HER2+")
er = np.where(hr_pos, np.where(rng.random(N) < 0.97, "Positive", "Negative"), "Negative")   # ER-/PR+ also occurs
pr = np.where(hr_pos, np.where((er == "Negative") | (rng.random(N) < 0.85), "Positive", "Negative"), "Negative")
her2 = np.where(her2_pos, "Positive", "Negative")
unknown_receptor = subtype == "Unknown"
er, pr, her2 = (np.where(unknown_receptor, "Unknown", x) for x in (er, pr, her2))

# ------------------------------------------------------------------ stage, T, N, M (AJCC 7th edition)
stage = pick({"IA": 38, "IB": 3, "IIA": 22, "IIB": 12, "IIIA": 7, "IIIB": 4, "IIIC": 5, "IV": 7, "0": 0.5, "UNK": 1.5}, N)
T_BY_STAGE = {"IA": {"T1mic": 4, "T1a": 10, "T1b": 26, "T1c": 58, "T1": 2}, "IB": {"T0": 30, "T1c": 70},
              "IIA": {"T0": 2, "T1c": 38, "T2": 60}, "IIB": {"T2": 70, "T3": 30}, "IIIA": {"T2": 40, "T3": 60},
              "IIIB": {"T4a": 15, "T4b": 50, "T4c": 10, "T4d": 25}, "IIIC": {"T1c": 15, "T2": 40, "T3": 25, "T4b": 15, "T4d": 5},
              "IV": {"T1c": 15, "T2": 35, "T3": 20, "T4b": 20, "T4d": 10}, "0": {"Tis": 100}, "UNK": {"TX": 100}}
N_BY_STAGE = {"IA": {"N0": 85, "N0(i-)": 5, "N0(i+)": 8, "N0(mol+)": 2}, "IB": {"N1mi": 100},
              "IIA": {"N0": 55, "N1mi": 15, "N1a": 30}, "IIB": {"N0": 30, "N1a": 50, "N1b": 10, "N1": 10},
              "IIIA": {"N1a": 25, "N2a": 60, "N2": 15}, "IIIB": {"N0": 40, "N1a": 40, "N2a": 20},
              "IIIC": {"N3a": 60, "N3b": 20, "N3": 20}, "IV": {"N0": 30, "N1a": 30, "N2a": 15, "N3a": 25},
              "0": {"N0": 100}, "UNK": {"NX": 100}}
t_stage = np.array([pick(T_BY_STAGE[s]) for s in stage])
n_stage = np.array([pick(N_BY_STAGE[s]) for s in stage])
m_stage = np.where(stage == "IV", np.where(rng.random(N) < 0.95, "M1", "M0(i+)"), np.where(rng.random(N) < 0.01, "M0(i+)", "M0"))
m_unknown = rng.random(N) < 0.005
mets = {site_: np.where(m_stage == "M1", np.where(rng.random(N) < p, "Yes", "No"), "No")
        for site_, p in [("bone", 0.6), ("brain", 0.1), ("liver", 0.25), ("lung", 0.3)]}
mets["bone"] = np.where(m_unknown, "Unknown", mets["bone"])

ROMAN_TO_CMB = {"IA": "1A", "IB": "1B", "IIA": "2A", "IIB": "2B", "IIIA": "3A", "IIIB": "3B", "IIIC": "3C", "IV": "4", "0": "0",
                "UNK": "99"}


T_COMBINED = {"T0": "0", "Tis": "is", "T1mic": "1mi", "T1a": "1a", "T1b": "1b", "T1c": "1c", "T1": "1", "T2": "2", "T3": "3",
              "T4a": "4a", "T4b": "4b", "T4c": "4c", "T4d": "4d"}
N_COMBINED = {"N0": "0", "N0(i-)": "0i-", "N0(i+)": "0i+", "N0(mol+)": "0m+", "N1mi": "1mi", "N1a": "1a", "N1b": "1b", "N1": "1",
              "N2a": "2a", "N2": "2", "N3a": "3a", "N3b": "3b", "N3": "3"}


def combined_t(t):   # SEER Combined T (2016-2017): clinical or pathological prefix + code
    return "cX" if t == "TX" else ("p" if rng.random() < 0.8 else "c") + T_COMBINED[t]


def combined_n(n):
    return "cX" if n == "NX" else ("p" if rng.random() < 0.8 else "c") + N_COMBINED[n]


def combined_m(m):
    return {"M0": "c0", "M0(i+)": "c0I+", "M1": "p1" if rng.random() < 0.3 else "c1"}[m]


era_1015 = year <= 2015
blank = "Blank(s)"
stage_1015 = np.where(era_1015, np.where(stage == "UNK", "UNK Stage", stage), blank)
t_1015 = np.where(era_1015, t_stage, blank)
n_1015 = np.where(era_1015, n_stage, blank)
m_1015 = np.where(era_1015, np.where(m_unknown, "MX", m_stage), blank)
stage_1617 = np.where(era_1015, blank, np.where(stage == "UNK", "UNK Stage", stage))
cmb_1617 = np.where(era_1015, blank, [ROMAN_TO_CMB[s] for s in stage])
t_1617 = np.where(era_1015, blank, [combined_t(t) for t in t_stage])
n_1617 = np.where(era_1015, blank, [combined_n(n) for n in n_stage])
m_1617 = np.where(era_1015, blank, np.where(m_unknown, "cX", [combined_m(m) for m in m_stage]))

# ------------------------------------------------------------------ tumor size and nodes (SEER codes)
size_mm = {"T0": 0, "T1mic": 1, "T1a": 4, "T1b": 8, "T1c": 15, "T1": 12, "T2": 30, "T3": 65}
size = np.array([size_mm.get(t, 45) for t in t_stage], dtype=float) * rng.uniform(0.7, 1.3, N)
size_code = np.char.zfill(np.clip(np.round(size), 1, 400).astype(int).astype(str), 3)
size_code = np.where(t_stage == "T0", "000", size_code)
size_code = np.where(t_stage == "T1mic", "990", size_code)
size_code = np.where(t_stage == "T4d", "998", size_code)
size_code = np.where(rng.random(N) < 0.03, "999", size_code)

positive = np.array([{"N0": 0, "N0(i-)": 0, "N0(i+)": 0, "N0(mol+)": 0, "N1mi": 1, "N1a": 2, "N1b": 2, "N1": 2, "N2a": 6,
                      "N2": 6, "N3a": 12, "N3b": 12, "N3": 12}.get(n, 0) for n in n_stage])
positive = positive + rng.integers(0, 3, N) * (positive > 0)
examined = positive + rng.integers(1, 20, N)
exam_code = np.char.zfill(np.minimum(examined, 90).astype(str), 2)
pos_code = np.char.zfill(np.minimum(positive, 90).astype(str), 2)
special = pick({"none": 92, "not_examined": 3, "aspiration": 1.5, "unknown_count": 1.5, "unknown": 2}, N)
exam_code = np.select([special == "not_examined", special == "aspiration", special == "unknown_count", special == "unknown"],
                      ["00", "95", "97", "99"], exam_code)
pos_code = np.select([special == "not_examined", special == "aspiration", special == "unknown_count", special == "unknown"],
                     ["98", "95", "97", "99"], pos_code)
# Nodes removed but the number examined not recorded, while the positive count is known (as occurs in registry data)
exam_code = np.where((special == "none") & (rng.random(N) < 0.03), rng.choice(["96", "97", "98"], N), exam_code)
axillary = np.char.zfill(np.maximum(positive - rng.integers(0, 2, N), 0).astype(str), 2)
axillary = np.where(axillary == "00", "All ipsilateral axillary nodes examined negative", axillary)
axillary = np.select([special == "not_examined", special == "aspiration", special == "unknown_count", special == "unknown"],
                     ["No axillary nodes examined", "Positive aspiration or needle core biopsy",
                      "Positive nodes, number unspecified", "Unknown; not stated"], axillary)
axillary = np.where(rng.random(N) < 0.08, "Not applicable", axillary)

# ------------------------------------------------------------------ survival: competing hazards per month
stage_lp = {"IA": 0.0, "IB": 0.3, "IIA": 0.7, "IIB": 1.1, "IIIA": 1.7, "IIIB": 2.2, "IIIC": 2.4, "IV": 3.6, "0": 0, "UNK": 1}
lp_bc = (np.array([stage_lp[s] for s in stage]) + 0.35 * (grade_n - 2) + 0.9 * (subtype == "HR-/HER2-")
         + 0.012 * (age - 60) - 0.2 * (subtype == "HR+/HER2+"))
h_bc = np.exp(-7.6 + lp_bc)
h_oc = np.exp(-7.4 + 0.075 * (age - 60))
t_event = rng.exponential(1 / (h_bc + h_oc))
cause = np.where(rng.random(N) < h_bc / (h_bc + h_oc), 1, 2)
admin = (2023 - year) * 12 + (12 - month)
loss = rng.exponential(1 / 0.0015, N)
c_time = np.minimum(admin, loss)
dead = t_event <= c_time
months = np.floor(np.where(dead, t_event, c_time)).astype(int)
months = np.maximum(months, 1)
cause = np.where(dead, cause, 0)
unknown_cod = dead & (rng.random(N) < 0.01)
vital = np.where(dead, "Dead", "Alive")
csd = np.select([unknown_cod, cause == 1], ["Dead (missing/unknown COD)", "Dead (attributable to this cancer dx)"],
                "Alive or dead of other cause")
ocd = np.select([unknown_cod, cause == 2], ["Dead (missing/unknown COD)", "Dead (attributable to causes other than this cancer dx)"],
                "Alive or dead due to cancer")
flag = np.where(rng.random(N) < 0.01, "Incomplete dates are available and there cannot be zero days of follow-up",
                "Complete dates are available and there are more than 0 days of survival")
cod_site = np.select([cause == 1, cause == 2], ["Breast", "Diseases of Heart"], "Alive")

# ------------------------------------------------------------------ write
ids = rng.choice(np.arange(10_000_000, 99_999_999), N, replace=False)
band = lambda a: "90+ years" if a >= 90 else f"{a // 5 * 5}-{a // 5 * 5 + 4} years"
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(COLUMNS)
    for i in range(N):
        code = hist[i]
        row = {
            "Age recode with <1 year olds and 90+": band(age[i]),
            "Age recode with single ages and 90+": "90+ years" if age[i] >= 90 else f"{age[i]} years",
            "Patient ID": str(ids[i]),
            "Sequence number": sequence[i],
            "Type of Reporting Source": source[i],
            "Total number of in situ/malignant tumors for patient": "1" if sequence[i] == "One primary only" else "2",
            "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)": race[i],
            "Marital status at diagnosis": marital[i],
            "Year of diagnosis": str(year[i]),
            "Median household income inflation adj to 2024": income[i],
            "Rural-Urban Continuum Code": rural[i],
            "Histologic Type ICD-O-3": code,
            "Histology recode - broad groupings": HIST[code][2],
            "ICD-O-3 Hist/behav, malignant": f"{code}/3: {HIST[code][1]}",
            "Grade Recode (thru 2017)": "Unknown" if grade_unknown[i] else GRADE[grade_n[i]],
            "Laterality": laterality[i],
            "Primary Site": site[i][1:5].replace(".", ""),
            "Primary Site - labeled": site[i],
            "Diagnostic Confirmation": "Positive histology",
            "Tumor Size Over Time Recode (1988+)": size_code[i],
            "Regional nodes examined (1988+)": exam_code[i],
            "Regional nodes positive (1988+)": pos_code[i],
            "LN Positive Axillary Level I-II Recode (2010+)": axillary[i],
            "Derived AJCC Stage Group, 7th ed (2010-2015)": stage_1015[i],
            "Derived AJCC T, 7th ed (2010-2015)": t_1015[i],
            "Derived AJCC N, 7th ed (2010-2015)": n_1015[i],
            "Derived AJCC M, 7th ed (2010-2015)": m_1015[i],
            "7th Edition Stage Group Recode (2016-2017)": stage_1617[i],
            "Derived SEER Cmb Stg Grp (2016-2017)": cmb_1617[i],
            "Derived SEER Combined T (2016-2017)": t_1617[i],
            "Derived SEER Combined N (2016-2017)": n_1617[i],
            "Derived SEER Combined M (2016-2017)": m_1617[i],
            "Derived SEER Combined T Src (2016-2017)": blank if era_1015[i] else "Pathological",
            "Derived SEER Combined N Src (2016-2017)": blank if era_1015[i] else "Pathological",
            "Derived SEER Combined M Src (2016-2017)": blank if era_1015[i] else "Clinical",
            "SEER Combined Mets at DX-bone (2010+)": mets["bone"][i],
            "SEER Combined Mets at DX-brain (2010+)": mets["brain"][i],
            "SEER Combined Mets at DX-liver (2010+)": mets["liver"][i],
            "SEER Combined Mets at DX-lung (2010+)": mets["lung"][i],
            "Breast Subtype (2010+)": subtype[i],
            "ER Status Recode Breast Cancer (1990+)": er[i],
            "PR Status Recode Breast Cancer (1990+)": pr[i],
            "Derived HER2 Recode (2010+)": her2[i],
            "RX Summ--Surg Prim Site (1998-2022)": "Partial mastectomy",
            "RX Summ--Surg/Rad Seq": "Radiation after surgery",
            "Radiation recode": "Beam radiation",
            "Chemotherapy recode (yes, no/unk)": "No/Unknown",
            "RX Summ--Systemic/Sur Seq (2007+)": "Systemic therapy after surgery",
            "Time from diagnosis to treatment in days recode": str(rng.integers(0, 120)),
            "SEER cause-specific death classification": csd[i],
            "SEER other cause of death classification": ocd[i],
            "Survival months": f"{months[i]:04d}",
            "Survival Days": str(int(months[i] * 30.4 + rng.integers(0, 30))),
            "Survival months flag": flag[i],
            "Vital status recode (study cutoff used)": vital[i],
            "Year of follow-up recode": str(min(2023, year[i] + months[i] // 12)),
            "COD to site recode ICD-O-3 2023 Revision Expanded (1999+)": cod_site[i],
        }
        w.writerow([row.get(c, blank) for c in COLUMNS])
print(f"Wrote {N:,} artificial records to {OUT}")
