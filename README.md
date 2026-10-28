# Censoring adjustment and synthetic oversampling in fixed-horizon risk prediction

Code for the article *Censoring adjustment and synthetic oversampling in fixed-horizon risk prediction: a factorial study using SEER breast cancer data* (Yıldız Karadayı, submitted to the International Journal of Data Science and Analytics).

The study crosses three ways of handling censored patients (labeling them as survivors, discarding them, inverse probability of censoring weighting) with two ways of handling class imbalance (none, SMOTE-NC to 1:1). Each combination is used for three classifiers (logistic regression, random forest, XGBoost), with cause-specific Cox and random survival forest models as references. The models predict the cumulative incidence of breast cancer death at 3, 5 and 10 years, with other-cause death as a competing event, in 257,462 women from the SEER program. All patients enter 3 × 5-fold cross-validation.

**No data are included.** The analysis uses the SEER Research Data, which are available to researchers who accept the SEER Research Data Use Agreement. This repository contains code, extraction instructions, aggregated reference results and a synthetic example that runs without SEER access.

## Contents

| Path | Content |
|---|---|
| `01_preprocess.ipynb` | Cohort selection (exclusions in Online Resource 1, Table S1), recoding of SEER fields, staging harmonization of the two staging eras, horizon-specific labels |
| `02_model_prep.ipynb` | The 28 predictors and their levels (reference levels, merges), the data partition (3 repeats × 5 folds, inner validation set), design-matrix specification, SHA-256 checksums |
| `03_modeling.ipynb` | Main analysis: censoring model and weights, the six factorial cells × three classifiers × three horizons, reference models (`seer_modeling.run_split`) |
| `04_evaluation.ipynb` | Evaluation on the test folds, research-question contrasts, supplementary analyses (i), (ii) and (iv), figures in the original style |
| `05_recalibration.py` | Post hoc recalibration, analysis (v), and the exploratory comparison of the censoring strategies after recalibration |
| `06_bootstrap_control.py` | Weighted-bootstrap control, analysis (vi) |
| `07_descriptive_checks.py` | Common threshold rule, analysis (vii); follow-up by year of diagnosis; computation times |
| `08_jdsa_figures.py` | Figs. 3 and 4 and Online Resource Figs. S1–S3 |
| `09_smotenc_crosscheck.py`, `10_smotenc_tiebreak_check.py` | Check of the SMOTE-NC implementation against imbalanced-learn, analysis (viii) |
| `11_decile_ipcw_check.py` | Decile-level check of the calibration estimates (Aalen–Johansen versus IPCW), analysis (ix) |
| `12_check_10y_2010_2013.py` | 10-year check in the test patients diagnosed in 2010–2013, analysis (iii) |
| `fig1_flowchart.py`, `fig2_workflow.py` | Figs. 1 and 2 (Fig. 1 reads the published counts in `figure_data/cohort_cascade.csv`) |
| `seer_modeling.py` | All modeling functions: encoder, censoring model and weights, cell construction, SMOTE-NC, classifiers, reference models, evaluation, corrected *t*-test |
| `config.py` | The two paths to set: the SEER extract and the working folder |
| `run_pipeline.py` | Runs the steps in order |
| `compare_with_expected.py` | Compares a run with the reference results |
| `seer_stat/` | SEER\*Stat session description and export dictionary (extraction instructions) |
| `synthetic/` | Generator of an artificial SEER-like extract |
| `expected_results/` | Aggregated results of the article (no patient-level data), for comparison |
| `environment.yml` | Software versions |

The section numbers and analyses (i)–(ix) refer to the article and its Online Resource 1.

## 1. Obtain the data (SEER access needed)

1. **Request access.** Request the SEER Research Data through the SEER data request system and accept the SEER Research Data Use Agreement (https://seer.cancer.gov/data/access.html). Install SEER\*Stat (the analysis used version 9.0.43.0).
2. **Create the case listing.** In SEER\*Stat, create a case-listing session on the database "Incidence – SEER Research Data, 17 Registries, Nov 2025 Sub (2000–2023) – Linked To County Attributes – Time Dependent (1990–2024) Income/Rurality, 1969–2024 Counties". Use the selection and the 65 variables listed in `seer_stat/Session_information.txt`. In short:
   - women, aged 20 years or older;
   - diagnosed in 2010–2017;
   - breast (sites C50.0–C50.9), malignant behavior;
   - microscopically confirmed;
   - sequence number "One primary only" or "1st of 2 or more primaries".
3. **Export.** Export the listing as described in `seer_stat/export.csv.dic`: labels (not codes), comma-delimited, fields with delimiters in quotes, variable names included, no thousands separators. The export has 392,728 records.
4. **Point the code to it.** Save the file as `data/breast_2010-2017_competing_risk_case_listing.csv`, or set the path in `config.py` or with the environment variable `SEER_EXTRACT`.

The SEER Research Data Use Agreement does not allow the data, or information about individuals, to be shared. Keep `data/` and the working folder private; both are excluded in `.gitignore`.

## 2. Install

```
conda env create -f environment.yml
conda activate seer-bc-ipcw-smotenc
```

The analysis was run on a desktop computer with an Intel Core i9-12900K processor (16 cores, 24 logical processors), 64 GB of memory and Windows 11 Pro, with Python 3.12 and the versions in `environment.yml`.

## 3. Run

```
python run_pipeline.py
```

**What the run does:**
- Runs the steps in order and writes every output below the working folder: `work/` by default, or the folder set in `config.py` or with the environment variable `SEER_WORK_DIR`.
- Runs the notebooks headless. Their executed copies, which contain outputs and therefore must not be shared, go to `work/executed_notebooks/`; the notebooks in this repository contain no outputs.
- Single steps can be run by prefix, e.g. `python run_pipeline.py 05 06`.

| Step | Time on the analysis computer |
|---|---|
| 01–02 | 1 min |
| 03 (main analysis, 15 splits) | 2.9 h |
| 04 | 3 min |
| 05 (recalibration) | 1 min |
| 06 (bootstrap control) | 17 min |
| 07 | < 1 min |
| 08, fig1, fig2 | 1 min |
| 09 (implementation check) | 54 min |
| 10 (tie-rule follow-up) | 22 min |
| 11 (decile-level calibration check) | < 1 min |
| 12 (2010–2013 check) | < 1 min |

The complete run took about 4.6 hours (re-run of 28 October 2026). Fitting times vary between runs: the main analysis of the article took 3.2 hours.

**Determinism.** All random seeds are fixed in the code:
- seed 42 in `seer_modeling.py` and the notebooks;
- the data partition seeds in `02_model_prep.ipynb`;
- one random stream per split, horizon and cell.

The data partition is saved with a checksum that is verified before every analysis.

## 4. Check a run against the article

```
python compare_with_expected.py
```

The comparison checks two things:
- **Checksums:** the SHA-256 checksums of the model-preparation files (`expected_results/outputs_model_prep/prep_manifest.json`). Identical checksums mean an identical extract, cohort, labels and partition.
- **Tables:** every table in `expected_results/`, column by column. The two files of fitting times (`compute_times_*.csv`) are shown for information only, because times vary between runs.

A complete re-run from the SEER extract on 28 October 2026 reproduced all checksums and all 44 reference tables exactly (`expected_results/validation_run_2026-10-28.txt`).

## 5. Synthetic example (no SEER access needed)

```
python synthetic/make_synthetic_extract.py
set SEER_EXTRACT=synthetic\synthetic_case_listing.csv
set SEER_WORK_DIR=work_synthetic
python run_pipeline.py
```

(On macOS or Linux, use `export` instead of `set`, and forward slashes.)

**What the synthetic data are:**
- **Artificial records:** the generator writes 16,000 artificial records with the columns and label formats of the SEER export. No SEER record, count or frequency is used.
- **Simulated survival:** survival follows a simple simulation with administrative censoring at the end of 2023 and some loss to follow-up, so every step of the pipeline has work to do.
- **Not the article's results:** the results have no clinical meaning and do not reproduce the article.

The complete synthetic run took about 8 minutes on the computer used for the article.

Step 08 also redraws the original-style figures and compares them pixel by pixel with those drawn by step 04. With the article's data they are identical. With other data, such as the synthetic example, the 10-year calibration figure can differ, because step 04 lets matplotlib choose the tick spacing, whereas step 08 fixes it at 0.1. A difference there does not indicate an error.

## 6. Where the mappings are

| Mapping | Location |
|---|---|
| Recoding of SEER special codes (tumor size, node counts) | `01_preprocess.ipynb`, function `add_raw_model_features` |
| Staging harmonization of the 2010–2015 and 2016–2017 eras | `01_preprocess.ipynb`, dictionaries `STAGE_MAP_ROMAN`, `STAGE_MAP_CMB`, `T_MAP_*`, `N_MAP_*`, `M_MAP_*` |
| The 28 predictors, reference levels and level merges | `02_model_prep.ipynb`, `NUMERIC_VARS`, `CATEGORICAL_SPEC`, `HISTOLOGY_KEEP` |
| Design matrix (log transformation, imputation, scaling, one-hot encoding) | `02_model_prep.ipynb`, `build_transformer` |
| Censoring weights, factorial cells, SMOTE-NC, models, evaluation | `seer_modeling.py` |

## 7. Data protection

The repository follows the SEER Research Data Use Agreement:
- **No patient data:** no patient-level data and no notebook outputs are included.
- **Small counts:** no aggregated count below 5 is included.
- **No model parameters:** no fitted models or model parameters are included.

Runs write patient-level files only to the working folder.

## Citation and license

- **Citation:** please cite the article and this code (`CITATION.cff`). Repository: https://github.com/YILDIZK/seer-bc-ipcw-smotenc
- **License:** the code is released under the MIT License; the documentation and aggregated results under CC BY 4.0 (`LICENSE`).
- **Contact:** Yıldız Karadayı, yildizkaradayi@topkapi.edu.tr.
