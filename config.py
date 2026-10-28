"""Paths used by every notebook and script. Edit the two defaults below, or set the environment variables
SEER_EXTRACT and SEER_WORK_DIR, before running the pipeline (see README.md)."""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent

# SEER*Stat case listing exported as described in seer_stat/Session_information.txt (not distributed with this code)
SEER_EXTRACT = Path(os.environ.get("SEER_EXTRACT", REPO_DIR / "data" / "breast_2010-2017_competing_risk_case_listing.csv"))

# Every output (intermediate files, tables, figures) is written below this folder
WORK_DIR = Path(os.environ.get("SEER_WORK_DIR", REPO_DIR / "work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
