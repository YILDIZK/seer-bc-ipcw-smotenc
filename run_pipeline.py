"""Run the analysis steps in order (README.md).

Notebooks are executed headless with the Python of this environment; the executed copies, which contain outputs, are written
to WORK_DIR/executed_notebooks, so that the notebooks in this folder stay free of outputs and therefore of SEER data.

Usage (from this folder):
    python run_pipeline.py                 all steps
    python run_pipeline.py 03 04           only the named steps (by their two-digit prefix, or "fig1", "fig2")
Paths are taken from config.py (or the environment variables SEER_EXTRACT and SEER_WORK_DIR).
"""
import subprocess
import sys
import time
from pathlib import Path

import config

HERE = Path(__file__).resolve().parent
STEPS = ["01_preprocess.ipynb", "02_model_prep.ipynb", "03_modeling.ipynb", "04_evaluation.ipynb",
         "05_recalibration.py", "06_bootstrap_control.py", "07_descriptive_checks.py", "08_jdsa_figures.py",
         "09_smotenc_crosscheck.py", "10_smotenc_tiebreak_check.py", "11_decile_ipcw_check.py",
         "12_check_10y_2010_2013.py", "fig1_flowchart.py",
         "fig2_workflow.py"]


def run(step: str) -> None:
    if step.endswith(".ipynb"):
        out = config.WORK_DIR / "executed_notebooks"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", step,
               "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3", "--output-dir", str(out)]
    else:
        cmd = [sys.executable, step]
    t0 = time.time()
    print(f"=== {step}", flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)
    print(f"=== {step} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    wanted = sys.argv[1:]
    steps = [s for s in STEPS if not wanted or any(s.startswith(w) for w in wanted)]
    print(f"SEER extract: {config.SEER_EXTRACT}\nWorking folder: {config.WORK_DIR}", flush=True)
    for s in steps:
        run(s)
