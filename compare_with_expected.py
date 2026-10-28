"""Compare a complete run (WORK_DIR) with the reference results of the article (expected_results/).

Checks:
1. The SHA-256 checksums of the model-preparation files (prep_manifest.json): identical files mean an identical extract,
   cohort, labels and data partition.
2. Every reference table: the same rows and columns, and the largest absolute difference of each numeric column.
   The fitting times (INFO_ONLY) differ between runs and computers; they are shown for information and not counted.

Usage (from this folder, after run_pipeline.py):  python compare_with_expected.py [tolerance]   (default 1e-9)
Prints one line per table and a summary; exit code 1 if any table differs by more than the tolerance.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

EXPECTED = Path(__file__).resolve().parent / "expected_results"
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-9
INFO_ONLY = {"outputs_descriptive/compute_times_by_cell.csv", "outputs_descriptive/compute_times_per_split.csv"}


def compare_manifest() -> bool:
    exp = json.loads((EXPECTED / "outputs_model_prep" / "prep_manifest.json").read_text(encoding="utf-8"))
    got_path = config.WORK_DIR / "outputs_model_prep" / "prep_manifest.json"
    if not got_path.exists():
        print("prep_manifest.json: not found in the working folder")
        return False
    got = json.loads(got_path.read_text(encoding="utf-8"))
    ok = True
    for name, digest in {**exp["input"], **exp["sha256"]}.items():
        same = {**got["input"], **got["sha256"]}.get(name) == digest
        ok &= same
        print(f"checksum {name:<28} {'identical' if same else 'DIFFERENT'}")
    return ok


def compare_table(exp_file: Path) -> tuple[bool, str]:
    got_file = config.WORK_DIR / exp_file.relative_to(EXPECTED)
    if not got_file.exists():
        return False, "missing in the working folder"
    a, b = pd.read_csv(exp_file), pd.read_csv(got_file)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, f"different shape or columns ({a.shape} vs {b.shape})"
    worst, col_worst = 0.0, ""
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                return False, f"missing values differ in {c}"
            d = float(np.nanmax(np.abs(x - y))) if np.isfinite(x).any() else 0.0
            if d > worst:
                worst, col_worst = d, c
        elif not a[c].astype(str).equals(b[c].astype(str)):
            return False, f"text column {c} differs"
    return worst <= TOL, f"largest difference {worst:.2e}" + (f" ({col_worst})" if col_worst else "")


if __name__ == "__main__":
    print(f"Working folder: {config.WORK_DIR}\nTolerance: {TOL:g}\n")
    checksums_ok = compare_manifest()
    print()
    n_ok = n = 0
    for f in sorted(EXPECTED.rglob("*.csv")):
        ok, msg = compare_table(f)
        name = f.relative_to(EXPECTED).as_posix()
        if name in INFO_ONLY:
            print(f"info {name:<75} {msg} (fitting times, not compared)")
            continue
        n += 1
        n_ok += ok
        print(f"{'ok  ' if ok else 'DIFF'} {name:<75} {msg}")
    print(f"\nChecksums: {'all identical' if checksums_ok else 'NOT all identical'}; tables: {n_ok} of {n} within tolerance"
          f" (plus {len(INFO_ONLY)} timing files for information)")
    sys.exit(0 if checksums_ok and n_ok == n else 1)
