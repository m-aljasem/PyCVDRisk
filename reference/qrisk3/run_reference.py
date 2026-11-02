"""Reference answers for QRISK3 from ClinRisk's own C code (source/qrisk3_2017.c), compiled
unmodified with harness.c. Run from repo root: python reference/qrisk3/run_reference.py
Needs a C compiler (cc). Writes fixtures/reference.csv column `clinrisk_c` and fixtures/VERSION.
"""

import datetime as dt
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.qrisk3 import DIABETES, ETHNICITY, SMOKING

HERE = Path(__file__).parent
FX = HERE / "fixtures"
C_FLAGS = ("af", "atypical_antipsychotics", "corticosteroids", "erectile_dysfunction", "migraine",
           "rheumatoid_arthritis", "ckd", "severe_mental_illness", "sle", "bp_treated")


def c_rows(p: pd.DataFrame) -> str:
    """The harness's input order; categories to ClinRisk codes, age truncated to int."""
    c = pd.DataFrame({"male": (p.sex == "male").astype(int), "age": np.floor(p.age).astype(int),
                      **{f: p[f] for f in C_FLAGS},
                      "type1": (p.diabetes == "type1").astype(int), "type2": (p.diabetes == "type2").astype(int),
                      "bmi": p.bmi.map(repr), "ethrisk": p.ethnicity.map(ETHNICITY.index) + 1,
                      "fh_cvd": p.family_history, "rati": p.chol_hdl_ratio.map(repr), "sbp": p.sbp.map(repr),
                      "sbps5": p.sbp_sd.map(repr), "smoke_cat": p.smoking.map(SMOKING.index),
                      "town": p.townsend.map(repr)})
    assert set(p.diabetes) <= set(DIABETES)
    return c.to_csv(header=False, index=False)


def run_c(p: pd.DataFrame) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "qrisk3"
        subprocess.run(["cc", "-O0", "-std=c99", "-o", exe, HERE / "harness.c", "-lm"], check=True)
        out = subprocess.run([exe], input=c_rows(p), capture_output=True, text=True, check=True).stdout
    return np.array(out.split(), dtype=float) / 100


def main():
    p = pd.read_csv(FX / "patients.csv")
    ref = run_c(p)
    assert len(ref) == len(p)
    pd.DataFrame({"id": p.id, "clinrisk_c": [f"{v:.17g}" for v in ref]}).to_csv(
        FX / "reference.csv", index=False)
    cc = subprocess.run(["cc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    (FX / "VERSION").write_text(
        "reference: ClinRisk QRISK3-2017 C source (reference/qrisk3/source/qrisk3_2017.c), unmodified\n"
        f"compiler: {cc} (-O0 -std=c99)\ngenerated: {dt.date.today()}\n")


if __name__ == "__main__":
    main()
