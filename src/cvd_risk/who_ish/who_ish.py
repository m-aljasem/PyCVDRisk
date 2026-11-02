"""WHO/ISH risk prediction charts (WHO 2007): 10-year risk band of a fatal or non-fatal
myocardial infarction or stroke, for the 14 WHO epidemiological subregions.

The charts are tables, not equations: the guideline gives no formula, only the coloured cells
(Annex 5: values tabulated from a simulated cohort). The function returns the band index:
0 "<10%", 1 "10% to <20%", 2 "20% to <30%", 3 "30% to <40%", 4 ">=40%" (see BANDS),
as float64 so that invalid rows are NaN.

With total_chol (mmol/L) the cholesterol charts are used; without it, the charts for settings
where cholesterol cannot be measured.

Cell selection. Age: the guideline's rule, "if age is 50-59 select 50, 60-69 select 60 etc"
(p. 17); the 40 chart covers 40-49 and the 70 chart 70-79 (Annex 5 simulated ages up to 79).
SBP and cholesterol: the guideline gives no rule; we follow whoishRisk, the R implementation by
Collins et al.: SBP <140 -> 120, 140-159 -> 140, 160-179 -> 160, >=180 -> 180; cholesterol
<4.5 -> 4, 4.5-5.4 -> 5, ..., >=7.5 -> 8.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex

BANDS = ("<10%", "10% to <20%", "20% to <30%", "30% to <40%", ">=40%")
REGIONS = ("afr_d", "afr_e", "amr_a", "amr_b", "amr_d", "emr_b", "emr_d",
           "eur_a", "eur_b", "eur_c", "sear_b", "sear_d", "wpr_a", "wpr_b")
SOURCE = ("World Health Organization. Prevention of cardiovascular disease: guidelines for assessment "
          "and management of cardiovascular risk. Geneva: WHO; 2007 (Annexes 3-5); cell values for the "
          "12 subregions not printed there from whoishRisk 1.0 (Collins DRJ et al., GPL-3)")


@cache
def _table():
    """band[region, diabetes, male, smoker, age 40/50/60/70, sbp 120/140/160/180, chol none/4..8]."""
    raw = np.loadtxt(Path(__file__).with_name("charts.csv"), delimiter=",", skiprows=1, dtype=np.int64)
    t = np.zeros((len(REGIONS), 2, 2, 2, 4, 4, 6))
    dm, male, sm, age, sbp, chol = raw[:, :6].T
    c = np.where(chol == 0, 0, chol - 3)
    t[:, dm, male, sm, (age - 40) // 10, (sbp - 120) // 20, c] = raw[:, 6:].T
    return t


def who_ish(age, sex, smoker, sbp, diabetes, region, total_chol=None) -> np.ndarray:
    """WHO/ISH 2007 chart band (index into BANDS). Source: SOURCE."""
    chol = np.float64(np.nan) if total_chol is None else as_float(total_chol)
    a, sx, sm, sb, dm, ri, ch = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_bool(smoker), as_float(sbp), as_bool(diabetes),
        as_category(region, REGIONS), chol)
    ai = np.floor((a - 40) / 10)
    si = np.clip(np.floor((sb - 120) / 20), 0, 3)
    ci = np.clip(np.floor(ch - 3.5), 0, 4) + 1 if total_chol is not None else np.zeros(a.shape)
    ok = ((ai >= 0) & (ai <= 3) & (sb > 0) & (sx >= 0) & (ri >= 0) & ~np.isnan(sm + dm)
          & ((ch > 0) if total_chol is not None else True))
    idx = [np.where(ok, v, 0).astype(np.int64) for v in (ri, dm, sx, sm, ai, si, ci)]
    return np.where(ok, _table()[tuple(idx)], np.nan)
