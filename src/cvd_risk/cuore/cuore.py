"""CUORE Project individual score (Italy): 10-year risk of a first major cardiovascular event (fatal or non-fatal
myocardial infarction or stroke), ages 35-69, no previous cardiovascular event.

risk = 1 - S0 ** exp(sum(beta * x) - G), sex-specific Cox models with uncentred inputs; coefficients, G (the linear
predictor at the risk-factor means) and the 10-year S0 as printed in Giampaoli S, Palmieri L, et al. Ital J Public
Health 2007;4(2):102-9, Table 2 (three decimals). The ISS online calculator uses these values
(reference/cuore/REPORT.md).

Inputs outside the ranges the ISS states for the score (SBP 90-200 mmHg, total cholesterol 130-320 mg/dL,
HDL 20-100 mg/dL) give NaN, as do ages outside 35-69.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = ("Giampaoli S, Palmieri L, Donfrancesco C, et al. Cardiovascular risk assessment in Italy: the CUORE "
          "Project risk score and risk chart. Ital J Public Health 2007;4(2):102-9; Palmieri L et al. Ann Ist Super "
          "Sanita 2004;40(4):393-9; www.cuore.iss.it")

# age, SBP, TC, HDL (mg/dL), smoker, diabetes, BP treatment; G; S0 (10-year)
COEF = {"male": ((0.076, 0.013, 0.006, -0.013, 0.508, 0.462, 0.490), 6.583, 0.953),
        "female": ((0.079, 0.016, 0.003, -0.015, 0.773, 0.339, 0.590), 6.016, 0.989)}


def _within(x, lo, hi):
    x = as_float(x)
    return np.where((x >= lo) & (x <= hi), x, np.nan)


def cuore(age, sex, sbp, tc, hdl, smoker, diabetes, bp_treatment) -> np.ndarray:
    """10-year risk (fraction). sbp mmHg (the calculator asks for two readings and takes their mean);
    tc, hdl mg/dL; smoker = current cigarette smoker; flags 0/1."""
    x = (_within(age, 35, 69), _within(sbp, 90, 200), _within(tc, 130, 320), _within(hdl, 20, 100),
         as_bool(smoker), as_bool(diabetes), as_bool(bp_treatment))
    sx = as_sex(sex)
    out = np.nan
    for code, name in ((1, "male"), (0, "female")):
        b, g, s0 = COEF[name]
        out = np.where(sx == code, 1 - s0 ** np.exp(sum(bi * xi for bi, xi in zip(b, x)) - g), out)
    return out
