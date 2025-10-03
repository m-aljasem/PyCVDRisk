"""SMART2: 10-year (or shorter-horizon) risk of recurrent myocardial infarction, stroke or
vascular death in people aged 40-80 with established atherosclerotic disease (CAD,
cerebrovascular disease, PAD and/or abdominal aortic aneurysm), recalibrated to a risk region.
Hageman SHJ et al., Eur Heart J 2022;43:1715-27, doi:10.1093/eurheartj/ehac056.

Source of every number: Supplementary Table 2 of the paper. Risk at horizon t years is
1 - (1 - baseline_t) ** exp(LP - LP_mean - ln(E/O ratio of the region)).
The paper's Table 2 gives SBP per 10 mmHg, so the SBP coefficient multiplies SBP/10. Table 2 gives
no centering constants, and the supplement's mean linear predictor (-0.0463729) does not reproduce
the calculator on raw inputs. U-Prevent's SMART2 results (30 patients, 0.1% display precision) fit
only with an extra +0.2107 on the linear predictor, i.e. the mean is -0.0463729 - 0.2107 when the
antithrombotic offset is in the predictor; 0.2107 = -ln(0.81), see reference/smart2/.
Units: SBP mmHg, non-HDL cholesterol mmol/L, eGFR (CKD-EPI) mL/min/1.73 m2, hsCRP mg/L.
Aspirin or other antithrombotic use enters as a fixed offset (HR 0.81). U-Prevent's further options
(dual antiplatelet, aspirin + low-dose DOAC) are its own treatment-effect layer, not SMART2, and
are not modelled; nor is HbA1c, which U-Prevent asks for but which does not change the risk. A row is NaN when an
input is invalid or no disease location is given.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex

# Supplementary Table 2.
_BETA = dict(aaa=0.330356631, age=-0.03496022, age_sq=0.000551072, antithrombotic=-0.21072103,
             cad=0.294701954, egfr=-0.03967521, egfr_sq=0.000218613, ln_crp=0.151760173,
             smoker=0.345583271, cerebrovascular=0.34831786, diabetes=0.318170659,
             ln_nonhdl=0.540364249, pad=0.22446658, sbp=0.018913154, male=0.287658743,
             years=0.047699585, years_sq=-0.00164973)
_LP_MEAN = -0.0463729 + _BETA["antithrombotic"]  # empirical: see docstring
_BASELINE = np.array([0.020867965, 0.03481669, 0.047574783, 0.06257936, 0.07887031, 0.094891922,
                      0.112217969, 0.129898521, 0.149001544, 0.165822797])  # risk at years 1-10
_EO_RATIO = {"europe_low": 0.81590, "europe_moderate": 0.6973285, "europe_high": 0.5085825,
             "europe_very_high": 0.2285371, "asia": 0.4043255, "australia": 1.0040808,
             "japan": 0.8825678, "north_america": 0.4714961, "latin_america": 0.5075670}
REGION_NAMES = tuple(_EO_RATIO)
_LN_RATIO = np.log(list(_EO_RATIO.values()))


def smart2(age, sex, smoker, diabetes, sbp, non_hdl_chol, egfr, hscrp, years_since_diagnosis,
           cad, cerebrovascular, pad, aaa, *, region, antithrombotic=0, horizon=10) -> np.ndarray:
    """SMART2 risk (fraction) at `horizon` years (whole number, 1-10), one value per person.

    cad, cerebrovascular, pad, aaa: 0/1 disease locations (at least one). region: one of
    REGION_NAMES. antithrombotic: aspirin or equivalent at baseline, 0/1.
    """
    if horizon not in range(1, 11):
        raise ValueError("horizon must be a whole number of years from 1 to 10")
    flags = [as_bool(v) for v in (smoker, diabetes, cad, cerebrovascular, pad, aaa, antithrombotic)]
    a, sx, s, nh, ef, crp, yr, reg, sm, dm, c, cv, p, aa, at = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_float(sbp), as_float(non_hdl_chol), as_float(egfr),
        as_float(hscrp), as_float(years_since_diagnosis), as_category(region, REGION_NAMES), *flags)
    ok = ((a >= 40) & (a <= 80) & (sx >= 0) & (reg >= 0) & (s > 0) & (nh > 0) & (ef > 0)
          & (crp > 0) & (yr >= 0) & np.all([f >= 0 for f in (sm, dm, c, cv, p, aa, at)], axis=0)
          & (c + cv + p + aa >= 1))
    x = lambda v: np.where(ok, v, 1.0)  # keeps log() quiet on rows that end up NaN
    b = _BETA
    lp = (b["age"] * x(a) + b["age_sq"] * x(a) ** 2 + b["male"] * (sx == 1) + b["smoker"] * np.where(ok, sm, 0)
          + b["diabetes"] * np.where(ok, dm, 0) + b["sbp"] * x(s) / 10 + b["ln_nonhdl"] * np.log(x(nh))
          + b["egfr"] * x(ef) + b["egfr_sq"] * x(ef) ** 2 + b["ln_crp"] * np.log(x(crp))
          + b["years"] * x(yr) + b["years_sq"] * x(yr) ** 2 + b["cad"] * np.where(ok, c, 0)
          + b["cerebrovascular"] * np.where(ok, cv, 0) + b["pad"] * np.where(ok, p, 0)
          + b["aaa"] * np.where(ok, aa, 0) + b["antithrombotic"] * np.where(ok, at, 0))
    out = 1 - (1 - _BASELINE[horizon - 1]) ** np.exp(lp - _LP_MEAN - _LN_RATIO[np.where(ok, reg, 0)])
    return np.where(ok, out, np.nan)
