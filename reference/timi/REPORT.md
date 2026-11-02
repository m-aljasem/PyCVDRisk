# TIMI risk score for UA/NSTEMI: verification report

Generated 2026-11-01 by `reference/timi/report.py`.

## 1. Source (evidence level C)

Antman et al., JAMA 2000 (`source/antman2000.pdf`): the 7 predictors that stayed significant in
the multivariate model (Table 1, bold) score 1 point each, because their coefficients were of
similar size. Risk factors for CAD (Table 1 footnote): family history of CAD, hypertension,
hypercholesterolaemia, diabetes, current smoking.

| Rule | Score | Expected |
|---|---:|---:|
| age 64 | 0 | 0 |
| age 65 | 1 | 1 |
| 2 risk factors | 0 | 0 |
| 3 risk factors | 1 | 1 |
| coronary stenosis >= 50% | 1 | 1 |
| ST deviation | 1 | 1 |
| 1 anginal event in 24 h | 0 | 0 |
| 2 anginal events in 24 h | 1 | 1 |
| aspirin in last 7 days | 1 | 1 |
| elevated cardiac markers | 1 | 1 |
| all seven | 7 | 7 |

11/11 rules give the published points.

## 2. RiskScorescvd (evidence level B)

- reference: RiskScorescvd (CRAN), TIMI() - vendored source, see reference/vendor/RiskScorescvd/DESCRIPTION
- package_version: 0.3.1
- R: R version 4.3.3 (2024-02-29)
- dplyr: 1.1.4
- generated: 2026-11-01
- rounding: none (integer points)

RiskScorescvd stands in previous PCI or CABG for known coronary stenosis, and hs-troponin I
above fixed sex-specific limits (34 and 16 ng/L) for elevated markers. `make_patients.py` draws both
forms consistently, with 30% of ages and troponins on or next to a cut point.

5000 random patients (seed 2000): 5000/5000 identical scores.
