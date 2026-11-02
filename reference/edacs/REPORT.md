# EDACS: verification report

Generated 2026-11-01 by `reference/edacs/report.py`.

## 1. Source (evidence level C)

Than et al., Emerg Med Australas 2014, Box 3 (`source/than2014.pdf`). Every age band is checked at
both ends; other items are checked on a woman aged 30 (2 points).

| Rule | Score | Expected |
|---|---:|---:|
| age 18 | 2 | 2 |
| age 45 | 2 | 2 |
| age 46 | 4 | 4 |
| age 50 | 4 | 4 |
| age 51 | 6 | 6 |
| age 55 | 6 | 6 |
| age 56 | 8 | 8 |
| age 60 | 8 | 8 |
| age 61 | 10 | 10 |
| age 65 | 10 | 10 |
| age 66 | 12 | 12 |
| age 70 | 12 | 12 |
| age 71 | 14 | 14 |
| age 75 | 14 | 14 |
| age 76 | 16 | 16 |
| age 80 | 16 | 16 |
| age 81 | 18 | 18 |
| age 85 | 18 | 18 |
| age 86 | 20 | 20 |
| age 110 | 20 | 20 |
| male | 8 | 8 |
| age 50, known CAD | 8 | 8 |
| age 50, 3 risk factors | 8 | 8 |
| age 50, 2 risk factors | 4 | 4 |
| age 51, known CAD | 6 | 6 |
| diaphoresis | 5 | 5 |
| radiates to arm or shoulder | 7 | 7 |
| pain with inspiration | -2 | -2 |
| pain on palpation | -4 | -4 |

29/29 rules give the published points.

## 2. RiskScorescvd (evidence level B)

- reference: RiskScorescvd (CRAN), EDACS() - vendored source, see reference/vendor/RiskScorescvd/DESCRIPTION
- package_version: 0.3.1
- R: R version 4.3.3 (2024-02-29)
- dplyr: 1.1.4
- generated: 2026-11-01
- rounding: none (integer points)

RiskScorescvd has no known-CAD input: its +4 applies to >= 3 risk factors only. The fixtures
therefore set known CAD to 0; the CAD rule is covered in section 1. 40% of ages are band edges.

5000 random patients aged 18-99 (seed 2014): 5000/5000 identical scores.

## 3. Differences by design

- Ages are taken in completed years (50.5 counts as 50); RiskScorescvd mixes `Age < 51` and `Age <= 50`.
- Ages under 18 give NaN (the score is for adults); RiskScorescvd gives 0 age points.
- The EDACS-ADP (score < 16, no new ischaemia, negative 0 and 2 h troponins) is left to the caller.
