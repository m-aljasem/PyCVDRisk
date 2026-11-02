# HEART score: verification report

Generated 2026-11-01 by `reference/heart/report.py`.

## 1. Source (evidence level C: the published rules)

Six et al. (Neth Heart J 2008) define the score; their paper is not in the project library. The point
table and its operational definitions are taken from Melki & Jernberg (Crit Pathw Cardiol 2013,
Table 1 and Methods; `source/melki2013.pdf`), which cites www.heartscore.nl. Table 1 reads
"<=normal limit / 1-3x / >=3x" for troponin and "<=45 / 45-65 / >=65" for age, which overlap at
the cut points; the Methods resolve both ("younger than 45" 0, "45-64" 1, "65 years or older" 2;
"below the local decision limit" 0, "between 1 and 3 times" 1, "higher than 3 times" 2).
These are the rules implemented.

| Rule | Points | Expected |
|---|---:|---:|
| history slightly suspicious | 0 | 0 |
| history moderately suspicious | 1 | 1 |
| history highly suspicious | 2 | 2 |
| ECG normal | 0 | 0 |
| ECG non-specific repolarisation | 1 | 1 |
| ECG significant ST depression | 2 | 2 |
| age 44 (<45) | 0 | 0 |
| age 45 | 1 | 1 |
| age 64 | 1 | 1 |
| age 65 | 2 | 2 |
| 1 risk factor | 1 | 1 |
| 2 risk factors | 1 | 1 |
| 3 risk factors | 2 | 2 |
| atherosclerosis, no risk factors | 2 | 2 |
| troponin below the limit | 0 | 0 |
| troponin at the limit | 1 | 1 |
| troponin 3x the limit | 1 | 1 |
| troponin above 3x | 2 | 2 |

18/18 rules give the published points.

## 2. RiskScorescvd (evidence level B)

- reference: RiskScorescvd (CRAN), HEART() - vendored source, see reference/vendor/RiskScorescvd/DESCRIPTION
- package_version: 0.3.1
- R: R version 4.3.3 (2024-02-29)
- dplyr: 1.1.4
- generated: 2026-11-01
- rounding: round(score, 0) on integer points, i.e. none

RiskScorescvd grades history from a count of typical symptoms (0-1, 2-4, 5-6) and troponin from
hs-troponin I against fixed sex-specific limits (34 and 16 ng/L). `make_patients.py` therefore
draws each patient in both forms: our history grade and a symptom count in the matching range,
and our troponin ratio equal to its hs-troponin I divided by its limit. 30% of troponin values
and 20% of ages sit on or next to a cut point.

5000 random patients (seed 2008): 5000/5000 identical scores.

## 3. Differences by design

- Ages are not restricted to whole years: 64.5 scores 1 here, while RiskScorescvd's `45 <= Age <= 64`
  gives NA.
- RiskScorescvd counts five risk factors; Melki also counts obesity and angina pectoris. Here the
  caller passes the count.
- RiskScorescvd scores an ECG given as all-zero flags as 1 point; here the ECG is the clinician's
  0-2 grade.
