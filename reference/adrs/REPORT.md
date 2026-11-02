# Abu Dhabi Risk Study (ADRS): verification report

Generated 2026-11-01 by `reference/adrs/report.py`.

Evidence level C, and weak: the paper (J Am Heart Assoc 2024;13:e035930, open access, PMC11681588) gives the
coefficients (Table 2) and the supplement (Data S1, `source/adrs_data_s1.pdf`) gives SPSS syntax with centring
values and baseline survival, but there is no worked example, no calculator and no independent implementation.

## 1. Transcription

| Model | Term | Coefficient | Centring | In Data S1 syntax | Equals Table 2 B |
|---|---|---:|---:|---|---|
| ascvd | vitamin_d | -0.007 | 35.53 | yes | yes |
| ascvd | dev_gfr | -0.029 | 0 | yes | yes |
| ascvd | male | 1.516 | 0.51 | yes | yes |
| ascvd | smoker | 0.611 | 0.09 | yes | yes |
| ascvd | age | 0.204 | 38.6 | yes | yes |
| ascvd | bp_treated | 0.305 | 0.13 | yes | yes |
| ascvd | tc_hdl | 0.068 | 4.1 | yes | yes |
| ascvd | diabetes | 0.332 | 0.21 | yes | yes |
| ascvd | dev_gfr_x_male | 0.031 | -1.28 | yes | yes |
| ascvd | map | 0.012 | 89.32 | yes | yes |
| ascvd | age2 | -0.001 | 1749.5 | yes | yes |
| cad | vitamin_d | -0.007 | 35.53 | yes | yes |
| cad | dev_gfr | -0.039 | 0 | yes | yes |
| cad | male | 2.136 | 0.51 | yes | yes |
| cad | smoker | 0.55 | 0.09 | yes | yes |
| cad | age | 0.067 | 38.6 | yes | yes |
| cad | hypertension | 0.327 | 0.18 | yes | yes |
| cad | tc_hdl | 0.067 | 4.1 | yes | yes |
| cad | diabetes | 1.543 | 0.21 | yes | yes |
| cad | dev_gfr_x_male | 0.027 | -1.28 | yes | yes |
| cad | diabetes_x_male | -0.669 | 0.12 | yes | yes |
| cad | diabetes_x_dev_gfr | 0.019 | -2.72 | yes | yes |
| cad | bp_normal | -1.118 | 0.46 | yes | yes |
| cad | bp_elevated | -1.148 | 0.15 | yes | yes |
| cad | bp_stage1 | -1.378 | 0.25 | yes | yes |
| cad | bp_stage2 | -1.093 | 0.14 | yes | yes |
| cad | bp_crisis | -0.698 | 0 | yes | yes |
| stroke | smoker | 1.053 | - | (see 3) | yes |
| stroke | sbp | 0.024 | - | (see 3) | yes |
| stroke | age | 0.084 | - | (see 3) | yes |
| stroke | hypertension_x_male | -0.646 | - | (see 3) | yes |
| stroke | age_x_male | 0.011 | - | (see 3) | yes |

32/32 terms confirmed in both sources (stroke: Table 2 only).
Baseline survival 0.994 (ASCVD, CAD) and 0.998 (stroke) found in Data S1: yes;
stroke mean score 6.37 found: yes.

## 2. Values the paper does not state

- **Cohort GFR mean** (the deviation is entered, the mean is not printed): Table 1 gives GFR and deviation by sex;
  GFR - deviation = 111.4 (women) and 111.4 (men). The package uses 111.4.
  Consistency: the Data S1 centring of deviation x sex, -1.28, against 0.51 x (-2.5) = -1.275.
- **Mean blood pressure** = DBP + (SBP - DBP)/3: Table 1 means give 85.8 (women; printed 85.8)
  and 92.6 (men; printed 92.7). Their sex-weighted mean, 89.32, against the Data S1 centring
  89.32 (the cohort used for the centring is not stated exactly).
- **CAD blood-pressure stages**: Table 2 names the ACC/AHA 2017 stages; crisis taken as SBP > 180 and/or DBP > 120.
  All five stages have a coefficient and their centring values sum to 1.00, so with no sixth (reference)
  category the five indicators are collinear in the fitted model. The printed formula is still well defined and
  is used as printed.

## 3. Stroke: two versions in Data S1

Data S1 computes `sumrfFSTRK` twice; the second assignment (which SPSS would use) drops the hypertension x sex
term that Table 2 lists. The mean score subtracted, 6.37, should equal the score at the cohort means. From Table 1
(sex-weighted, male share 0.51): 6.38 with the term, 6.45 without it. The printed
6.37 fits the version with the term, which the package therefore uses.

A smaller slip: the ASCVD block stores exp(score) in `ASCVD_SUMRF_` but raises 0.994 to `ASCVD_SUMRF` (no
trailing underscore); the intent, 1 - S0 ** exp(score), is clear and is what the package computes.

## 4. Owner action

Ask the authors to confirm the cohort GFR mean (111.4?), the stroke formula, and to provide a few calculated
risks; with those the evidence would rise above weak C.

