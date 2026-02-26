# Suita CVD risk model (2020): verification report

Evidence level C (+ partial B): the paper's own coefficients and tables, and the R package Jcvrisk (an independent
implementation; no calculator found).

## Source

Nakai M, Watanabe M, Kokubo Y, Nishimura K, Higashiyama A, Takegami M, Nakao YM, Okamura T, Miyamoto Y. Development
of a cardiovascular disease risk prediction model using the Suita study, a population-based prospective cohort study
in Japan. J Atheroscler Thromb 2020;27(11):1160-75. doi:10.5551/jat.48843 (`source/nakai2020.pdf`, J-STAGE advance
publication, CC BY-NC-SA; sha256 0ef10402...7361). Table 3: Cox coefficients and points, with and without ECG;
Table 4: 10-year CVD probability by score band and the 10-year baseline survival (0.99411 / 0.99389).

## Model

Points (Table 3) for sex, 7 age bands (30-79), 4 blood-pressure categories (on medication = top category), high
non-HDL-C or LDL-C, 3 HDL-C bands, smoking, diabetes, urinary protein 1+, and with ECG also atrial fibrillation and
left ventricular hypertrophy. Where SBP and DBP, or LDL-C and non-HDL-C, fall in different categories the higher one
applies. Table 4 maps the total to a printed probability. `cvd_risk.suita_cvd`, `cvd_risk.suita_cvd_risk`.

## Checks (`tests/suita/test_suita_cvd.py`)

- Every score equals 10 x its coefficient, rounded, as the Methods state: 21/21 entries with ECG, 19/19 without.
- Table 4's bands and labels, both models; hand-summed worked cases, including the higher-category rule and treated
  blood pressure.
- **Jcvrisk 0.1.3** (CRAN, Okumiyama and Fujii; vendored unmodified in `reference/vendor/Jcvrisk`, run by
  `reference/jcvrisk/run_jcvrisk.R` on 3,000 seeded patients): `suita_wECG` and `suita_woECG` return the same Table 4
  band for 2,409/2,416 and 2,411/2,416 patients aged 30-79. Every difference has SBP 139 or DBP 89: its cut points
  (`sbp < 139`, `dbp < 89`) leave those values in the lowest category. Partial B: it computes non-HDL-C as TC - HDL-C
  and only warns outside ages 30-79.

## Findings

1. **Table 4 is not a function of S10 alone.** The paper computes each band's probability as the average predicted
   risk of the participants in it. 1 - S10^exp(score / 10) at mid-band (33) gives 14.8% with ECG and 15.3% without,
   against the printed 15% and 14%. The package returns the printed band labels, not a continuous risk.
2. **Supplemental Table 3** (derivation subsample, n = 4,585) prints different coefficients and points (e.g. atrial
   fibrillation 1.12, 11 points, against 0.83, 8); the package uses Table 3 (full cohort), which the paper presents as the model.
3. **Model without CKD.** Unlike the 2014 Suita (CHD) score, this model has no CKD item (the paper explains why).
