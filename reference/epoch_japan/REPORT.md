# EPOCH-JAPAN (2021): verification report

Evidence level C + B: the paper's two worked examples, and the R package Jcvrisk on every patient for stroke and CVD
death (its CHD function has an input error, below).

## Sources

- **Paper**: Li Y, Yatsuya H, Tanaka S, Iso H, Okayama A, Tsuji I, Sakata K, Miyamoto Y, Ueshima H, Miura K,
  Murakami Y, Okamura T; EPOCH-JAPAN Research Group. Estimation of 10-year risk of death from coronary heart
  disease, stroke, and cardiovascular disease in a pooled analysis of Japanese cohorts: EPOCH-JAPAN. J Atheroscler
  Thromb 2021;28(8):816-25. doi:10.5551/jat.58958 (`source/li2021.pdf`, J-STAGE, CC BY-NC-SA; sha256
  47ea1a81...2bff; `source/li2021.txt` extracted with Ghostscript). Table 2: coefficients (5 decimals) and S10;
  Supplementary Table 2: the formula, the means used and two worked examples.
- **R package Jcvrisk 0.1.3**: `epoch_chd()`, `epoch_stroke()`, `epoch_cvd()` (`reference/vendor/Jcvrisk`).

## Model

risk = 1 - S10^exp(sum(beta x) - sum(beta xbar)); x = ln age, ln SBP, ln(TC/HDL-C) (CHD), current smoking, diabetes,
proteinuria, sex (CHD and CVD), ln age x ln SBP, ln age x smoking. S10 0.9981 (CHD), 0.9961 (stroke), 0.9942 (CVD).
The mean term follows Supplementary Table 2: ln 57.0, ln 131.1, ln 4.1 and proportions 0.480 men, 0.267 smokers,
0.046 diabetes, 0.028 proteinuria. Outcome: death. `cvd_risk.epoch_japan`.

## Checks (`tests/epoch_japan/test_epoch_japan.py`)

- Example 1 (CHD): mean sum 265.45, individual 267.24, risk 1.13%; Example 2 (stroke): 163.44, 165.43, 2.82%. All
  equal.
- Jcvrisk on 2,416 seeded patients aged 40-79 (`reference/jcvrisk`): stroke and CVD equal to 1e-12.

## Findings

1. **Jcvrisk `epoch_chd()` enters ln(TC) where the model has ln(TC/HDL-C)** (it computes the ratio but does not use
   it; the mean term uses ln 4.1, the mean ratio). It differs from the paper's model for every patient; with ln TC
   substituted in ours the two agree to 1e-12.
2. Table 1 gives 47.9% men; Supplementary Table 2 uses 0.480 in the mean term. The package follows Supplementary
   Table 2, which reproduces the examples.
3. The mean term uses ln(mean), not mean(ln), and 0.267 x ln 57 for the interaction, as the examples do.

## Search for a developer tool

No level-A source found. Searched (English and Japanese) for an EPOCH-JAPAN web calculator, app, risk chart or
spreadsheet from the group, JAS, JCS or NCVC: none exists. The paper (Discussion) only proposes the models for
"education and risk communication tools"; its supplement holds no spreadsheet; the JAS 2022 guideline uses the
Hisayama model, not EPOCH-JAPAN. The MHLW grant reports of the group (mhlw-grants.niph.go.jp, 201909001A0005,
201909001A0006, 201909001B0005) describe validation and recalibration of the death models, with no tool or
risk table. Jcvrisk remains the only implementation.

## Author test set

`make_author_test_patients.py` writes `fixtures/author_test_patients.csv` (20 patients: the two Supplementary
Table 2 examples, then 18 seeded with seed 2021, ages 40-79, mg/dL) and `fixtures/author_test_expected.csv` (our
CHD, stroke and CVD 10-year risks in %, held back).
