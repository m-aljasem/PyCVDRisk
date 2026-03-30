# JALS score (2019): verification report

Evidence level C: the paper's points, rate ratios and risk table. The R package Jcvrisk implements it with several
departures from the paper (below), so it is not counted.

## Sources

- **Paper**: Harada A, Ueshima H, Kinoshita Y, Miura K, Ohkubo T, Asayama K, Ohashi Y; Japan Arteriosclerosis
  Longitudinal Study Group. Absolute risk score for stroke, myocardial infarction, and all cardiovascular disease:
  Japan Arteriosclerosis Longitudinal Study. Hypertens Res 2019;42(4):567-79. doi:10.1038/s41440-019-0220-z
  (`source/harada2019.pdf`, nature.com, subscription article; sha256 c686dcec...4b6). Tables 2-5: incidence rate ratios and points for
  stroke, AMI, stroke + AMI and CVD death, each for a model without and with atrial fibrillation (AF); Table 6: 5-
  and 10-year absolute risk by score (0, 5, ..., 60) and age group, AF models. Text extraction drops cells in Tables
  3-5; the tables were read from the page images. The online supplement (model variants for the ACC/AHA and
  ESC/ESH blood-pressure thresholds) was not available.
- **R package Jcvrisk 0.1.3**: `jals_*_wECG()` / `jals_*_woECG()` (`reference/vendor/Jcvrisk`).

## Model

Points = 10 x log2(IRR) for sex, age (40-49 ... 80-89), JSH 2014 blood-pressure category (the higher of SBP and DBP)
scored separately with and without antihypertensive medication, diabetes, current smoking, and per outcome HDL-C
and non-HDL-C (AMI), HDL-C (composite), BMI, HDL-C and eGFR (CVD death), and AF in the AF model. Table 6 "score"
= total minus age points. 5-year risk = 1 - exp(-(5 x 1-year rate at score 0) x 1.0718^score), 1.0718 = 2^0.1;
10-year = 1 - (1 - first 5 years) x (1 - next 5 years at the next age group). `cvd_risk.jals`, `cvd_risk.jals_risk`.

## Recovering the risk constants (`fit_table6.py`)

The paper does not print the score-0 rates. Table 6 (`fixtures/table6.csv`, 260 cells x 2 horizons) fits one
constant per outcome and age group (`fixtures/table6_constants.json`): 257/260 printed 5-year and 235/260 10-year
values equal, every other value within 0.01 point. The per-point factor is 2^0.1 (fitting it free gives 1.07172-
1.07178 in the stroke, composite and CVD columns). The age effect is not the rounded age points: one constant per
outcome scaled by 2^(age points / 10) misses by up to 0.8 point, one scaled by the printed age IRRs by up to 0.04;
the fitted constants run in the ratio of the AF model's age IRRs.

## Checks (`tests/jals/test_jals.py`)

- Points = 10 x log2(IRR), rounded: 174/176 non-reference entries from the printed IRR, the other 2 from an IRR
  within its 2-decimal rounding (stroke AF model, treated optimal BP, 2.22 -> 11.51, printed 11; composite AF model,
  normal BP, 1.57 -> 6.51, printed 6).
- Table 6 as above; the worked example (man 75, grade II hypertension untreated: stroke score 25, total 48; 5-year
  3.08%, 10-year 8.07%); the higher-category blood-pressure rule.

## Findings

1. **No risk table for the models without AF.** Table 6 covers the AF models only; `jals_risk` applies to AF-model
   scores (give af=0 when AF is absent).
2. **Jcvrisk departs from the paper** (3,000 seeded patients, `reference/jcvrisk`; its scores equal ours for 32-70%
   of patients, by model):
   - blood pressure: "normal" is SBP 120-129 with DBP < 80 only, so DBP 80-84 falls in high-normal, and when SBP
     and DBP disagree the lower category is taken (its nested conditions test the lower categories first). Several
     category points differ, e.g. in the AF models: stroke untreated normal 8 (paper 7), treated optimal, normal,
     high-normal 12, 13, 14 (paper 11, 14, 13); AMI untreated normal and high-normal 4, 4 (paper 3, 6), treated 15, 16
     (paper 14, 17); CVD death untreated high-normal 3 (paper 4), treated normal and high-normal 7, 5 (paper 8, 4);
   - other points: AMI AF model smoking 11 and AF 12 (paper 10 and 13) and an eGFR term (3, 3, 0, 0) the AMI
     model does not have; CVD death AF model eGFR >= 90 scores 1 (paper 2); CVD death model without AF, men 6
     (paper 7);
   - its per-score risk maps equal Table 6 at 39, 23, 50 and 25 of the 65 printed cells (stroke, AMI, composite, CVD
     death), and it applies separate maps, of unstated origin, to the models without AF.
