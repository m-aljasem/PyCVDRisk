# JALS score (2019): verification report

Evidence level C + B. C: the paper's points, rate ratios and risk table (main-text model, JSH 2014 blood-pressure
categories, the package default). B: the online supplement refits the same models with the ACC/AHA 2017 categories
(`variant="acc_aha"`); the R package Jcvrisk turns out to implement that supplement, not the main text, and equals
our supplement variant for every one of 3,000 seeded patients in all 8 models once its blood-pressure rule is
used (below). No calculator, app or chart from the JALS group was found, so there is no level A.

## Sources

- **Paper**: Harada A, Ueshima H, Kinoshita Y, Miura K, Ohkubo T, Asayama K, Ohashi Y; Japan Arteriosclerosis
  Longitudinal Study Group. Absolute risk score for stroke, myocardial infarction, and all cardiovascular disease:
  Japan Arteriosclerosis Longitudinal Study. Hypertens Res 2019;42(4):567-79. doi:10.1038/s41440-019-0220-z
  (`source/harada2019.pdf`, nature.com, subscription article; sha256 c686dcec...4b6). Tables 2-5: incidence rate ratios and points for
  stroke, AMI, stroke + AMI and CVD death, each for a model without and with atrial fibrillation (AF); Table 6: 5-
  and 10-year absolute risk by score (0, 5, ..., 60) and age group, AF models. Text extraction drops cells in Tables
  3-5; the tables were read from the page images. - **Online supplement** (static-content.springer.com, `.../41440_2019_220_MOESM{1..4}_ESM.pdf`, open; `source/supp1-4.pdf`, text in `supp*.txt`): MOESM1 contents page (sha256 2eec8ffa...7d9), MOESM2 the
  study group (7d43e337...85), MOESM3 Supplementary Tables 1-5 (2db0acae...5e5), MOESM4 the flow diagram
  (f1323eeb...4ff). Tables 1-4: IRRs and points for the ACC/AHA blood-pressure rows (optimal < 120 and < 80,
  120-129 and < 80, 130-139 / 80-89, 140-159 / 90-99, 160-179 / 100-109, >= 180 / >= 110); the AMI AF model adds
  eGFR; several other points differ from the main text by 1-2. Table 5: the 5- and 10-year risk table of this
  variant (AF models), `fixtures/supp_table5.csv`; it differs from main Table 6 in 172 of 260 cells (mostly by 0.01).
- **R package Jcvrisk 0.1.3** (Okumiyama H, Fujii R; not the JALS group): `jals_*_wECG()` / `jals_*_woECG()`
  (`reference/vendor/Jcvrisk`), run unmodified on 3,000 seeded patients in `reference/jcvrisk`.
- **Search for a developer tool**: web searches in English and Japanese; the MHLW research report
  20FA1002 (2021) lists Japanese prediction tools (Suita score, Suita AF and CVD scores, JAS app) and names JALS
  only as a published model; the Japan Atherosclerosis Society's tools implement Suita and Hisayama. No JALS
  calculator, app, spreadsheet or guideline chart was found.

## Model

Points = 10 x log2(IRR) for sex, age (40-49 ... 80-89), JSH 2014 blood-pressure category (the higher of SBP and DBP)
scored separately with and without antihypertensive medication, diabetes, current smoking, and per outcome HDL-C
and non-HDL-C (AMI), HDL-C (composite), BMI, HDL-C and eGFR (CVD death), and AF in the AF model. Table 6 "score"
= total minus age points. 5-year risk = 1 - exp(-(5 x 1-year rate at score 0) x 1.0718^score), 1.0718 = 2^0.1;
10-year = 1 - (1 - first 5 years) x (1 - next 5 years at the next age group). `cvd_risk.jals`, `cvd_risk.jals_risk`;
`variant="acc_aha"` uses Supplementary Tables 1-5 instead (same structure, the higher of the SBP and DBP row).

## Recovering the risk constants (`fit_table6.py`)

The paper does not print the score-0 rates. Table 6 (`fixtures/table6.csv`, 260 cells x 2 horizons) fits one
constant per outcome and age group (`fixtures/table6_constants.json`): 257/260 printed 5-year and 235/260 10-year
values equal, every other value within 0.01 point. The per-point factor is 2^0.1 (fitting it free gives 1.07172-
1.07178 in the stroke, composite and CVD columns). The age effect is not the rounded age points: one constant per
outcome scaled by 2^(age points / 10) misses by up to 0.8 point, one scaled by the printed age IRRs by up to 0.04;
the fitted constants run in the ratio of the AF model's age IRRs. Supplementary Table 5 (`fixtures/supp_table5.csv`,
`fixtures/supp_table5_constants.json`, same script): 260/260 printed 5-year and 242/260 10-year values equal, the rest
within 0.01.

## Checks (`tests/jals/test_jals.py`)

- Points = 10 x log2(IRR), rounded: 174/176 non-reference entries from the printed IRR, the other 2 from an IRR
  within its 2-decimal rounding (stroke AF model, treated optimal BP, 2.22 -> 11.51, printed 11; composite AF model,
  normal BP, 1.57 -> 6.51, printed 6).
- Table 6 as above; the worked example (man 75, grade II hypertension untreated: stroke score 25, total 48; 5-year
  3.08%, 10-year 8.07%); the higher-category blood-pressure rule.
- Supplement: Supplementary Table 5 as above; its worked example (same man: score 25, 3.08%, 10-year 8.06%); the
  ACC/AHA rows.
- Jcvrisk (`compare_jcvrisk.py`, 3,000 patients aged 40-89, all 8 functions): scores equal our supplement variant
  for all 2,107 patients whose SBP and DBP give the same row under both blood-pressure rules, and for 3,000/3,000
  when Jcvrisk's row is used (finding 2); 5-year risk from its score (AF models) equal for 2,902, 2,938, 2,799 and
  2,932 of 3,000 (stroke, AMI, composite, CVD death), every other within 0.01. Its per-score maps equal
  Supplementary Table 5 at all 260 printed cells, and 1,182 of 1,220 cells (scores 0-60) equal 1 - exp(-c 2^(s/10))
  with our constants, the rest within 0.01. Against the main-text model the same scores agree for 32-70%.

## Findings

1. **No risk table for the models without AF.** Table 6 covers the AF models only; `jals_risk` applies to AF-model
   scores (give af=0 when AF is absent).
2. **Jcvrisk implements the online supplement, not the main text.** Every point that earlier looked like a
   departure from Tables 2-5 (blood-pressure rows, AMI smoking 11, AF 12 and its eGFR term, CVD death eGFR >= 90
   = 1, men 6, ...) is the supplement's value, and its risk maps are Supplementary Table 5 (it does not say which
   tables it follows). Classification: not an error, a different published variant; we now offer both.
3. **Blood-pressure rule when SBP and DBP fall in different rows (ambiguity; Jcvrisk departs).** The supplement's
   rows read "130-139 and/or 80-89" etc.; the main text says "whichever was greater", which we apply to both
   variants. Jcvrisk's nested `ifelse` takes the first row that matches from the bottom, so e.g. SBP 170 with DBP 85
   scores as 130-139 / 80-89; this changes the row for 893 of 3,000 patients (and the score for 733-843 of them,
   by model). Asked of the authors.
4. **Jcvrisk's risk maps for the models without AF** follow the same law (1 - exp(-c 2^(score/10)), per-point
   slope 2^0.1) with constants of their own, but are printed in neither the article nor the supplement. We give no
   risk for those models; asked of the authors.
5. **Main text vs supplement.** Table 6 and Supplementary Table 5 differ in 172 of 260 cells (by up to 0.2 point in
   the 5-year and 0.4 in the 10-year values); the worked example's 10-year stroke risk is 8.07% in one and 8.06% in
   the other.

## Author request

`make_author_test_patients.py` writes `fixtures/author_test_patients.csv` (20 patients: the worked example, 4 with
SBP and DBP in different rows, 15 seeded) and `fixtures/author_test_expected.csv` (our scores for the 8 models and
the AF models' 5- and 10-year risks, both variants).
