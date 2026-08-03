# Suita score (2014): verification report

Functions: `cvd_risk.suita.suita` (points), `suita_risk` (the sheet's printed 10-year CHD probability). Checks:
`tests/suita/test_suita.py`.

## 1. Sources

- **Paper (evidence C).** Nishimura K, Okamura T, Watanabe M, et al. Predicting coronary heart disease using risk
  factor categories for a Japanese urban population, and comparison with the Framingham risk score: the Suita
  study. J Atheroscler Thromb 2014;21(8):784-798 (open access, J-STAGE; `source/nishimura2014.pdf`). Table 3
  (TC model) and Table 4 (LDL model) give the Cox coefficients, with and without CKD; Table 5 gives the prediction
  score sheets (TC and LDL, both with CKD) and their probability tables.
- **Developers' chart (evidence A, LDL sheet only).** NCVC press release "冠動脈疾患を予測する新しいリスクスコアの開発"
  (Development of a new risk score predicting coronary heart disease; 2014-05-07, updated 2014-05-09), Table 1
  "吹田スコア": the LDL sheet with CKD and its probability table, issued by the developers' institution (Nishimura's
  division). Page http://www.ncvc.go.jp/pr/release/006484.html (Wayback 20140525195300); image
  http://www.ncvc.go.jp/20140507_press_h1.jpg (Wayback 20140526013703), `source/ncvc_20140507_press_h1.jpg`, sha256
  1f12d261caa6c90debc5492b98b8f7d21989f59e82e3cee423fe74a4d281ef7e; transcribed in
  `fixtures/ncvc_2014_press_sheet.json`. The release has no TC sheet. (Section 6.)
- **No developers' calculator for the 2014 sheets was found** (re-searched: NCVC site and press
  releases via Wayback, Japanese and English web search). The Japan Atherosclerosis Society's web tool
  ("これりすくん", archived at j-athero.org/general/ge_app/general2) implements the 2017 guideline's adaptation
  (ages 35-74, glucose intolerance and family history, CKD moved to the high-risk conditions), a different score;
  the current tool follows the 2022 guideline (Hisayama-based).

## 2. What can and cannot be reproduced

The paper computes risk as P = 1 - S(t)^exp(f), with f the Cox linear predictor centred on the cohort means, and
for the sheets f = points/10. S(t) and the means are not published, so the Cox models cannot be evaluated. The
score sheets can, and they are what the package implements.

## 3. Checks against the paper

- **Points = 10 x beta.** The paper says categorical points are 10 x beta rounded and age points 10 x beta x band
  midpoint. The LDL sheet (Table 5b) matches Table 4a for every entry, age included (midpoints 39.5, 49.5, 59.5, 67).
  The TC sheet (Table 5a) matches Table 3a except three entries (findings 1-3).
- **One baseline survival.** With P = 1 - S^exp(points/10), a single S reproduces every bounded row of each
  probability table at the row's middle score, rounded to whole percent: S in [0.999490, 0.999502] for the TC
  sheet and [0.999718, 0.999730] for the LDL sheet. At those S the open rows also agree (TC ">= 61: 24%" is the
  value at 63; LDL ">= 71: > 28%" holds from 71 on).

## 4. Findings

1. **TC sheet, HDL >= 60 mg/dL: -5 points**, where 10 x beta (-0.4322771) gives -4. The sheet repeats the 50-59 value.
2. **TC sheet, CKD stage 4-5: 15 points**, where 10 x beta (1.315004) gives 13. The LDL sheet's 14 matches its own
   beta (1.388216).
3. **TC sheet age bands.** The sheet labels them 36-45, 46-55, 56-65, > 65, but the first band's 30 points are
   10 x beta x 39.5 (the 35-44 midpoint); 40.5 would give 31. The other bands fit the printed labels. The package
   follows the printed labels (36-45 ...); a 35-year-old gets NaN on the TC sheet.
4. **Gaps between bands.** TC sheet: HDL "< 35" and "36-49" leave 35 uncovered (Table 3: 35-50 is the reference),
   TC "240-279" and "> 280" leave 280 (Table 3: "over 280"); the package puts 35 and 280 with the higher band.
   The paper's CKD stage 3 is "eGFR 30-60" and the sheets' no-CKD row is "eGFR > 60"; the package uses K/DOQI
   (stage 3 = 30-59), which the paper cites.
5. **Upper age.** The sheets' top age bands are open-ended; the cohort was aged 30-79, so the package gives NaN
   from 80.

## 5. Web calculators (JAS 2017 adaptation)

Two independent web calculators, calculator.jp/health/suita and dataclock.jp/calculation-of-suita-score (`fixtures/jas2017_web_calculators.json`), implement the Japan Atherosclerosis Society 2017 guideline's
version of the LDL sheet: diabetes (6) becomes glucose intolerance (5), CKD is dropped, and family history of
premature CHD (+5) is added. On every item they share with Table 5b (age 5 bands, sex, smoking, blood pressure
5 categories, HDL 3, LDL 5) and on the 9 probability rows their values equal the package's
(`tests/suita/test_suita.py`). They are third-party implementations of a different variant, so this is a partial
level-B check of the LDL sheet, not of its diabetes or CKD rows. The JAS 2017 variant itself is not implemented: its
primary source (Kinoshita M et al. J Atheroscler Thromb 2018;25:846-984) is not in `source/`.

## 6. Developers' chart (NCVC press release, May 2014)

The NCVC sheet (`fixtures/ncvc_2014_press_sheet.json`) equals the package's LDL sheet on all 28 point entries
(age 5, sex, smoking, diabetes, blood pressure 4 rows, LDL 5, HDL 3, CKD 3) and all 9 probability rows
(`test_ncvc_2014_press_release_sheet_equals_the_ldl_sheet`). Two label differences, no value differences:

- Blood pressure has four rows; "正常血圧" (normal, 0) stands for the paper's "normal and high normal" (0).
- CKD stage 3 is labelled "eGFR 30-59" and stage 4-5 "eGFR < 30", which settles finding 4's stage-3 band for the
  LDL sheet in favour of the K/DOQI cut-points the package uses (eGFR 60 is no CKD).

This is a reprint of Table 5b by the developers, not a calculator: it confirms the published sheet is the one the
developers intended and fixes the eGFR band, but it adds no computation beyond the sheet. The TC sheet stays at
evidence C, and its three departures from Table 3a (findings 1-3) are unresolved.

## 7. Author request

`make_author_test_patients.py` writes `fixtures/author_test_patients.csv` (20 patients, inputs only; S01-S06 target
the TC-sheet entries in question, S07-S20 seeded 2014) and `fixtures/author_test_expected.csv` (our points and
printed probabilities on both sheets, kept back).
