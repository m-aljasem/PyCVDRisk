# Suita score (2014): verification report

Functions: `cvd_risk.suita.suita` (points), `suita_risk` (the sheet's printed 10-year CHD probability). Checks:
`tests/suita/test_suita.py`.

## 1. Sources

- **Paper (evidence C).** Nishimura K, Okamura T, Watanabe M, et al. Predicting coronary heart disease using risk
  factor categories for a Japanese urban population, and comparison with the Framingham risk score: the Suita
  study. J Atheroscler Thromb 2014;21(8):784-798 (open access, J-STAGE; `source/nishimura2014.pdf`). Table 3
  (TC model) and Table 4 (LDL model) give the Cox coefficients, with and without CKD; Table 5 gives the prediction
  score sheets (TC and LDL, both with CKD) and their probability tables.
- **No developers' calculator for the 2014 sheets was found.** The Japan Atherosclerosis Society's web tool
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
